import re
from typing import Dict, Any, Optional, Callable


class MetricStreamParser:
    """Parses streaming stdout lines from server and client processes to extract metric events."""

    def __init__(self, on_metric_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.on_metric_callback = on_metric_callback
        self.current_round = 0
        self.total_rounds = 0

        # Regex patterns for Server
        self.re_server_round = re.compile(r'(?:Starting\s+Round|Round|ROUND)\s*\[?(\d+)[/\s]*(\d+)?\]?', re.IGNORECASE)
        self.re_server_metrics = re.compile(
            r'(?:Loss|test_loss)\s*[:=]\s*([\d\.]+)'
            r'|(?:\bBTA\b)\s*[:=]\s*([\d\.]+)%?'
            r'|(?:\bBMCTA\b)\s*[:=]\s*([\d\.]+)%?'
            r'|(?:\bTA\b)\s*[:=]\s*([\d\.]+)%?'
            r'|(?:\bMCTA\b)\s*[:=]\s*([\d\.]+)%?',
            re.IGNORECASE
        )

        # Regex patterns for Client
        self.re_client_epoch = re.compile(
            r'Epoch\s*\[?(\d+)/(\d+)\]?\s*(?:LR=[\d\.]+)?\s*\|\s*Loss=([\d\.]+)\s*Acc=([\d\.]+)',
            re.IGNORECASE
        )
        self.re_client_fit_done = re.compile(
            r'\[Client\s+(\d+)\]\s+fit\s+done\s*\|\s*Round\s+(\d+).*?loss=([\d\.]+)\s+acc=([\d\.]+)',
            re.IGNORECASE
        )
        self.re_client_evaluate_done = re.compile(
            r'\[Client\s+(\d+)\]\s+evaluate\s*\|\s*loss=([\d\.]+)\s+acc=([\d\.]+)',
            re.IGNORECASE
        )
        self.re_client_local_test = re.compile(
            r'Local\s+Test.*?(?:Loss\s*[:=]\s*([\d\.]+))?.*?(?:Acc(?:uracy)?|TA)\s*[:=]\s*([\d\.]+)%?.*?(?:BTA\s*[:=]\s*([\d\.]+)%?)?',
            re.IGNORECASE
        )

    def parse_line(self, line: str, source: str = "server"):
        """Parse a single output line and trigger metric updates if data found."""
        clean = line.strip()
        if not clean:
            return

        if source == "server":
            self._parse_server_line(clean)
        else:
            self._parse_client_line(clean)

    def _parse_server_line(self, line: str):
        # Check for final client bundle
        if "[CLIENT_FINAL_METRICS]" in line:
            try:
                import json
                idx = line.find("[CLIENT_FINAL_METRICS]") + len("[CLIENT_FINAL_METRICS]")
                payload = json.loads(line[idx:].strip())
                if self.on_metric_callback:
                    self.on_metric_callback("server_client_bundle", payload)
                return
            except Exception:
                pass

        # Check for round start / progress
        m_round = self.re_server_round.search(line)
        if m_round:
            r = int(m_round.group(1))
            self.current_round = r
            if m_round.group(2):
                self.total_rounds = int(m_round.group(2))

        # Check for metrics
        metrics = {}
        # Explicit key-value search
        patterns = {
            "loss": r'(?:Global\s+Test\s+Loss|test_loss|Loss)\s*[:=]\s*([\d\.]+)',
            "bta": r'\bBTA\s*[:=]\s*([\d\.]+)%?',
            "bmcta": r'\bBMCTA\s*[:=]\s*([\d\.]+)%?',
            "ta": r'\bTA\s*[:=]\s*([\d\.]+)%?',
            "mcta": r'\bMCTA\s*[:=]\s*([\d\.]+)%?',
        }
        for k, pat in patterns.items():
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if val > 1.0 and k in ("bta", "bmcta", "ta", "mcta"):
                    val = val / 100.0  # normalize percentage e.g. 74.2% -> 0.742
                metrics[k] = val

        if metrics:
            payload = {
                "round": self.current_round,
                "total_rounds": self.total_rounds,
                **metrics,
            }
            if self.on_metric_callback:
                self.on_metric_callback("server", payload)

    def _parse_client_line(self, line: str):
        # 1. Check for client fit completion per round
        m_fit = self.re_client_fit_done.search(line)
        if m_fit:
            payload = {
                "client_id": int(m_fit.group(1)),
                "round": int(m_fit.group(2)),
                "train_loss": float(m_fit.group(3)),
                "train_acc": float(m_fit.group(4)),
            }
            if self.on_metric_callback:
                self.on_metric_callback("client_round_done", payload)
            return

        # 2. Check for evaluate done per round
        m_eval_done = self.re_client_evaluate_done.search(line)
        if m_eval_done:
            payload = {
                "client_id": int(m_eval_done.group(1)),
                "loss": float(m_eval_done.group(2)),
                "accuracy": float(m_eval_done.group(3)),
            }
            if self.on_metric_callback:
                self.on_metric_callback("client_eval_done", payload)
            return

        # 3. Check for epoch training progression (within current round)
        m_epoch = self.re_client_epoch.search(line)
        if m_epoch:
            ep = int(m_epoch.group(1))
            payload = {
                "epoch": ep,
                "total_epochs": int(m_epoch.group(2)),
                "train_loss": float(m_epoch.group(3)),
                "train_acc": float(m_epoch.group(4)),
                "is_new_round": bool(ep == 1),
            }
            if self.on_metric_callback:
                self.on_metric_callback("client_epoch", payload)
            return

        # 4. Check for local test evaluation (legacy or fallback)
        m_eval = self.re_client_local_test.search(line)
        if m_eval:
            metrics = {}
            if m_eval.group(1):
                metrics["test_loss"] = float(m_eval.group(1))
            if m_eval.group(2):
                val = float(m_eval.group(2))
                metrics["ta"] = val / 100.0 if val > 1.0 else val
            if m_eval.group(3):
                val = float(m_eval.group(3))
                metrics["bta"] = val / 100.0 if val > 1.0 else val
            if metrics and self.on_metric_callback:
                self.on_metric_callback("client_eval", metrics)
