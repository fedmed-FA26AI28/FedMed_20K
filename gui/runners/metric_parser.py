import re
from typing import Dict, Any, Optional, Callable


class MetricStreamParser:
    """Parses streaming stdout lines from server and client processes to extract metric events."""

    def __init__(self, on_metric_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.on_metric_callback = on_metric_callback
        self.current_round = 0
        self.total_rounds = 0

        # Regex patterns for Server
        self.re_sim_matrix = re.compile(
            r'>>>\s*\[Simulation\s+Matrix\s+(\d+)/(\d+)\]\s*Starting:\s*Dataset=([^\s|]+)\s*\|\s*Partition=([^\s|]+)\s*\|\s*Strategy=([^\s|]+)',
            re.IGNORECASE,
        )
        self.re_server_round = re.compile(
            r'(?:Starting\s+Round|(?:\[|\b)(?:Round|ROUND))\s*\[?(\d+)(?:\s*(?:/|\bof\b)\s*(\d+))?\]?(?!\s*:\s*[-+]?\d+(?:\.\d+)?(?:\s*$|\s*[,)]))',
            re.IGNORECASE,
        )
        self.re_server_ignore = re.compile(
            r'\[SUMMARY\]'
            r'|\bHistory\s*\('
            r'|\b(?:losses_distributed|losses_centralized|metrics_distributed|metrics_centralized|app_fit)\b'
            r'|(?:^|\s)round\s+\d+\s*:\s*[-+]?\d+(?:\.\d+)?(?:\s*$|\s*[,)])'
            r'|\b(?:Best\s+Metric|BMCTA\s+Loss|Best\s+Eval\s+Loss|Final\s+Eval\s+Loss|Final\s+Accuracy|Final\s+Loss|Final\s+Summary|Evaluation\s+Summary|Training\s+Summary|Run\s+finished)\b'
            r'|\[SERVER\s+(?:LAUNCH|TERMINATED)\]',
            re.IGNORECASE,
        )
        self.re_server_metrics = re.compile(
            r'(?:Global\s+Test\s+Loss|Global\s+Loss|test_loss|(?<!BMCTA\s)(?<!Eval\s)(?<!Final\s)(?<!Best\s)\bLoss)\s*[:=]\s*([\d\.]+)'
            r'|(?:\b(?:BTA|Best))\s*[:=]\s*([\d\.]+)\s*%?'
            r'|(?:\bBMCTA)\s*[:=]\s*([\d\.]+)\s*%?'
            r'|(?:Global\s+Test\s+Acc(?:uracy)?|\bCombined|\bAccuracy|\bTA)\s*[:=]\s*([\d\.]+)\s*%?'
            r'|(?:\bMCTA)\s*[:=]\s*([\d\.]+)\s*%?',
            re.IGNORECASE
        )

        # Regex patterns for Client
        self.re_client_epoch = re.compile(
            r'Epoch\s*\[?(\d+)/(\d+)\]?\s*(?:LR=[\d\.]+)?\s*\|\s*Loss=([\d\.]+)\s*Acc=([\d\.]+)',
            re.IGNORECASE
        )
        self.re_client_fit_done = re.compile(
            r'\[Client\s+(\d+)\]\s+fit\s+done\s*\|\s*Round\s+(\d+).*?loss=([\d\.]+)\s+acc=([\d\.]+)'
            r'(?:.*?round_time=([\d\.]+)s)?(?:.*?ping=([-\d\.]+)ms)?',
            re.IGNORECASE,
        )
        self.re_client_evaluate_done = re.compile(
            r'\[Client\s+(\d+)\]\s+evaluate\s*\|\s*(?:Round\s+(\d+)\s*\|\s*)?loss=([\d\.]+)\s+acc=([\d\.]+)'
            r'(?:.*?ping=([-\d\.]+)ms)?',
            re.IGNORECASE,
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
        # Check for Simulation Matrix transition between sub-runs
        m_sim = self.re_sim_matrix.search(line)
        if m_sim:
            self.current_round = 0
            if self.on_metric_callback:
                self.on_metric_callback(
                    "simulation_matrix_step",
                    {
                        "run_idx": int(m_sim.group(1)),
                        "total_runs": int(m_sim.group(2)),
                        "dataset": m_sim.group(3).strip(),
                        "partition": m_sim.group(4).strip(),
                        "strategy": m_sim.group(5).strip(),
                    },
                )
            return

        # Check for real-time per-round client metrics
        if "[CLIENT_ROUND_METRICS]" in line:
            try:
                import json
                idx = line.find("[CLIENT_ROUND_METRICS]") + len("[CLIENT_ROUND_METRICS]")
                payload = json.loads(line[idx:].strip())
                if self.on_metric_callback:
                    self.on_metric_callback("server_client_round", payload)
                return
            except Exception:
                pass

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

        # Ignore post-run summaries, banners, and Flower history dumps
        if not line or not line.strip() or self.re_server_ignore.search(line):
            return

        # Check for round start / progress
        m_round = self.re_server_round.search(line)
        if m_round:
            r = int(m_round.group(1))
            self.current_round = r
            if m_round.group(2):
                self.total_rounds = int(m_round.group(2))

        # Ignore initial/Round-0 evaluations before Round 1 starts
        if self.current_round < 1:
            return

        # Check for metrics (ordered cleanly to avoid summary/BMCTA-loss collisions)
        metrics = {}
        patterns = {
            "loss": r'(?:Global\s+Test\s+Loss|Global\s+Loss|test_loss|(?<!BMCTA\s)(?<!Eval\s)(?<!Final\s)(?<!Best\s)\bLoss)\s*[:=]\s*([\d\.]+)',
            "bta": r'\b(?:BTA|Best)\s*[:=]\s*([\d\.]+)\s*(%?)',
            "bmcta": r'\bBMCTA\s*[:=]\s*([\d\.]+)\s*(%?)',
            "ta": r'(?:Global\s+Test\s+Acc(?:uracy)?|\bCombined|\bAccuracy|\bTA)\s*[:=]\s*([\d\.]+)\s*(%?)',
            "mcta": r'\bMCTA\s*[:=]\s*([\d\.]+)\s*(%?)',
        }
        for k, pat in patterns.items():
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                has_pct = m.lastindex and m.lastindex >= 2 and bool(m.group(2))
                if k in ("bta", "bmcta", "ta", "mcta") and (has_pct or val > 1.0):
                    val = val / 100.0  # normalize percentage e.g. 74.2% -> 0.742
                metrics[k] = val

        if metrics and self.current_round >= 1:
            payload = {
                "round": self.current_round,
                "total_rounds": self.total_rounds,
                **metrics,
            }
            if self.on_metric_callback:
                self.on_metric_callback("server", payload)

    def _parse_client_line(self, line: str):
        # 0. Structured lifecycle & resume markers emitted on client reconnect
        if "[CLIENT_RESUME_HISTORY]" in line:
            try:
                import json
                idx = line.find("[CLIENT_RESUME_HISTORY]") + len("[CLIENT_RESUME_HISTORY]")
                payload = json.loads(line[idx:].strip())
                if self.on_metric_callback:
                    self.on_metric_callback("client_resume_history", payload)
                return
            except Exception:
                pass

        if "[CLIENT_SESSION_START]" in line:
            try:
                import json
                idx = line.find("[CLIENT_SESSION_START]") + len("[CLIENT_SESSION_START]")
                payload = json.loads(line[idx:].strip())
                if self.on_metric_callback:
                    self.on_metric_callback("client_session_start", payload)
                return
            except Exception:
                pass

        # Ignore checkpoint / best-model status lines so they never emit roundless client_eval events
        if "New Best Local Test" in line or "Checkpoint Saved:" in line:
            return

        # 1. Check for client fit completion per round
        m_fit = self.re_client_fit_done.search(line)
        if m_fit:
            payload = {
                "client_id": int(m_fit.group(1)),
                "round": int(m_fit.group(2)),
                "train_loss": float(m_fit.group(3)),
                "train_acc": float(m_fit.group(4)),
            }
            if m_fit.group(5) is not None:
                payload["round_time"] = float(m_fit.group(5))
            if m_fit.group(6) is not None:
                payload["ping_ms"] = float(m_fit.group(6))
            if self.on_metric_callback:
                self.on_metric_callback("client_round_done", payload)
            return

        # 2. Check for evaluate done per round
        m_eval_done = self.re_client_evaluate_done.search(line)
        if m_eval_done:
            payload = {
                "client_id": int(m_eval_done.group(1)),
                "loss": float(m_eval_done.group(3)),
                "accuracy": float(m_eval_done.group(4)),
            }
            if m_eval_done.group(2) is not None:
                payload["round"] = int(m_eval_done.group(2))
            if m_eval_done.group(5) is not None:
                payload["ping_ms"] = float(m_eval_done.group(5))
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
            m_rnd = re.search(r'\bRound\s+(\d+)', line, re.IGNORECASE)
            if m_rnd:
                metrics["round"] = int(m_rnd.group(1))
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
