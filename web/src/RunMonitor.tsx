import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, Download, Pause, Play, Search } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, active, names, pct, type Overview, type Run } from "./api";
import "./monitor.css";

type Row = Record<string, unknown>;
type Evaluation = {
  round: number;
  reporting_clients: number;
  samples: number;
  accuracy: number;
  balanced_accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  matrix: number[][];
  classes: {
    class_id: number;
    label: string;
    support: number;
    precision: number | null;
    recall: number | null;
    f1: number | null;
  }[];
};
type Monitor = {
  run_id: string;
  configuration: Row;
  partition_hash: string | null;
  dataset_hash: string | null;
  event_count: number;
  rounds: Row[];
  client_rounds: Row[];
  epochs: Row[];
  systems: Row[];
  logs: Row[];
  latest: Row[];
  support: Row[];
  evaluations: Evaluation[];
  limits: { epochs_total: number; systems_total: number; logs_total: number };
  summary: {
    current_accuracy: number | null;
    final_accuracy: number | null;
    bta: number | null;
    bta_round: number | null;
    bmcta: number | null;
    completed_rounds: number;
    upload_payload_kb: number | null;
  };
};
const palette = [
  "#0e9387",
  "#3972b8",
  "#8070c8",
  "#c17b19",
  "#c95473",
  "#52734d",
  "#637da9",
  "#9d684a",
  "#426b73",
  "#ad5699",
  "#837b45",
  "#696ac4",
];
const n = (row: Row, key: string): number | null =>
  typeof row[key] === "number" && Number.isFinite(row[key])
    ? (row[key] as number)
    : null;
const fmt = (value: unknown, places = 3) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: places })
    : "—";
const words = (value: unknown) =>
  String(value ?? "Awaiting telemetry").replaceAll("_", " ");

function Plot({
  title,
  caption,
  rows,
  fields,
  x = "round",
  split = false,
  percent = false,
  clientName,
}: {
  title: string;
  caption: string;
  rows: Row[];
  fields: [string, string][];
  x?: string;
  split?: boolean;
  percent?: boolean;
  clientName: (id: number) => string;
}) {
  const map = new Map<number, Row>();
  const series = new Map<string, string>();
  for (const row of rows) {
    const step = n(row, x);
    if (step == null) continue;
    const point = map.get(step) || { [x]: step };
    for (const [field, label] of fields) {
      const value = n(row, field);
      if (value == null) continue;
      const key = split ? `${field}_c${row.client_id}` : field;
      series.set(key, split ? clientName(Number(row.client_id)) : label);
      point[key] = percent ? value * 100 : value;
    }
    map.set(step, point);
  }
  const points = [...map.values()].sort((a, b) => Number(a[x]) - Number(b[x]));
  return (
    <article className="card monitor-plot">
      <div className="monitor-card-title">
        <h3>{title}</h3>
        <p>{caption}</p>
      </div>
      {series.size ? (
        <div className="monitor-chart">
          <ResponsiveContainer>
            <LineChart
              data={points}
              margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey={x}
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v) => fmt(v, 1)}
                tickLine={false}
              />
              <YAxis
                width={58}
                domain={percent ? [0, 100] : ["auto", "auto"]}
                tickFormatter={(v) => `${fmt(v, 2)}${percent ? "%" : ""}`}
                tickLine={false}
              />
              <Tooltip
                labelFormatter={(v) =>
                  `${x === "step" ? "Local epoch step" : x === "elapsed" ? "Elapsed seconds" : "Round"} ${fmt(Number(v), 1)}`
                }
                formatter={(v) => `${fmt(Number(v), 5)}${percent ? "%" : ""}`}
              />
              <Legend />
              {[...series].map(([key, label], i) => (
                <Line
                  key={key}
                  dataKey={key}
                  name={label}
                  stroke={palette[i % palette.length]}
                  strokeWidth={2}
                  dot={points.length < 30 ? { r: 2 } : false}
                  connectNulls={split}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="monitor-empty">No measurements recorded yet.</div>
      )}
    </article>
  );
}

export default function RunMonitor({
  run,
  overview,
}: {
  run: Run;
  overview: Overview;
}) {
  const [tab, setTab] = useState("scalars"),
    [client, setClient] = useState("all"),
    [round, setRound] = useState("latest");
  const [reference, setReference] = useState("");
  const [paused, setPaused] = useState(false),
    [search, setSearch] = useState(""),
    [severity, setSeverity] = useState("all"),
    [follow, setFollow] = useState(true);
  const logRef = useRef<HTMLPreElement>(null);
  const query = useQuery({
    queryKey: ["monitor", run.id, client],
    queryFn: () =>
      api<Monitor>(
        `/runs/${run.id}/monitor${client === "all" ? "" : `?client_id=${client}`}`,
      ),
    refetchInterval: paused ? false : 2000,
  });
  const data = query.data;
  const referenceRun = overview.runs.find((r) => r.id === reference);
  const comparison = useQuery({
    queryKey: ["monitor-comparison", reference],
    queryFn: () => api<Monitor>(`/runs/${reference}/monitor`),
    enabled: !!reference,
    refetchInterval:
      paused || !referenceRun || !active.has(referenceRun.status)
        ? false
        : 2000,
  });
  const differences: string[] = [];
  const compareRows: Row[] = [];
  if (data && comparison.data && referenceRun) {
    for (const key of [
      "dataset",
      "model",
      "partition",
      "seed",
      "rounds",
      "local_epochs",
      "batch_size",
      "learning_rate",
      "momentum",
      "weight_decay",
      "node_ids",
    ]) {
      if (
        JSON.stringify(data.configuration[key]) !==
        JSON.stringify(comparison.data.configuration[key])
      )
        differences.push(words(key));
    }
    if (!data.partition_hash || !comparison.data.partition_hash)
      differences.push("partition identity unknown");
    else if (data.partition_hash !== comparison.data.partition_hash)
      differences.push("partition identity");
    if (!data.dataset_hash || !comparison.data.dataset_hash)
      differences.push("dataset identity unknown");
    else if (data.dataset_hash !== comparison.data.dataset_hash)
      differences.push("dataset identity");
    const merged = new Map<number, Row>();
    for (const [prefix, records] of [
      ["current", data.rounds],
      ["reference", comparison.data.rounds],
    ] as const) {
      for (const row of records) {
        const rnd = Number(row.round),
          point = merged.get(rnd) || { round: rnd };
        for (const field of ["eval_accuracy", "eval_loss", "wall_seconds"])
          point[`${prefix}_${field}`] = n(row, field);
        merged.set(rnd, point);
      }
    }
    compareRows.push(
      ...[...merged.values()].sort((a, b) => Number(a.round) - Number(b.round)),
    );
  }
  const clientName = (cid: number) => {
    if (cid === -1) return "Coordinator";
    const node = Object.entries(run.participants || {}).find(
      ([, p]) => p.client_id === cid,
    )?.[0];
    return (
      overview.nodes.find((n) => n.id === node)?.name ||
      (overview.mode === "institution" ? overview.name : `Client ${cid}`)
    );
  };
  const clientIds =
    overview.mode === "coordinator"
      ? Object.values(run.participants || {}).map((p) => p.client_id)
      : [run.client_id ?? 0];
  const evaluations = data?.evaluations || [];
  const evaluation =
    evaluations.find((e) => String(e.round) === round) || evaluations.at(-1);
  const epochRows = (data?.epochs || []).map((e) => ({
    ...e,
    step: (Number(e.round) - 1) * run.spec.local_epochs + Number(e.epoch),
  }));
  const systemRows = (data?.systems || []).map((e) => ({
    ...e,
    elapsed:
      n(e, "worker_elapsed_seconds") ??
      Number(e.time) - (run.started_at || Number(e.time)),
  }));
  const selectedRound =
    round === "latest"
      ? Math.max(0, ...(data?.client_rounds || []).map((r) => Number(r.round)))
      : Number(round);
  const clients = (data?.client_rounds || []).filter(
    (r) => Number(r.round) === selectedRound,
  );
  const logBlocks = (data?.logs || [])
    .map((e): Row & { clean: string } => ({
      ...e,
      clean: String(e.text || "").replace(/\x1b\[[0-9;]*[A-Za-z]/g, ""),
    }))
    .filter((e) => {
      const text = e.clean.toLowerCase();
      return (
        (!search || text.includes(search.toLowerCase())) &&
        (severity === "all" ||
          (severity === "errors"
            ? /error|exception|traceback|failed/.test(text)
            : /warn/.test(text)))
      );
    });
  const logText = logBlocks
    .map(
      (e) =>
        `[${new Date(Number(e.time) * 1000).toLocaleTimeString()} · ${clientName(Number(e.client_id))} · R${e.round}]\n${e.clean}`,
    )
    .join("\n");
  useEffect(() => {
    if (follow && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logText, follow, tab]);
  const plot = (
    title: string,
    caption: string,
    rows: Row[],
    fields: [string, string][],
    extra: Partial<{ x: string; split: boolean; percent: boolean }> = {},
  ) => <Plot {...{ title, caption, rows, fields, clientName, ...extra }} />;
  return (
    <section className="experiment-monitor" aria-label="Experiment monitor">
      <div className="monitor-header">
        <div>
          <span className="eyebrow">RESEARCH OBSERVABILITY</span>
          <h2>Experiment monitor</h2>
          <p>
            {overview.mode === "coordinator"
              ? "Recorded measurements from the coordinator and participating institutions."
              : "Recorded measurements from this institution only."}
          </p>
        </div>
        <div className="monitor-actions">
          <span
            className={`monitor-live ${active.has(run.status) && !paused ? "on" : ""}`}
          >
            <Activity size={14} />
            {paused
              ? "Updates paused"
              : active.has(run.status)
                ? "Live · 2s refresh"
                : "Saved run"}
          </span>
          <button className="secondary" onClick={() => setPaused(!paused)}>
            {paused ? <Play size={14} /> : <Pause size={14} />}{" "}
            {paused ? "Resume" : "Pause"}
          </button>
          <a
            className="secondary"
            href={`/api/v1/runs/${run.id}/monitor/export?format=csv`}
          >
            <Download size={14} /> CSV
          </a>
          <a
            className="secondary"
            href={`/api/v1/runs/${run.id}/monitor/export?format=json`}
          >
            <Download size={14} /> JSON
          </a>
        </div>
      </div>
      {query.error && (
        <div className="alert error">
          {query.error.message === "Not Found"
            ? "The running service has not loaded the monitoring API yet. Restart it after the active training session finishes."
            : query.error.message}
        </div>
      )}
      {!data ? (
        <div className="monitor-empty">
          {query.error
            ? "Detailed monitoring is currently unavailable."
            : "Loading recorded metrics…"}
        </div>
      ) : (
        <>
          <div className="monitor-stats">
            {[
              [
                "Current test accuracy",
                pct(data.summary.current_accuracy),
                "Latest evaluated round",
              ],
              [
                "Final test accuracy",
                pct(data.summary.final_accuracy),
                "Available after completion",
              ],
              [
                "BTA",
                pct(data.summary.bta),
                data.summary.bta_round
                  ? `Best test round: ${data.summary.bta_round}`
                  : "No evaluated rounds",
              ],
              [
                "BMCTA",
                pct(data.summary.bmcta),
                "Best mean client test accuracy",
              ],
            ].map(([label, value, hint]) => (
              <div key={label}>
                <small>{label}</small>
                <strong>{value}</strong>
                <span>{hint}</span>
              </div>
            ))}
          </div>
          <div className="monitor-toolbar">
            <div
              className="monitor-tabs"
              role="tablist"
              aria-label="Monitoring views"
            >
              {[
                ["scalars", "Learning curves"],
                ["clients", "Client diagnostics"],
                ["classes", "Class performance"],
                ["system", "System & timing"],
                ["logs", "Live logs"],
                ["config", "Configuration"],
              ].map(([key, label]) => (
                <button
                  role="tab"
                  id={`monitor-tab-${key}`}
                  aria-controls="monitor-panel"
                  aria-selected={tab === key}
                  key={key}
                  onClick={() => setTab(key)}
                >
                  {label}
                </button>
              ))}
            </div>
            <label className="monitor-filter">
              Source
              <select
                aria-label="Monitor source"
                value={client}
                onChange={(e) => setClient(e.target.value)}
              >
                <option value="all">All participants</option>
                {overview.mode === "coordinator" && (
                  <option value="-1">Coordinator</option>
                )}
                {clientIds.map((id) => (
                  <option value={id} key={id}>
                    {clientName(id)} (#{id})
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div
            id="monitor-panel"
            role="tabpanel"
            aria-labelledby={`monitor-tab-${tab}`}
          >
            {tab === "scalars" && (
              <>
                <p className="monitor-note">
                  Round curves summarize the session. The source filter selects
                  local epoch diagnostics. Training accuracy is measured during
                  local updates; test accuracy is measured after aggregation.
                  BTA/BMCTA use the test set and remain exploratory.
                </p>
                <div className="monitor-section-head">
                  <div>
                    <h3>Compare with another run</h3>
                    <p>
                      Overlay measured curves and inspect the recorded
                      conditions.
                    </p>
                  </div>
                  <select
                    aria-label="Compare run"
                    value={reference}
                    onChange={(e) => setReference(e.target.value)}
                  >
                    <option value="">No comparison</option>
                    {overview.runs
                      .filter((r) => r.id !== run.id)
                      .map((r) => (
                        <option key={r.id} value={r.id}>
                          {r.name} · {names[r.spec.strategy]}
                        </option>
                      ))}
                  </select>
                </div>
                {comparison.error && (
                  <div className="alert error">{comparison.error.message}</div>
                )}
                {referenceRun && comparison.data && (
                  <>
                    <p className="monitor-note" role="status">
                      {differences.length
                        ? `Conditions differ or are unverified: ${differences.join(", ")}. Treat this as a descriptive comparison.`
                        : "Recorded data, roster, and shared training settings match. Strategy settings may differ; device/runtime conditions still need checking."}
                    </p>
                    <div className="monitor-grid">
                      {plot(
                        "Test accuracy comparison",
                        "Each curve uses its own completed rounds.",
                        compareRows,
                        [
                          ["current_eval_accuracy", run.name],
                          ["reference_eval_accuracy", referenceRun.name],
                        ],
                        { percent: true },
                      )}
                      {plot(
                        "Test loss comparison",
                        "Unweighted evaluation cross-entropy",
                        compareRows,
                        [
                          ["current_eval_loss", run.name],
                          ["reference_eval_loss", referenceRun.name],
                        ],
                      )}
                      {plot(
                        "Round time comparison",
                        "Hardware, workload, and monitoring overhead affect runtime.",
                        compareRows,
                        [
                          ["current_wall_seconds", run.name],
                          ["reference_wall_seconds", referenceRun.name],
                        ],
                      )}
                    </div>
                  </>
                )}
                <div className="monitor-grid">
                  {plot(
                    "Accuracy across rounds",
                    "Sample-weighted local training accuracy and combined local test accuracy",
                    data.rounds,
                    [
                      ["train_accuracy", "Train"],
                      ["eval_accuracy", "Test"],
                      ["mcta", "Mean client test"],
                    ],
                    { percent: true },
                  )}
                  {plot(
                    "Loss across rounds",
                    "Training objective and unweighted evaluation cross-entropy have different meanings for SCR.",
                    data.rounds,
                    [
                      ["train_loss", "Training objective"],
                      ["eval_loss", "Test cross-entropy"],
                    ],
                  )}
                  {plot(
                    "Local training objective",
                    "Each line is one institution. X = (round − 1) × configured epochs + local epoch.",
                    epochRows,
                    [["loss", "Objective"]],
                    { x: "step", split: true },
                  )}
                  {plot(
                    "Learning rate",
                    "Actual optimizer learning rate used in each local epoch",
                    epochRows,
                    [["learning_rate", "Learning rate"]],
                    { x: "step", split: true },
                  )}
                  {plot(
                    "Gradient norm",
                    "L2 norm of the last minibatch gradient in each epoch; not an epoch average.",
                    epochRows,
                    [["last_batch_grad_l2", "Gradient L2"]],
                    { x: "step", split: true },
                  )}
                  {plot(
                    "Model update size",
                    "L2 change from received/local starting state, including floating BatchNorm buffers.",
                    data.client_rounds,
                    [["model_update_l2", "Update L2"]],
                    { split: true },
                  )}
                </div>
                {run.spec.strategy === "scr" && (
                  <div className="monitor-grid">
                    {plot(
                      "Unweighted training cross-entropy",
                      "Computed from raw logits without SCR sample weights.",
                      epochRows,
                      [["raw_ce_loss", "Raw CE"]],
                      { x: "step", split: true },
                    )}
                    {plot(
                      "SCR weighted risk",
                      "Sample-weighted cross-entropy before the head anchor penalty.",
                      epochRows,
                      [["data_objective_loss", "Weighted CE"]],
                      { x: "step", split: true },
                    )}
                    {plot(
                      "SCR head anchor penalty",
                      "Actual additive penalty, averaged across samples seen in the epoch.",
                      epochRows,
                      [["head_penalty_loss", "Head penalty"]],
                      { x: "step", split: true },
                    )}
                  </div>
                )}
                {data.limits.epochs_total > data.epochs.length && (
                  <p className="monitor-note">
                    Showing the latest {data.epochs.length} epoch records of{" "}
                    {data.limits.epochs_total}. Choose a source or export all
                    records.
                  </p>
                )}
              </>
            )}
            {tab === "clients" && (
              <>
                <div className="monitor-section-head">
                  <div>
                    <h3>Local contributions</h3>
                    <p>
                      Compare test performance, runtime, learning rate, and
                      update size under the same round.
                    </p>
                  </div>
                  <label>
                    Round{" "}
                    <select
                      aria-label="Client diagnostics round"
                      value={round}
                      onChange={(e) => setRound(e.target.value)}
                    >
                      <option value="latest">Latest</option>
                      {[
                        ...new Set(
                          data.client_rounds.map((r) => Number(r.round)),
                        ),
                      ]
                        .sort((a, b) => a - b)
                        .map((r) => (
                          <option key={r}>{r}</option>
                        ))}
                    </select>
                  </label>
                </div>
                <div className="monitor-table">
                  <table>
                    <thead>
                      <tr>
                        {[
                          "Institution",
                          "Latest phase",
                          "Train / test samples",
                          "Train loss",
                          "Train acc.",
                          "Test loss",
                          "Test acc.",
                          "Fit seconds",
                          "LR",
                          "Relative update",
                        ].map((h) => (
                          <th key={h}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {clients.map((c) => {
                        const cid = Number(c.client_id),
                          last = data.latest.find(
                            (e) => Number(e.client_id) === cid,
                          );
                        return (
                          <tr key={cid}>
                            <td>
                              <strong>{clientName(cid)}</strong>
                              <small>
                                Client {cid} · R{c.round as number}
                              </small>
                            </td>
                            <td>{words(last?.phase)}</td>
                            <td>
                              {fmt(c.num_samples, 0)} / {fmt(c.test_samples, 0)}
                            </td>
                            <td>{fmt(c.train_loss)}</td>
                            <td>{pct(n(c, "train_accuracy"))}</td>
                            <td>{fmt(c.eval_loss)}</td>
                            <td>
                              {c.evaluation_skipped
                                ? "Skipped (empty test)"
                                : pct(n(c, "eval_accuracy"))}
                            </td>
                            <td>{fmt(c.training_time, 2)}</td>
                            <td>{fmt(c.learning_rate, 6)}</td>
                            <td>{fmt(c.relative_update_l2, 5)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  {!clients.length && (
                    <div className="monitor-empty">
                      Client fit/evaluation reports appear here as they finish.
                    </div>
                  )}
                </div>
                <div className="monitor-grid">
                  {plot(
                    "Test accuracy by institution",
                    "Local test sets can have different class composition and sample counts.",
                    data.client_rounds,
                    [["eval_accuracy", "Test accuracy"]],
                    { split: true, percent: true },
                  )}
                  {plot(
                    "Local training throughput",
                    "Samples processed per second during each epoch",
                    epochRows,
                    [["samples_per_second", "Samples/s"]],
                    { split: true, x: "step" },
                  )}
                </div>
                {run.spec.strategy === "scr" &&
                  data.support
                    .filter((e) => Number(e.round) === selectedRound)
                    .map((e) => (
                      <article
                        className="card monitor-support"
                        key={`${e.client_id}-${e.round}`}
                      >
                        <h3>
                          {clientName(Number(e.client_id))} · SCR support · R
                          {e.round as number}
                        </h3>
                        <div className="monitor-table">
                          <table>
                            <thead>
                              <tr>
                                <th>Class ID</th>
                                <th>Training samples</th>
                                <th>Sample weight</th>
                                <th>Anchor coefficient</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(e.class_counts as number[]).map((count, i) => (
                                <tr key={i}>
                                  <td>{i}</td>
                                  <td>{count}</td>
                                  <td>
                                    {fmt((e.sample_weights as number[])[i], 5)}
                                  </td>
                                  <td>{fmt((e.anchors as number[])[i], 5)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </article>
                    ))}
              </>
            )}
            {tab === "classes" && (
              <>
                <div className="monitor-section-head">
                  <div>
                    <h3>Class performance</h3>
                    <p>
                      Pooled confusion counts from the selected institutions. No
                      images leave the institution.
                    </p>
                  </div>
                  <label>
                    Evaluation round{" "}
                    <select
                      aria-label="Class performance round"
                      value={round}
                      onChange={(e) => setRound(e.target.value)}
                    >
                      <option value="latest">Latest available</option>
                      {evaluations.map((e) => (
                        <option key={e.round}>{e.round}</option>
                      ))}
                    </select>
                  </label>
                </div>
                {!evaluation ? (
                  <div className="monitor-empty">
                    No confusion matrix is available. New runs record it during
                    non-empty local test evaluations; older runs cannot
                    reconstruct it from accuracy alone.
                  </div>
                ) : (
                  <>
                    <p className="monitor-note">
                      Round {evaluation.round} · {evaluation.reporting_clients}{" "}
                      institution(s) reporting ·{" "}
                      {evaluation.samples.toLocaleString()} test images.{" "}
                      {client === "all" && overview.mode === "coordinator"
                        ? `Roster: ${clientIds.length} institutions; missing or empty evaluations are excluded.`
                        : ""}{" "}
                      FedBN/FedBNS pool predictions from institution-specific
                      models.
                    </p>
                    <div className="monitor-stats">
                      <div>
                        <small>Macro F1</small>
                        <strong>{pct(evaluation.macro_f1)}</strong>
                        <span>Classes with test support</span>
                      </div>
                      <div>
                        <small>Balanced accuracy</small>
                        <strong>{pct(evaluation.balanced_accuracy)}</strong>
                        <span>Mean recall of supported classes</span>
                      </div>
                      <div>
                        <small>Weighted F1</small>
                        <strong>{pct(evaluation.weighted_f1)}</strong>
                        <span>Weighted by test support</span>
                      </div>
                      <div>
                        <small>Pooled accuracy</small>
                        <strong>{pct(evaluation.accuracy)}</strong>
                        <span>Selected reported matrices</span>
                      </div>
                    </div>
                    <article className="card monitor-support">
                      <h3>Confusion matrix</h3>
                      <p className="monitor-note">
                        Rows = true class · columns = predicted class · cell
                        intensity = fraction of that true class
                      </p>
                      <div className="monitor-table">
                        <table className="confusion-table">
                          <thead>
                            <tr>
                              <th>True ↓ / Predicted →</th>
                              {evaluation.classes.map((c) => (
                                <th key={c.class_id} title={c.label}>
                                  {c.class_id}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {evaluation.matrix.map((row, i) => (
                              <tr key={i}>
                                <th>
                                  {i} · {evaluation.classes[i].label}
                                </th>
                                {row.map((value, j) => (
                                  <td
                                    key={j}
                                    style={{
                                      backgroundColor: `rgba(14,147,135,${0.04 + (0.65 * value) / Math.max(1, evaluation.classes[i].support)})`,
                                    }}
                                    title={`${evaluation.classes[i].label} → ${evaluation.classes[j].label}: ${value}`}
                                  >
                                    {value}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </article>
                    <div className="monitor-table">
                      <table>
                        <thead>
                          <tr>
                            <th>Class</th>
                            <th>Test support</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evaluation.classes.map((c) => (
                            <tr key={c.class_id}>
                              <td>
                                {c.class_id} · {c.label}
                              </td>
                              <td>{c.support}</td>
                              <td>{pct(c.precision)}</td>
                              <td>{pct(c.recall)}</td>
                              <td>{pct(c.f1)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="monitor-note">
                      A dash means the metric is undefined (for example, no true
                      samples or no predicted positives). Macro F1 and balanced
                      accuracy include only classes with true test support.
                    </p>
                  </>
                )}
              </>
            )}
            {tab === "system" && (
              <>
                <div className="monitor-grid">
                  {plot(
                    "Round wall time",
                    "Fit includes aggregation and waiting; evaluation follows fit. Full round time includes both.",
                    data.rounds,
                    [
                      ["wall_seconds", "Full round (s)"],
                      ["fit_seconds", "Fit (s)"],
                      ["evaluation_seconds", "Evaluation (s)"],
                    ],
                  )}
                  {plot(
                    "Training time by institution",
                    "Time spent inside each client’s local training loop",
                    data.client_rounds,
                    [["training_time", "Fit seconds"]],
                    { split: true },
                  )}
                  {plot(
                    "Parameter payload",
                    "Uncompressed arrays sent/received during fit, in KiB. Includes SCAFFOLD control arrays; excludes transport headers and evaluation transfers.",
                    data.rounds,
                    [
                      ["upload_payload_kb", "Client upload KiB"],
                      ["download_payload_kb", "Client download KiB"],
                    ],
                  )}
                  {plot(
                    "Host CPU utilization",
                    "Host-wide samples about every five seconds. X = seconds since each worker launched.",
                    systemRows,
                    [["cpu_percent", "CPU %"]],
                    { split: true, x: "elapsed" },
                  )}
                  {plot(
                    "Training process memory",
                    "RSS summed over the owned process tree, in MiB; shared pages can be counted more than once.",
                    systemRows,
                    [["process_tree_rss_mb", "RSS MiB"]],
                    { split: true, x: "elapsed" },
                  )}
                  {plot(
                    "Host RAM utilization",
                    "Percentage of host physical memory in use",
                    systemRows,
                    [["ram_percent", "RAM %"]],
                    { split: true, x: "elapsed" },
                  )}
                  {plot(
                    "PyTorch GPU memory",
                    "Allocated memory in the training process at epoch end, in MiB. CPU runs show unavailable.",
                    epochRows,
                    [["gpu_memory_mb", "GPU MiB"]],
                    { split: true, x: "step" },
                  )}
                  {plot(
                    "Client fit-time spread",
                    "Slowest minus fastest client training time; a timing diagnostic, not a measured network delay.",
                    data.rounds,
                    [["straggler_time_seconds", "Spread (s)"]],
                  )}
                </div>
                <p className="monitor-note">
                  Resource series retain the latest 2,000 samples in this view;
                  JSON/CSV exports include all stored samples. No GPU
                  utilization, power, or network traffic is inferred.
                </p>
              </>
            )}
            {tab === "logs" && (
              <>
                <div className="monitor-log-tools">
                  <label className="monitor-search">
                    <Search size={16} />
                    <input
                      aria-label="Search training logs"
                      placeholder="Search loss, round, error, traceback…"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />
                  </label>
                  <select
                    aria-label="Log severity"
                    value={severity}
                    onChange={(e) => setSeverity(e.target.value)}
                  >
                    <option value="all">All output</option>
                    <option value="errors">Errors & tracebacks</option>
                    <option value="warnings">Warnings</option>
                  </select>
                  <label>
                    <input
                      type="checkbox"
                      checked={follow}
                      onChange={(e) => setFollow(e.target.checked)}
                    />{" "}
                    Follow output
                  </label>
                </div>
                <p className="monitor-note">
                  Actual process stdout/stderr, relayed with institution
                  heartbeats (about 5s). Prefixes show capture time and the last
                  observed round. Showing {data.logs.length} of{" "}
                  {data.limits.logs_total} chunks; JSON exports contain the full
                  captured log.
                </p>
                <pre className="monitor-log" ref={logRef} tabIndex={0}>
                  {logText || "No matching process output yet."}
                </pre>
              </>
            )}
            {tab === "config" && (
              <>
                <div className="monitor-section-head">
                  <div>
                    <h3>Reproducible run configuration</h3>
                    <p>
                      Run ID: <code>{run.id}</code> ·{" "}
                      {data.event_count.toLocaleString()} persisted events
                    </p>
                  </div>
                </div>
                <div className="monitor-table">
                  <table>
                    <tbody>
                      {Object.entries(data.configuration).map(
                        ([key, value]) => (
                          <tr key={key}>
                            <th>{words(key)}</th>
                            <td>
                              {Array.isArray(value)
                                ? value.join(", ")
                                : String(value)}
                            </td>
                          </tr>
                        ),
                      )}
                      <tr>
                        <th>partition hash</th>
                        <td>
                          <code>{data.partition_hash || "Unknown"}</code>
                        </td>
                      </tr>
                      <tr>
                        <th>dataset hash</th>
                        <td>
                          <code>{data.dataset_hash || "Unknown"}</code>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p className="monitor-note">
                  Method: {names[run.spec.strategy]}. Compare runs only when
                  dataset, partition identity, roster size, seed, model,
                  optimizer, and training budget match. Local training metrics
                  use the actual method objective. Diagnostics observe training
                  and do not change aggregation or prediction rules.
                </p>
              </>
            )}
          </div>
        </>
      )}
    </section>
  );
}
