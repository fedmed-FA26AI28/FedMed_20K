import type { components } from "./api.generated";
export type RunSpec = Required<components["schemas"]["RunSpec"]>;
export type DatasetName = RunSpec["dataset"];
export type Strategy = RunSpec["strategy"];
export type ModelImport = Required<components["schemas"]["ImportModel"]>;

export async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    method,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-FedMed-Request": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response
    .json()
    .catch(() => ({ detail: "The service did not return a response." }));
  if (!response.ok) {
    const message =
      typeof payload.detail === "string"
        ? payload.detail
        : JSON.stringify(payload.detail);
    throw new Error(message || `Request failed (${response.status})`);
  }
  return payload.data as T;
}

export interface Session {
  username: string;
  mode: "coordinator" | "institution";
  name: string;
}
export interface Capabilities {
  ready: boolean;
  errors: string[];
  python?: string;
  torch?: string;
  flwr?: string;
  cuda?: boolean;
  gpu?: string;
  platform?: string;
  jetpack_release?: string;
}
export interface Telemetry {
  cpu_percent?: number;
  ram_percent?: number;
  ram_used_mb?: number;
  ram_total_mb?: number;
}
export interface Model {
  id: string;
  dataset: DatasetName;
  model: string;
  strategy: Strategy;
  classes: Record<string, string>;
  scope: "global" | "institution";
  institution_id?: string;
  source_run: string;
  source_round: number;
  selection_rule: string;
  checksum: string;
  verified: boolean;
  created_at: number;
  evaluation?: {
    eval_accuracy?: number | null;
    eval_loss?: number | null;
    skipped?: boolean;
  } | null;
}
export interface Node {
  id: string;
  name: string;
  device_type: string;
  connection_status: string;
  last_seen: number;
  capabilities: Capabilities;
  telemetry: Telemetry;
  models?: Model[];
}
export interface Participation {
  client_id: number;
  status: string;
  error?: string;
  partition_hash?: string;
  train_samples?: number;
  test_samples?: number;
}
export interface Run {
  id: string;
  name: string;
  status: string;
  spec: RunSpec;
  phase?: string;
  current_round?: number;
  completed_rounds?: number;
  created_at: number;
  started_at?: number;
  ended_at?: number;
  participants?: Record<string, Participation>;
  partition_hash?: string;
  client_id?: number;
  error?: string;
  job_id?: string;
  coordinator_status?: string;
}
export interface Job {
  id: string;
  action: string;
  status: string;
  created_at: number;
  context: { run_id?: string };
  result?: unknown;
  error?: string;
}
export interface Dataset {
  id: string;
  dataset: DatasetName;
  train_samples: number;
  test_samples: number;
  classes: Record<string, string>;
  class_distribution: number[];
  assigned: boolean;
  previews: { index: number; label: number; image: string }[];
}
export interface Overview {
  mode: Session["mode"];
  name: string;
  instance_id: string;
  capabilities: Capabilities | null;
  telemetry: Telemetry;
  connection: { url: string; last_seen: number; error?: string } | null;
  nodes: Node[];
  runs: Run[];
  jobs: Job[];
  models: Model[];
  datasets: Dataset[];
  service_error?: string;
}
export interface Event {
  id: string;
  seq: number;
  run_id: string;
  client_id: number;
  round: number;
  time: number;
  phase: string;
  epoch?: number;
  loss?: number;
  accuracy?: number;
  metrics?: Record<string, number | null>;
}
export interface Experiment {
  id: string;
  name: string;
  dataset: DatasetName;
  strategy: Strategy;
  model: string;
  partition: string;
  seed: string;
  num_clients: number;
  source: string;
  status: string;
  bta: number | null;
  bmcta: number | null;
  final_accuracy: number | null;
  loss: number | null;
  time_seconds: number | null;
  rounds: number;
  partition_hash: string | null;
  configuration: Record<string, string | null>;
  curves: Record<string, number>[];
}
export interface Prediction {
  label: string;
  top: { label: string; score: number; class_id: number }[];
  latency_ms: number;
  model_id: string;
  dataset: string;
}
export const datasets: DatasetName[] = [
  "bloodmnist",
  "organamnist",
  "pathmnist",
];
export const strategies: Strategy[] = [
  "fedavg",
  "fedbn",
  "scaffold",
  "fedbns",
  "scr",
];
export const names: Record<string, string> = {
  bloodmnist: "BloodMNIST",
  organamnist: "OrganAMNIST",
  pathmnist: "PathMNIST",
  fedavg: "FedAvg",
  fedbn: "FedBN",
  scaffold: "SCAFFOLD",
  fedbns: "FedBNS",
  scr: "SCR",
};
export const active = new Set([
  "preparing",
  "waiting",
  "starting",
  "running",
  "stopping",
  "training",
  "checking",
  "ready",
  "finalizing",
  "awaiting_clients",
]);
export const pct = (x: number | null | undefined) =>
  x == null ? "—" : `${(x * 100).toFixed(2)}%`;
export const date = (x?: number) =>
  x
    ? new Date(x * 1000).toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";
export const duration = (x?: number | null) =>
  x == null ? "—" : x < 60 ? `${Math.round(x)}s` : `${(x / 60).toFixed(1)} min`;
