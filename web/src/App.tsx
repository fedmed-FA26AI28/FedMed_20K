import { useState, type FormEvent, type ReactNode } from "react";
import RunMonitor from './RunMonitor';
import { NavLink, Routes, Route, Link, useLocation } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Building2,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Cpu,
  Database,
  FlaskConical,
  Globe2,
  ImagePlus,
  Layers3,
  LayoutDashboard,
  Loader2,
  LogOut,
  Network,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  Square,
  Stethoscope,
  X,
  XCircle,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import {
  api,
  active,
  datasets,
  strategies,
  names,
  pct,
  date,
  duration,
  type Overview,
  type Session,
  type Run,
  type RunSpec,
  type Job,
  type Event,
  type Experiment,
  type Model,
  type ModelImport,
  type Prediction,
  type DatasetName,
} from "./api";

type Context = {
  o: Overview;
  action: <T>(path: string, body?: unknown, method?: string) => Promise<T>;
  busy: boolean;
};
const colors = ["#0e9387", "#3972b8", "#8070c8", "#d79338", "#de7373"];

function Badge({ value }: { value: string }) {
  return (
    <span className={`badge ${value}`}>
      <i />
      {value.replaceAll("_", " ")}
    </span>
  );
}
function Empty({
  icon = <Layers3 size={25} />,
  title,
  children,
}: {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <span className="empty-icon">{icon}</span>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
function Heading({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div className="heading-actions">{children}</div>
    </div>
  );
}
function Card({
  title,
  subtitle,
  children,
  action,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      <div className="card-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
function Metric({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: ReactNode;
  detail: string;
  icon: ReactNode;
}) {
  return (
    <div className="metric">
      <span className="metric-icon">{icon}</span>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}
function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-heading">
          <h2>{title}</h2>
          <button
            className="icon-button"
            aria-label="Close dialog"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        {children}
      </section>
    </div>
  );
}
function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

function Login({ onLogin }: { onLogin: () => void }) {
  const status = useQuery({
    queryKey: ["status"],
    queryFn: () => api<{ mode: string; name: string }>("/status"),
  });
  const [username, setUsername] = useState(""),
    [password, setPassword] = useState(""),
    [error, setError] = useState(""),
    [pending, setPending] = useState(false);
  async function submit(e: FormEvent) {
    e.preventDefault();
    setPending(true);
    setError("");
    try {
      await api("/session", "POST", { username, password });
      onLogin();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  return (
    <div className="login">
      <div className="login-story">
        <div className="brand">
          <img src="/mark.svg" />
          <strong>
            FedMed<span>AI</span>
          </strong>
        </div>
        <div className="login-copy">
          <span className="eyebrow">COLLABORATIVE MEDICAL IMAGING</span>
          <h1>
            Shared intelligence.
            <br />
            <em>Local expertise.</em>
          </h1>
          <p>
            A connected workspace for institutions to train, evaluate, and
            explore medical image models together.
          </p>
          <div className="login-diagram">
            <span>
              <Building2 size={27} />
            </span>
            <b />
            <span className="hub">
              <Network size={38} />
            </span>
            <b />
            <span>
              <Cpu size={27} />
            </span>
          </div>
          <div className="login-tags">
            <span>
              <CheckCircle2 size={16} /> Federated training
            </span>
            <span>
              <CheckCircle2 size={16} /> Local prediction
            </span>
          </div>
        </div>
        <small>FedMedAI · Federated learning research platform</small>
      </div>
      <div className="login-form-wrap">
        <div className="login-form">
          <span className="small-logo">
            <Stethoscope size={24} />
          </span>
          <span className="eyebrow">
            {status.data?.mode === "institution"
              ? "INSTITUTION WORKSPACE"
              : "COORDINATOR WORKSPACE"}
          </span>
          <h2>Welcome to your workspace</h2>
          <p>Sign in to {status.data?.name || "FedMedAI"}.</p>
          <form onSubmit={submit}>
            <Field label="Username">
              <input
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                placeholder="Your operator username"
              />
            </Field>
            <Field label="Password">
              <input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="Enter your password"
              />
            </Field>
            {error && <div className="error-inline">{error}</div>}
            <button className="primary full" disabled={pending}>
              {pending ? <Loader2 size={17} className="spin" /> : null}Sign in
              to workspace <ArrowRight size={17} />
            </button>
          </form>
          <p className="login-help">
            <CircleHelp size={16} /> Use the account created during device
            setup.
          </p>
          <div className="research-label">
            RESEARCH DEMONSTRATION <span>Public MedMNIST benchmarks</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const cache = useQueryClient(),
    location = useLocation();
  const session = useQuery({
    queryKey: ["session"],
    queryFn: () => api<Session>("/session"),
    staleTime: 60000,
  });
  const overview = useQuery({
    queryKey: ["overview"],
    queryFn: () => api<Overview>("/overview"),
    enabled: !!session.data,
    refetchInterval: 2000,
  });
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [help, setHelp] = useState(false);
  async function action<T>(
    path: string,
    body?: unknown,
    method = "POST",
  ): Promise<T> {
    setBusy(true);
    setError("");
    try {
      const result = await api<T>(path, method, body);
      await cache.invalidateQueries();
      return result;
    } catch (e) {
      setError((e as Error).message);
      throw e;
    } finally {
      setBusy(false);
    }
  }
  if (session.isPending)
    return (
      <div className="boot">
        <img src="/mark.svg" />
        <Loader2 className="spin" /> Opening workspace
      </div>
    );
  if (!session.data)
    return (
      <Login
        onLogin={() => cache.invalidateQueries({ queryKey: ["session"] })}
      />
    );
  const isCoordinator = session.data.mode === "coordinator";
  const nav = isCoordinator
    ? ([
        ["/", "Overview", LayoutDashboard],
        ["/institutions", "Institutions", Building2],
        ["/training", "Training", Activity],
        ["/experiments", "Experiments", FlaskConical],
        ["/models", "Models", Layers3],
      ] as const)
    : ([
        ["/", "Overview", LayoutDashboard],
        ["/dataset", "Dataset", Database],
        ["/training", "Training", Activity],
        ["/predict", "Predict", ImagePlus],
        ["/models", "Models", Layers3],
      ] as const);
  const current =
    nav.find(([p]) => p === location.pathname)?.[1] || "Workspace";
  const context = overview.data ? { o: overview.data, action, busy } : null;
  return (
    <div className="shell">
      <aside className="sidebar">
        <Link to="/" className="brand">
          <img src="/mark.svg" />
          <strong>
            FedMed<span>AI</span>
          </strong>
        </Link>
        <div className="workspace-label">
          {isCoordinator ? "COORDINATOR" : "INSTITUTION"} WORKSPACE
        </div>
        <nav>
          {nav.map(([path, label, Icon]) => (
            <NavLink key={path} to={path} end={path === "/"} aria-label={label}>
              <Icon size={19} />
              <span>{label}</span>
              {label === current && <span className="nav-dot" />}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="note-icon">
            <Network size={23} />
          </span>
          <strong>Connected by research</strong>
          <p>One federation. Independent institutions. Shared progress.</p>
          <span className="version">
            RESEARCH DEMO <b>v1.0</b>
          </span>
        </div>
        <div className="sidebar-user">
          <span className="avatar">
            {session.data.username.slice(0, 2).toUpperCase()}
          </span>
          <div>
            <strong>{session.data.username}</strong>
            <small>
              {isCoordinator ? "Network coordinator" : "Institution operator"}
            </small>
          </div>
          <button
            title="Sign out"
            className="icon-button"
            onClick={() =>
              action("/session", undefined, "DELETE")
                .then(() => cache.clear())
                .catch(() => {})
            }
          >
            <LogOut size={17} />
          </button>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={14} />
            <strong>{current}</strong>
          </div>
          <div className="topbar-right">
            <span className="network-label">
              <i className={overview.isError ? "danger-dot" : ""} />
              {overview.isError ? "Service unreachable" : "Local network"}
            </span>
            <button
              className="icon-button"
              title="Demo walkthrough"
              onClick={() => setHelp(true)}
            >
              <CircleHelp size={19} />
            </button>
            <span className="top-avatar">{isCoordinator ? "CO" : "IN"}</span>
          </div>
        </header>
        <main>
          {(error || overview.isError) && (
            <div role="alert" className="alert error global-alert">
              <XCircle size={18} />
              <span>
                {error ||
                  "Cannot reach this service. Check that the portal is running."}
              </span>
              <button
                className="icon-button"
                onClick={() => {
                  setError("");
                  overview.refetch();
                }}
              >
                <RefreshCw size={16} />
              </button>
            </div>
          )}
          {context ? (
            <Routes>
              <Route path="/" element={<Dashboard {...context} />} />
              <Route
                path="/institutions"
                element={
                  isCoordinator ? <Institutions {...context} /> : <Denied />
                }
              />
              <Route
                path="/dataset"
                element={
                  !isCoordinator ? <DatasetPage {...context} /> : <Denied />
                }
              />
              <Route path="/training" element={<Training {...context} />} />
              <Route
                path="/experiments"
                element={
                  isCoordinator ? <Experiments {...context} /> : <Denied />
                }
              />
              <Route path="/models" element={<Models {...context} />} />
              <Route
                path="/predict"
                element={!isCoordinator ? <Predict {...context} /> : <Denied />}
              />
              <Route
                path="*"
                element={
                  <Empty title="Page not found">
                    <Link to="/">Return to overview</Link>
                  </Empty>
                }
              />
            </Routes>
          ) : (
            <Empty title="Connecting to your workspace">
              <Loader2 className="spin" />
            </Empty>
          )}
        </main>
        <footer>
          <span>
            FedMedAI <i /> Collaborative medical imaging research
          </span>
          <span>MedMNIST benchmarks · Model scores are research outputs</span>
        </footer>
      </div>
      {help && (
        <Modal
          title="Your demonstration, step by step"
          onClose={() => setHelp(false)}
        >
          <ol className="walkthrough">
            <li>
              <strong>Prepare each device</strong>
              <p>
                Check its training environment and download the benchmark
                dataset before the presentation.
              </p>
            </li>
            <li>
              <strong>Connect institutions</strong>
              <p>
                Create a pairing token on the coordinator and enter it on the
                institution’s overview.
              </p>
            </li>
            <li>
              <strong>Train collaboratively</strong>
              <p>
                Create a session, accept the invitation on every client, then
                start from the coordinator.
              </p>
            </li>
            <li>
              <strong>Explore a trained model</strong>
              <p>
                Use a verified full-run model on the institution’s Predict page.
                Short live sessions demonstrate the training workflow.
              </p>
            </li>
          </ol>
        </Modal>
      )}
    </div>
  );
}

function Denied() {
  return (
    <Empty title="This page belongs to the other workspace">
      Open the appropriate coordinator or institution service.
    </Empty>
  );
}

function Dashboard(ctx: Context) {
  const { o, action, busy } = ctx,
    coordinator = o.mode === "coordinator";
  const [connect, setConnect] = useState(false),
    [url, setUrl] = useState(""),
    [token, setToken] = useState(""),
    [taskLog, setTaskLog] = useState<string | null>(null);
  const log = useQuery({
    queryKey: ["task-log", taskLog],
    queryFn: () => api<string>(`/jobs/${taskLog}/log`),
    enabled: !!taskLog,
    refetchInterval: 2000,
  });
  const history = useQuery({
    queryKey: ["experiments"],
    queryFn: () => api<Experiment[]>("/experiments"),
    enabled: coordinator,
  });
  const current = o.runs.find((r) => active.has(r.status)),
    online = o.nodes.filter((n) => n.connection_status === "online").length;
  const environment = o.capabilities;
  const completed = o.runs.filter((r) => r.status === "completed").length;
  return (
    <>
      <Heading
        eyebrow={coordinator ? "FEDERATION OVERVIEW" : "INSTITUTION OVERVIEW"}
        title={coordinator ? "Your collaborative workspace" : o.name}
        description={
          coordinator
            ? "A clear view of your institutions, training sessions, and shared models."
            : "Your local data, device, and contribution to collaborative learning."
        }
      >
        <button
          className="secondary"
          disabled={busy}
          onClick={() => action("/preflight").catch(() => {})}
        >
          <RefreshCw size={16} />
          Check environment
        </button>
        {coordinator ? (
          <Link className="primary" to="/training">
            <Plus size={17} />
            New training session
          </Link>
        ) : (
          <button className="primary" onClick={() => setConnect(true)}>
            <Network size={17} />
            {o.connection ? "Reconnect" : "Connect to coordinator"}
          </button>
        )}
      </Heading>
      <div className="hero-panel">
        <div>
          <span className="eyebrow">
            {current ? "COLLABORATION IN PROGRESS" : "READY FOR COLLABORATION"}
          </span>
          <h2>
            {current ? current.name : "Better research starts with connection."}
          </h2>
          <p>
            {current
              ? `${names[current.spec.dataset]} · ${names[current.spec.strategy]} · ${current.spec.node_ids.length} institutions`
              : "Bring institutions together, follow every round, and explore what your models learn."}
          </p>
          <Link to={coordinator ? "/institutions" : "/dataset"}>
            {coordinator ? "Manage your network" : "Explore local dataset"}
            <ArrowRight size={17} />
          </Link>
        </div>
        <div
          className="network-art"
          aria-label="Federation architecture illustration"
        >
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="art-hub">
            <img src="/mark.svg" />
          </div>
          {[Building2, Cpu, Building2, Cpu].map((Icon, i) => (
            <div key={i} className={`art-node art-node-${i}`}>
              <Icon size={22} />
            </div>
          ))}
          <span className="art-label">FEDERATED NETWORK</span>
        </div>
      </div>
      <div className="metrics-grid">
        <Metric
          label={
            coordinator ? "Connected institutions" : "Coordinator connection"
          }
          value={
            coordinator ? (
              <>
                {online}
                <em> / {o.nodes.length}</em>
              </>
            ) : o.connection &&
              Date.now() / 1000 - o.connection.last_seen < 15 ? (
              "Online"
            ) : o.connection &&
              Date.now() / 1000 - o.connection.last_seen < 30 ? (
              "Stale"
            ) : (
              "Not connected"
            )
          }
          detail={
            coordinator
              ? "Devices reporting within 15 seconds"
              : "Direct connection over your local network"
          }
          icon={<Building2 size={19} />}
        />
        <Metric
          label="Training sessions"
          value={o.runs.length}
          detail={`${completed} completed${current ? " · 1 active" : ""}`}
          icon={<Activity size={19} />}
        />
        <Metric
          label={coordinator ? "Recorded experiments" : "Prepared datasets"}
          value={coordinator ? (history.data?.length ?? 0) : o.datasets.length}
          detail={
            coordinator
              ? "Imported from your real result folders"
              : "Public MedMNIST benchmark data"
          }
          icon={<FlaskConical size={19} />}
        />
        <Metric
          label="Verified models"
          value={o.models.length}
          detail="Checkpoint and architecture verified"
          icon={<Layers3 size={19} />}
        />
      </div>
      <div className="dashboard-grid">
        <Card
          title={
            current ? "Active training session" : "Recent training sessions"
          }
          subtitle="Progress from this live federation"
          action={
            <Link className="text-link" to="/training">
              View sessions <ArrowUpRight size={15} />
            </Link>
          }
        >
          {o.runs.length ? (
            <RunList runs={o.runs.slice(0, 4)} />
          ) : (
            <div className="onboarding">
              <div className="step-number">01</div>
              <div>
                <h3>
                  {coordinator
                    ? "Connect your first institution"
                    : "Prepare your local workspace"}
                </h3>
                <p>
                  {coordinator
                    ? "Pair each client using a one-time token. Once devices are ready, create your first collaborative session."
                    : "Prepare a benchmark dataset, connect to the coordinator, and accept a training invitation."}
                </p>
                <Link
                  className="secondary"
                  to={coordinator ? "/institutions" : "/dataset"}
                >
                  {coordinator ? "Add institution" : "Prepare dataset"}
                  <ArrowRight size={16} />
                </Link>
              </div>
            </div>
          )}
        </Card>
        <Card
          title="Device readiness"
          subtitle="Checks from your training environment"
          action={
            environment ? (
              <Badge value={environment.ready ? "ready" : "failed"} />
            ) : (
              <Badge value="checking" />
            )
          }
        >
          <div className="readiness">
            <div>
              <span>Python environment</span>
              <strong>{environment?.python || "Checking…"}</strong>
            </div>
            <div>
              <span>Flower runtime</span>
              <strong>{environment?.flwr || "—"}</strong>
            </div>
            <div>
              <span>Compute</span>
              <strong>
                {environment?.cuda
                  ? "CUDA GPU"
                  : environment?.ready
                    ? "CPU"
                    : "—"}
              </strong>
            </div>
            <div>
              <span>Memory used</span>
              <strong>{o.telemetry.ram_percent?.toFixed(0)}%</strong>
            </div>
            <div className="meter">
              <span style={{ width: `${o.telemetry.ram_percent || 0}%` }} />
            </div>
            {environment?.gpu && <small>{environment.gpu}</small>}
            {environment?.errors?.map((e, i) => (
              <p className="error-inline" key={i}>
                {e}
              </p>
            ))}
          </div>
        </Card>
      </div>
      <div className="dashboard-grid">
        <Card title="Recent activity" subtitle="Tasks executed by this service">
          <div className="activity-list">
            {o.jobs.slice(0, 4).map((j) => (
              <div key={j.id}>
                <span
                  className={`activity-icon ${j.status === "failed" ? "failed" : ""}`}
                >
                  {j.status === "completed" ? (
                    <Check size={16} />
                  ) : (
                    <Activity size={16} />
                  )}
                </span>
                <div>
                  <strong>
                    {j.action === "preflight"
                      ? "Environment compatibility check"
                      : j.action === "train"
                        ? "Federated training"
                        : j.action === "model"
                          ? "Checkpoint verification"
                          : j.action === "download"
                            ? "Dataset preparation"
                            : j.action.replaceAll("_", " ")}
                  </strong>
                  <small>{date(j.created_at)}</small>
                  {j.error && <small className="error-inline">{j.error}</small>}
                  <button
                    className="text-link"
                    onClick={() => setTaskLog(j.id)}
                  >
                    View task log
                  </button>
                </div>
                <Badge value={j.status} />
              </div>
            ))}
          </div>
        </Card>
        <Card
          title="Prepare for your next session"
          subtitle="Set up once, present with confidence"
        >
          <div className="next-steps">
            <Link to={coordinator ? "/institutions" : "/dataset"}>
              <span>
                <Database size={18} />
              </span>
              <div>
                <strong>
                  {coordinator
                    ? "Check your institutions"
                    : "Prepare benchmark images"}
                </strong>
                <small>
                  {coordinator
                    ? "Verify that the selected clients are online"
                    : "BloodMNIST, OrganAMNIST, and PathMNIST"}
                </small>
              </div>
              <ChevronRight size={17} />
            </Link>
            <Link to="/models">
              <span>
                <Layers3 size={18} />
              </span>
              <div>
                <strong>Bring a trained model</strong>
                <small>Use a verified checkpoint for image prediction</small>
              </div>
              <ChevronRight size={17} />
            </Link>
          </div>
        </Card>
      </div>
      {coordinator && <CoordinatorData {...ctx} />}
      {taskLog && (
        <Modal title="Task log" onClose={() => setTaskLog(null)}>
          <pre className="log-view">{log.data || "Loading task log…"}</pre>
        </Modal>
      )}
      {connect && (
        <Modal
          title="Connect this institution"
          onClose={() => setConnect(false)}
        >
          <p className="modal-description">
            Create a pairing token in the coordinator’s Institutions page, then
            enter it here. Each token works once and expires in 15 minutes.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              action("/connection", { coordinator_url: url, token })
                .then(() => setConnect(false))
                .catch(() => {});
            }}
          >
            <Field label="Coordinator URL">
              <input
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="http://192.168.1.10:8000"
                required
              />
            </Field>
            <Field label="Pairing token">
              <input
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Paste the one-time token"
                required
              />
            </Field>
            <button className="primary full" disabled={busy}>
              Connect institution
              <ArrowRight size={16} />
            </button>
          </form>
        </Modal>
      )}
    </>
  );
}

function CoordinatorData({ o, action, busy }: Context) {
  const [dataset, setDataset] = useState<DatasetName>("bloodmnist");
  const working = o.jobs.some((j) => ["queued", "running"].includes(j.status));
  return (
    <Card
      title="Benchmark datasets"
      subtitle="Prepare the coordinator copy used to create and verify client partitions."
    >
      <div className="coordinator-data">
        <Field label="Dataset to prepare">
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value as DatasetName)}
          >
            {datasets.map((d) => (
              <option value={d} key={d}>
                {names[d]}
              </option>
            ))}
          </select>
        </Field>
        <button
          className="secondary"
          disabled={busy || working}
          onClick={() =>
            action("/datasets/prepare", { dataset }).catch(() => {})
          }
        >
          <Database size={16} /> Prepare dataset
        </button>
        <span className="subtle">
          Uses the local cache; downloads public data if missing.
        </span>
      </div>
      <div className="dataset-readiness">
        {datasets.map((d) => (
          <span key={d}>
            {names[d]}{" "}
            <Badge
              value={
                o.datasets.some((x) => x.dataset === d) ? "ready" : "unprepared"
              }
            />
          </span>
        ))}
      </div>
    </Card>
  );
}

function RunList({
  runs,
  onSelect,
}: {
  runs: Run[];
  onSelect?: (r: Run) => void;
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Session</th>
            <th>Method</th>
            <th>Progress</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr
              key={r.id}
              onClick={() => onSelect?.(r)}
              className={onSelect ? "clickable" : ""}
            >
              <td>
                <strong>{r.name}</strong>
                <small>
                  {names[r.spec.dataset]} · {date(r.created_at)}
                </small>
              </td>
              <td>{names[r.spec.strategy]}</td>
              <td>
                <span className="progress-label">
                  {r.completed_rounds ?? r.current_round ?? 0} / {r.spec.rounds}{" "}
                  rounds
                </span>
                <div className="progress">
                  <i
                    style={{
                      width: `${((r.completed_rounds ?? r.current_round ?? 0) / r.spec.rounds) * 100}%`,
                    }}
                  />
                </div>
              </td>
              <td>
                <Badge value={r.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Institutions({ o, action, busy }: Context) {
  const [pair, setPair] = useState<{
    token: string;
    coordinator_url: string;
  } | null>(null);
  return (
    <>
      <Heading
        eyebrow="RESEARCH NETWORK"
        title="Institutions"
        description="Connect and monitor the independent devices in your federation."
      >
        <button
          className="primary"
          disabled={busy}
          onClick={() =>
            action<{ token: string; coordinator_url: string }>(
              "/pairing-tokens",
            )
              .then(setPair)
              .catch(() => {})
          }
        >
          <Plus size={17} />
          Pair an institution
        </button>
      </Heading>
      <div className="metrics-grid three">
        <Metric
          label="Registered institutions"
          value={o.nodes.length}
          detail="Each device has its own identity"
          icon={<Building2 size={19} />}
        />
        <Metric
          label="Online now"
          value={o.nodes.filter((n) => n.connection_status === "online").length}
          detail="Heartbeat received within 15 seconds"
          icon={<Globe2 size={19} />}
        />
        <Metric
          label="Training environments ready"
          value={o.nodes.filter((n) => n.capabilities?.ready).length}
          detail="Python and training imports verified"
          icon={<ShieldCheck size={19} />}
        />
      </div>
      {o.nodes.length ? (
        <div className="device-grid">
          {o.nodes.map((n, i) => (
            <section className="card device-card" key={n.id}>
              <div className="device-top">
                <span className="device-symbol">
                  {n.device_type === "pc" ? (
                    <Building2 size={25} />
                  ) : (
                    <Cpu size={25} />
                  )}
                </span>
                <Badge value={n.connection_status} />
              </div>
              <span className="eyebrow">
                INSTITUTION {String(i + 1).padStart(2, "0")}
              </span>
              <h2>{n.name}</h2>
              <p>
                {n.device_type.replaceAll("_", " ")} ·{" "}
                {n.capabilities?.cuda ? "GPU enabled" : "CPU / checking"}
              </p>
              <div className="device-stats">
                <span>
                  CPU
                  <strong>
                    {n.telemetry?.cpu_percent == null
                      ? "—"
                      : `${n.telemetry.cpu_percent.toFixed(0)}%`}
                  </strong>
                </span>
                <span>
                  Memory
                  <strong>
                    {n.telemetry?.ram_percent == null
                      ? "—"
                      : `${n.telemetry.ram_percent.toFixed(0)}%`}
                  </strong>
                </span>
                <span>
                  Flower<strong>{n.capabilities?.flwr || "—"}</strong>
                </span>
              </div>
              <div className="device-footer">
                <span>
                  <i
                    className={
                      n.capabilities?.ready ? "ready-dot" : "neutral-dot"
                    }
                  />
                  {n.capabilities?.ready
                    ? "Environment ready"
                    : "Readiness check needed"}
                </span>
                <small>Seen {date(n.last_seen)}</small>
              </div>
              <p className="assignment-note">
                {(() => {
                  const assignment = o.runs.find(
                    (r) => r.participants?.[n.id]?.train_samples != null,
                  );
                  const part = assignment?.participants?.[n.id];
                  return part
                    ? `${names[assignment!.spec.dataset]} · ${part.train_samples?.toLocaleString()} train / ${part.test_samples?.toLocaleString()} test`
                    : "Sample assignment pending a training session";
                })()}
              </p>
              {n.capabilities?.errors?.map((e, j) => (
                <p className="error-inline" key={j}>
                  {e}
                </p>
              ))}
            </section>
          ))}
        </div>
      ) : (
        <Card
          title="Your federation starts here"
          subtitle="Register the three client laptops, then add the Jetson devices after preflight checks."
        >
          <Empty
            icon={<Network size={30} />}
            title="No institutions connected yet"
          >
            Create a pairing token and open the Institution portal on a client
            device to connect it.
          </Empty>
        </Card>
      )}
      {pair && (
        <Modal title="Pair a new institution" onClose={() => setPair(null)}>
          <p className="modal-description">
            On the client device, open Overview → Connect to coordinator and
            enter these details.
          </p>
          <Field label="Coordinator URL">
            <input
              readOnly
              value={pair.coordinator_url}
              onFocus={(e) => e.target.select()}
            />
          </Field>
          <Field
            label="One-time pairing token"
            hint="Expires in 15 minutes. Create a separate token for each device."
          >
            <input
              readOnly
              className="mono"
              value={pair.token}
              onFocus={(e) => e.target.select()}
            />
          </Field>
          <div className="info-note">
            <ShieldCheck size={19} />
            <p>
              Device credentials are created during pairing. The operator
              password stays on the institution’s service.
            </p>
          </div>
          <button className="primary full" onClick={() => setPair(null)}>
            Done
          </button>
        </Modal>
      )}
    </>
  );
}

const defaultSpec: RunSpec = {
  name: "BloodMNIST · collaborative session",
  dataset: "bloodmnist",
  strategy: "scr",
  model: "baseline_bn",
  partition: "dirichlet_0.3",
  seed: 42,
  rounds: 3,
  local_epochs: 1,
  batch_size: 16,
  learning_rate: 0.01,
  momentum: 0.9,
  weight_decay: 0,
  scr_gamma: 1,
  scr_head_lambda: 0.01,
  round_timeout: 600,
  node_ids: [],
};
function Training(ctx: Context) {
  const { o, action, busy } = ctx;
  const [create, setCreate] = useState(false),
    [spec, setSpec] = useState<RunSpec>({ ...defaultSpec }),
    [selected, setSelected] = useState<string | null>(null),
    [confirm, setConfirm] = useState<Run | null>(null),
    [log, setLog] = useState<string | null>(null);
  const coordinator = o.mode === "coordinator",
    run = o.runs.find((r) => r.id === selected) || o.runs[0];
  const events = useQuery({
    queryKey: ["events", run?.id],
    queryFn: () => api<Event[]>(`/runs/${run!.id}/events`),
    enabled: !!run,
    refetchInterval: 2000,
  });
  const rounds = (events.data || [])
    .filter((e) =>
      coordinator
        ? e.phase === "round_completed"
        : ["evaluation_completed", "evaluation_skipped"].includes(e.phase),
    )
    .map((e) => ({
      round: e.round,
      accuracy:
        (coordinator ? e.metrics?.eval_accuracy : e.accuracy) == null
          ? null
          : Number(coordinator ? e.metrics?.eval_accuracy : e.accuracy) * 100,
      loss: coordinator ? e.metrics?.eval_loss : e.loss,
    }));
  const job = o.jobs.find(
    (j) => j.context.run_id === run?.id && j.action === "train",
  );
  function update<K extends keyof RunSpec>(key: K, value: RunSpec[K]) {
    setSpec((s) => ({ ...s, [key]: value }));
  }
  return (
    <>
      <Heading
        eyebrow="FEDERATED LEARNING"
        title="Training sessions"
        description="Configure a shared experiment and follow every institution’s contribution."
      >
        {coordinator && (
          <button className="primary" onClick={() => setCreate(true)}>
            <Plus size={17} />
            New session
          </button>
        )}
      </Heading>
      {!o.runs.length ? (
        <Card
          title="Start your first collaborative session"
          subtitle="Each selected institution accepts its invitation before training begins."
        >
          <Empty
            icon={<Activity size={30} />}
            title={
              coordinator
                ? "Ready when your institutions are"
                : "Waiting for an invitation"
            }
          >
            {coordinator
              ? "Connect your institutions and prepare the same dataset on every device."
              : "The coordinator will select this institution when creating a session. Your invitation will appear here."}
          </Empty>
        </Card>
      ) : (
        <>
          <div className="card">
            <RunList runs={o.runs} onSelect={(r) => setSelected(r.id)} />
          </div>
          {run && (
            <section className="run-detail">
              <div className="detail-heading">
                <div>
                  <span className="eyebrow">SESSION DETAILS</span>
                  <h2>{run.name}</h2>
                  <p>
                    {names[run.spec.strategy]} · {names[run.spec.dataset]} ·{" "}
                    {run.spec.partition} · Seed {run.spec.seed}
                  </p>
                </div>
                <div className="heading-actions">
                  <Badge value={run.status} />
                  {coordinator && run.status === "waiting" && (
                    <button
                      className="primary"
                      disabled={busy}
                      onClick={() =>
                        action(`/runs/${run.id}/start`).catch(() => {})
                      }
                    >
                      <Play size={16} />
                      Start training
                    </button>
                  )}
                  {!coordinator &&
                    ["invited", "failed"].includes(run.status) &&
                    run.coordinator_status === "waiting" && (
                      <button
                        className="primary"
                        disabled={busy}
                        onClick={() =>
                          action(`/runs/${run.id}/accept`).catch(() => {})
                        }
                      >
                        <Check size={16} />
                        Accept & check readiness
                      </button>
                    )}
                  {active.has(run.status) && (
                    <button
                      className="secondary danger"
                      disabled={busy}
                      onClick={() => setConfirm(run)}
                    >
                      <Square size={14} />
                      Cancel session
                    </button>
                  )}
                </div>
              </div>
              {run.error && (
                <div className="alert error">
                  <XCircle size={18} />
                  {run.error}
                </div>
              )}
              <div className="metrics-grid three">
                <Metric
                  label="Communication rounds"
                  value={`${run.completed_rounds ?? 0} / ${run.spec.rounds}`}
                  detail={run.phase?.replaceAll("_", " ") || run.status}
                  icon={<RefreshCw size={19} />}
                />
                <Metric
                  label="Local training"
                  value={`${run.spec.local_epochs} epoch${run.spec.local_epochs === 1 ? "" : "s"}`}
                  detail={`Batch ${run.spec.batch_size} · SGD · LR ${run.spec.learning_rate}`}
                  icon={<Cpu size={19} />}
                />
                <Metric
                  label="Elapsed time"
                  value={duration(
                    run.started_at
                      ? (run.ended_at ?? Date.now() / 1000) - run.started_at
                      : null,
                  )}
                  detail={
                    run.started_at
                      ? `Started ${date(run.started_at)}`
                      : "Waiting for the session to start"
                  }
                  icon={<Activity size={19} />}
                />
              </div>
              <div className="dashboard-grid">
                <Card
                  title={
                    coordinator
                      ? "Round accuracy"
                      : "This institution’s accuracy"
                  }
                  subtitle={
                    coordinator
                      ? "Combined local test accuracy · actual completed rounds"
                      : "Local test evaluation · empty tests are skipped"
                  }
                >
                  {rounds.length ? (
                    <div className="chart">
                      <ResponsiveContainer>
                        <LineChart data={rounds}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            vertical={false}
                          />
                          <XAxis
                            dataKey="round"
                            tickLine={false}
                            axisLine={false}
                          />
                          <YAxis
                            unit="%"
                            domain={[0, 100]}
                            tickLine={false}
                            axisLine={false}
                          />
                          <Tooltip />
                          <Line
                            dataKey="accuracy"
                            name="Accuracy"
                            stroke="#0e9387"
                            strokeWidth={2.5}
                            dot={{ r: 3 }}
                            isAnimationActive={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <Empty title="Awaiting completed rounds">
                      The chart updates as evaluation results arrive.
                    </Empty>
                  )}
                </Card>
                <Card
                  title={
                    coordinator ? "Institution readiness" : "Local progress"
                  }
                  subtitle="A fixed roster keeps the session configuration consistent"
                >
                  {coordinator ? (
                    <div className="readiness-list">
                      {Object.entries(run.participants || {}).map(([id, p]) => (
                        <div key={id}>
                          <span className="avatar small">
                            {p.client_id + 1}
                          </span>
                          <div>
                            <strong>
                              {o.nodes.find((n) => n.id === id)?.name ||
                                id.slice(0, 8)}
                            </strong>
                            <small>
                              Client {p.client_id}
                              {p.train_samples != null
                                ? ` · ${p.train_samples.toLocaleString()} train / ${p.test_samples?.toLocaleString()} test`
                                : ""}
                              {p.test_samples === 0
                                ? " · evaluation skipped"
                                : ""}
                              {p.error ? ` · ${p.error}` : ""}
                            </small>
                          </div>
                          <Badge value={p.status} />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="event-list">
                      {(events.data || [])
                        .slice(-7)
                        .reverse()
                        .map((e) => (
                          <div key={e.id}>
                            <span className="event-dot" />
                            <div>
                              <strong>{e.phase.replaceAll("_", " ")}</strong>
                              <small>
                                Round {e.round}
                                {e.epoch ? ` · epoch ${e.epoch}` : ""} ·{" "}
                                {new Date(e.time * 1000).toLocaleTimeString()}
                              </small>
                            </div>
                            {e.accuracy != null && <b>{pct(e.accuracy)}</b>}
                          </div>
                        ))}
                    </div>
                  )}
                </Card>
              </div>
              <div className="run-meta">
                <span>
                  Partition identity:{" "}
                  <code>
                    {run.partition_hash?.slice(0, 20) || "Preparing…"}
                  </code>
                </span>
                <span>Model: {run.spec.model}</span>
                <span>Weight decay: {run.spec.weight_decay}</span>
                {job && (
                  <button
                    className="text-link"
                    onClick={() =>
                      api<string>(`/jobs/${job.id}/log`)
                        .then(setLog)
                        .catch(() => {})
                    }
                  >
                    Open process log <ArrowUpRight size={14} />
                  </button>
                )}
              </div>
              <RunMonitor key={run.id} run={run} overview={o} />
            </section>
          )}
        </>
      )}
      {create && (
        <Modal
          title="Create a training session"
          onClose={() => setCreate(false)}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault();
              action<Run>("/runs", spec)
                .then((r) => {
                  setSelected(r.id);
                  setCreate(false);
                })
                .catch(() => {});
            }}
          >
            <div className="info-note">
              <FlaskConical size={20} />
              <p>
                The demonstration preset runs three rounds with one local epoch.
                Use a full-run model for the prediction demonstration.
              </p>
            </div>
            <Field label="Session name">
              <input
                value={spec.name}
                onChange={(e) => update("name", e.target.value)}
                required
              />
            </Field>
            <div className="form-grid">
              <Field label="Dataset">
                <select
                  value={spec.dataset}
                  onChange={(e) =>
                    update("dataset", e.target.value as DatasetName)
                  }
                >
                  {datasets.map((d) => (
                    <option key={d} value={d}>
                      {names[d]}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Method">
                <select
                  value={spec.strategy}
                  onChange={(e) =>
                    update("strategy", e.target.value as RunSpec["strategy"])
                  }
                >
                  {strategies.map((d) => (
                    <option key={d} value={d}>
                      {names[d]}
                      {d === "fedbns" ? " · FedBN + SCAFFOLD" : ""}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Partition">
                <select
                  value={spec.partition}
                  onChange={(e) =>
                    update("partition", e.target.value as RunSpec["partition"])
                  }
                >
                  {[
                    "dirichlet_0.3",
                    "dirichlet_0.1",
                    "dirichlet_1.0",
                    "practical",
                  ].map((x) => (
                    <option key={x}>{x}</option>
                  ))}
                </select>
              </Field>
              <Field label="Model">
                <select
                  value={spec.model}
                  onChange={(e) =>
                    update("model", e.target.value as RunSpec["model"])
                  }
                >
                  {["baseline_bn", "baseline", "cnn"].map((x) => (
                    <option key={x}>{x}</option>
                  ))}
                </select>
              </Field>
              {(
                [
                  ["rounds", "Rounds"],
                  ["local_epochs", "Local epochs"],
                  ["batch_size", "Batch size"],
                  ["seed", "Seed"],
                ] as const
              ).map(([key, label]) => (
                <Field label={label} key={key}>
                  <input
                    type="number"
                    min={key === "seed" ? 0 : 1}
                    value={spec[key]}
                    onChange={(e) => update(key, Number(e.target.value))}
                    required
                  />
                </Field>
              ))}
            </div>
            <details className="advanced">
              <summary>
                <Settings2 size={16} />
                Optimizer & method settings
              </summary>
              <div className="form-grid">
                {(
                  [
                    ["learning_rate", "Learning rate"],
                    ["momentum", "Momentum"],
                    ["weight_decay", "Weight decay"],
                    ["round_timeout", "Round timeout (seconds)"],
                    ...(spec.strategy === "scr"
                      ? [
                          ["scr_gamma", "SCR gamma"],
                          ["scr_head_lambda", "SCR head lambda"],
                        ]
                      : []),
                  ] as [keyof RunSpec, string][]
                ).map(([key, label]) => (
                  <Field label={label} key={key}>
                    <input
                      type="number"
                      step="any"
                      min="0"
                      value={spec[key] as number}
                      onChange={(e) => update(key, Number(e.target.value))}
                    />
                  </Field>
                ))}
              </div>
              <p>
                SGD · fixed learning rate · no augmentation · local test
                evaluation
              </p>
            </details>
            <Field
              label="Participating institutions"
              hint="Select the full roster. A partition is generated for exactly this client count."
            >
              <div className="node-choices">
                {o.nodes.map((n) => (
                  <label key={n.id}>
                    <input
                      type="checkbox"
                      disabled={n.connection_status !== "online"}
                      checked={spec.node_ids.includes(n.id)}
                      onChange={(e) =>
                        update(
                          "node_ids",
                          e.target.checked
                            ? [...spec.node_ids, n.id]
                            : spec.node_ids.filter((x) => x !== n.id),
                        )
                      }
                    />
                    <span>{n.name}</span>
                    <Badge value={n.connection_status} />
                  </label>
                ))}
                {!o.nodes.length && (
                  <p>Pair an institution before creating a session.</p>
                )}
              </div>
            </Field>
            <button
              className="primary full"
              disabled={busy || !spec.node_ids.length}
            >
              Create session & invite institutions
              <ArrowRight size={16} />
            </button>
          </form>
        </Modal>
      )}
      {confirm && (
        <Modal
          title="Cancel this training session?"
          onClose={() => setConfirm(null)}
        >
          <p className="modal-description">
            The service will stop this session’s processes. Partial results
            remain available. A cancelled session is restarted as a new run.
          </p>
          <div className="heading-actions">
            <button className="secondary" onClick={() => setConfirm(null)}>
              Keep training
            </button>
            <button
              className="primary destructive"
              disabled={busy}
              onClick={() =>
                action(`/runs/${confirm.id}/cancel`)
                  .then(() => setConfirm(null))
                  .catch(() => {})
              }
            >
              Cancel session
            </button>
          </div>
        </Modal>
      )}
      {log !== null && (
        <Modal title="Training process log" onClose={() => setLog(null)}>
          <pre className="log">{log}</pre>
        </Modal>
      )}
    </>
  );
}

function DatasetPage({ o, action, busy }: Context) {
  const [selected, setSelected] = useState<DatasetName>("bloodmnist");
  const data = o.datasets.find((d) => d.dataset === selected);
  return (
    <>
      <Heading
        eyebrow="LOCAL DATA"
        title="Dataset workspace"
        description="Inspect this device’s assigned data and prepare public benchmarks before training."
      >
        <select
          className="compact-select"
          value={selected}
          onChange={(e) => setSelected(e.target.value as DatasetName)}
        >
          {datasets.map((d) => (
            <option value={d} key={d}>
              {names[d]}
            </option>
          ))}
        </select>
        <button
          className="primary"
          disabled={busy}
          onClick={() =>
            action("/datasets/prepare", { dataset: selected }).catch(() => {})
          }
        >
          <ArrowDownToLine size={17} />
          {data ? "Check prepared data" : "Prepare dataset"}
        </button>
      </Heading>
      {data ? (
        <>
          <div className="metrics-grid three">
            <Metric
              label={
                data.assigned
                  ? "Assigned training samples"
                  : "Benchmark training samples"
              }
              value={data.train_samples.toLocaleString()}
              detail={
                data.assigned
                  ? "Partition verified against local labels"
                  : "A local partition is assigned when you accept a session"
              }
              icon={<Database size={19} />}
            />
            <Metric
              label="Test samples"
              value={data.test_samples.toLocaleString()}
              detail={
                data.assigned
                  ? "Local evaluation partition"
                  : "Public benchmark test split"
              }
              icon={<ImagePlus size={19} />}
            />
            <Metric
              label="Classes"
              value={Object.keys(data.classes).length}
              detail={names[data.dataset]}
              icon={<Layers3 size={19} />}
            />
          </div>
          <Card
            title="Training class distribution"
            subtitle={
              data.assigned
                ? "Samples assigned to this institution"
                : "Full public benchmark · assignment pending"
            }
          >
            <div className="chart">
              <ResponsiveContainer>
                <BarChart
                  data={data.class_distribution.map((count, i) => ({
                    label: data.classes[String(i)],
                    count,
                  }))}
                >
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 11 }}
                    interval={0}
                    tickFormatter={(s) => String(s).slice(0, 13)}
                  />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#0e9387" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card
            title="Benchmark image previews"
            subtitle="28 × 28 input patches · images served directly by this institution"
            className="section-gap"
          >
            <div className="sample-grid">
              {data.previews.map((p) => (
                <div className="sample-card" key={p.index}>
                  <img
                    src={p.image}
                    alt={`${data.classes[String(p.label)]} benchmark sample ${p.index}`}
                  />
                  <strong>{data.classes[String(p.label)]}</strong>
                  <small>Sample #{p.index}</small>
                </div>
              ))}
            </div>
            {!data.previews.length && (
              <Empty title="No local test images">
                This partition has an empty local test split. Evaluation will be
                skipped for this client.
              </Empty>
            )}
          </Card>
        </>
      ) : (
        <Card
          title={names[selected]}
          subtitle="Prepare once before going offline"
        >
          <Empty
            icon={<Database size={30} />}
            title="Dataset has not been prepared"
          >
            Prepare dataset checks the local MedMNIST files and downloads this
            public benchmark if it is missing. Follow the task status on
            Overview.
          </Empty>
        </Card>
      )}
    </>
  );
}

function Experiments({ action, busy }: Context) {
  const query = useQuery({
    queryKey: ["experiments"],
    queryFn: () => api<Experiment[]>("/experiments"),
  });
  const [dataset, setDataset] = useState("bloodmnist"),
    [partition, setPartition] = useState("dirichlet_0.3"),
    [clientCount, setClientCount] = useState("12"),
    [model, setModel] = useState("baseline_bn"),
    [source, setSource] = useState("recorded");
  const rows = (query.data || []).filter(
    (e) =>
      e.dataset === dataset &&
      e.partition === partition &&
      e.model === model &&
      e.source === source &&
      String(e.num_clients) === clientCount,
  );
  const grouped = strategies
    .map((strategy) => {
      const rs = rows.filter(
        (r) =>
          r.strategy === strategy && r.bta != null && r.status === "completed",
      );
      const vals = rs.map((r) => r.bta!);
      const mean = vals.length
        ? vals.reduce((a, b) => a + b, 0) / vals.length
        : null;
      return {
        strategy,
        label: names[strategy],
        n: vals.length,
        accuracy: mean == null ? null : mean * 100,
        mean,
        sd:
          vals.length > 1
            ? Math.sqrt(
                vals.reduce((sum, v) => sum + (v - mean!) ** 2, 0) /
                  (vals.length - 1),
              )
            : null,
      };
    })
    .filter((x) => x.n);
  return (
    <>
      <Heading
        eyebrow="RESEARCH EVIDENCE"
        title="Experiment library"
        description="Explore completed results with their original dataset, partition, and configuration."
      >
        <button
          className="primary"
          disabled={busy}
          onClick={() => action("/experiments/import").catch(() => {})}
        >
          <ArrowDownToLine size={17} />
          Import result folders
        </button>
      </Heading>
      <div className="filter-bar">
        <span>
          <Search size={17} />
          Compare experiments
        </span>
        <select
          aria-label="Dataset filter"
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
        >
          {datasets.map((x) => (
            <option value={x} key={x}>
              {names[x]}
            </option>
          ))}
        </select>
        <select
          aria-label="Partition filter"
          value={partition}
          onChange={(e) => setPartition(e.target.value)}
        >
          {["dirichlet_0.3", "dirichlet_0.1", "practical", "dirichlet_1.0"].map(
            (x) => (
              <option key={x}>{x}</option>
            ),
          )}
        </select>
        <select
          aria-label="Model filter"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          {["baseline_bn", "baseline", "cnn"].map((x) => (
            <option key={x}>{x}</option>
          ))}
        </select>
        <select
          aria-label="Client count filter"
          value={clientCount}
          onChange={(e) => setClientCount(e.target.value)}
        >
          {[
            ...new Set([
              3,
              4,
              5,
              12,
              ...(query.data || []).map((r) => r.num_clients).filter(Boolean),
            ]),
          ]
            .sort((a, b) => a - b)
            .map((x) => (
              <option value={x} key={x}>
                {x} clients
              </option>
            ))}
        </select>
        <select
          aria-label="Source filter"
          value={source}
          onChange={(e) => setSource(e.target.value)}
        >
          <option value="recorded">Saved experiments</option>
          <option value="live">Live federation sessions</option>
        </select>
      </div>
      {rows.length ? (
        <>
          <div className="dashboard-grid">
            <Card
              title="Best test accuracy"
              subtitle="Mean across completed runs · exploratory test-selected metric"
            >
              <div className="chart">
                <ResponsiveContainer>
                  <BarChart data={grouped} barSize={42}>
                    <CartesianGrid vertical={false} strokeDasharray="3 3" />
                    <XAxis dataKey="label" axisLine={false} tickLine={false} />
                    <YAxis
                      domain={[0, 100]}
                      unit="%"
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
                    <Bar
                      dataKey="accuracy"
                      name="Mean BTA"
                      radius={[5, 5, 0, 0]}
                    >
                      {grouped.map((r, i) => (
                        <Cell key={r.strategy} fill={colors[i]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card
              title="Across-run summary"
              subtitle="Sample standard deviation; inspect individual configurations below"
            >
              <table>
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Runs</th>
                    <th>BTA · mean ± SD</th>
                  </tr>
                </thead>
                <tbody>
                  {grouped.map((g) => (
                    <tr key={g.strategy}>
                      <td>
                        <strong>{g.label}</strong>
                      </td>
                      <td>{g.n}</td>
                      <td>
                        {pct(g.mean)}{" "}
                        {g.sd != null
                          ? `± ${(g.sd * 100).toFixed(2)} pp`
                          : "· SD unavailable"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="info-note compact">
                <FlaskConical size={19} />
                <p>
                  Shared filters do not establish a matched comparison. Check
                  seed, complete configuration, and partition hash for each run.
                </p>
              </div>
            </Card>
          </div>
          <Card
            title={`${rows.length} experiment records`}
            subtitle="Historical files are read without modification"
          >
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Method / source</th>
                    <th>Seed</th>
                    <th>Rounds</th>
                    <th>BTA</th>
                    <th>BMCTA</th>
                    <th>Final accuracy</th>
                    <th>Final loss</th>
                    <th>Time</th>
                    <th>Partition identity</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.id}>
                      <td>
                        <strong>{names[r.strategy] || r.strategy}</strong>
                        <small>
                          {r.source} · {r.model}
                        </small>
                      </td>
                      <td>{r.seed}</td>
                      <td>{r.rounds}</td>
                      <td>{pct(r.bta)}</td>
                      <td>{pct(r.bmcta)}</td>
                      <td>{pct(r.final_accuracy)}</td>
                      <td>{r.loss?.toFixed(4) ?? "—"}</td>
                      <td>{duration(r.time_seconds)}</td>
                      <td>
                        <code>
                          {r.partition_hash?.slice(0, 12) || "Unknown"}
                        </code>
                        <details>
                          <summary>Configuration</summary>
                          <pre className="mini-log">
                            {JSON.stringify(r.configuration, null, 2)}
                          </pre>
                        </details>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      ) : (
        <Card
          title="Experiments will appear here"
          subtitle="Import the existing result folders, then choose matching filters."
        >
          <Empty
            icon={<FlaskConical size={30} />}
            title="No records for these filters"
          >
            Your saved 12-client experiments and live device sessions have
            different client counts. Use the filters to inspect each group.
          </Empty>
        </Card>
      )}
    </>
  );
}

function Models({ o, action, busy }: Context) {
  const query = useQuery({
    queryKey: ["models"],
    queryFn: () =>
      api<{ local: Model[]; remote: Model[]; institution: Model[] }>("/models"),
    refetchInterval: 2000,
  });
  const [show, setShow] = useState(false),
    [form, setForm] = useState<ModelImport>({
      path: "",
      dataset: "bloodmnist",
      model: "baseline_bn",
      strategy: "scr",
      selection_rule: "final_round",
      source_run: "",
      source_round: 80,
    });
  const candidates = useQuery({
    queryKey: ["model-candidates"],
    queryFn: () => api<ModelImport[]>("/models/candidates"),
    enabled: show,
  });
  const local = query.data?.local || [],
    remote = (query.data?.remote || []).filter(
      (m) => !local.some((x) => x.id === m.id),
    );
  function modelCard(m: Model, download = false, remoteSite = false) {
    return (
      <section className="card model-card" key={`${m.id}-${remoteSite}`}>
        <div className="device-top">
          <span className="model-symbol">
            <Layers3 size={26} />
          </span>
          <span className="badge ready">
            <CheckCircle2 size={12} />
            {download ? "Available" : "Verified"}
          </span>
        </div>
        <span className="eyebrow">
          {m.scope === "institution" ? "INSTITUTION MODEL" : "SHARED MODEL"}
        </span>
        <h2>{names[m.dataset]}</h2>
        <p>
          {names[m.strategy]} · {m.model}
        </p>
        <div className="model-meta">
          <div>
            <span>Source round</span>
            <strong>{m.source_round}</strong>
          </div>
          <div>
            <span>Classes</span>
            <strong>{Object.keys(m.classes).length}</strong>
          </div>
          <div>
            <span>Selection</span>
            <strong>{m.selection_rule.replaceAll("_", " ")}</strong>
          </div>
        </div>
        {m.selection_rule === "best_test_round" && (
          <p className="selection-note">
            Exploratory checkpoint selected using test results.
          </p>
        )}
        <div className="model-hash">
          <ShieldCheck size={14} />
          <code>{m.checksum.slice(0, 18)}</code>
        </div>
        <details className="model-details">
          <summary>Provenance & evaluation</summary>
          <p>
            Source: {m.source_run} · round {m.source_round}
          </p>
          <p>
            {m.evaluation?.skipped
              ? "Evaluation skipped: no local test samples."
              : `Evaluation accuracy: ${pct(m.evaluation?.eval_accuracy)} · loss: ${m.evaluation?.eval_loss?.toFixed(4) ?? "Unavailable"}`}
          </p>
          {m.scope === "institution" && (
            <p>
              Institution:{" "}
              {o.nodes.find((n) => n.id === m.institution_id)?.name ||
                m.institution_id}
            </p>
          )}
        </details>
        {download ? (
          <button
            className="secondary full"
            disabled={busy}
            onClick={() => action(`/models/${m.id}/download`).catch(() => {})}
          >
            <ArrowDownToLine size={15} />
            Download & verify locally
          </button>
        ) : o.mode === "institution" ? (
          <Link className="secondary full" to="/predict">
            Use for prediction
            <ArrowRight size={15} />
          </Link>
        ) : (
          <small>
            {remoteSite
              ? "Stored on its owning institution"
              : "Available for paired institutions to download"}
          </small>
        )}
      </section>
    );
  }
  return (
    <>
      <Heading
        eyebrow="MODEL REGISTRY"
        title="Verified models"
        description="Trace every prediction to a checkpoint, training method, and source experiment."
      >
        <button className="primary" onClick={() => setShow(true)}>
          <Plus size={17} />
          Import checkpoint
        </button>
      </Heading>
      {local.length ? (
        <div className="device-grid">{local.map((m) => modelCard(m))}</div>
      ) : (
        <Card
          title="Build your model library"
          subtitle="Completed sessions register their final checkpoints automatically."
        >
          <Empty
            icon={<Layers3 size={30} />}
            title="No verified local checkpoints yet"
          >
            Import a project-generated checkpoint or complete a training
            session. Each checkpoint is checked against its architecture and
            dataset.
          </Empty>
        </Card>
      )}
      {remote.length > 0 && (
        <>
          <div className="section-title">
            <h2>Available from your coordinator</h2>
            <p>
              Download a global model to run predictions on this institution.
            </p>
          </div>
          <div className="device-grid">
            {remote.map((m) => modelCard(m, true))}
          </div>
        </>
      )}
      {!!query.data?.institution.length && (
        <>
          <div className="section-title">
            <h2>Institution-specific models</h2>
            <p>
              FedBN and FedBNS retain each institution’s local BatchNorm state.
            </p>
          </div>
          <div className="device-grid">
            {query.data.institution.map((m) => modelCard(m, false, true))}
          </div>
        </>
      )}
      {show && (
        <Modal
          title="Import a research checkpoint"
          onClose={() => setShow(false)}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault();
              action("/models/import", form)
                .then(() => setShow(false))
                .catch(() => {});
            }}
          >
            <p className="modal-description">
              Use a trusted .pth file inside this device’s configured results
              directories. FedBN/FedBNS checkpoints are registered by their
              owning institution after a live session.
            </p>
            {!!candidates.data?.length && (
              <Field
                label="Existing project checkpoint"
                hint="Metadata is filled from the experiment. The checkpoint is validated before registration."
              >
                <select
                  value={candidates.data.findIndex((c) => c.path === form.path)}
                  onChange={(e) => {
                    const candidate = candidates.data?.[Number(e.target.value)];
                    if (candidate) setForm(candidate);
                  }}
                >
                  <option value={-1}>
                    Choose a saved checkpoint, or enter details below
                  </option>
                  {candidates.data.map((c, i) => (
                    <option value={i} key={c.path}>
                      {names[c.dataset]} · {names[c.strategy]} · {c.source_run}{" "}
                      · R{c.source_round}
                    </option>
                  ))}
                </select>
              </Field>
            )}
            <Field label="Checkpoint path on this device">
              <input
                value={form.path}
                onChange={(e) => setForm({ ...form, path: e.target.value })}
                placeholder="D:\…\results\…\final_model.pth"
                required
              />
            </Field>
            <div className="form-grid">
              <Field label="Dataset">
                <select
                  value={form.dataset}
                  onChange={(e) =>
                    setForm({ ...form, dataset: e.target.value as DatasetName })
                  }
                >
                  {datasets.map((x) => (
                    <option value={x} key={x}>
                      {names[x]}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Architecture">
                <select
                  value={form.model}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      model: e.target.value as ModelImport["model"],
                    })
                  }
                >
                  {["baseline_bn", "baseline", "cnn"].map((x) => (
                    <option key={x}>{x}</option>
                  ))}
                </select>
              </Field>
              <Field label="Method">
                <select
                  value={form.strategy}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      strategy: e.target.value as ModelImport["strategy"],
                    })
                  }
                >
                  {["fedavg", "scaffold", "scr"].map((x) => (
                    <option value={x} key={x}>
                      {names[x]}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Checkpoint selection">
                <select
                  value={form.selection_rule}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      selection_rule: e.target
                        .value as ModelImport["selection_rule"],
                    })
                  }
                >
                  {[
                    "final_round",
                    "best_test_round",
                    "validation_selected",
                  ].map((x) => (
                    <option value={x} key={x}>
                      {x.replaceAll("_", " ")}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Source run name">
                <input
                  required
                  value={form.source_run}
                  onChange={(e) =>
                    setForm({ ...form, source_run: e.target.value })
                  }
                />
              </Field>
              <Field label="Source round">
                <input
                  type="number"
                  min="0"
                  required
                  value={form.source_round}
                  onChange={(e) =>
                    setForm({ ...form, source_round: Number(e.target.value) })
                  }
                />
              </Field>
            </div>
            <button className="primary full" disabled={busy}>
              Verify & register checkpoint
              <ArrowRight size={16} />
            </button>
          </form>
        </Modal>
      )}
    </>
  );
}

function Predict({ o, action, busy }: Context) {
  const [modelId, setModelId] = useState(""),
    [sample, setSample] = useState<number | null>(null),
    [image, setImage] = useState<string | null>(null),
    [fileName, setFileName] = useState(""),
    [jobId, setJobId] = useState<string | null>(null),
    [uploadError, setUploadError] = useState("");
  const model = o.models.find((m) => m.id === modelId) || o.models[0];
  const data = o.datasets.find((d) => d.dataset === model?.dataset);
  const job = useQuery({
    queryKey: ["prediction", jobId],
    queryFn: () => api<Job>(`/jobs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (q) =>
      ["queued", "running"].includes(q.state.data?.status || "queued")
        ? 1000
        : false,
  });
  const prediction =
    job.data?.status === "completed"
      ? (job.data.result as Prediction)
      : undefined;
  const pending = job.data && ["queued", "running"].includes(job.data.status);
  function selectModel(value: string) {
    setModelId(value);
    setSample(null);
    setImage(null);
    setJobId(null);
  }
  function upload(file?: File) {
    if (!file) return;
    setUploadError("");
    if (
      !["image/png", "image/jpeg"].includes(file.type) ||
      file.size > 5_000_000
    ) {
      setUploadError("Choose a PNG/JPEG image smaller than 5 MB.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setImage(String(reader.result));
      setFileName(file.name);
      setSample(null);
      setJobId(null);
    };
    reader.readAsDataURL(file);
  }
  return (
    <>
      <Heading
        eyebrow="LOCAL IMAGE CLASSIFICATION"
        title="Explore a prediction"
        description="Run a verified model on this institution. The image stays with this service."
      >
        <span className="local-pill">
          <ShieldCheck size={16} />
          Local inference
        </span>
      </Heading>
      {!model ? (
        <Card title="A model is needed to begin">
          <Empty
            icon={<Layers3 size={30} />}
            title="Download or import a checkpoint"
          >
            <Link className="primary" to="/models">
              Open model registry
              <ArrowRight size={16} />
            </Link>
          </Empty>
        </Card>
      ) : (
        <div className="prediction-grid">
          <div>
            <Card
              title="Choose your input"
              subtitle="Use an image patch appropriate to the selected benchmark."
            >
              <div className="prediction-input">
                <Field label="Verified model">
                  <select
                    value={model.id}
                    onChange={(e) => selectModel(e.target.value)}
                  >
                    {o.models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {names[m.dataset]} · {names[m.strategy]} · Round{" "}
                        {m.source_round}
                      </option>
                    ))}
                  </select>
                </Field>
                <label
                  className="upload-zone"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    upload(e.dataTransfer.files[0]);
                  }}
                >
                  <input
                    type="file"
                    accept="image/png,image/jpeg"
                    onChange={(e) => upload(e.target.files?.[0])}
                  />
                  {image ? (
                    <>
                      <img src={image} alt="Selected image patch" />
                      <strong>{fileName}</strong>
                      <small>Click to choose another image</small>
                    </>
                  ) : (
                    <>
                      <ImagePlus size={32} />
                      <strong>Drop an image here</strong>
                      <small>or click to browse · PNG/JPEG · up to 5 MB</small>
                    </>
                  )}
                </label>
                {uploadError && <p className="error-inline">{uploadError}</p>}
                <div className="input-divider">
                  <span />
                  or use a local benchmark sample
                  <span />
                </div>
                {data?.previews.length ? (
                  <div className="prediction-samples">
                    {data.previews.slice(0, 8).map((p) => (
                      <button
                        className={sample === p.index ? "selected" : ""}
                        key={p.index}
                        title={data.classes[String(p.label)]}
                        onClick={() => {
                          setSample(p.index);
                          setImage(null);
                          setJobId(null);
                        }}
                      >
                        <img
                          src={p.image}
                          alt={`Benchmark sample ${p.index}`}
                        />
                        <small>#{p.index}</small>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="subtle">
                    Prepare {names[model.dataset]} on the Dataset page to use
                    benchmark samples.
                  </p>
                )}
                <button
                  className="primary full"
                  disabled={
                    busy ||
                    !!pending ||
                    o.jobs.some((j) =>
                      ["queued", "running"].includes(j.status),
                    ) ||
                    (sample === null && !image)
                  }
                  onClick={() =>
                    action<Job>("/predict", {
                      model_id: model.id,
                      sample_index: sample,
                      image_base64: image?.split(",")[1] ?? null,
                    })
                      .then((j) => setJobId(j.id))
                      .catch(() => {})
                  }
                >
                  {pending ? (
                    <Loader2 size={17} className="spin" />
                  ) : (
                    <Play size={16} />
                  )}
                  Classify image
                </button>
                <small className="preprocess">
                  Model input: 28 × 28 · deterministic preprocessing · no
                  augmentation
                </small>
              </div>
            </Card>
          </div>
          <div>
            <Card
              title="Prediction result"
              subtitle="Model scores describe this classifier’s output."
            >
              {prediction ? (
                <div className="prediction-result">
                  <span className="prediction-check">
                    <CheckCircle2 size={26} />
                  </span>
                  <span className="eyebrow">PREDICTED CLASS</span>
                  <h2>{prediction.label}</h2>
                  <p>
                    {names[prediction.dataset]} · {names[model.strategy]}
                  </p>
                  <div className="score-list">
                    {prediction.top.map((p, i) => (
                      <div key={p.class_id}>
                        <div>
                          <span>
                            {i + 1}. {p.label}
                          </span>
                          <strong>{pct(p.score)}</strong>
                        </div>
                        <div className="score-track">
                          <i
                            style={{
                              width: `${p.score * 100}%`,
                              background: colors[i],
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="prediction-foot">
                    <span>
                      Inference time<strong>{prediction.latency_ms} ms</strong>
                    </span>
                    <span>
                      Model version
                      <strong>{prediction.model_id.slice(0, 10)}</strong>
                    </span>
                  </div>
                  <div className="info-note compact">
                    <FlaskConical size={18} />
                    <p>
                      Research classification on MedMNIST-style images. Scores
                      are not calibrated diagnostic confidence.
                    </p>
                  </div>
                </div>
              ) : pending ? (
                <Empty
                  icon={<Loader2 className="spin" size={28} />}
                  title="Running local inference"
                >
                  The prediction worker is loading the verified checkpoint.
                </Empty>
              ) : job.data?.status === "failed" ? (
                <Empty
                  icon={<XCircle size={28} />}
                  title="Prediction could not complete"
                >
                  {job.data.error}
                </Empty>
              ) : (
                <Empty
                  icon={<ImagePlus size={30} />}
                  title="Your result will appear here"
                >
                  Select a model and an image, then run classification to see
                  its top predictions.
                </Empty>
              )}
            </Card>
            <div className="info-note section-gap">
              <Layers3 size={20} />
              <p>
                <strong>Traceable by design</strong>
                <br />
                Source: {model.source_run} · round {model.source_round}
                <br />
                {model.selection_rule.replaceAll("_", " ")} · {model.scope}{" "}
                checkpoint
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
