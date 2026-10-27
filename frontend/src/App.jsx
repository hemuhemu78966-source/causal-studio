import React, { useState, useEffect, useRef } from "react";
import {
  ArrowUpRight,
  ArrowRight,
  ArrowDownToLine,
  BrainCircuit,
  ChartNoAxesCombined,
  Check,
  ChevronDown,
  ChevronRight,
  Database,
  FlaskConical,
  GitBranch,
  Layers,
  LoaderCircle,
  LockKeyhole,
  Menu,
  MousePointer2,
  Network,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  Upload,
  X,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Info,
} from "lucide-react";
import Papa from "papaparse";
const NAV = [
  ["Overview", Layers],
  ["Causal graph", Network],
  ["Interventions", SlidersHorizontal],
  ["Action planner", Target],
  ["Model lab", FlaskConical],
  ["Dataset", Database],
];
const DOMAINS = {
  fintech: ["Credit intelligence", "Finance"],
  healthcare: ["Clinical outcomes", "Healthcare"],
  ecommerce: ["Customer retention", "Commerce"],
  education: ["Student success", "Education"],
  hr: ["Workforce growth", "People"],
  custom: ["Your dataset", "Custom CSV"],
};
const ALGOS = {
  hill: "Hill-Climbing · BIC",
  pc: "PC · Independence tests",
  tan: "Tree-Augmented Naive Bayes",
  reference: "Known benchmark graph",
};
const pretty = (s) => String(s).replaceAll("_", " ");
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const fmt = (x) => Number(x).toLocaleString();
async function api(path, body) {
  const r = await fetch(
    "/api/" + path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  const j = await r.json();
  if (!r.ok)
    throw Error(
      typeof j.detail === "string"
        ? j.detail
        : "Check your inputs and try again.",
    );
  return j;
}
function Badge({ children, tone = "" }) {
  return <span className={"badge " + tone}>{children}</span>;
}
function Heading({ eyebrow, title, description, children }) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {children}
    </div>
  );
}
function Card({ title, kicker, children, className = "", action }) {
  return (
    <section className={"card " + className}>
      <div className="card-head">
        <div>
          {kicker && <span className="eyebrow">{kicker}</span>}
          <h3>{title}</h3>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}
function Select({ label, value, onChange, options, hint }) {
  return (
    <Field label={label} hint={hint}>
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((x) => (
          <option key={x.value ?? x} value={x.value ?? x}>
            {x.label ?? x}
          </option>
        ))}
      </select>
    </Field>
  );
}
function Probability({ value, label, color = "lime" }) {
  return (
    <div className="prob">
      <div>
        <span>{label}</span>
        <strong>{pct(value)}</strong>
      </div>
      <div className="track">
        <i className={color} style={{ width: pct(value) }} />
      </div>
    </div>
  );
}
function Empty({ title, description, icon: Icon = Sparkles }) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Icon size={28} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}
function download(data, name) {
  const u = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}
function Graph({ model, reference = false, compact = false, cut = null }) {
  const nodes = model.nodes,
    edges = (reference ? model.reference_edges : model.edges) || [];
  const [selected, setSelected] = useState(null),
    [zoom, setZoom] = useState(1),
    [positions, setPositions] = useState({});
  const svg = useRef(null),
    drag = useRef(null);
  const uid = useRef("g" + Math.random().toString(36).slice(2)).current;
  useEffect(() => {
    setPositions({});
    setSelected(null);
    setZoom(1);
  }, [model.id, reference]);
  const levels = {};
  for (const n of nodes) levels[n] = 0;
  for (let i = 0; i < nodes.length; i++)
    for (const [a, b] of edges)
      levels[b] = Math.max(
        levels[b],
        Math.min(nodes.length - 1, levels[a] + 1),
      );
  const max = Math.max(...Object.values(levels), 1),
    groups = {};
  nodes.forEach((n) => (groups[levels[n]] ??= []).push(n));
  const point = (n) =>
    positions[n] || {
      x: 90 + levels[n] * (650 / max),
      y:
        50 +
        ((groups[levels[n]].indexOf(n) + 0.5) * 320) / groups[levels[n]].length,
    };
  const connected = new Set([
    selected,
    ...edges.filter((e) => e.includes(selected)).flat(),
  ]);
  function move(e) {
    if (!drag.current) return;
    const rect = svg.current.getBoundingClientRect();
    setPositions((p) => ({
      ...p,
      [drag.current]: {
        x: Math.min(
          780,
          Math.max(
            50,
            (((e.clientX - rect.left) / rect.width) * 840 - 420 * (1 - zoom)) /
              zoom,
          ),
        ),
        y: Math.min(
          375,
          Math.max(
            35,
            (((e.clientY - rect.top) / rect.height) * 440 - 220 * (1 - zoom)) /
              zoom,
          ),
        ),
      },
    }));
  }
  return (
    <div className={"graph " + (compact ? "compact" : "")}>
      <div className="graph-top">
        <span>
          <i className="live-dot" />{" "}
          {reference ? "KNOWN STRUCTURE" : "LEARNED STRUCTURE"}
        </span>
        <span>
          {nodes.length} nodes <b>·</b> {edges.length} edges
        </span>
      </div>
      <div className="graph-viewport">
        <svg
          ref={svg}
          viewBox="0 0 840 440"
          role="img"
          aria-label="Interactive directed Bayesian network. Drag nodes to rearrange."
          onPointerMove={move}
          onPointerUp={() => (drag.current = null)}
          onPointerCancel={() => (drag.current = null)}
        >
          <defs>
            <pattern
              id={uid + "grid"}
              width="24"
              height="24"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="1" cy="1" r=".7" fill="#ffffff18" />
            </pattern>
            <marker
              id={uid + "arrow"}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#86aac5" />
            </marker>
          </defs>
          <rect width="840" height="440" fill={`url(#${uid}grid)`} />
          <g
            transform={`translate(${420 * (1 - zoom)},${220 * (1 - zoom)}) scale(${zoom})`}
          >
            {edges.map(([a, b], i) => {
              let p = point(a),
                q = point(b);
              const d = `M ${p.x + 13} ${p.y} C ${p.x + (q.x - p.x) * 0.55} ${p.y},${p.x + (q.x - p.x) * 0.45} ${q.y},${q.x - 26} ${q.y}`;
              const removed = b === cut;
              return (
                <g
                  key={a + b}
                  opacity={selected && !connected.has(a) ? ".15" : "1"}
                >
                  <path
                    d={d}
                    fill="none"
                    stroke={removed ? "#f78787" : "#527991"}
                    strokeWidth="1.4"
                    strokeDasharray={removed ? "6 5" : undefined}
                    markerEnd={`url(#${uid}arrow)`}
                  />
                  {!removed && (
                    <circle r="2.8" fill="#5ce6da" className="flow-particle">
                      <animateMotion
                        dur={`${3 + i * 0.35}s`}
                        repeatCount="indefinite"
                        path={d}
                      />
                    </circle>
                  )}
                </g>
              );
            })}
            {nodes.map((n) => {
              const p = point(n),
                target = n === model.target;
              return (
                <g
                  key={n}
                  className="graph-node"
                  role="button"
                  tabIndex={0}
                  aria-label={`Inspect ${pretty(n)}`}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelected(n);
                    }
                  }}
                  transform={`translate(${p.x},${p.y})`}
                  opacity={selected && !connected.has(n) ? ".3" : "1"}
                  onPointerDown={(e) => {
                    e.currentTarget.setPointerCapture(e.pointerId);
                    drag.current = n;
                    setSelected(n);
                  }}
                  onDoubleClick={() => setSelected(null)}
                >
                  <circle
                    r={target ? 29 : 23}
                    fill={target ? "#5ce6da" : "#142d40"}
                    stroke={
                      target
                        ? "#8ef3ea"
                        : selected === n
                          ? "#5ce6da"
                          : "#5d8ca7"
                    }
                    strokeWidth="1.4"
                  />
                  {target ? (
                    <path
                      d="M -7 0 L -2 5 L 8 -6"
                      fill="none"
                      stroke="#092426"
                      strokeWidth="2.5"
                    />
                  ) : (
                    <circle r="5" fill="#bad4e8" />
                  )}
                  <text
                    y={target ? 49 : 44}
                    textAnchor="middle"
                    fill={target ? "#8ef3ea" : "#eef5ff"}
                    fontSize="14"
                    fontWeight="500"
                  >
                    {pretty(n)}
                  </text>
                  <title>
                    {pretty(n)} · {Object.keys(model.labels[n]).length} states
                    {target ? " · Target outcome" : ""}
                  </title>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      {selected && (
        <div className="node-inspector" role="status">
          <span className="inspector-icon">
            <Network size={18} />
          </span>
          <div>
            <strong>{pretty(selected)}</strong>
            <span>
              {edges.filter(([, b]) => b === selected).length} parents ·{" "}
              {edges.filter(([a]) => a === selected).length} children ·{" "}
              {Object.keys(model.labels[selected]).length} states
            </span>
          </div>
          <button
            className="icon-btn"
            aria-label="Clear node selection"
            onClick={() => setSelected(null)}
          >
            <X size={15} />
          </button>
        </div>
      )}
      <div className="graph-footer">
        <span>
          <MousePointer2 size={13} /> Drag to explore <b>·</b> Click to focus
        </span>
        <div>
          <button
            aria-label="Zoom out"
            onClick={() => setZoom(Math.max(0.6, zoom - 0.15))}
          >
            <ZoomOut size={15} />
          </button>
          <button
            aria-label="Reset graph"
            onClick={() => {
              setZoom(1);
              setPositions({});
              setSelected(null);
            }}
          >
            <RotateCcw size={14} />
          </button>
          <button
            aria-label="Zoom in"
            onClick={() => setZoom(Math.min(1.8, zoom + 0.15))}
          >
            <ZoomIn size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
function Donut({ value }) {
  return (
    <div className="donut">
      <svg viewBox="0 0 160 160">
        <circle
          cx="80"
          cy="80"
          r="61"
          fill="none"
          stroke="#edf0e8"
          strokeWidth="13"
        />
        <circle
          cx="80"
          cy="80"
          r="61"
          fill="none"
          stroke="#315d44"
          strokeWidth="13"
          strokeLinecap="round"
          strokeDasharray={`${value * 383} 383`}
          transform="rotate(-90 80 80)"
          className="donut-value"
        />
      </svg>
      <div>
        <strong>{pct(value)}</strong>
        <span>held-out accuracy</span>
      </div>
    </div>
  );
}
function MiniBars({ values }) {
  return (
    <div className="mini-bars" aria-hidden="true">
      {values.map((v, i) => (
        <i
          key={i}
          style={{
            height: `${25 + (v / Math.max(...values)) * 65}%`,
            animationDelay: `${i * 60}ms`,
          }}
        />
      ))}
    </div>
  );
}
function Overview({ m, go }) {
  const ev = m.evaluation[0];
  const improvement = 100 * (ev.accuracy - m.evaluation[2].accuracy);
  return (
    <>
      <div className="overview-intro">
        <div className="intro-copy">
          <div className="eyebrow">
            <span className="section-tick" /> A WORKSPACE FOR CAUSAL REASONING
          </div>
          <h1>
            Understand the system.
            <br />
            <span>Explore what changes.</span>
          </h1>
          <p>
            Trace relationships, test an intervention, and compare the paths to
            a better outcome.
          </p>
          <div className="hero-actions">
            <button className="primary" onClick={() => go("Interventions")}>
              <Play size={16} /> Launch a simulation <ArrowUpRight size={17} />
            </button>
            <button className="text-btn" onClick={() => go("Causal graph")}>
              Explore the network <ArrowRight size={16} />
            </button>
          </div>
        </div>
        <div className="experiment-ledger">
          <div className="ledger-heading">
            <span className="ledger-cross">+</span> ACTIVE EXPERIMENT{" "}
            <span className="live-dot" />
          </div>
          <strong>{m.title}</strong>
          <dl>
            <div>
              <dt>Learning method</dt>
              <dd>{ALGOS[m.algorithm]}</dd>
            </div>
            <div>
              <dt>Outcome</dt>
              <dd>{pretty(m.target)}</dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>{m.source}</dd>
            </div>
            <div>
              <dt>Random seed</dt>
              <dd>{m.seed.toString().padStart(4, "0")}</dd>
            </div>
          </dl>
          <div className="ledger-footer">
            <ShieldCheck size={15} /> Reproducible 80 / 20 train-test split
          </div>
        </div>
      </div>
      <div className="data-ribbon" aria-label="Workspace statistics">
        <div>
          <span className="ribbon-index">01</span>
          <div>
            <label>RECORDS ANALYZED</label>
            <strong>
              {fmt(m.rows)}
              <small>
                {fmt(m.train_rows)} train / {fmt(m.test_rows)} test
              </small>
            </strong>
          </div>
        </div>
        <div>
          <span className="ribbon-index">02</span>
          <div>
            <label>NETWORK STRUCTURE</label>
            <strong>
              {m.edges.length}
              <small>edges across {m.nodes.length} variables</small>
            </strong>
          </div>
        </div>
        <div>
          <span className="ribbon-index">03</span>
          <div>
            <label>HELD-OUT ACCURACY</label>
            <strong>
              {pct(ev.accuracy)}
              <small className="positive">
                {improvement >= 0 ? "+" : ""}
                {improvement.toFixed(1)} pp vs. baseline
              </small>
            </strong>
          </div>
        </div>
        <div>
          <span className="ribbon-index">04</span>
          <div>
            <label>MODEL STATUS</label>
            <strong className="status-value">
              <Check size={20} /> Validated
              <small>Acyclic · normalized probabilities</small>
            </strong>
          </div>
        </div>
      </div>
      <div className="observatory-grid">
        <Card
          title="The causal network"
          kicker="01 / STRUCTURE OBSERVATORY"
          className="graph-card"
          action={
            <button className="text-btn" onClick={() => go("Causal graph")}>
              Full explorer <ArrowUpRight size={16} />
            </button>
          }
        >
          <Graph model={m} compact />
          <div className="card-bottom">
            <span>
              <i className="dot lime" /> {pretty(m.target)} is the target
              outcome
            </span>
            <span>Drag nodes · Select to inspect</span>
          </div>
        </Card>
        <div className="decision-column">
          <section className="decision-panel">
            <div className="eyebrow">02 / INTERVENTION ENGINE</div>
            <div className="intervention-glyph" aria-hidden="true">
              <span>X</span>
              <i />
              <span className="glyph-target">Y</span>
              <b>do(X)</b>
            </div>
            <h2>
              Change a variable.
              <br />
              Measure the shift.
            </h2>
            <p>
              Compare what the model observes with what it estimates under a
              deliberate action.
            </p>
            <button
              className="primary full"
              onClick={() => go("Interventions")}
            >
              Open simulator <ArrowUpRight size={18} />
            </button>
          </section>
          <button className="action-link" onClick={() => go("Action planner")}>
            <span className="action-link-icon">
              <Target size={21} />
            </span>
            <span>
              <b>Find an actionable path</b>
              <small>Respect constraints. Compare costs.</small>
            </span>
            <ArrowUpRight size={20} />
          </button>
        </div>
      </div>
      <div className="bottom-grid">
        <Card
          title="Performance, in context"
          kicker="03 / MODEL SIGNAL"
          action={
            <button className="text-btn" onClick={() => go("Model lab")}>
              View evidence <ArrowUpRight size={15} />
            </button>
          }
        >
          <div className="performance">
            <Donut value={ev.accuracy} />
            <div className="performance-detail">
              <Probability
                value={ev.f1}
                label="Predictive macro F1"
                color="forest"
              />
              {m.reference_edges ? (
                <Probability
                  value={m.metrics.f1_score}
                  label="Directed graph F1"
                  color="sage"
                />
              ) : (
                <p className="muted">
                  Causal ground truth is unavailable for this upload.
                </p>
              )}
              <small>
                Prediction quality and structural recovery measure different
                things.
              </small>
            </div>
          </div>
        </Card>
        <Card
          title="One experiment. Three perspectives."
          kicker="04 / ANALYSIS WORKFLOW"
        >
          <div className="workflow">
            {[
              ["01", "Discover", "Inspect the relationships", Network],
              [
                "02",
                "Intervene",
                "Compare action scenarios",
                SlidersHorizontal,
              ],
              ["03", "Optimize", "Find feasible changes", Target],
            ].map(([n, t, d, I], i) => (
              <button
                key={n}
                onClick={() =>
                  go(["Causal graph", "Interventions", "Action planner"][i])
                }
              >
                <span className="workflow-icon">
                  <I size={21} />
                </span>
                <small>{n}</small>
                <b>{t}</b>
                <span>{d}</span>
                {i < 2 && <ChevronRight className="step-arrow" size={17} />}
              </button>
            ))}
          </div>
          <div className="note">
            <Info size={16} />
            {m.source}. Estimated intervention effects depend on the graph
            assumptions.
          </div>
        </Card>
      </div>
    </>
  );
}

function GraphPage({ m }) {
  const [ref, setRef] = useState(false);
  return (
    <>
      <Heading
        eyebrow="STRUCTURE EXPLORER"
        title="Follow the connections."
        description="Explore how variables connect, and inspect the assumptions behind each arrow."
      />
      <div className="section-toolbar">
        <div className="segmented">
          <button
            className={!ref ? "active" : ""}
            onClick={() => setRef(false)}
          >
            Learned graph
          </button>
          {m.reference_edges && (
            <button
              className={ref ? "active" : ""}
              onClick={() => setRef(true)}
            >
              Known benchmark graph
            </button>
          )}
        </div>
        <Badge tone="green">
          <Check size={12} /> Directed acyclic graph
        </Badge>
      </div>
      <Card
        title={ref ? "Synthetic ground truth" : "Your learned Bayesian network"}
        kicker={ALGOS[m.algorithm]}
      >
        <Graph model={m} reference={ref} />
      </Card>
      <div className="three-grid">
        <Card title="What an arrow means">
          <p>
            A proposed direction of influence in the model. Observational
            structure learning can leave directions ambiguous.
          </p>
        </Card>
        <Card title="What the target means">
          <p>
            The lime node is <b>{pretty(m.target)}</b>, the outcome selected for
            prediction and decision analysis.
          </p>
        </Card>
        <Card title="Read it responsibly">
          <p>
            The known graph is available because benchmark data is simulated. It
            is never available automatically for real uploads.
          </p>
        </Card>
      </div>
      <Card
        title="Connection inventory"
        kicker={`${m.edges.length} LEARNED EDGES`}
      >
        <div className="edge-list">
          {m.edges.map(([a, b]) => (
            <span key={a + b}>
              {pretty(a)}
              <ArrowRight size={14} />
              {pretty(b)}
            </span>
          ))}
          {!m.edges.length && (
            <p>No edges were discovered at the current sample size.</p>
          )}
        </div>
      </Card>
    </>
  );
}
function Interventions({ m }) {
  const [desired, setDesired] = useState(m.desired);
  const [treatment, setTreatment] = useState(m.treatment),
    [value, setValue] = useState(m.states[m.treatment].at(-1)),
    [control, setControl] = useState(m.states[m.treatment][0]),
    [result, setResult] = useState(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  function change(v) {
    setTreatment(v);
    setValue(m.states[v].at(-1));
    setControl(m.states[v][0]);
    setResult(null);
  }
  async function run() {
    setBusy(true);
    setError("");
    try {
      setResult(
        await api("intervene", {
          session_id: m.id,
          treatment,
          value,
          control,
          target: m.target,
          desired,
        }),
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading
        eyebrow="THE WHAT-IF LAB"
        title="The intervention simulator."
        description="Separate observing a condition from deliberately setting it. Compare both through the same model."
      />
      <div className="lab-grid">
        <Card title="Design your scenario" kicker="01 / CONFIGURE">
          <Select
            label="Intervention variable"
            value={treatment}
            onChange={change}
            options={m.nodes
              .filter((n) => n !== m.target)
              .map((n) => ({ value: n, label: pretty(n) }))}
          />
          <Select
            label="Current / control setting"
            value={control}
            onChange={(v) => {
              setControl(Number(v));
              setResult(null);
            }}
            options={m.states[treatment].map((s) => ({
              value: s,
              label: m.labels[treatment][s],
            }))}
          />
          <Select
            label="Action setting"
            value={value}
            onChange={(v) => {
              setValue(Number(v));
              setResult(null);
            }}
            options={m.states[treatment].map((s) => ({
              value: s,
              label: m.labels[treatment][s],
            }))}
          />
          <Select
            label="Desired outcome"
            value={desired}
            onChange={(v) => {
              setDesired(Number(v));
              setResult(null);
            }}
            options={m.states[m.target].map((value) => ({
              value,
              label: m.labels[m.target][value],
            }))}
          />
          <div className="formula">
            do({pretty(treatment)} = {m.labels[treatment][value]})
          </div>
          <button
            className="primary full"
            disabled={busy || control === value}
            onClick={run}
          >
            {busy ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Play size={16} />
            )}{" "}
            Run simulation
          </button>
          {control === value && (
            <small className="muted">
              Choose two different settings to compare.
            </small>
          )}
          {error && (
            <div role="alert" className="error">
              {error}
            </div>
          )}
          <div className="note">
            <Info size={15} /> Outcome: {pretty(m.target)} →{" "}
            {m.labels[m.target][desired]}
          </div>
        </Card>
        <Card title="A decision, quantified" kicker="02 / COMPARE">
          {!result ? (
            <Empty
              title="Your next insight starts here"
              description="Choose a variable and two settings, then run a simulation to compare outcome probabilities."
              icon={SlidersHorizontal}
            />
          ) : (
            <div className="result-enter">
              <div className="effect-hero">
                <span>MODEL-ESTIMATED EFFECT</span>
                <strong>
                  {result.effect >= 0 ? "+" : ""}
                  {(result.effect * 100).toFixed(1)}
                  <small> pp</small>
                </strong>
                <p>
                  Change in probability of{" "}
                  {m.labels[m.target][desired].toLowerCase()} under the action
                  versus control.
                </p>
              </div>
              <Probability
                value={result.control[desired]}
                label={`Control · ${m.labels[treatment][control]}`}
                color="sage"
              />
              <Probability
                value={result.interventional[desired]}
                label={`Intervention · ${m.labels[treatment][value]}`}
                color="forest"
              />
              <Probability
                value={result.observational[desired]}
                label="Observation · selected state"
              />
              <div className="insight">
                <BrainCircuit size={20} />
                <div>
                  <b>Why the probabilities can differ</b>
                  <p>
                    Observation conditions on existing cases. Intervention cuts
                    incoming arrows into {pretty(treatment)} and propagates the
                    action through the graph.
                  </p>
                </div>
              </div>
              <div className="chips">
                <Badge>
                  {result.removed_edges.length} incoming edges removed
                </Badge>
                <Badge>
                  Adjustment:{" "}
                  {result.adjustment === null
                    ? "not identified"
                    : result.adjustment.length
                      ? result.adjustment.map(pretty).join(", ")
                      : "none needed"}
                </Badge>
              </div>
              <button
                className="text-btn"
                onClick={() =>
                  download(
                    {
                      ...result,
                      treatment,
                      value,
                      control,
                      target: m.target,
                      desired,
                    },
                    "intervention.json",
                  )
                }
              >
                <ArrowDownToLine size={15} /> Export scenario
              </button>
            </div>
          )}
        </Card>
      </div>
      <Card title="Watch the intervention" kicker="GRAPH SURGERY">
        <Graph model={m} cut={result ? treatment : null} />
      </Card>
    </>
  );
}
function Planner({ m }) {
  const [desired, setDesired] = useState(m.desired);
  const [record, setRecord] = useState(0);
  const [profile, setProfile] = useState({ ...m.sample_profiles[0] }),
    [immutable, setImmutable] = useState(m.immutable),
    [threshold, setThreshold] = useState(0.65),
    [maxActions, setMaxActions] = useState(2),
    [costs, setCosts] = useState({}),
    [result, setResult] = useState(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const features = m.nodes.filter((n) => n !== m.target);
  const reset = () => setResult(null);
  async function run() {
    setBusy(true);
    setError("");
    try {
      setResult(
        await api("recourse", {
          session_id: m.id,
          profile,
          target: m.target,
          desired,
          immutable,
          threshold,
          max_actions: maxActions,
          costs,
        }),
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading
        eyebrow="ACTIONABLE INTELLIGENCE"
        title="Find a path forward."
        description="Find small, cost-aware action sets while respecting the features you cannot change."
      />
      <div className="lab-grid planner">
        <Card
          title="Build an individual profile"
          kicker="01 / PROFILE & CONSTRAINTS"
        >
          <Select
            label="Start from a sample record"
            value={record}
            onChange={(i) => {
              setRecord(Number(i));
              setProfile({ ...m.sample_profiles[Number(i)] });
              reset();
            }}
            options={[
              { value: "", label: "Choose a profile…" },
              ...m.sample_profiles.map((p, i) => ({
                value: i,
                label: `Record ${i + 1} · ${m.labels[m.target][p[m.target]]}`,
              })),
            ]}
          />
          <div className="profile-grid">
            {features.map((n) => (
              <Select
                key={n}
                label={pretty(n)}
                value={profile[n]}
                onChange={(v) => {
                  setRecord("");
                  setProfile({ ...profile, [n]: Number(v) });
                  reset();
                }}
                options={m.states[n].map((s) => ({
                  value: s,
                  label: m.labels[n][s],
                }))}
              />
            ))}
          </div>
          <Select
            label="Desired outcome"
            value={desired}
            onChange={(v) => {
              setDesired(Number(v));
              setResult(null);
            }}
            options={m.states[m.target].map((value) => ({
              value,
              label: m.labels[m.target][value],
            }))}
          />
          <div className="goal">
            <Target size={20} />
            <div>
              <small>DESIRED OUTCOME</small>
              <b>
                {pretty(m.target)} → {m.labels[m.target][desired]}
              </b>
            </div>
          </div>
          <Field label={`Required model probability · ${pct(threshold)}`}>
            <input
              type="range"
              min=".1"
              max=".95"
              step=".05"
              value={threshold}
              onChange={(e) => {
                setThreshold(Number(e.target.value));
                reset();
              }}
            />
          </Field>
          <Select
            label="Maximum simultaneous changes"
            value={maxActions}
            onChange={(v) => {
              setMaxActions(Number(v));
              reset();
            }}
            options={[1, 2, 3]}
          />
          <details>
            <summary>
              <LockKeyhole size={15} /> Immutable features & action costs{" "}
              <ChevronDown size={14} />
            </summary>
            <p className="muted">
              Lock features that cannot change. Higher weights make changes more
              expensive.
            </p>
            {features.map((n) => (
              <div className="constraint-row" key={n}>
                <label>
                  <input
                    type="checkbox"
                    checked={immutable.includes(n)}
                    onChange={(e) => {
                      setImmutable(
                        e.target.checked
                          ? [...immutable, n]
                          : immutable.filter((v) => v !== n),
                      );
                      reset();
                    }}
                  />
                  {pretty(n)}
                </label>
                <input
                  aria-label={`Action cost for ${pretty(n)}`}
                  type="number"
                  min=".1"
                  max="100"
                  step=".5"
                  value={costs[n] ?? 1}
                  onChange={(e) => {
                    setCosts({ ...costs, [n]: Number(e.target.value) });
                    reset();
                  }}
                />
              </div>
            ))}
          </details>
          <button
            className="primary full"
            disabled={busy || Object.values(costs).some((c) => c <= 0)}
            onClick={run}
          >
            {busy ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Sparkles size={17} />
            )}{" "}
            Find action plans
          </button>
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
        </Card>
        <Card title="Your possible paths forward" kicker="02 / COMPARE PLANS">
          {!result ? (
            <Empty
              title="Small changes. New possibilities."
              description="Set a profile and your constraints. We’ll search for up to three low-cost plans that meet your goal."
              icon={Target}
            />
          ) : !result.plans.length ? (
            <Empty
              title="No feasible plan at this threshold"
              description="Try lowering the probability threshold, allowing another action, or reviewing immutable features. Some graphs contain no actionable path to the outcome."
              icon={Search}
            />
          ) : (
            <div className="plans result-enter">
              {result.plans.map((p, i) => (
                <div className="plan" key={i}>
                  <div className="plan-head">
                    <Badge tone={i === 0 ? "green" : ""}>
                      {i === 0 ? "LOWEST COST" : `ALTERNATIVE ${i}`}
                    </Badge>
                    <span>
                      Cost <b>{p.total_cost.toFixed(2)}</b>
                    </span>
                  </div>
                  <h2>
                    {Object.keys(p.interventions).length
                      ? `${Object.keys(p.interventions).length} ${Object.keys(p.interventions).length === 1 ? "focused change" : "focused changes"}`
                      : "No action needed"}
                  </h2>
                  {Object.entries(p.feature_changes).map(([n, c]) => (
                    <div className="change" key={n}>
                      <b>{pretty(n)}</b>
                      <span>
                        {m.labels[n][c.from]}
                        <ArrowRight size={14} />
                        <strong>{m.labels[n][c.to]}</strong>
                      </span>
                    </div>
                  ))}
                  <div className="plan-outcome">
                    <div>
                      <small>Current settings</small>
                      <b>{pct(p.baseline_probability)}</b>
                    </div>
                    <ArrowRight size={20} />
                    <div>
                      <small>With the action</small>
                      <b>{pct(p.target_probability)}</b>
                    </div>
                    <Badge tone="green">
                      {(
                        (p.target_probability - p.baseline_probability) *
                        100
                      ).toFixed(1)}{" "}
                      pp
                    </Badge>
                  </div>
                </div>
              ))}
              <div className="note">
                <Info size={16} />
                {result.meaning} Each plan compares the changed features with
                those same features set to their current values.
              </div>
              <button
                className="text-btn"
                onClick={() => download(result, "action-plans.json")}
              >
                <ArrowDownToLine size={15} /> Export action plans
              </button>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
function ModelLab({ m }) {
  return (
    <>
      <Heading
        eyebrow="EVIDENCE, NOT JUST OUTPUT"
        title="Evidence behind the model."
        description="A reproducible held-out evaluation, with transparent baselines and separate structural metrics."
      />
      <div className="validation-banner">
        <ShieldCheck size={25} />
        <div>
          <b>Training and test data stay separate.</b>
          <span>
            80/20 stratified split · Seed {m.seed} · Preprocessing fitted on
            training rows · {fmt(m.test_rows)} unseen test records
          </span>
        </div>
        <Badge tone="green">Reproducible</Badge>
      </div>
      <Card title="The predictive benchmark" kicker="SAME DATA. SAME SPLIT.">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy ↑</th>
                <th>Macro F1 ↑</th>
                <th>Log loss ↓</th>
              </tr>
            </thead>
            <tbody>
              {m.evaluation.map((e, i) => (
                <tr key={e.model}>
                  <td>
                    <span className={"model-dot model-" + i} />
                    <b>{e.model}</b>
                    {i === 0 && <Badge>Our model</Badge>}
                  </td>
                  <td>{pct(e.accuracy)}</td>
                  <td>{e.f1.toFixed(3)}</td>
                  <td>{e.log_loss.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note">
          <Info size={15} /> Accuracy measures correct predictions. Macro F1
          balances classes. Log loss measures probability quality; lower is
          better. Predictive performance does not prove causality.
        </div>
      </Card>
      <div className="two-grid">
        <Card title="Where predictions land" kicker="CONFUSION MATRIX">
          <div className="matrix-label">Predicted outcome →</div>
          <div
            className="matrix"
            style={{
              gridTemplateColumns: `100px repeat(${m.confusion.length},1fr)`,
            }}
          >
            <span>Actual ↓</span>
            {m.states[m.target].map((s) => (
              <b key={s}>{m.labels[m.target][s]}</b>
            ))}
            {m.confusion.map((row, i) => (
              <React.Fragment key={i}>
                <b>{m.labels[m.target][m.states[m.target][i]]}</b>
                {row.map((v, j) => (
                  <div key={j} className={i === j ? "correct" : ""}>
                    {v}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </Card>
        <Card title="How close is the structure?" kicker="GRAPH QUALITY">
          {m.reference_edges ? (
            <>
              <div className="metric-pair">
                <div>
                  <span>Directed graph F1</span>
                  <strong>{m.metrics.f1_score.toFixed(3)}</strong>
                </div>
                <div>
                  <span>Structural distance</span>
                  <strong>{m.metrics.structural_hamming_distance}</strong>
                </div>
              </div>
              <Probability
                label="Edge precision"
                value={m.metrics.precision}
                color="forest"
              />
              <Probability
                label="Edge recall"
                value={m.metrics.recall}
                color="sage"
              />
              <p className="muted">
                Compared with the known synthetic graph. A reversed edge counts
                as one structural edit. Lower distance is better.
              </p>
            </>
          ) : (
            <Empty
              title="Ground truth isn’t available"
              description="Real uploaded data has no automatically known causal graph. Structural precision and recall cannot be claimed."
            />
          )}
        </Card>
      </div>
      <Card title="Assumptions & limits" kicker="MODEL ASSUMPTIONS">
        <div className="three-grid plain">
          <div>
            <b>01 / Causal identification</b>
            <p>
              Observational discovery needs assumptions. Hidden confounding,
              equivalent graphs, and wrong directions can invalidate
              intervention estimates.
            </p>
          </div>
          <div>
            <b>02 / Data & generalization</b>
            <p>
              Benchmarks are synthetic. One held-out split measures performance
              on this dataset, not on a real population.
            </p>
          </div>
          <div>
            <b>03 / Decision uncertainty</b>
            <p>
              Action scores are model probabilities, not confidence intervals or
              outcome guarantees. Plans do not identify personal
              counterfactuals.
            </p>
          </div>
        </div>
      </Card>
    </>
  );
}
function Dataset({ m }) {
  const [col, setCol] = useState(m.target),
    [search, setSearch] = useState("");
  const rows = m.preview.filter((r) =>
    Object.entries(r)
      .map(([n, v]) => pretty(n) + " " + m.labels[n][v])
      .join(" ")
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  return (
    <>
      <Heading
        eyebrow="KNOW YOUR INPUT"
        title="Inside the dataset."
        description="Inspect the records, state meanings and training-only transformations behind your model."
      >
        <a className="secondary" href={`/api/dataset/${m.id}`} download>
          <ArrowDownToLine size={16} /> Download CSV
        </a>
      </Heading>
      <div className="two-grid">
        <Card title="A transparent data source" kicker={m.source}>
          <div className="dataset-summary">
            <strong>
              {fmt(m.rows)}
              <small>rows</small>
            </strong>
            <strong>
              {m.nodes.length}
              <small>variables</small>
            </strong>
            <strong>
              0<small>missing values</small>
            </strong>
          </div>
          <p>
            {m.source === "Synthetic benchmark"
              ? "Generated from explicit mechanisms with a known graph. Suitable for controlled demonstrations, not a claim of real-world performance."
              : "Uploaded records remain on this local server and are validated before training."}
          </p>
          <Badge>{m.train_rows} training</Badge>{" "}
          <Badge>{m.test_rows} test</Badge>
        </Card>
        <Card title="State distribution">
          <Select
            label="Variable"
            value={col}
            onChange={setCol}
            options={m.nodes.map((n) => ({ value: n, label: pretty(n) }))}
          />
          {Object.entries(m.distribution[col]).map(([s, n]) => (
            <Probability
              key={s}
              value={n / m.rows}
              label={`${m.labels[col][s]} · ${fmt(n)}`}
              color="forest"
            />
          ))}
        </Card>
      </div>
      <Card
        title="Record explorer"
        kicker="FIRST 100 ENCODED RECORDS"
        action={
          <div className="search-input">
            <Search size={15} />
            <input
              aria-label="Search records"
              placeholder="Search records…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        }
      >
        <div className="table-wrap records">
          <table>
            <thead>
              <tr>
                <th>#</th>
                {m.nodes.map((n) => (
                  <th key={n}>{pretty(n)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  {m.nodes.map((n) => (
                    <td key={n}>
                      <span className={n === m.target ? "outcome-cell" : ""}>
                        {m.labels[n][r[n]]}
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && (
            <p className="muted">No records match your search.</p>
          )}
        </div>
      </Card>
      <Card title="Encoding & transformation reference">
        <div className="schema-grid">
          {m.nodes.map((n) => (
            <div key={n}>
              <b>{pretty(n)}</b>
              <span>
                {Object.entries(m.labels[n])
                  .map(([s, l]) => `${s} = ${l}`)
                  .join(" · ")}
              </span>
              {m.preprocessing[n].kind === "bins" && (
                <small>
                  Training quantile boundaries:{" "}
                  {m.preprocessing[n].cuts.map((x) => x.toFixed(2)).join(", ")}
                </small>
              )}
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}
export default function App() {
  const [page, setPage] = useState("Overview"),
    [model, setModel] = useState(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [config, setConfig] = useState(false),
    [mobile, setMobile] = useState(false);
  const [domain, setDomain] = useState("fintech"),
    [algorithm, setAlgorithm] = useState("hill"),
    [samples, setSamples] = useState(2000),
    [seed, setSeed] = useState(42),
    [upload, setUpload] = useState(null),
    [target, setTarget] = useState(""),
    [uploadError, setUploadError] = useState("");
  const request = useRef(0);
  async function train() {
    const id = ++request.current;
    setBusy(true);
    setError("");
    try {
      const m = await api("train", {
        domain,
        algorithm,
        samples,
        seed,
        ...(domain === "custom" ? { records: upload?.records, target } : {}),
      });
      if (id === request.current) {
        setModel(m);
        setConfig(false);
        setPage("Overview");
      }
    } catch (e) {
      if (id === request.current) setError(e.message);
    } finally {
      if (id === request.current) setBusy(false);
    }
  }
  useEffect(() => {
    train();
  }, []);
  useEffect(() => {
    if (!config) return;
    const previous = document.activeElement;
    const dialog = () => document.querySelector('[role="dialog"]');
    const frame = requestAnimationFrame(() =>
      dialog()?.querySelector("button,select,input")?.focus(),
    );
    function key(e) {
      if (e.key === "Escape" && !busy) setConfig(false);
      if (e.key === "Tab") {
        const all = [
          ...(dialog()?.querySelectorAll(
            "button:not(:disabled),select,input",
          ) || []),
        ];
        if (!all.length) return;
        const first = all[0],
          last = all.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    }
    document.addEventListener("keydown", key);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("keydown", key);
      previous?.focus();
    };
  }, [config, busy]);
  function go(p) {
    setPage(p);
    setMobile(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  function loadFile(file) {
    setUploadError("");
    setUpload(null);
    setTarget("");
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) {
      setUploadError("Use a CSV smaller than 5 MB.");
      return;
    }
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (r) => {
        if (r.errors.length) {
          setUploadError(
            "Could not parse this CSV. Check its headers and delimiters.",
          );
          return;
        }
        if (
          r.data.length < 100 ||
          r.data.length > 10000 ||
          r.meta.fields.length < 2 ||
          r.meta.fields.length > 12
        ) {
          setUploadError("Use 100–10,000 rows and 2–12 columns.");
          return;
        }
        setUpload({ name: file.name, records: r.data, columns: r.meta.fields });
        setTarget(r.meta.fields.at(-1));
      },
    });
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="application-header">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            go("Overview");
          }}
        >
          <span className="brand-mark">
            <Network size={24} />
          </span>
          <span className="brand-word">
            CAUSAL<span>STUDIO</span>
          </span>
          <span className="brand-edition">
            DECISION
            <br />
            INTELLIGENCE
          </span>
        </a>
        <nav className={mobile ? "open" : ""} aria-label="Main navigation">
          {NAV.map(([n, I], index) => (
            <button
              key={n}
              aria-label={n}
              aria-current={page === n ? "page" : undefined}
              className={page === n ? "active" : ""}
              onClick={() => go(n)}
            >
              <span className="nav-number">0{index + 1}</span>
              <I size={16} />
              <span>{n}</span>
            </button>
          ))}
        </nav>
        <div className="header-tools">
          <button
            className="icon-btn"
            aria-label="Workspace settings"
            onClick={() => setConfig(true)}
          >
            <Settings2 size={19} />
          </button>
          <button
            className="icon-btn mobile-menu"
            aria-label="Open navigation"
            aria-expanded={mobile}
            onClick={() => setMobile(!mobile)}
          >
            {mobile ? <X size={21} /> : <Menu size={21} />}
          </button>
        </div>
      </header>
      {mobile && <div className="backdrop" onClick={() => setMobile(false)} />}
      <div className="main-shell">
        <header className="topbar">
          <div>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <b>{page}</b>
          </div>
          <div>
            <span className="system-status">
              <i className="live-dot" />
              {busy ? "Training model" : "Local environment"}
            </span>
            <button
              className="secondary small"
              onClick={() =>
                model && download(model, "causal-studio-report.json")
              }
              disabled={!model || busy}
            >
              <ArrowDownToLine size={14} />
              <span>Export report</span>
            </button>
          </div>
        </header>
        <main id="main">
          <div className="workspace-bar">
            <span>
              <i className="dot green" />
              {model?.title || "Initializing workspace"}
              <Badge>{model?.source || "Synthetic benchmark"}</Badge>
            </span>
            <button className="text-btn" onClick={() => setConfig(true)}>
              <Settings2 size={14} /> Configure model <ChevronDown size={13} />
            </button>
          </div>
          {error && !config && (
            <div className="error" role="alert">
              {error}
              <button className="text-btn" onClick={train}>
                Try again
              </button>
            </div>
          )}
          {!model ? (
            <div className="initial-loading">
              <div className="loading-orbit">
                <BrainCircuit size={40} />
              </div>
              <h2>
                {busy ? "Connecting the dots…" : "Let’s build your workspace."}
              </h2>
              <p>
                {busy
                  ? "Learning a graph, fitting probabilities, and evaluating unseen records."
                  : "Configure your data to get started."}
              </p>
              {!busy && (
                <button className="primary" onClick={() => setConfig(true)}>
                  Configure workspace
                </button>
              )}
            </div>
          ) : (
            <div key={page + model.id} className="page-enter">
              {page === "Overview" ? (
                <Overview m={model} go={go} />
              ) : page === "Causal graph" ? (
                <GraphPage m={model} />
              ) : page === "Interventions" ? (
                <Interventions m={model} />
              ) : page === "Action planner" ? (
                <Planner m={model} />
              ) : page === "Model lab" ? (
                <ModelLab m={model} />
              ) : page === "Dataset" ? (
                <Dataset m={model} />
              ) : (
                <Dataset m={model} />
              )}
            </div>
          )}
          <footer>
            <span>
              CAUSAL STUDIO <b>RESEARCH PLATFORM</b>
            </span>
            <span>Discover dependencies. Explore interventions.</span>
            <span>
              <i className="dot green" />{" "}
              {model
                ? `${model.seconds}s model fit & evaluation`
                : "Python + React"}
            </span>
          </footer>
        </main>
      </div>
      {config && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget && !busy) setConfig(false);
          }}
        >
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="Configure workspace"
          >
            <div className="modal-head">
              <div>
                <span className="eyebrow">YOUR RESEARCH ENVIRONMENT</span>
                <h2>Make it your workspace.</h2>
              </div>
              <button
                className="icon-btn"
                aria-label="Close settings"
                disabled={busy}
                onClick={() => setConfig(false)}
              >
                <X size={20} />
              </button>
            </div>
            <Select
              label="Dataset"
              value={domain}
              onChange={(v) => {
                setDomain(v);
                if (v === "custom" && algorithm === "reference")
                  setAlgorithm("hill");
              }}
              options={Object.entries(DOMAINS).map(([v, [l, c]]) => ({
                value: v,
                label: `${l} · ${c}`,
              }))}
            />
            {domain === "custom" && (
              <>
                <label className="upload-area">
                  <Upload size={24} />
                  <b>{upload?.name || "Choose your CSV"}</b>
                  <span>100–10,000 rows · 2–12 columns · Up to 5 MB</span>
                  <input
                    aria-label="Upload CSV"
                    type="file"
                    accept=".csv"
                    onChange={(e) => loadFile(e.target.files[0])}
                  />
                </label>
                {uploadError && <div className="error">{uploadError}</div>}
                {upload && (
                  <Select
                    label="Target outcome"
                    value={target}
                    onChange={setTarget}
                    options={upload.columns}
                  />
                )}
              </>
            )}
            <Select
              label="Learning method"
              value={algorithm}
              onChange={setAlgorithm}
              options={Object.entries(ALGOS)
                .filter(([v]) => domain !== "custom" || v !== "reference")
                .map(([value, label]) => ({ value, label }))}
            />
            <div className="two-grid">
              <Select
                label="Benchmark sample size"
                value={samples}
                onChange={(v) => setSamples(Number(v))}
                options={[500, 1000, 2000, 5000]}
              />
              <Field label="Random seed">
                <input
                  type="number"
                  min="0"
                  max="99999"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </Field>
            </div>
            <div className="note">
              <Info size={17} /> Every training run uses a separate held-out
              test set. Uploaded numeric features with more than six values are
              discretized using training-only quantile bins.
            </div>
            {error && (
              <div role="alert" className="error">
                {error}
              </div>
            )}
            <button
              className="primary full"
              disabled={
                busy ||
                (domain === "custom" && (!upload || Boolean(uploadError)))
              }
              onClick={train}
            >
              {busy ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <Sparkles size={17} />
              )}{" "}
              {busy ? "Learning & evaluating…" : "Build workspace"}{" "}
              {!busy && <ArrowRight size={16} />}
            </button>
          </section>
        </div>
      )}
      {busy && model && (
        <div className="training-toast" role="status">
          <LoaderCircle size={19} className="spin" />
          <div>
            <b>Building your workspace</b>
            <span>
              Learning, fitting, and evaluating. Your current results stay
              available.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
