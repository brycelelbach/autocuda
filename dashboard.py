#!/usr/bin/env python3
"""Dashboard server for autocuda experiment logs.

Reads `*-log.csv` files from ./experiments/ and serves them as a single
HTML page on http://localhost:8000/. Stdlib only, no dependencies.

The page auto-refreshes so you can leave it open while the optimizer runs.

Consumes the one-row-per-trial schema produced by the autocuda:optimize
skill. The CSV header looks like

    timestamp,<bench_1>,<bench_2>,...,<bench_N>,status,description

with `timestamp`, `status`, `description` as fixed metadata columns and
every other column being a benchmark. Benchmark columns hold the
measured metric for that trial (or `N/A` for compactions, failures, or
benchmarks outside the active set). The dashboard renders one coloured
curve per benchmark column, each against its own baseline.
"""

import argparse
import csv
import html
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


STATUS_COLORS = {
    "baseline":         "#888",
    "improved":         "#3a8",
    "regressed":        "#c62",
    "build_error":      "#c33",
    "validation_error": "#c33",
    "runtime_error":    "#c33",
    "compaction":       "#557",
}

# Palette for per-benchmark chart lines. Cycles if there are more benchmarks.
BENCHMARK_COLORS = [
    "#3a8", "#e6a23c", "#9966ff", "#ef476f",
    "#14b8a6", "#f59e0b", "#6366f1", "#f472b6",
]

METADATA_COLUMNS = {"timestamp", "status", "description"}


def load_log(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return columns, rows


def benchmark_columns(columns):
    return [c for c in columns if c not in METADATA_COLUMNS]


def parse_metric(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse_timestamp(row):
    ts = row.get("timestamp", "") or ""
    # Tolerate trailing "Z" (UTC designator) — Python 3.10's fromisoformat doesn't.
    if ts.endswith("Z"):
        ts = ts[:-1]
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def fmt_duration(sec: float) -> str:
    sec = max(int(sec), 0)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m"
    h, m = sec // 3600, (sec % 3600) // 60
    return f"{h}h{m:02d}m" if m else f"{h}h"


def parse_rows(rows, bench_cols):
    """Parse rows, clamping timestamps monotonically across the sequence.

    Returns one dict per row with: idx, raw_t, t, values (dict bm -> float or
    None), status, description, timestamp_str. Benchmarks whose cell is "N/A"
    or unparseable land as None in values.
    """
    parsed = []
    running_max = None
    for i, r in enumerate(rows):
        raw_t = parse_timestamp(r)
        if raw_t is None:
            t = running_max
        elif running_max is None:
            t = raw_t
        else:
            t = raw_t if raw_t >= running_max else running_max
        if t is not None:
            running_max = t
        values = {bm: parse_metric(r.get(bm, "")) for bm in bench_cols}
        parsed.append({
            "idx": i,
            "raw_t": raw_t,
            "t": t,
            "values": values,
            "status": r.get("status", "") or "",
            "description": r.get("description", "") or "",
            "timestamp_str": r.get("timestamp", "") or "",
        })
    return parsed


def benchmark_baseline(parsed, bm):
    for p in parsed:
        if p["status"] == "baseline":
            v = p["values"].get(bm)
            if v is not None:
                return v
    for p in parsed:
        v = p["values"].get(bm)
        if v is not None:
            return v
    return None


def benchmark_lower_better(parsed, bm, baseline):
    """Infer direction from improved rows: if the committed 'improved' trials
    average below baseline, the benchmark is lower-is-better; otherwise higher.
    Defaults to higher-is-better if no 'improved' rows exist yet.
    """
    if baseline is None:
        return False
    improved = [p["values"][bm] for p in parsed
                if p["status"] == "improved" and p["values"].get(bm) is not None]
    if not improved:
        return False
    return (sum(improved) / len(improved)) < baseline


def compute_speedup(metric, baseline, lower_better):
    if metric is None or baseline is None or not baseline:
        return None
    if lower_better:
        if not metric:
            return None
        return baseline / metric
    return metric / baseline


def render_chart(parsed, bench_cols, width=900, height=280, pad=52):
    """Overlay chart: one speedup curve per benchmark column, shared x
    (walltime) and y (speedup vs that benchmark's own baseline, so every
    benchmark's 1.0 coincides with the dashed reference line)."""
    series = []
    for idx, bm in enumerate(bench_cols):
        baseline = benchmark_baseline(parsed, bm)
        if baseline is None:
            continue
        lower = benchmark_lower_better(parsed, bm, baseline)
        pts = []
        for p in parsed:
            v = p["values"].get(bm)
            if v is None or p["t"] is None:
                continue
            sp = compute_speedup(v, baseline, lower)
            if sp is None:
                continue
            pts.append({
                "idx": p["idx"],
                "raw_t": p["raw_t"],
                "t": p["t"],
                "status": p["status"],
                "metric": v,
                "speedup": sp,
            })
        if pts:
            series.append({
                "benchmark": bm,
                "baseline": baseline,
                "lower": lower,
                "points": pts,
                "color": BENCHMARK_COLORS[idx % len(BENCHMARK_COLORS)],
            })

    if not series:
        return '<div class="empty">no numeric data</div>'

    all_pts = [q for s in series for q in s["points"]]
    t0 = min(q["t"] for q in all_pts)
    tmax = max((q["t"] - t0).total_seconds() for q in all_pts)
    if tmax <= 0:
        tmax = 1.0

    vmax = max(q["speedup"] for q in all_pts)
    vmax = max(vmax, 1.0)  # keep baseline in view
    vmin = 0.0

    def x(dt): return pad + (width - 2 * pad) * dt / tmax
    def y(sp): return height - pad - (height - 2 * pad) * (sp - vmin) / (vmax - vmin)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    parts.append(f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#333"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#333"/>')
    for k in range(3):
        sp = vmin + (vmax - vmin) * (k / 2)
        yv = y(sp)
        parts.append(f'<line x1="{pad}" y1="{yv:.1f}" x2="{width - pad}" y2="{yv:.1f}" stroke="#1f1f1f"/>')
        parts.append(f'<text x="{pad - 6}" y="{yv + 4:.1f}" text-anchor="end" fill="#888" font-size="11">{sp:.2f}×</text>')
    for k in range(3):
        dt = tmax * (k / 2)
        xv = x(dt)
        anchor = "start" if k == 0 else ("end" if k == 2 else "middle")
        parts.append(f'<text x="{xv:.1f}" y="{height - pad + 14:.1f}" text-anchor="{anchor}" fill="#888" font-size="11">{fmt_duration(dt)}</text>')
    yb = y(1.0)
    parts.append(f'<line x1="{pad}" y1="{yb:.1f}" x2="{width - pad}" y2="{yb:.1f}" stroke="#555" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{width / 2:.0f}" y="{height - 6}" text-anchor="middle" fill="#666" font-size="11">walltime since start</text>')

    for s in series:
        color = s["color"]
        bm_esc = html.escape(s["benchmark"])
        kept = [q for q in s["points"] if q["status"] in ("baseline", "improved")]
        if len(kept) >= 2:
            d = "M " + " L ".join(
                f"{x((q['t'] - t0).total_seconds()):.1f},{y(q['speedup']):.1f}"
                for q in kept
            )
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85"/>')
        for q in s["points"]:
            dot_fill = STATUS_COLORS.get(q["status"], color)
            dt = (q["t"] - t0).total_seconds()
            if q["raw_t"] is None:
                note = " [timestamp missing; pinned to prior trial]"
            elif (q["raw_t"] - t0).total_seconds() < dt - 0.5:
                note = f" [raw timestamp {q['raw_t'].isoformat(timespec='seconds')} clamped]"
            else:
                note = ""
            parts.append(
                f'<circle cx="{x(dt):.1f}" cy="{y(q["speedup"]):.1f}" r="3.5"'
                f' fill="{dot_fill}" stroke="{color}" stroke-width="1.5">'
                f'<title>{bm_esc} row {q["idx"]}: {q["speedup"]:.3f}× ({q["metric"]:g})'
                f' @ {fmt_duration(dt)} ({q["status"]}){note}</title>'
                f'</circle>'
            )

    parts.append("</svg>")

    legend_items = "".join(
        f'<span class="legend-item">'
        f'<span class="legend-swatch" style="background:{s["color"]}"></span>'
        f'{html.escape(s["benchmark"])}'
        f'</span>'
        for s in series
    )
    return "".join(parts) + f'<div class="legend">{legend_items}</div>'


def render_summary(parsed, bench_cols):
    """Global run-level summary plus a per-benchmark line for baseline/best/delta."""
    trial_ts = {
        p["timestamp_str"]
        for p in parsed
        if p["status"] != "compaction" and p["timestamp_str"]
    }
    n_trials = len(trial_ts)
    n_benchmarks = len(bench_cols)

    failed_trial_ts = {
        p["timestamp_str"]
        for p in parsed
        if p["status"].endswith("_error") and p["timestamp_str"]
    }
    n_failures = len(failed_trial_ts)

    usable_ts = [p["t"] for p in parsed if p["t"] is not None]
    rate_span = ""
    if len(usable_ts) >= 2 and n_trials >= 2:
        elapsed = (max(usable_ts) - min(usable_ts)).total_seconds()
        if elapsed >= 60:
            rate_span = f'<span><b>{(n_trials - 1) * 3600 / elapsed:.1f}</b> trials/hr</span>'

    bench_label = "benchmark" if n_benchmarks == 1 else "benchmarks"
    global_bits = [
        f'<span><b>{n_trials}</b> trials</span>',
        f'<span><b>{n_benchmarks}</b> {bench_label}</span>',
    ]
    if rate_span:
        global_bits.append(rate_span)
    if n_failures:
        global_bits.append(f'<span class="muted">{n_failures} failures</span>')
    global_summary = f'<div class="summary">{"".join(global_bits)}</div>'

    bench_rows = []
    for bm in bench_cols:
        baseline = benchmark_baseline(parsed, bm)
        if baseline is None:
            bench_rows.append(
                f'<div class="bench-summary">'
                f'<span class="bench-name">{html.escape(bm)}</span>'
                f'<span class="muted">no baseline recorded</span>'
                f'</div>'
            )
            continue
        lower = benchmark_lower_better(parsed, bm, baseline)
        kept = [p for p in parsed
                if p["status"] in ("baseline", "improved") and p["values"].get(bm) is not None]
        if kept:
            best_row = (min(kept, key=lambda p: p["values"][bm]) if lower
                        else max(kept, key=lambda p: p["values"][bm]))
            best = best_row["values"][bm]
            best_idx = best_row["idx"]
        else:
            best, best_idx = baseline, parsed[0]["idx"] if parsed else 0
        delta = (best - baseline) / baseline * 100 if baseline else 0
        if lower:
            delta = -delta
        bench_rows.append(
            f'<div class="bench-summary">'
            f'<span class="bench-name">{html.escape(bm)}</span>'
            f'<span>baseline <b>{baseline:g}</b></span>'
            f'<span>best <b>{best:g}</b> @ row {best_idx}</span>'
            f'<span class="delta {"pos" if delta >= 0 else "neg"}">{delta:+.2f}%</span>'
            f'</div>'
        )
    return global_summary + "".join(bench_rows)


def render_table(rows, columns):
    bench_cols = benchmark_columns(columns)
    header_cells = (
        '<th>#</th><th>timestamp</th>'
        + "".join(f'<th class="bench-col">{html.escape(bm)}</th>' for bm in bench_cols)
        + '<th>status</th><th>description</th>'
    )
    out = [f'<table><thead><tr>{header_cells}</tr></thead><tbody>']
    for i, row in enumerate(rows):
        status = row.get("status", "") or ""
        color = STATUS_COLORS.get(status, "#777")
        out.append('<tr>')
        out.append(f'<td class="num">{i}</td>')
        out.append(f'<td class="ts">{html.escape(row.get("timestamp", "") or "")}</td>')
        for bm in bench_cols:
            cell = row.get(bm, "") or ""
            cls = "metric muted" if cell.strip().upper() == "N/A" else "metric"
            out.append(f'<td class="{cls}">{html.escape(cell)}</td>')
        out.append(f'<td><span class="badge" style="background:{color}">{html.escape(status)}</span></td>')
        out.append(f'<td>{html.escape(row.get("description", "") or "")}</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return "".join(out)


def render_section(path: Path, collapsed: bool = False):
    name = html.escape(path.name)
    try:
        columns, rows = load_log(path)
        bench_cols = benchmark_columns(columns)
        parsed = parse_rows(rows, bench_cols)
        body = (
            f'<h2>{name}</h2>'
            f'{render_summary(parsed, bench_cols)}'
            f'{render_chart(parsed, bench_cols)}'
            f'<details><summary>show {len(rows)} rows</summary>{render_table(rows, columns)}</details>'
        )
        count = len(rows)
    except OSError as e:
        body = f'<h2>{name}</h2><p class="empty">read error: {html.escape(str(e))}</p>'
        count = 0

    if not collapsed:
        return f'<section data-run="{name}">{body}</section>'
    summary = f'{name} · {count} rows' if count else name
    return (f'<details class="run" data-run="{name}">'
            f'<summary>{summary}</summary>'
            f'<section>{body}</section>'
            f'</details>')


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>autocuda dashboard</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; padding: 24px; background: #0e0e0e; color: #ddd; }}
  header {{ display: flex; align-items: baseline; gap: 16px; margin-bottom: 20px; }}
  h1 {{ margin: 0; font-size: 20px; font-weight: 600; }}
  header .path {{ color: #888; font-size: 12px; font-family: ui-monospace, monospace; }}
  section {{ background: #161616; border: 1px solid #262626; border-radius: 6px;
             padding: 16px 18px; margin-bottom: 20px; }}
  section h2 {{ margin: 0 0 10px; font-size: 14px; font-weight: 500; color: #fff;
                font-family: ui-monospace, SFMono-Regular, monospace; }}
  .summary {{ display: flex; flex-wrap: wrap; gap: 18px; font-size: 12.5px; color: #aaa;
              margin-bottom: 10px; }}
  .summary b {{ color: #fff; font-weight: 600; }}
  .bench-summary {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: #aaa;
                    padding: 4px 0 4px 12px; border-left: 2px solid #262626;
                    margin-bottom: 2px; }}
  .bench-summary:last-of-type {{ margin-bottom: 12px; }}
  .bench-summary b {{ color: #fff; font-weight: 600; }}
  .bench-name {{ color: #ddd; font-weight: 500;
                 font-family: ui-monospace, SFMono-Regular, monospace; }}
  .delta.pos {{ color: #3a8; }}
  .delta.neg {{ color: #c62; }}
  .muted {{ color: #666; }}
  .chart {{ display: block; width: 100%; height: auto; background: #0a0a0a;
            border-radius: 4px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 11px; color: #aaa;
             margin-top: 6px; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 5px;
                  font-family: ui-monospace, SFMono-Regular, monospace; }}
  .legend-swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px;
           font-family: ui-monospace, SFMono-Regular, monospace; }}
  th, td {{ padding: 4px 10px; text-align: left; border-bottom: 1px solid #1f1f1f;
            vertical-align: top; }}
  th {{ color: #777; font-weight: 500; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.04em; }}
  th.bench-col {{ text-align: right; }}
  td.num, td.metric {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.ts {{ color: #888; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; color: #000;
            font-size: 10px; text-transform: uppercase; letter-spacing: 0.03em;
            font-weight: 600; }}
  details summary {{ cursor: pointer; color: #777; font-size: 12px; margin-top: 12px;
                     user-select: none; }}
  details summary:hover {{ color: #aaa; }}
  details.run {{ margin-bottom: 20px; }}
  details.run > summary {{ padding: 10px 14px; background: #161616; border: 1px solid #262626;
                          border-radius: 6px; color: #aaa; font-size: 13px; margin-top: 0;
                          font-family: ui-monospace, SFMono-Regular, monospace; }}
  details.run > summary:hover {{ color: #fff; }}
  details.run[open] > summary {{ border-radius: 6px 6px 0 0; border-bottom: none; }}
  details.run[open] > section {{ border-radius: 0 0 6px 6px; border-top: none;
                                  margin-bottom: 0; }}
  .empty {{ color: #666; font-size: 13px; padding: 12px 0; }}
  code {{ background: #222; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
</style>
</head><body>
<header>
  <h1>autocuda dashboard</h1>
  <span class="path">{root}</span>
</header>
<main id="dashboard-body">
{body}
</main>
<script>
(function () {{
  const REFRESH_MS = 10000;

  function snapshot(root) {{
    const state = {{}};
    root.querySelectorAll('[data-run]').forEach(el => {{
      const key = el.getAttribute('data-run');
      if (el.tagName === 'DETAILS') state[key + '|outer'] = el.open;
      el.querySelectorAll('details').forEach((d, i) => {{
        state[key + '|' + i] = d.open;
      }});
    }});
    return state;
  }}

  function apply(root, state) {{
    root.querySelectorAll('[data-run]').forEach(el => {{
      const key = el.getAttribute('data-run');
      if (el.tagName === 'DETAILS' && state[key + '|outer']) el.open = true;
      el.querySelectorAll('details').forEach((d, i) => {{
        if (state[key + '|' + i]) d.open = true;
      }});
    }});
  }}

  async function refresh() {{
    const host = document.getElementById('dashboard-body');
    if (!host) return;
    try {{
      const resp = await fetch(window.location.pathname, {{cache: 'no-store'}});
      if (!resp.ok) return;
      const text = await resp.text();
      const doc = new DOMParser().parseFromString(text, 'text/html');
      const fresh = doc.getElementById('dashboard-body');
      if (!fresh) return;
      apply(fresh, snapshot(host));
      const x = window.scrollX, y = window.scrollY;
      host.replaceWith(document.adoptNode(fresh));
      window.scrollTo(x, y);
    }} catch (e) {{ /* transient; try again next tick */ }}
  }}

  setInterval(refresh, REFRESH_MS);
}})();
</script>
</body></html>
"""


def render_page(experiments_dir: Path) -> str:
    if not experiments_dir.exists():
        body = (f'<p class="empty">No <code>{html.escape(experiments_dir.name)}/</code> '
                f'directory found at <code>{html.escape(str(experiments_dir))}</code>.</p>')
    else:
        files = sorted(experiments_dir.glob("*.csv"), reverse=True)
        if not files:
            body = (f'<p class="empty">No <code>*.csv</code> logs in '
                    f'<code>{html.escape(str(experiments_dir))}</code>.</p>')
        else:
            body = "\n".join(render_section(f, collapsed=(i > 0)) for i, f in enumerate(files))
    return PAGE.format(root=html.escape(str(experiments_dir)), body=body)


class Handler(BaseHTTPRequestHandler):
    experiments_dir: Path = Path("experiments")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            data = render_page(self.experiments_dir).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/raw/"):
            name = self.path[len("/raw/"):]
            target = (self.experiments_dir / name).resolve()
            if (target.parent == self.experiments_dir.resolve()
                    and target.suffix == ".csv" and target.is_file()):
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--experiments", default="experiments",
                        help="directory containing *-log.csv files (default: ./experiments)")
    args = parser.parse_args()

    Handler.experiments_dir = Path(args.experiments).resolve()
    server = HTTPServer((args.bind, args.port), Handler)
    print(f"Serving {Handler.experiments_dir} on http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
