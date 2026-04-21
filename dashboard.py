#!/usr/bin/env python3
"""Dashboard server for autocuda experiment logs.

Reads `*-log.csv` files from ./experiments/ and serves them as a single
HTML page on http://localhost:8000/. Stdlib only, no dependencies.

The page auto-refreshes so you can leave it open while the optimizer runs.
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
}

LOWER_IS_BETTER_UNITS = {"ms", "s", "us", "ns", "\u00b5s", "sec", "seconds"}


def is_lower_better(unit: str) -> bool:
    return unit.strip().lower() in LOWER_IS_BETTER_UNITS


def load_log(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_metric(row):
    try:
        return float(row.get("metric_value", ""))
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


def render_chart(rows, lower_better, width=900, height=260, pad=44):
    # CSV row order is the true trial order. Clamp each timestamp to the
    # running max so backward or missing timestamps don't distort the x-axis.
    parsed = []
    running_max = None
    for i, r in enumerate(rows):
        v = parse_metric(r)
        if v is None:
            continue
        raw_t = parse_timestamp(r)
        if raw_t is None:
            t = running_max
        elif running_max is None:
            t = raw_t
        else:
            t = raw_t if raw_t >= running_max else running_max
        if t is None:
            continue
        running_max = t
        parsed.append((i, raw_t, t, v, r.get("status", "")))
    if not parsed:
        return '<div class="empty">no numeric data</div>'

    baseline = next((v for (_, _, _, v, s) in parsed if s == "baseline"), parsed[0][3])
    if not baseline:
        return '<div class="empty">baseline is zero; cannot compute speedup</div>'

    def speedup(v):
        if lower_better:
            return baseline / v if v else None
        return v / baseline

    t0 = min(t for _, _, t, _, _ in parsed)
    data = []
    for i, raw_t, t, v, s in parsed:
        sp = speedup(v)
        if sp is None:
            continue
        data.append((i, raw_t, (t - t0).total_seconds(), v, sp, s))
    if not data:
        return '<div class="empty">no numeric data</div>'

    tmin = 0.0
    tmax = max(dt for _, _, dt, _, _, _ in data)
    if tmax <= tmin:
        tmax = tmin + 1.0

    vmin = 0.0
    vmax = max(sp for _, _, _, _, sp, _ in data)
    vmax = max(vmax, 1.0)  # keep baseline reference in view

    def x(dt): return pad + (width - 2 * pad) * (dt - tmin) / (tmax - tmin)
    def y(sp): return height - pad - (height - 2 * pad) * (sp - vmin) / (vmax - vmin)

    kept = [(dt, sp) for (_, _, dt, _, sp, s) in data if s in ("baseline", "improved")]

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    parts.append(f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#333"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#333"/>')
    # y gridlines + speedup labels
    for k in range(3):
        sp = vmin + (vmax - vmin) * (k / 2)
        yv = y(sp)
        parts.append(f'<line x1="{pad}" y1="{yv:.1f}" x2="{width - pad}" y2="{yv:.1f}" stroke="#1f1f1f"/>')
        parts.append(f'<text x="{pad - 6}" y="{yv + 4:.1f}" text-anchor="end" fill="#888" font-size="11">{sp:.2f}\u00d7</text>')
    # x tick labels (walltime since start)
    for k in range(3):
        dt = tmin + (tmax - tmin) * (k / 2)
        xv = x(dt)
        anchor = "start" if k == 0 else ("end" if k == 2 else "middle")
        parts.append(f'<text x="{xv:.1f}" y="{height - pad + 14:.1f}" text-anchor="{anchor}" fill="#888" font-size="11">{fmt_duration(dt)}</text>')
    # dashed reference line at baseline speedup (1.0)
    yb = y(1.0)
    parts.append(f'<line x1="{pad}" y1="{yb:.1f}" x2="{width - pad}" y2="{yb:.1f}" stroke="#555" stroke-dasharray="4 4"/>')
    # x-axis title
    parts.append(f'<text x="{width / 2:.0f}" y="{height - 6}" text-anchor="middle" fill="#666" font-size="11">walltime since start</text>')
    # kept trajectory line
    if len(kept) >= 2:
        d = "M " + " L ".join(f"{x(dt):.1f},{y(sp):.1f}" for dt, sp in kept)
        parts.append(f'<path d="{d}" fill="none" stroke="#3a8" stroke-width="2"/>')
    # all-trial dots
    for i, raw_t, dt, v, sp, s in data:
        color = STATUS_COLORS.get(s, "#888")
        if raw_t is None:
            note = " [timestamp missing; pinned to prior trial]"
        elif (raw_t - t0).total_seconds() < dt - 0.5:
            raw_iso = raw_t.isoformat(timespec="seconds")
            note = f" [raw timestamp {raw_iso} clamped]"
        else:
            note = ""
        parts.append(f'<circle cx="{x(dt):.1f}" cy="{y(sp):.1f}" r="3.5" fill="{color}"><title>trial {i}: {sp:.3f}\u00d7 ({v:g}) @ {fmt_duration(dt)} ({s}){note}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def render_summary(rows, unit, lower_better):
    if not rows:
        return ""
    metrics = [(i, parse_metric(r), r.get("status", "")) for i, r in enumerate(rows)]
    numeric = [(i, v, s) for (i, v, s) in metrics if v is not None]
    if not numeric:
        return f'<div class="summary">{len(rows)} trials · no numeric data</div>'

    baseline = next((v for (_, v, s) in numeric if s == "baseline"), numeric[0][1])
    kept = [(i, v) for (i, v, s) in numeric if s in ("baseline", "improved")]
    if kept:
        best_i, best = (min(kept, key=lambda x: x[1]) if lower_better
                        else max(kept, key=lambda x: x[1]))
    else:
        best_i, best = numeric[0][0], numeric[0][1]
    delta = (best - baseline) / baseline * 100 if baseline else 0
    if lower_better:
        delta = -delta
    failures = sum(1 for r in rows if r.get("status", "").endswith("_error"))

    # Trials/hr uses clamped timestamps (see render_chart) so reverse timestamps don't poison the rate.
    running = None
    clamped = []
    for r in rows:
        raw = parse_timestamp(r)
        if raw is None:
            t = running
        elif running is None:
            t = raw
        else:
            t = raw if raw >= running else running
        if t is not None:
            running = t
        clamped.append(t)
    usable = [t for t in clamped if t is not None]
    rate_span = ""
    if len(rows) >= 2 and len(usable) >= 2:
        elapsed = (usable[-1] - usable[0]).total_seconds()
        if elapsed >= 60:
            rate_span = f'<span><b>{(len(rows) - 1) * 3600 / elapsed:.1f}</b> trials/hr</span>'

    return (
        '<div class="summary">'
        f'<span><b>{len(rows)}</b> trials</span>'
        f'{rate_span}'
        f'<span>baseline <b>{baseline:g}</b> {html.escape(unit)}</span>'
        f'<span>best <b>{best:g}</b> {html.escape(unit)} @ trial {best_i}</span>'
        f'<span class="delta {"pos" if delta >= 0 else "neg"}">{delta:+.2f}%</span>'
        f'<span class="muted">{failures} failures</span>'
        '</div>'
    )


def render_table(rows):
    out = ['<table><thead><tr>',
           '<th>#</th><th>timestamp</th><th>metric</th><th>unit</th><th>status</th><th>description</th>',
           '</tr></thead><tbody>']
    for i, row in enumerate(rows):
        status = row.get("status", "")
        color = STATUS_COLORS.get(status, "#777")
        out.append(f'<tr>')
        out.append(f'<td class="num">{i}</td>')
        out.append(f'<td class="ts">{html.escape(row.get("timestamp", ""))}</td>')
        out.append(f'<td class="metric">{html.escape(row.get("metric_value", ""))}</td>')
        out.append(f'<td>{html.escape(row.get("unit", ""))}</td>')
        out.append(f'<td><span class="badge" style="background:{color}">{html.escape(status)}</span></td>')
        out.append(f'<td>{html.escape(row.get("description", ""))}</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return "".join(out)


def render_section(path: Path, collapsed: bool = False):
    name = html.escape(path.name)
    try:
        rows = load_log(path)
        unit = next((r.get("unit", "") for r in rows if r.get("unit")), "")
        lower_better = is_lower_better(unit)
        body = (
            f'<h2>{name}</h2>'
            f'{render_summary(rows, unit, lower_better)}'
            f'{render_chart(rows, lower_better)}'
            f'<details><summary>show {len(rows)} trials</summary>{render_table(rows)}</details>'
        )
        count = len(rows)
    except OSError as e:
        body = f'<h2>{name}</h2><p class="empty">read error: {html.escape(str(e))}</p>'
        count = 0

    if not collapsed:
        return f'<section data-run="{name}">{body}</section>'
    summary = f'{name} · {count} trials' if count else name
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
              margin-bottom: 14px; }}
  .summary b {{ color: #fff; font-weight: 600; }}
  .delta.pos {{ color: #3a8; }}
  .delta.neg {{ color: #c62; }}
  .muted {{ color: #666; }}
  .chart {{ display: block; width: 100%; height: auto; background: #0a0a0a;
            border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px;
           font-family: ui-monospace, SFMono-Regular, monospace; }}
  th, td {{ padding: 4px 10px; text-align: left; border-bottom: 1px solid #1f1f1f;
            vertical-align: top; }}
  th {{ color: #777; font-weight: 500; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.04em; }}
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
