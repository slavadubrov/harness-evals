"""Markdown tables for summaries and comparisons, and an SVG plot of the frontier."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from html import escape
from math import floor, log10
from typing import Any

from harness_evals.compare import Comparison
from harness_evals.core import Grade
from harness_evals.metrics import pareto
from harness_evals.usecase import UseCase

REQUIRED = [
    ("success", "Success"),
    ("pass^k", "pass^k"),
    ("violations", "Violations"),
    ("cost_per_task", "$ per task"),
    ("cost_per_success", "$ per success"),
    ("latency_p50", "p50 latency, s"),
    ("latency_p95", "p95 latency, s"),
    ("calls_per_task", "Calls per task"),
    ("tokens_per_task", "Tokens per task"),
    ("infra_errors", "Infra errors"),
    ("unknown_verdicts", "Undecided"),
]


def _num(key: str, v: float) -> str:
    if key.startswith("cost"):
        return f"${v:.5f}"
    return f"{v:.0f}" if key == "tokens_per_task" else f"{v:.2f}"


def _fmt(key: str, v: Any, s: Mapping[str, Any]) -> str:
    if v is None:
        return "unknown"
    if key == "pass^k":
        return f"{v:.2f} (k={s['trials']})"
    numeric = isinstance(v, int | float) and not isinstance(v, bool)
    if not numeric or (isinstance(v, int) and f"{key}_ci" not in s):
        return str(v)  # counts such as violations
    out = _num(key, v)
    if ci := s.get(f"{key}_ci"):
        out += f" ({_num(key, ci[0]).lstrip('$')}–{_num(key, ci[1]).lstrip('$')})"
    if key.startswith("cost") and s.get("unknown_cost"):
        out += f" ({s['unknown_cost']} unknown)"
    return out


def table(summaries: Mapping[str, Mapping[str, Any]], extra: list[str] | None = None) -> str:
    """One row per harness. A rate, cost, latency or usage value is followed by its 95%
    interval over tasks in brackets; pass^k is followed by its k."""
    cols = REQUIRED + [(k, k) for k in extra or []]
    lines = [
        "| Harness | " + " | ".join(label for _, label in cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for name, s in summaries.items():
        cells = [_fmt(k, s.get(k), s) for k, _ in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison(c: Comparison) -> str:
    lo, hi = c.ci
    lines = [
        f"**{c.candidate}** against **{c.base}** on {c.tasks} tasks: "
        f"success {c.diff:+.3f} ({lo:+.3f} to {hi:+.3f}).",
        f"Better on {_tasks(len(c.better))}, worse on {_tasks(len(c.worse))}.",
    ]
    others = []
    for name, d in c.deltas.items():
        if name == "success":
            continue
        if d is None:
            others.append(f"{name} unknown")
            continue
        v, (dlo, dhi) = d
        digits = 5 if name.startswith("cost") else 0 if name == "tokens_per_task" else 2
        others.append(f"{name} {v:+.{digits}f} ({dlo:+.{digits}f} to {dhi:+.{digits}f})")
    if others:
        lines.append("Candidate minus base: " + "; ".join(others) + ".")
    lines.append(f"Decision: **{c.decision}**. " + " ".join(c.reasons))
    return "\n".join(lines)


def _tasks(n: int) -> str:
    return f"{n} task" + "s" * (n != 1)


def frontier_svg(
    summaries: Mapping[str, Mapping[str, Any]],
    x: str,
    y: str,
    *,
    lower: Iterable[str] = (),
    title: str = "",
) -> str:
    """Every harness that has both metrics, with each metric's 95% interval as an error
    bar and a line through the frontier (metrics.pareto). Others are left out."""
    pts = {h: s for h, s in summaries.items() if s.get(x) is not None and s.get(y) is not None}
    if not pts:
        raise ValueError(f"no harness has both {x} and {y}")
    front = pareto(pts, x, y, lower=lower)
    w, h, left, right, top, bottom = 640, 440, 76, 28, 64, 56
    (x0, x1), (y0, y1) = _axis(pts, x), _axis(pts, y)

    def px(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * (w - left - right)

    def py(v: float) -> float:
        return h - bottom - (v - y0) / (y1 - y0) * (h - top - bottom)

    title = title or f"{y} against {x}"
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" '
        f'font-family="system-ui, sans-serif" font-size="13" fill="#222">',
        f"<title>{escape(title)}</title>",
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
        f'<text x="{left}" y="24" font-size="15" font-weight="600">{escape(title)}</text>',
        f'<text x="{left}" y="44" fill="#666">Bars: 95% intervals over tasks. '
        f'<tspan fill="#c2185b">Line: no other harness beats these on both.</tspan></text>',
    ]
    for i in range(6):
        vx, vy = x0 + (x1 - x0) * i / 5, y0 + (y1 - y0) * i / 5
        o += [
            f'<path d="M{px(vx):.1f} {top}V{h - bottom}M{left} {py(vy):.1f}H{w - right}" '
            f'stroke="#e6e6e6"/>',
            f'<text x="{px(vx):.1f}" y="{h - bottom + 18}" text-anchor="middle" fill="#666">'
            f"{_tick(vx, x1 - x0)}</text>",
            f'<text x="{left - 8}" y="{py(vy) + 4:.1f}" text-anchor="end" fill="#666">'
            f"{_tick(vy, y1 - y0)}</text>",
        ]
    o += [
        f'<text x="{(left + w - right) / 2}" y="{h - 14}" text-anchor="middle">{escape(x)}</text>',
        f'<text transform="translate(20 {(top + h - bottom) / 2}) rotate(-90)" '
        f'text-anchor="middle">{escape(y)}</text>',
    ]
    line = sorted((pts[n][x], pts[n][y]) for n in front)
    o.append(
        '<polyline fill="none" stroke="#c2185b" stroke-width="2" points="'
        + " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in line)
        + '"/>'
    )
    placed: list[tuple[float, float, float]] = []  # label boxes: x, y, width
    for name, s in pts.items():
        cx, cy = px(s[x]), py(s[y])
        color = "#c2185b" if name in front else "#444"
        if ci := s.get(f"{x}_ci"):
            o.append(f'<path d="M{px(ci[0]):.1f} {cy:.1f}H{px(ci[1]):.1f}" stroke="{color}"/>')
        if ci := s.get(f"{y}_ci"):
            o.append(f'<path d="M{cx:.1f} {py(ci[0]):.1f}V{py(ci[1]):.1f}" stroke="{color}"/>')
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}"/>')
        lx, ly, lw = cx + 8, cy - 8, 7.2 * len(name)
        while any(abs(ly - b) < 15 and lx < a + c and a < lx + lw for a, b, c in placed):
            ly += 15  # Greedy nudge downward: enough for a handful of harnesses.
        placed.append((lx, ly, lw))
        if ly > cy - 8:  # moved: a leader line keeps the label with its point
            o.append(f'<path d="M{cx:.1f} {cy:.1f}L{lx - 2:.1f} {ly - 4:.1f}" stroke="#bbb"/>')
        o.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{color}">{escape(name)}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def _axis(pts: Mapping[str, Mapping[str, Any]], m: str) -> tuple[float, float]:
    """From a nice bound below the values and intervals, or zero, to one above, or zero."""
    vals = [v for s in pts.values() for v in (s[m], *(s.get(f"{m}_ci") or ()))]
    lo, hi = min(vals), max(vals)
    bottom = -_nice(-lo) if lo < 0 else 0.0
    top = _nice(hi) if hi > 0 else 0.0
    return (bottom, top) if top > bottom else (0.0, 1.0)


def _nice(v: float) -> float:
    """The smallest of 1, 2, 2.5 or 5 times a power of ten that is at least v."""
    mag = 10 ** floor(log10(v))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= v * (1 - 1e-9))


def _tick(v: float, span: float) -> str:
    step = span / 5  # the fewest decimals that show every tick exactly
    decimals = next((d for d in range(7) if abs(round(step, d) - step) < 1e-9 * step), 6)
    return f"{v:.{decimals}f}"


def check_kind(name: str) -> str:
    """The column a check falls in: its name without arguments, so "did refund(order='A1')"
    and "did refund(order='B1')" share the column "did refund"."""
    return name.split("(", 1)[0].strip()


def failure_map_svg(
    usecase: UseCase,
    grades: Mapping[str, Sequence[Grade]],
    *,
    by: str = "checks",
    group: Callable[[str], str] = check_kind,
    title: str = "",
) -> str:
    """Where each harness fails: one row per task and, in a panel per harness, one column
    per kind of check (``by="checks"``) or per run (``by="runs"``). A cell shows the share
    of the task's runs that failed there; hovering over it gives the count."""
    if by not in ("checks", "runs"):
        raise ValueError("by must be 'checks' or 'runs'")
    graded = {g.episode.task_id for gs in grades.values() for g in gs}
    tasks = [t for t in usecase.tasks if t.id in graded]
    if not tasks:
        raise ValueError("no graded runs of the use case's tasks")
    cols, cells = _failure_cells(tasks, grades, by, group)

    size, pitch, char = 13, 16, 7.2  # cell, grid step, width of a 13px character
    label_w = 16 + char * min(44, max(len(t.id) for t in tasks))
    head = 28 + (char * max(map(len, cols)) if by == "checks" else 14)  # labels stand upright
    panel_w = {h: max(len(cols) * pitch, char * len(h)) + 28 for h in grades}
    top = 78 + head
    legend_w = 16 + 5 * (size + 22) + 6.6 * len("every run passedsome runs failedevery run failed")
    legend_w += 6.6 * len("undecidednot checked on this task")
    width = max(label_w + sum(panel_w.values()) + 16, legend_w)
    height = top + len(tasks) * pitch + 24
    red, green, grey, pink = "#c2185b", "#2e7d32", "#9e9e9e", "#fce4ec"
    title = title or f"Where each harness fails, by {'check' if by == 'checks' else 'run'}"

    def box(x: float, y: float, fill: str, inner: str = "", stroke: str = "none") -> str:
        attrs = f'x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" rx="2"'
        return f'<rect {attrs} fill="{fill}" stroke="{stroke}">{inner}</rect>'

    def dot(x: float, y: float) -> str:
        return f'<circle cx="{x + size / 2:.1f}" cy="{y + size / 2:.1f}" r="1.6" fill="{grey}"/>'

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" font-family="system-ui, sans-serif" font-size="13" fill="#222">',
        f"<title>{escape(title)}</title>",
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#fff"/>',
        f'<text x="16" y="26" font-size="15" font-weight="600">{escape(title)}</text>',
    ]
    half = f'<rect y="{size / 2}" width="{size}" height="{size / 2}" fill="{red}"/>'
    legend = [
        (box(0, 0, green), "every run passed"),
        (box(0, 0, pink, stroke=red) + half, "some runs failed"),
        (box(0, 0, red), "every run failed"),
        (box(0, 0, grey), "undecided"),
        (dot(0, 0), "not checked on this task"),
    ]
    if by == "runs":
        legend = [(box(0, 0, green), "passed"), (box(0, 0, red), "failed"), legend[3]]
    x = 16.0
    for glyph, text in legend:
        o.append(f'<g transform="translate({x:.0f} 42)">{glyph}</g>')
        o.append(f'<text x="{x + size + 6:.0f}" y="53" fill="#555">{text}</text>')
        x += size + 22 + 6.6 * len(text)
    for i, t in enumerate(tasks):
        name = t.id if len(t.id) <= 44 else t.id[:43] + "…"
        y = top + i * pitch + 11
        o.append(f'<text x="{label_w - 10:.0f}" y="{y}" text-anchor="end">{escape(name)}</text>')
    x0 = label_w
    for h in grades:
        o.append(f'<text x="{x0:.0f}" y="90" font-weight="600">{escape(h)}</text>')
        for j, c in enumerate(cols):
            cx = x0 + j * pitch + size / 2
            at = (
                f'transform="translate({cx + 4:.1f} {top - 6:.0f}) rotate(-90)"'
                if by == "checks"
                else f'x="{cx:.1f}" y="{top - 6:.0f}" text-anchor="middle"'
            )
            o.append(f'<text {at} fill="#555">{escape(c)}</text>')
        for i, t in enumerate(tasks):
            for j, c in enumerate(cols):
                x, y = x0 + j * pitch, top + i * pitch
                v = cells[h, t.id, c]
                if v is None:
                    o.append(dot(x, y))
                if not v or not v[2]:
                    continue
                bad, unsure, n = v
                tip = f"<title>{escape(f'{h} · {t.id} · {c}: {bad} of {n} runs failed')}</title>"
                if 0 < bad < n:  # the failed share fills the cell from the bottom
                    bar = size * bad / n
                    part = (
                        f'<rect x="{x}" y="{y + size - bar:.1f}" width="{size}" height="{bar:.1f}"'
                    )
                    o.append(f'<g>{box(x, y, pink, stroke=red)}{part} fill="{red}"/>{tip}</g>')
                else:
                    o.append(box(x, y, red if bad else grey if unsure else green, tip))
        x0 += panel_w[h]
    o.append("</svg>")
    return "\n".join(o) + "\n"


def _failure_cells(
    tasks: Sequence[Any],
    grades: Mapping[str, Sequence[Grade]],
    by: str,
    group: Callable[[str], str],
) -> tuple[list[str], dict[tuple[str, str, str], tuple[int, int, int] | None]]:
    """The columns, and (failed, undecided, runs) for every cell; None where a check does
    not apply to the task."""
    runs = {
        (h, t.id): [g for g in gs if g.episode.task_id == t.id]
        for h, gs in grades.items()
        for t in tasks
    }
    cells: dict[tuple[str, str, str], tuple[int, int, int] | None] = {}
    if by == "runs":
        trials = max(g.episode.trial for v in runs.values() for g in v) + 1
        cols = [str(i + 1) for i in range(trials)]
        for (h, task), gs in runs.items():
            for i, c in enumerate(cols):
                mine = [g for g in gs if g.episode.trial == i]
                bad = sum(bool(g.failed) for g in mine)
                unsure = sum(bool(g.unknown and not g.failed) for g in mine)
                cells[h, task, c] = (bad, unsure, len(mine))
        return cols, cells
    cols = list(dict.fromkeys(group(c.name) for t in tasks for c in t.checks))
    if any("no run error" in g.failed for v in runs.values() for g in v):
        cols.append("no run error")
    applies = {t.id: {group(c.name) for c in t.checks} | {"no run error"} for t in tasks}
    for (h, task), gs in runs.items():
        for c in cols:
            if c not in applies[task]:
                cells[h, task, c] = None
                continue
            bad = [any(group(f) == c for f in g.failed) for g in gs]
            unsure = [
                not b and any(group(u) == c for u in g.unknown)
                for g, b in zip(gs, bad, strict=True)
            ]
            cells[h, task, c] = (sum(bad), sum(unsure), len(gs))
    return cols, cells
