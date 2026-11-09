"""Markdown tables for summaries and comparisons, and an SVG plot of the frontier."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import escape
from math import floor, log10
from typing import Any

from harness_evals.compare import Comparison
from harness_evals.metrics import pareto

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
    if not isinstance(v, float):
        return str(v)
    out = _num(key, v)
    if ci := s.get(f"{key}_ci"):
        out += f" ({_num(key, ci[0]).lstrip('$')}–{_num(key, ci[1]).lstrip('$')})"
    if key.startswith("cost") and s.get("unknown_cost"):
        out += f" ({s['unknown_cost']} unknown)"
    return out


def table(summaries: Mapping[str, Mapping[str, Any]], extra: list[str] | None = None) -> str:
    """One row per harness; a value in brackets is its 95% interval over tasks."""
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
    """Every harness on two metrics, with each metric's 95% interval as an error bar and
    a line through the harnesses no other beats on both (metrics.pareto)."""
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
            f"{_tick(vx, x1)}</text>",
            f'<text x="{left - 8}" y="{py(vy) + 4:.1f}" text-anchor="end" fill="#666">'
            f"{_tick(vy, y1)}</text>",
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
    """From zero, or a nice negative bound, to a nice upper bound over values and intervals."""
    vals = [v for s in pts.values() for v in (s[m], *(s.get(f"{m}_ci") or ()))]
    lo, hi = min(vals), max(vals)
    top = _nice(hi) if hi > 0 else 1.0
    return (-_nice(-lo) if lo < 0 else 0.0), top


def _nice(v: float) -> float:
    """The smallest of 1, 2, 2.5 or 5 times a power of ten that is at least v."""
    mag = 10 ** floor(log10(v))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= v * (1 - 1e-9))


def _tick(v: float, top: float) -> str:
    decimals = max(0, -floor(log10(top / 5) + 1e-9))
    return f"{v:.{decimals}f}"
