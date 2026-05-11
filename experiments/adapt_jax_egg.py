#!/usr/bin/env python3
"""Inject `TiledHDot` / `TiledHConv`, tiling + algebraic rulesets, and a two-phase
`(run-schedule …)` into an HLO `.egg`. Default out: sibling `.tiled.egg`.

    python3 experiments/adapt_jax_egg.py experiments/jax_egglog/bert_tiny.egg
        [--out OUT] [--tiles 32 64 128] [--k-algebraic 1]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_TILES = (32, 64, 128)


def _inject_into_datatype(src: str, new_ctors: list[str]) -> str:
    """Insert `new_ctors` right before the closing ')' of `(datatype Hlo …)`."""
    m = re.search(r"\(datatype\s+Hlo\b", src)
    if m is None:
        raise RuntimeError("could not find `(datatype Hlo …)` block")
    start = m.start()

    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                inject = "\n  ;; ----- flow_prune injected tiled targets -----\n"
                inject += "\n".join(f"  {ctor}" for ctor in new_ctors) + "\n"
                return src[:i] + inject + src[i:]
    raise RuntimeError("unbalanced parens in `(datatype Hlo …)`")


def _tag_rewrites_with_ruleset(src: str, ruleset: str) -> str:
    """Append `:ruleset <ruleset>` to every top-level `(rewrite …)` /
    `(birewrite …)` that doesn't already specify one."""
    out_lines = []
    in_rewrite = False
    depth = 0
    buf = []

    def flush_rewrite(text: str) -> str:
        if ":ruleset" in text:
            return text
        stripped = text.rstrip()
        trailing = text[len(stripped):]
        if not stripped.endswith(")"):
            return text
        return stripped[:-1] + f" :ruleset {ruleset})" + trailing

    for line in src.splitlines(keepends=True):
        stripped = line.lstrip()
        if not in_rewrite and (stripped.startswith("(rewrite") or stripped.startswith("(birewrite")):
            in_rewrite = True
            depth = 0
            buf = []
        if in_rewrite:
            buf.append(line)
            for ch in line:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            if depth == 0:
                out_lines.append(flush_rewrite("".join(buf)))
                in_rewrite = False
                buf = []
        else:
            out_lines.append(line)
    return "".join(out_lines)


def _strip_trailing_run(src: str) -> tuple[str, int]:
    """Remove trailing `(run N)` or `(run-schedule …)`; return body and N (default 3)."""
    m = re.search(r"\(\s*run\s+(\d+)\s*\)\s*$", src)
    if m:
        return src[: m.start()].rstrip() + "\n", int(m.group(1))
    m = re.search(r"\(run-schedule[\s\S]*?\)\s*$", src)
    if m:
        return src[: m.start()].rstrip() + "\n", 3
    return src, 3


def adapt(src: str, tiles=DEFAULT_TILES, k_algebraic: int | None = None) -> str:
    new_ctors = [
        "(TiledHDot  Hlo Hlo String i64)",
        "(TiledHConv Hlo Hlo String i64)",
    ]
    s = _inject_into_datatype(src, new_ctors)

    decls = (
        "\n;; ----- flow_prune ruleset declarations -----\n"
        "(ruleset algebraic)\n"
        "(ruleset tiling)\n"
    )
    m = re.search(r"\(datatype\s+Hlo\b", s)
    depth = 0
    insert_at = None
    for i in range(m.start(), len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                insert_at = i + 1
                break
    s = s[:insert_at] + decls + s[insert_at:]

    s = _tag_rewrites_with_ruleset(s, "algebraic")
    s, default_k = _strip_trailing_run(s)
    if k_algebraic is None:
        k_algebraic = default_k

    tiling_rules = [
        "",
        ";; ----- flow_prune injected tiling rewrites -----",
    ]
    for T in tiles:
        tiling_rules.append(
            f"(rewrite (HDot a c dt) (TiledHDot a c dt {T}) :ruleset tiling)"
        )
    for T in tiles:
        tiling_rules.append(
            f"(rewrite (HConv a k ct) (TiledHConv a k ct {T}) :ruleset tiling)"
        )

    schedule = [
        "",
        ";; ----- flow_prune two-phase schedule -----",
        "(run-schedule",
        f"  (repeat {k_algebraic} (run algebraic))",
        "  (run tiling))",
        "",
    ]

    return s + "\n".join(tiling_rules) + "\n" + "\n".join(schedule)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tiles", type=int, nargs="+", default=list(DEFAULT_TILES))
    ap.add_argument("--k-algebraic", type=int, default=None,
                    help="iterations of the algebraic ruleset per round "
                         "(default: copy the original (run N))")
    args = ap.parse_args(argv)

    src = args.input.read_text()
    out_path = args.out or args.input.with_suffix(".tiled.egg")
    out_path.write_text(adapt(src, tuple(args.tiles), args.k_algebraic))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
