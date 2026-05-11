"""XLA HLO text (e.g. from jax) → egglog: datatype + lets, inline `call`, reduce bodies by reducer root op.

Non-topological attrs become string tags. API: `parse_hlo_module`, `emit_egglog`, `convert`."""

from __future__ import annotations

import re
from typing import Optional


_HDR_RE = re.compile(
    r"^\s*(?:(ENTRY)\s+)?([\w.]+)"
    r"(?:\s*\(.*\))?(?:\s*->[^{]*)?\s*\{\s*$"
)
_INSTR_RE = re.compile(r"^\s*(ROOT\s+)?([\w.]+)\s*=\s*(.*)$")


def _find_matching_paren(s: str, start: int) -> int:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced parens in: {s!r}")


def _split_top_level_commas(s: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "({[":
            depth += 1
            cur.append(ch)
        elif ch in ")}]":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _parse_attrs(s: str) -> dict:
    s = s.lstrip(",").strip()
    if not s:
        return {}
    out: dict[str, str] = {}
    for part in _split_top_level_commas(s):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_instruction(line: str) -> Optional[dict]:
    m = _INSTR_RE.match(line)
    if not m:
        return None
    is_root = bool(m.group(1))
    name = m.group(2)
    rest = m.group(3)
    lparen = rest.find("(")
    if lparen < 0:
        return None
    type_op = rest[:lparen].strip()
    parts = type_op.rsplit(None, 1)
    if len(parts) < 2:
        # e.g. `() tuple()` — keep just the opcode
        type_str = ""
        opcode = parts[0]
    else:
        type_str, opcode = parts[0], parts[1]
    rparen = _find_matching_paren(rest, lparen)
    args_str = rest[lparen + 1 : rparen].strip()
    attrs_str = rest[rparen + 1 :].strip()
    args = [a.strip() for a in _split_top_level_commas(args_str)]
    return {
        "name": name,
        "opcode": opcode,
        "args": args,
        "attrs": _parse_attrs(attrs_str),
        "is_root": is_root,
        "type": type_str,
    }


def parse_hlo_module(text: str) -> dict:
    """Returns {comp_name: {is_entry, params: [(idx, name)], instrs, root}}."""
    computations: dict[str, dict] = {}
    current: Optional[dict] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("HloModule") or stripped.startswith("//"):
            continue
        if current is None:
            m = _HDR_RE.match(stripped)
            if m:
                current = {
                    "name": m.group(2),
                    "is_entry": m.group(1) is not None,
                    "params": [],
                    "instrs": [],
                    "root": None,
                }
                computations[current["name"]] = current
            continue
        if stripped == "}":
            current = None
            continue
        instr = _parse_instruction(line)
        if instr is None:
            continue
        if instr["opcode"] == "parameter":
            idx = int(instr["args"][0]) if instr["args"] else len(current["params"])
            current["params"].append((idx, instr["name"]))
        else:
            current["instrs"].append(instr)
        if instr["is_root"]:
            current["root"] = instr["name"]
    return computations


def _inline_calls(entry: dict, module: dict) -> dict:
    """Inline `call` chains from ENTRY; rename locals with prefixes to avoid collisions."""
    flat_instrs: list[dict] = []
    counter = [0]
    entry_root_holder = {"root": entry["root"]}

    def emit(instrs: list[dict], scope: dict, prefix: str,
             is_entry_scope: bool) -> None:
        for instr in instrs:
            op = instr["opcode"]
            name = instr["name"]
            if op == "call":
                sub_name = instr["attrs"].get("to_apply")
                sub = module.get(sub_name)
                if sub is not None:
                    sub_scope: dict = {}
                    for idx, pname in sorted(sub["params"], key=lambda p: p[0]):
                        if idx < len(instr["args"]):
                            arg = instr["args"][idx]
                            sub_scope[pname] = scope.get(arg, arg)
                    counter[0] += 1
                    sub_prefix = f"{prefix}inl{counter[0]}_{sub_name.replace('.', '_')}_"
                    emit(sub["instrs"], sub_scope, sub_prefix,
                         is_entry_scope=False)
                    sub_root = sub["root"]
                    if sub_root in sub_scope:
                        scope[name] = sub_scope[sub_root]
                        if is_entry_scope and entry_root_holder["root"] == name:
                            entry_root_holder["root"] = sub_scope[sub_root]
                    continue
            new_name = prefix + name
            copy = dict(instr)
            copy["name"] = new_name
            copy["args"] = [scope.get(a, a) for a in instr["args"]]
            flat_instrs.append(copy)
            scope[name] = new_name
            if is_entry_scope and entry_root_holder["root"] == name:
                entry_root_holder["root"] = new_name

    emit(entry["instrs"], scope={}, prefix="", is_entry_scope=True)

    return {
        "name": entry["name"],
        "is_entry": True,
        "params": list(entry["params"]),
        "instrs": flat_instrs,
        "root": entry_root_holder["root"],
    }


PRELUDE = r"""
;; Auto-generated HLO -> egglog datatype.

(datatype Hlo
  (HParam String)
  (HConst String)

  ;; element-wise unary
  (HNeg Hlo)
  (HAbs Hlo)
  (HExp Hlo)
  (HLog Hlo)
  (HSqrt Hlo)
  (HRsqrt Hlo)
  (HTanh Hlo)
  (HSigmoid Hlo)
  (HRelu Hlo)
  (HCopy Hlo)
  (HConvert Hlo String)

  ;; element-wise binary
  (HAdd Hlo Hlo)
  (HSub Hlo Hlo)
  (HMul Hlo Hlo)
  (HDiv Hlo Hlo)
  (HMax Hlo Hlo)
  (HMin Hlo Hlo)
  (HPow Hlo Hlo)
  (HCompare Hlo Hlo String)

  ;; ternary
  (HSelect Hlo Hlo Hlo)

  ;; data movement
  (HTranspose Hlo String)
  (HReshape Hlo String)
  (HBroadcast Hlo String)
  (HSlice Hlo String)
  (HConcat Hlo Hlo String)
  (HPad Hlo Hlo String)
  (HReverse Hlo String)

  ;; linear algebra
  (HDot Hlo Hlo String)
  (HConv Hlo Hlo String)

  ;; reductions (input, init, dims-tag)
  (HReduceSum Hlo Hlo String)
  (HReduceMax Hlo Hlo String)
  (HReduceMin Hlo Hlo String)
  (HReduceProd Hlo Hlo String)

  (HIota String String)

  ;; generic fallback for opcodes the bridge doesn't model individually
  (HOp1 String Hlo)
  (HOp2 String Hlo Hlo)
  (HOp3 String Hlo Hlo Hlo)
  (HOp4 String Hlo Hlo Hlo Hlo)
  (HOp5 String Hlo Hlo Hlo Hlo Hlo))
"""


# Optional rewrites (default off).
SAFE_REWRITES = r"""
;; ---- structural simplifications ----------------------------------------
(rewrite (HRelu (HRelu x)) (HRelu x))
(rewrite (HCopy x) x)
(rewrite (HConvert (HConvert x s1) s2) (HConvert x s2))
(rewrite (HReshape (HReshape x s1) s2) (HReshape x s2))
(rewrite (HNeg (HNeg x)) x)

;; ---- algebraic equivalences (these grow the e-graph) -------------------
(rewrite (HAdd a b) (HAdd b a))
(rewrite (HMul a b) (HMul b a))
(rewrite (HMax a b) (HMax b a))
(rewrite (HMin a b) (HMin b a))

(rewrite (HAdd (HAdd a b) c) (HAdd a (HAdd b c)))
(rewrite (HAdd a (HAdd b c)) (HAdd (HAdd a b) c))
(rewrite (HMul (HMul a b) c) (HMul a (HMul b c)))
(rewrite (HMul a (HMul b c)) (HMul (HMul a b) c))

(rewrite (HSub a b) (HAdd a (HNeg b)))
(rewrite (HAdd a (HNeg b)) (HSub a b))
"""


_UNARY = {
    "negate": "HNeg",
    "abs": "HAbs",
    "exponential": "HExp",
    "log": "HLog",
    "sqrt": "HSqrt",
    "rsqrt": "HRsqrt",
    "tanh": "HTanh",
    "logistic": "HSigmoid",
}

_BINARY = {
    "add": "HAdd",
    "subtract": "HSub",
    "multiply": "HMul",
    "divide": "HDiv",
    "maximum": "HMax",
    "minimum": "HMin",
    "power": "HPow",
}


def _sanitize(name: str) -> str:
    s = name.replace(".", "_").replace("-", "_")
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "v_" + s
    return s


def _emit_instruction(instr: dict, module: dict, neuter_ops: set | None = None) -> str:
    op = instr["opcode"]
    args = ["$" + _sanitize(a) for a in instr["args"]]
    attrs = instr["attrs"]

    if neuter_ops and op in neuter_ops and args:
        return f"(HCopy {args[0]})"

    if op == "constant":
        val_tag = (instr["args"][0] if instr["args"] else instr["name"]).replace('"', '')
        return f'(HConst "{val_tag}")'

    if op == "parameter":
        return f'(HParam "{instr["name"]}")'

    if op in _UNARY:
        return f"({_UNARY[op]} {args[0]})"

    if op in _BINARY:
        return f"({_BINARY[op]} {args[0]} {args[1]})"

    if op == "compare":
        cmp = attrs.get("direction", "EQ").strip()
        return f'(HCompare {args[0]} {args[1]} "{cmp}")'

    if op == "select":
        return f"(HSelect {args[0]} {args[1]} {args[2]})"

    if op == "dot":
        tag = ";".join(
            f"{k}={attrs[k]}"
            for k in (
                "lhs_contracting_dims",
                "rhs_contracting_dims",
                "lhs_batch_dims",
                "rhs_batch_dims",
            )
            if k in attrs
        )
        return f'(HDot {args[0]} {args[1]} "{tag}")'

    if op == "convolution":
        tag = ";".join(f"{k}={attrs[k]}" for k in attrs)
        return f'(HConv {args[0]} {args[1]} "{tag}")'

    if op == "transpose":
        return f'(HTranspose {args[0]} "{attrs.get("dimensions", "")}")'

    if op == "reshape":
        return f'(HReshape {args[0]} "{instr["type"]}")'

    if op == "broadcast":
        return f'(HBroadcast {args[0]} "{attrs.get("dimensions", "")}")'

    if op == "slice":
        slice_attrs = ";".join(
            f"{k}={attrs[k]}"
            for k in attrs if k in ("slice", "start_indices", "limit_indices", "strides")
        )
        return f'(HSlice {args[0]} "{slice_attrs}")'

    if op == "pad":
        return f'(HPad {args[0]} {args[1]} "{attrs.get("padding", "")}")'

    if op == "reverse":
        return f'(HReverse {args[0]} "{attrs.get("dimensions", "")}")'

    if op == "convert":
        return f'(HConvert {args[0]} "{instr["type"]}")'

    if op == "concatenate":
        if not args:
            return '(HConst "concat_empty")'
        if len(args) == 1:
            return args[0]
        dim = attrs.get("dimensions", "0")
        result = args[0]
        for nxt in args[1:]:
            result = f'(HConcat {result} {nxt} "{dim}")'
        return result

    if op == "iota":
        return f'(HIota "{attrs.get("iota_dimension","")}" "{instr["type"]}")'

    if op == "tuple":
        return args[0] if args else '(HConst "tuple_empty")'

    if op == "get-tuple-element":
        return args[0] if args else '(HConst "gte_empty")'

    if op == "reduce":
        sub_name = attrs.get("to_apply")
        sub = module.get(sub_name)
        kind = "HReduceSum"
        if sub is not None:
            root_op = None
            for i in sub["instrs"]:
                if i["is_root"]:
                    root_op = i["opcode"]
                    break
            kind = {
                "add": "HReduceSum",
                "multiply": "HReduceProd",
                "maximum": "HReduceMax",
                "minimum": "HReduceMin",
            }.get(root_op or "", "HReduceSum")
        dims = attrs.get("dimensions", "{}")
        in_arg = args[0] if args else '(HConst "reduce_in_missing")'
        init_arg = args[1] if len(args) > 1 else '(HConst "reduce_init_missing")'
        return f'({kind} {in_arg} {init_arg} "{dims}")'

    n = len(args)
    if n == 0:
        return f'(HConst "{op}_op")'
    if n == 1:
        return f'(HOp1 "{op}" {args[0]})'
    if n == 2:
        return f'(HOp2 "{op}" {args[0]} {args[1]})'
    if n == 3:
        return f'(HOp3 "{op}" {args[0]} {args[1]} {args[2]})'
    if n == 4:
        return f'(HOp4 "{op}" {args[0]} {args[1]} {args[2]} {args[3]})'
    if n == 5:
        return f'(HOp5 "{op}" {args[0]} {args[1]} {args[2]} {args[3]} {args[4]})'
    head = f'(HOp5 "{op}" {args[0]} {args[1]} {args[2]} {args[3]} {args[4]})'
    for nxt in args[5:]:
        head = f'(HOp2 "{op}_chain" {head} {nxt})'
    return head


def emit_egglog(
    module: dict,
    *,
    enable_rewrites: bool = False,
    run_iters: int = 3,
    neuter_ops: set | None = None,
) -> str:
    entry = next((c for c in module.values() if c["is_entry"]), None)
    if entry is None:
        raise ValueError("HLO module has no ENTRY computation")
    flat = _inline_calls(entry, module)

    out: list[str] = [PRELUDE]
    if enable_rewrites:
        out.append(SAFE_REWRITES)

    for _, pname in sorted(flat["params"], key=lambda p: p[0]):
        out.append(f'(let ${_sanitize(pname)} (HParam "{pname}"))')

    seen_let = set(p[1] for p in flat["params"])
    for instr in flat["instrs"]:
        nm = instr["name"]
        if nm in seen_let:
            continue
        body = _emit_instruction(instr, module, neuter_ops=neuter_ops)
        out.append(f"(let ${_sanitize(nm)} {body})")
        seen_let.add(nm)

    out.append(f"(run {run_iters})")
    return "\n".join(out) + "\n"


def convert(
    text: str,
    *,
    enable_rewrites: bool = False,
    run_iters: int = 3,
    neuter_ops: set | None = None,
) -> str:
    return emit_egglog(
        parse_hlo_module(text),
        enable_rewrites=enable_rewrites,
        run_iters=run_iters,
        neuter_ops=neuter_ops,
    )


def op_frequency(text: str) -> dict[str, int]:
    """Count opcode occurrences across every computation in the module."""
    module = parse_hlo_module(text)
    out: dict[str, int] = {}
    for comp in module.values():
        for instr in comp["instrs"]:
            out[instr["opcode"]] = out.get(instr["opcode"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
