"""
Rule Evaluator - Python implementation of GoAmzn-AWSAppConfigRuleEvaluator.

Maps the Go rule engine to Python:

  rules.(*expressionBuilder).buildSexp        -> _parse()
  rules.(*contextExpression).eval             -> _eval: ('$', name) -> context lookup
  rules.(*staticExpression[T]).eval           -> _eval: str/int/float/bool literal
  rules.(*comparisonExpression).eval          -> _eval: eq / gt / lt / gte / lte
  rules.(*variadicLogicExpression).eval       -> _eval: and / or
  rules.(*notExpression).eval                 -> _eval: not
  rules.(*inExpression).eval                  -> _eval: in
  rules.(*existsExpression).eval              -> _eval: exists
  rules.(*stringMatchExpression).eval         -> _eval: begins_with / ends_with / contains
  rules.(*splitExpression).eval               -> _eval: split
  rules.(*regexpExpression).eval              -> _eval: matches
  rules.(*Renderer).RenderJson                -> evaluate_config()
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any

LOG = logging.getLogger(__name__)

# Keys that are AppConfig metadata, not user-defined flag attributes.
# Mirrors the _FLAG_META_KEYS logic in the original feature_flags.py.
_META_KEYS = frozenset[str](
    {
        "_variant",
        "_createdAt",
        "_updatedAt",
        "enabled",
        "name",
        "description",
        "_variants",
        "attributes",
        "attributeValues",
    }
)


# ---------------------------------------------------------------------------
# Tokenizer
# Maps to: rules.(*expressionBuilder).buildSexp (lexing phase)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"\(|\)"  # parens
    r'|"(?:[^"\\]|\\.)*"'  # quoted strings
    r"|\'(?:[^\'\\]|\\.)*\'"  # single-quoted strings
    r"|-?\d+\.\d+"  # floats
    r"|-?\d+"  # ints
    r"|true|false|null"  # literals
    r"|\$[\w.]+"  # context variables: $foo, $foo.bar
    r"|[\w_][\w_.:-]*"  # operators and identifiers
)


def _tokenize(expr: str) -> list[str]:
    return _TOKEN_RE.findall(expr)


# ---------------------------------------------------------------------------
# Parser
# Maps to: rules.(*expressionBuilder).buildSexp (parse phase)
# Output is a nested structure:
#   list         -> S-expression: [operator, arg1, arg2, ...]
#   ('$', name)  -> context variable reference
#   str/int/float/bool/None -> static literal
# ---------------------------------------------------------------------------
def _parse_tokens(tokens: list[str], pos: int) -> tuple[Any, int]:
    if pos >= len(tokens):
        raise ValueError("unexpected end of expression")

    token = tokens[pos]

    if token == "(":
        pos += 1
        items: list[Any] = []
        while pos < len(tokens) and tokens[pos] != ")":
            item, pos = _parse_tokens(tokens, pos)
            items.append(item)
        if pos >= len(tokens):
            raise ValueError("missing closing ')'")
        pos += 1  # consume ')'
        return items, pos

    if token == ")":
        raise ValueError("unexpected ')'")

    # Quoted string — strip delimiters and unescape.
    if token.startswith('"') or token.startswith("'"):
        return token[1:-1].replace('\\"', '"').replace("\\'", "'"), pos + 1

    if token == "true":
        return True, pos + 1
    if token == "false":
        return False, pos + 1
    if token == "null":
        return None, pos + 1

    # Context variable: $foo
    if token.startswith("$"):
        return ("$", token[1:]), pos + 1

    # Numeric literals
    with contextlib.suppress(ValueError):
        return (float(token), pos + 1) if "." in token else (int(token), pos + 1)
    # Operator or bare identifier
    return token, pos + 1


def _parse(expr: str) -> Any:
    """Parse an S-expression rule string into a nested Python structure."""
    tokens = _tokenize(expr.strip())
    if not tokens:
        raise ValueError(f"empty expression: {expr!r}")
    node, consumed = _parse_tokens(tokens, 0)
    if consumed != len(tokens):
        raise ValueError(f"trailing tokens in expression: {tokens[consumed:]}")
    return node


# ---------------------------------------------------------------------------
# Evaluator
# Maps each Go expression type to a branch in _eval().
# ---------------------------------------------------------------------------
def _fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit hash.

    Maps to hash/fnv.(*sum32a) confirmed in the Go binary.
    Used by splitExpression for deterministic percentage bucketing.
    """
    h = 0x811C9DC5  # FNV offset basis
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime
    return h


def _coerce(value: Any, target: Any) -> Any:
    """
    Coerce a context string value to match the type of the rule literal.

    Maps to rules.safeFloat / rules.safeString / rules.safeBool in the Go binary.

    Context values are always strings (passed as key=value pairs). Rules use
    typed literals (int, float, bool). Before comparing, the Go binary coerces
    the context value to the same type as the literal. We do the same here.
    """
    if not isinstance(value, str):
        return value
    if isinstance(target, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(target, int):
        with contextlib.suppress(ValueError, TypeError):
            return int(value)
    if isinstance(target, float):
        with contextlib.suppress(ValueError, TypeError):
            return float(value)
    return value


def _resolve(node: Any, context: dict[str, Any], coerce_to: Any = None) -> Any:
    """Resolve a context variable or return a static value.

    If coerce_to is provided, coerce a string context value to that type.
    Maps to rules.(*contextExpression).eval with safeFloat/safeString/safeBool.
    """
    if isinstance(node, tuple) and node[0] == "$":
        val = context.get(node[1])
        return _coerce(val, coerce_to) if coerce_to is not None else val
    return node


def _eval(node: Any, context: dict[str, Any]) -> Any:
    """
    Recursively evaluate a parsed S-expression node against a context dict.

    Maps to the .eval() methods on each expression type in the Go package.
    """
    # Static literal (str, int, float, bool, None)
    if not isinstance(node, (list, tuple)):
        return node

    # Context variable reference: ('$', 'varName')
    if isinstance(node, tuple) and node[0] == "$":
        return context.get(node[1])

    # S-expression: [operator, arg1, arg2, ...]
    if not isinstance(node, list) or not node:
        return None

    op = node[0]
    args = node[1:]

    # -- rules.(*variadicLogicExpression).eval --
    if op == "and":
        # earlyReturnAnd: short-circuits on first False
        return all(_eval(a, context) for a in args)

    if op == "or":
        # earlyReturnedOr: short-circuits on first True
        return any(_eval(a, context) for a in args)

    # -- rules.(*notExpression).eval --
    if op == "not":
        return not _eval(args[0], context)

    # -- rules.(*comparisonExpression).eval --
    # Coerce context string to the type of the rule literal before comparing.
    # Mirrors rules.safeFloat / rules.safeString / rules.safeBool in Go.
    if op == "eq":
        right = _resolve(args[1], context)
        left = _resolve(args[0], context, coerce_to=right)
        return left == right

    if op == "gt":
        right = _resolve(args[1], context)
        left = _resolve(args[0], context, coerce_to=right)
        return left is not None and right is not None and left > right

    if op == "lt":
        right = _resolve(args[1], context)
        left = _resolve(args[0], context, coerce_to=right)
        return left is not None and right is not None and left < right

    if op == "gte":
        right = _resolve(args[1], context)
        left = _resolve(args[0], context, coerce_to=right)
        return left is not None and right is not None and left >= right

    if op == "lte":
        right = _resolve(args[1], context)
        left = _resolve(args[0], context, coerce_to=right)
        return left is not None and right is not None and left <= right

    # -- rules.(*inExpression).eval --
    if op == "in":
        val = _resolve(args[0], context)
        collection = [_resolve(a, context) for a in args[1:]]
        return val in collection

    # -- rules.(*existsExpression).eval --
    if op == "exists":
        inner = args[0]
        if isinstance(inner, tuple) and inner[0] == "$":
            return inner[1] in context
        return inner is not None

    # -- rules.(*stringMatchExpression).eval --
    if op == "begins_with":
        val = _resolve(args[0], context)
        prefix = _resolve(args[1], context)
        return (
            isinstance(val, str) and isinstance(prefix, str) and val.startswith(prefix)
        )

    if op == "ends_with":
        val = _resolve(args[0], context)
        suffix = _resolve(args[1], context)
        return isinstance(val, str) and isinstance(suffix, str) and val.endswith(suffix)

    if op == "contains":
        val = _resolve(args[0], context)
        substr = _resolve(args[1], context)
        return isinstance(val, str) and isinstance(substr, str) and substr in val

    # -- rules.(*splitExpression).eval --
    # Percentage split: (split by:: $var pct::N seed:: "str")
    #
    # Hashing confirmed from binary (hash/fnv.(*sum32a)):
    #   fnv1a_32(val + seed) % 10000 < pct * 100
    #
    # - FNV-1a 32-bit (Go's hash/fnv sum32a)
    # - Input order: value first, seed second
    # - Modulus 10000 for 0.01% granularity
    # - pct::20 means 20% → threshold 2000
    if op == "split":
        by_val = None
        pct = None
        seed = ""

        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, str) and a == "by::":
                if i + 1 < len(args):
                    by_val = _resolve(args[i + 1], context)
                    i += 2
                    continue
            elif isinstance(a, str) and a.startswith("pct::"):
                pct = int(a[5:])
            elif isinstance(a, str) and a == "seed::":
                if i + 1 < len(args):
                    seed = _resolve(args[i + 1], context) or ""
                    i += 2
                    continue
            elif isinstance(a, str) and a.startswith("seed::"):
                seed = a[6:]
            i += 1

        if by_val is None or pct is None:
            return False

        bucket = _fnv1a_32(f"{by_val}{seed}".encode()) % 10000
        return bucket < pct * 100

    # -- rules.(*regexpExpression).eval --
    if op in ("matches", "match"):
        val = _resolve(args[0], context)
        pattern = _resolve(args[1], context)
        if isinstance(val, str) and isinstance(pattern, str):
            return bool(re.search(pattern, val))
        return False

    LOG.warning("unknown rule operator: %r", op)
    return False


def evaluate_rule(rule: str, context: dict[str, Any]) -> bool:
    """
    Parse and evaluate a single S-expression rule string.

    Args:
        rule:    S-expression string, e.g. '(and (eq $tier "vip") (begins_with $env "prod"))'
        context: Flat key-value dict resolved from the request context.

    Returns:
        True if the rule matches the context, False otherwise.
    """
    try:
        node = _parse(rule)
        result = _eval(node, context)
        return bool(result)
    except Exception as exc:
        LOG.warning("rule evaluation failed for %r: %s", rule, exc)
        return False


# ---------------------------------------------------------------------------
# Renderer
# Maps to: rules.(*Renderer).RenderJson
# ---------------------------------------------------------------------------
def evaluate_config(
    config: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate a raw AppConfig feature flag config against a context.

    Mirrors rules.(*Renderer).RenderJson:
      - Iterates flags in the config.
      - For each flag, evaluates each variant's rule against the context.
      - Sets _variant to the first matching variant name.
      - Promotes variant attributes to the root level of the flag (same as
        the Go agent's response format consumed by the original feature_flags.py).
      - Skips disabled flags.

    Args:
        config:  Raw AppConfig feature flag JSON, parsed to a dict.
        context: Flat key-value dict, equivalent to the Context header sent
                 to localhost:2772 in the original implementation.

    Returns:
        Evaluated config dict with _variant and attributes resolved per flag.
    """
    result: dict[str, Any] = {}

    for flag_name, flag_value in config.items():
        if not isinstance(flag_value, dict):
            result[flag_name] = flag_value
            continue

        if not flag_value.get("enabled", True):
            continue

        evaluated = dict(flag_value)
        variants: list[dict[str, Any]] = flag_value.get("_variants", [])

        for variant in variants:
            if not variant.get("enabled", True):
                continue
            rule = variant.get("rule")

            if rule is None or evaluate_rule(rule, context):
                evaluated["_variant"] = variant.get("name")
                # Promote variant attributes to root level, same as Go agent.
                # AppConfig uses "attributeValues"; some configs use "attributes".
                attrs = variant.get("attributeValues", variant.get("attributes", {}))
                for k, v in attrs.items():
                    evaluated[k] = v
                break

        result[flag_name] = evaluated

    return result


def extract_attributes(flag_value: dict[str, Any]) -> dict[str, Any]:
    """Extract user-defined attributes from an evaluated flag, stripping metadata.

    Args:
        flag_value: Evaluated flag value from AppConfig.

    Returns:
        Dictionary of user-defined attributes.
    """
    return {k: v for k, v in flag_value.items() if k not in _META_KEYS}
