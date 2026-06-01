# Copyright 2026 Bartłomiej Czech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DDC-3 static analyser: detect and rewrite tainted-argument sink calls.

The analyser operates on a single Python code block (no markdown fences).
It performs two passes:

  Pass 1 — TaintCollector:
    Walk the AST and build a taint set.  A name is tainted if it was assigned
    from a ``query_ai_assistant(...)`` call, or from an expression that
    transitively references a tainted name via attribute access (``x.field``),
    subscript (``x[k]``), or plain assignment (``y = x``).

  Pass 2 — SinkRewriter:
    Walk the AST again using the full taint set (initial + newly collected).
    For each statement that is a call to a state-changing sink function where
    at least one argument resolves to a tainted name, replace the entire
    statement with ``decline("DDC-3: ...")``.

Scope boundaries (per the plan):
  - No transitive taint across function calls (other than query_ai_assistant).
  - No taint through list/dict operations (only through Name/Attribute/Subscript).
  - Only statement-level rewrites (not sub-expression rewrites) to keep the
    rewritten code syntactically clean.

Usage
-----
  from camel.scope_guard.ddc3_analyser import analyse

  rewritten, new_taint = analyse(python_code, existing_taint)
  # rewritten: source with tainted sink calls replaced by decline(...)
  # new_taint: union of existing_taint and newly tainted names from this block
"""

from __future__ import annotations

import ast
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TAINT_SOURCE: str = "query_ai_assistant"

#: State-changing tools that must not receive tainted arguments.
SINK_FUNCTIONS: frozenset[str] = frozenset({
    # AgentDojo workspace
    "send_email",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
    "share_file",
    "delete_file",
    "create_file",
    "update_file",
    "move_file",
    "append_to_file",
    # AgentDojo travel
    "book_hotel",
    "book_flight",
    "book_car_rental",
    # AgentDojo banking
    "transfer_money",
    "pay_bill",
    # Generic
    "send_message",
    "post_tweet",
    "send_slack_message",
})


# ---------------------------------------------------------------------------
# Pass 1: Taint collection
# ---------------------------------------------------------------------------


class _TaintCollector(ast.NodeVisitor):
    """Collect variable names tainted by ``query_ai_assistant`` calls.

    Taint propagates through:
    - ``x = query_ai_assistant(...)``                    → x tainted
    - ``x = tainted_var``                                → x tainted
    - ``x = tainted_var.field``                          → x tainted
    - ``x = tainted_var[key]``                           → x tainted
    - ``x = a if tainted_cond else b``                   → x tainted (IfExp)
    - ``x = y = query_ai_assistant(...)``                → both tainted
    - Annotated assignment: ``x: T = tainted_expr``      → x tainted
    """

    def __init__(self, initial: set[str]) -> None:
        self.taint: set[str] = set(initial)

    def _expr_is_tainted(self, node: ast.expr) -> bool:
        """Return True if the expression transitively touches a tainted name."""
        if isinstance(node, ast.Name):
            return node.id in self.taint
        if isinstance(node, ast.Attribute):
            return self._expr_is_tainted(node.value)
        if isinstance(node, ast.Subscript):
            return self._expr_is_tainted(node.value)
        if isinstance(node, ast.IfExp):
            # x = a if cond else b — tainted if any branch or the condition is tainted
            return (
                self._expr_is_tainted(node.body)
                or self._expr_is_tainted(node.test)
                or self._expr_is_tainted(node.orelse)
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == TAINT_SOURCE:
                return True
            # Taint from a function call is not propagated further (per scope boundary).
            return False
        return False

    def _mark_target(self, target: ast.expr) -> None:
        """Mark a simple assignment target as tainted."""
        if isinstance(target, ast.Name):
            self.taint.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._mark_target(elt)
        # ast.Starred, ast.Attribute subscript targets: ignored (conservative)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._expr_is_tainted(node.value):
            for target in node.targets:
                self._mark_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and self._expr_is_tainted(node.value):
            self._mark_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # x += tainted_expr  →  x becomes tainted
        if self._expr_is_tainted(node.value):
            if isinstance(node.target, ast.Name):
                self.taint.add(node.target.id)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Pass 2: Sink rewriter
# ---------------------------------------------------------------------------


class _SinkRewriter(ast.NodeTransformer):
    """Replace statement-level tainted sink calls with ``decline(...)``."""

    def __init__(self, taint: set[str]) -> None:
        self.taint = taint
        self.rewrites: list[str] = []

    def _arg_is_tainted(self, node: ast.expr) -> bool:
        """Recursively check whether any Name in the expression is tainted."""
        if isinstance(node, ast.Name):
            return node.id in self.taint
        if isinstance(node, ast.Attribute):
            return self._arg_is_tainted(node.value)
        if isinstance(node, ast.Subscript):
            return self._arg_is_tainted(node.value) or self._arg_is_tainted(node.slice)
        if isinstance(node, ast.IfExp):
            return (
                self._arg_is_tainted(node.body)
                or self._arg_is_tainted(node.test)
                or self._arg_is_tainted(node.orelse)
            )
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return any(self._arg_is_tainted(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return any(
                (k is not None and self._arg_is_tainted(k)) or self._arg_is_tainted(v)
                for k, v in zip(node.keys, node.values)
            )
        if isinstance(node, ast.Call):
            return any(self._arg_is_tainted(a) for a in node.args) or any(
                self._arg_is_tainted(kw.value) for kw in node.keywords
            )
        return False

    def _tainted_sink_param(self, call: ast.Call) -> str | None:
        """Return a description of the first tainted param, or None if clean."""
        if not isinstance(call.func, ast.Name):
            return None
        if call.func.id not in SINK_FUNCTIONS:
            return None
        for i, arg in enumerate(call.args):
            if self._arg_is_tainted(arg):
                return f"arg[{i}]"
        for kw in call.keywords:
            if self._arg_is_tainted(kw.value):
                return kw.arg or "**kwargs"
        return None

    def _decline_node(self, sink_name: str, param: str) -> ast.Expr:
        reason = (
            f"DDC-3 detected: value from {TAINT_SOURCE}() flows into "
            f"{sink_name}({param}=...). Declining to prevent potential "
            "prompt-injection exfiltration."
        )
        node = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="decline", ctx=ast.Load()),
                args=[ast.Constant(value=reason)],
                keywords=[],
            )
        )
        return ast.fix_missing_locations(node)

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        if isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name):
                param = self._tainted_sink_param(call)
                if param is not None:
                    desc = f"{call.func.id}({param})"
                    logger.info("DDC-3 analyser: rewriting %s → decline()", desc)
                    self.rewrites.append(desc)
                    return self._decline_node(call.func.id, param)
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Also rewrite assignments whose RHS is a tainted sink call.

        Example: ``result = send_email(recipients=[tainted.email], ...)``
        """
        if isinstance(node.value, ast.Call):
            param = self._tainted_sink_param(node.value)
            if param is not None:
                call = node.value
                assert isinstance(call.func, ast.Name)
                desc = f"{call.func.id}({param})"
                logger.info("DDC-3 analyser: rewriting %s (assignment RHS) → decline()", desc)
                self.rewrites.append(desc)
                return self._decline_node(call.func.id, param)
        return self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyse(
    python_code: str,
    taint_set: set[str] | None = None,
) -> tuple[str, set[str]]:
    """Analyse *python_code* and rewrite tainted DDC-3 sink calls.

    Args:
        python_code: Raw Python source (no markdown fences).
        taint_set: Mutable set of tainted variable names accumulated from
            previous code blocks in the same task.  Updated in-place.

    Returns:
        ``(rewritten_code, updated_taint_set)`` where:
        - ``rewritten_code`` is *python_code* with each tainted sink call
          statement replaced by a ``decline(...)`` call.
        - ``updated_taint_set`` is the union of *taint_set* and any new names
          tainted in this block.

    The function never raises: parse errors or unparse errors fall back to
    returning the original code unchanged.
    """
    if taint_set is None:
        taint_set = set()

    try:
        tree = ast.parse(python_code)
    except SyntaxError:
        logger.debug("DDC-3 analyser: SyntaxError, passing through unchanged")
        return python_code, taint_set

    # Pass 1: collect taint
    collector = _TaintCollector(taint_set)
    collector.visit(tree)
    updated_taint = collector.taint  # superset of taint_set

    # Pass 2: rewrite sinks
    rewriter = _SinkRewriter(updated_taint)
    new_tree = rewriter.visit(tree)

    if not rewriter.rewrites:
        # No rewrites — still update the caller's taint_set via reference
        taint_set.update(updated_taint)
        return python_code, updated_taint

    taint_set.update(updated_taint)

    ast.fix_missing_locations(new_tree)
    try:
        rewritten = ast.unparse(new_tree)
    except Exception:
        logger.debug("DDC-3 analyser: ast.unparse failed, passing through unchanged")
        return python_code, updated_taint

    logger.info(
        "DDC-3 analyser: rewrote %d sink call(s): %s",
        len(rewriter.rewrites),
        ", ".join(rewriter.rewrites),
    )
    return rewritten, updated_taint
