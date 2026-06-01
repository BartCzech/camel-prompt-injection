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

"""Core scope-guard builtins: scoped_branch, decline, restrict_namespace."""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.models import KnownModelName, Model

from camel.capabilities import Capabilities
from camel.interpreter import namespace as ns
from camel.interpreter import value as v
from camel.interpreter.library import BUILT_IN_CLASSES, BUILT_IN_FUNCTIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Names that are always preserved when restricting a namespace.
# ---------------------------------------------------------------------------
_STDLIB_NAMES: frozenset[str] = frozenset(BUILT_IN_FUNCTIONS) | frozenset(BUILT_IN_CLASSES)

# Scope-guard builtins themselves must survive restriction so that nested
# scoped_branch calls work without the P-LLM having to declare them.
_SCOPE_GUARD_NAMES: frozenset[str] = frozenset({"scoped_branch", "decline"})

# query_ai_assistant is registered by the pipeline (not stdlib) but branch bodies
# must be able to call the Q-LLM for data extraction, so always keep it.
_ALWAYS_ALLOW: frozenset[str] = _STDLIB_NAMES | _SCOPE_GUARD_NAMES | frozenset({"query_ai_assistant"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ScopedBranchClassifierError(Exception):
    """Raised when the Q-LLM classifier returns a label not in the declared set.

    This may indicate a prompt-injection attempt targeting the classifier step.
    """

    def __init__(self, returned: str, declared: list[str]) -> None:
        self.returned = returned
        self.declared = declared
        super().__init__(
            f"Classifier returned {returned!r} which is not in the declared label set "
            f"{declared}.  This may indicate a prompt injection attempt against the "
            "classifier."
        )


class ScopedBranchDeclineError(Exception):
    """Raised when the P-LLM explicitly declines via the decline() builtin.

    Callers (the runner) catch this to record a structured refusal in the trace.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Agent declined: {reason}")


# ---------------------------------------------------------------------------
# CaMeLBuiltin subclasses with custom wrap_output
# ---------------------------------------------------------------------------


class _ScopedBranchBuiltin(v.CaMeLBuiltin):
    """CaMeLBuiltin that returns a fully-trusted label string (taint-laundering by design)
    and enforces the declared per-branch tool scope at runtime.

    Security argument (taint laundering)
    -----------------
    The returned label is an element of a finite set declared by the P-LLM *before*
    any untrusted data is read.  The Q-LLM classifier only *selects* among those
    pre-approved options; it cannot expand, modify, or escape the declared set.
    The set of possible outputs is therefore known at plan-construction time and
    carries no information from the untrusted data beyond "which of N pre-approved
    options was selected."  Treating this as CaMeL-trusted (Capabilities.camel())
    is therefore sound — it is semantically equivalent to the P-LLM having written
    a hard-coded conditional whose branches are pre-approved.

    Importantly, the raw untrusted input is NOT included as a CaMeL dependency of the
    returned value.  It is hashed and written to the Python logger for audit purposes
    but does not participate in CaMeL's taint-propagation graph.

    Hard namespace enforcement
    --------------------------
    When ``allowed_tools`` is provided in the ``scoped_branch`` call (which is
    recorded in ``_scope_state`` by the underlying callable), ``wrap_output``
    restricts the live namespace to the tools declared for the chosen label.
    Any tool not in ``allowed_tools[chosen_label]`` is removed from the namespace
    dict immediately after the label is returned.  The subsequent ``if``/``elif``
    chain therefore runs in a namespace where undeclared tools are absent; any
    reference to them raises a ``NameError`` from the interpreter.

    The restriction is permanent for the remainder of the code block.  This is
    intentional: the scope-guard design principle requires that all tool calls
    happen inside branch bodies.  Code written after the ``if``/``elif`` chain is
    expected to be tool-free (logging, variable binding, etc.).

    The ``_scope_state`` dict is shared with the underlying callable via the
    factory function ``_make_scoped_branch_fn``; it is populated just before the
    callable returns its result and is read here in ``wrap_output``.
    """

    def __init__(
        self,
        name: str,
        callable: Any,
        metadata: Capabilities,
        dependencies: tuple,
        scope_state: dict,
        is_class_method: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            callable=callable,
            metadata=metadata,
            dependencies=dependencies,
            is_class_method=is_class_method,
        )
        self._scope_state = scope_state

    def wrap_output(
        self,
        value_out: Any,
        args: v.CaMeLTuple,
        kwargs: v.CaMeLDict,  # type: ignore[type-arg]
        namespace: ns.Namespace,
    ) -> v.CaMeLStr:
        # No CaMeL dependencies on classifier_input — by design.
        # The security argument is documented on this class.
        label = str(value_out)

        allowed_tools_map: dict[str, list[str]] | None = self._scope_state.get("allowed_tools")
        if allowed_tools_map is not None and label in allowed_tools_map:
            permitted: set[str] = set(allowed_tools_map[label])
            restricted_ns = restrict_namespace(namespace, permitted)
            # Enforce restriction in-place: remove any name from the live namespace
            # that is absent from the restricted namespace.  Because Namespace.variables
            # is a mutable dict (despite Namespace being a frozen dataclass), this
            # mutation is immediately visible to subsequent interpreter lookups.
            disallowed = [
                name for name in list(namespace.variables.keys())
                if name not in restricted_ns.variables
            ]
            for name in disallowed:
                del namespace.variables[name]
            if disallowed:
                logger.debug(
                    "scoped_branch: hard namespace enforcement — label=%r, "
                    "removed %d disallowed tool(s): %s",
                    label,
                    len(disallowed),
                    sorted(disallowed),
                )

        return v.CaMeLStr.from_raw(label, Capabilities.camel(), ())


class _DeclineBuiltin(v.CaMeLBuiltin):
    """CaMeLBuiltin whose callable raises ScopedBranchDeclineError.

    The runner catches this exception and records it as a structured refusal.
    wrap_output is defined for completeness but is never reached in normal use
    because the underlying function always raises.
    """

    def wrap_output(
        self,
        value_out: Any,
        args: v.CaMeLTuple,
        kwargs: v.CaMeLDict,  # type: ignore[type-arg]
        namespace: ns.Namespace,
    ) -> v.CaMeLNone:
        return v.CaMeLNone(Capabilities.camel(), ())


# ---------------------------------------------------------------------------
# Namespace restriction
# ---------------------------------------------------------------------------


def restrict_namespace(parent_ns: ns.Namespace, allowed_tools: set[str]) -> ns.Namespace:
    """Return a Namespace that keeps only stdlib builtins, scope-guard builtins,
    *query_ai_assistant*, and the tools named in *allowed_tools*.

    Any AgentDojo tool whose name is NOT in ``allowed_tools ∪ _ALWAYS_ALLOW`` will be
    absent from the returned namespace; attempting to call it raises a ``NameError``
    from the interpreter — no runtime policy check needed.

    This implements per-branch tool scope enforcement as declared in a
    ``ScopedBranchSpec``.  It is a free function (not a method on ``Namespace``) to
    keep the diff localised and avoid modifying the existing CaMeL code base.

    Args:
        parent_ns: Full runtime namespace (typically already contains AgentDojo tools).
        allowed_tools: Names of the tools permitted inside this branch body.

    Returns:
        A new ``Namespace`` restricted to the allowed names.
    """
    restricted: dict[str, v.CaMeLValue] = {
        name: val
        for name, val in parent_ns.variables.items()
        if name in _ALWAYS_ALLOW or name in allowed_tools
    }
    return ns.Namespace(variables=restricted)


# ---------------------------------------------------------------------------
# Factory functions for the raw Python callables
# ---------------------------------------------------------------------------


def _make_scoped_branch_fn(q_llm_model: KnownModelName | Model, scope_state: dict) -> Any:
    """Return the raw Python function registered as the ``scoped_branch`` builtin.

    ``scope_state`` is a shared mutable dict populated here (just before returning
    the chosen label) and consumed in ``_ScopedBranchBuiltin.wrap_output`` to
    enforce hard namespace restrictions.
    """

    def scoped_branch(
        classifier_input: Any,
        classifier_query: str,
        labels: list[str],
        allowed_tools: dict[str, list[str]] | None = None,
    ) -> str:
        """Classify *classifier_input* into one of the pre-declared *labels*.

        Returns the chosen label as a plain Python string; the CaMeLBuiltin wrapper
        (_ScopedBranchBuiltin) then promotes it to a fully-trusted CaMeLStr and
        enforces the declared per-branch tool scope on the live namespace.

        Args:
            classifier_input: The untrusted data item to classify.  Converted to
                ``str`` before being sent to the Q-LLM.
            classifier_query: Natural-language description of the classification task.
            labels: The finite, pre-declared set of branch labels (at least two).
            allowed_tools: Optional per-label tool-scope declaration.  When provided,
                tools not declared for the chosen label are removed from the interpreter
                namespace immediately after this call returns.

        Returns:
            The chosen label — always a member of *labels*.

        Raises:
            ScopedBranchClassifierError: If the Q-LLM returns a label outside *labels*.
            ValueError: If fewer than two labels are declared.
        """
        from camel.scope_guard.classifier import classify, input_hash  # noqa: PLC0415

        if len(labels) < 2:
            raise ValueError("scoped_branch requires at least 2 labels")

        input_str = str(classifier_input)
        logger.debug(
            "scoped_branch: classifying input_hash=%s labels=%s",
            input_hash(input_str),
            labels,
        )

        chosen = classify(q_llm_model, input_str, classifier_query, labels)

        # Populate scope_state BEFORE returning so wrap_output can read it.
        scope_state["allowed_tools"] = allowed_tools

        logger.debug(
            "scoped_branch: result input_hash=%s → %r declared_scope=%s",
            input_hash(input_str),
            chosen,
            (allowed_tools or {}).get(chosen),
        )
        return chosen

    return scoped_branch


def _make_decline_fn() -> Any:
    """Return the raw Python function registered as the ``decline`` builtin."""

    def decline(reason: str = "") -> None:
        """Explicitly decline to process the current task.

        Call this when the task exhibits DDC-3 or DDC-4 patterns that
        scope-guarded branching cannot safely handle.  The runner catches the
        resulting ``ScopedBranchDeclineError`` and records it as a structured
        refusal in the trace.

        Args:
            reason: Human-readable explanation for the decline.

        Raises:
            ScopedBranchDeclineError: Always.
        """
        raise ScopedBranchDeclineError(reason or "Agent declined to process this task")

    return decline


# ---------------------------------------------------------------------------
# Namespace registration
# ---------------------------------------------------------------------------


def register_scope_guard_builtins(
    namespace: ns.Namespace,
    q_llm_model: KnownModelName | Model,
) -> ns.Namespace:
    """Register ``scoped_branch`` and ``decline`` into *namespace*.

    A shared ``scope_state`` dict is created here and passed to both the underlying
    ``scoped_branch`` callable (which writes to it just before returning the label)
    and to the ``_ScopedBranchBuiltin`` wrapper (which reads it in ``wrap_output``
    to enforce hard namespace restrictions on the live interpreter namespace).

    Args:
        namespace: Existing namespace (usually already contains AgentDojo tools).
        q_llm_model: Q-LLM model name forwarded to the classifier.

    Returns:
        A new Namespace with ``scoped_branch`` and ``decline`` added.
    """
    scope_state: dict = {}
    scoped_branch_builtin = _ScopedBranchBuiltin(
        name="scoped_branch",
        callable=_make_scoped_branch_fn(q_llm_model, scope_state),
        metadata=Capabilities.camel(),
        dependencies=(),
        scope_state=scope_state,
    )
    decline_builtin = _DeclineBuiltin(
        name="decline",
        callable=_make_decline_fn(),
        metadata=Capabilities.camel(),
        dependencies=(),
    )
    return namespace.add_variables(
        {
            "scoped_branch": scoped_branch_builtin,
            "decline": decline_builtin,
        }
    )
