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

"""Tests for restrict_namespace — per-branch tool scope enforcement."""

from __future__ import annotations

import pytest

from camel.capabilities import Capabilities
from camel.interpreter import namespace as ns
from camel.interpreter import value as v
from camel.interpreter.library import BUILT_IN_CLASSES, BUILT_IN_FUNCTIONS
from camel.scope_guard.builtin import (
    _ALWAYS_ALLOW,
    _SCOPE_GUARD_NAMES,
    _STDLIB_NAMES,
    register_scope_guard_builtins,
    restrict_namespace,
)

_STUB_MODEL = "anthropic:claude-sonnet-4"

_TOOL_NAMES = ["search_emails", "send_email", "create_calendar_event", "share_file", "read_file"]


def _make_tool_builtin(name: str) -> v.CaMeLBuiltin:
    return v.CaMeLBuiltin(name, lambda: None, Capabilities.camel(), (), False)


def _full_namespace(*extra_tool_names: str) -> ns.Namespace:
    """Stdlib builtins + scope_guard builtins + a set of fake tool functions."""
    base = ns.Namespace.with_builtins()
    extended = register_scope_guard_builtins(base, _STUB_MODEL)
    tools = {name: _make_tool_builtin(name) for name in extra_tool_names}
    return extended.add_variables(tools)


# ---------------------------------------------------------------------------
# Stdlib names must always survive restriction
# ---------------------------------------------------------------------------


class TestStdlibAlwaysPreserved:
    @pytest.mark.parametrize("name", list(BUILT_IN_FUNCTIONS.keys()))
    def test_stdlib_function_survives(self, name: str):
        full_ns = _full_namespace(*_TOOL_NAMES)
        restricted = restrict_namespace(full_ns, set())
        assert name in restricted.variables, f"stdlib function {name!r} was removed"

    @pytest.mark.parametrize("name", list(BUILT_IN_CLASSES.keys()))
    def test_stdlib_class_survives(self, name: str):
        full_ns = _full_namespace(*_TOOL_NAMES)
        restricted = restrict_namespace(full_ns, set())
        assert name in restricted.variables, f"stdlib class {name!r} was removed"


# ---------------------------------------------------------------------------
# Scope-guard builtins must survive restriction (enables nested branching)
# ---------------------------------------------------------------------------


class TestScopeGuardBuiltinsSurvive:
    def test_scoped_branch_survives(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        restricted = restrict_namespace(full_ns, set())
        assert "scoped_branch" in restricted.variables

    def test_decline_survives(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        restricted = restrict_namespace(full_ns, set())
        assert "decline" in restricted.variables


# ---------------------------------------------------------------------------
# Tools outside allowed_tools must be absent
# ---------------------------------------------------------------------------


class TestToolsOutsideAllowedAreRemoved:
    def test_non_allowed_tool_is_absent(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        restricted = restrict_namespace(full_ns, {"search_emails"})
        assert "send_email" not in restricted.variables
        assert "create_calendar_event" not in restricted.variables
        assert "share_file" not in restricted.variables
        assert "read_file" not in restricted.variables

    def test_allowed_tool_is_present(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        restricted = restrict_namespace(full_ns, {"search_emails", "send_email"})
        assert "search_emails" in restricted.variables
        assert "send_email" in restricted.variables

    def test_empty_allowed_set_removes_all_tools(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        restricted = restrict_namespace(full_ns, set())
        for tool in _TOOL_NAMES:
            assert tool not in restricted.variables, f"tool {tool!r} should be absent"

    def test_full_tool_set_preserves_all_tools(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        restricted = restrict_namespace(full_ns, set(_TOOL_NAMES))
        for tool in _TOOL_NAMES:
            assert tool in restricted.variables

    def test_allowed_set_is_intersected_not_added(self):
        """Naming a tool in allowed_tools that is NOT in the namespace must not add it."""
        full_ns = _full_namespace()  # no extra tools
        restricted = restrict_namespace(full_ns, {"tool_that_doesnt_exist"})
        assert "tool_that_doesnt_exist" not in restricted.variables


# ---------------------------------------------------------------------------
# query_ai_assistant must always survive (branch bodies may call Q-LLM)
# ---------------------------------------------------------------------------


class TestQueryAiAssistantSurvives:
    def test_query_ai_assistant_survives_when_present(self):
        full_ns = _full_namespace()
        qa = _make_tool_builtin("query_ai_assistant")
        full_ns_with_qa = full_ns.add_variables({"query_ai_assistant": qa})
        restricted = restrict_namespace(full_ns_with_qa, set())
        assert "query_ai_assistant" in restricted.variables


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestRestrictNamespaceReturnType:
    def test_returns_namespace_instance(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        result = restrict_namespace(full_ns, {"search_emails"})
        assert isinstance(result, ns.Namespace)

    def test_does_not_mutate_parent(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        before = set(full_ns.variables.keys())
        restrict_namespace(full_ns, {"search_emails"})
        after = set(full_ns.variables.keys())
        assert before == after

    def test_restricted_namespace_is_independent(self):
        full_ns = _full_namespace(*_TOOL_NAMES)
        r1 = restrict_namespace(full_ns, {"search_emails"})
        r2 = restrict_namespace(full_ns, {"send_email"})
        # Modifying one restricted namespace does not affect the other
        assert "search_emails" in r1.variables
        assert "search_emails" not in r2.variables
        assert "send_email" not in r1.variables
        assert "send_email" in r2.variables


# ---------------------------------------------------------------------------
# Nested scoped_branch — the restriction must not lock out the builtin
# ---------------------------------------------------------------------------


class TestNestedBranchingEnabled:
    def test_restricted_namespace_can_be_further_restricted(self):
        """A restricted namespace can be restricted again (supports nested branching)."""
        full_ns = _full_namespace("tool_a", "tool_b", "tool_c")
        level1 = restrict_namespace(full_ns, {"tool_a", "tool_b"})
        level2 = restrict_namespace(level1, {"tool_a"})

        assert "tool_a" in level2.variables
        assert "tool_b" not in level2.variables
        assert "tool_c" not in level2.variables
        # scope-guard builtins must survive both restrictions
        assert "scoped_branch" in level2.variables
        assert "decline" in level2.variables
