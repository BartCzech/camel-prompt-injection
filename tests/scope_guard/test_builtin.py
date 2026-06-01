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

"""Tests for the scoped_branch builtin and the decline builtin."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from camel.capabilities import Capabilities, get_all_sources
from camel.capabilities.sources import SourceEnum
from camel.interpreter import namespace as ns
from camel.interpreter import value as v
from camel.scope_guard.branch_spec import BranchDefinition, ScopedBranchSpec
from camel.scope_guard.builtin import (
    ScopedBranchClassifierError,
    ScopedBranchDeclineError,
    _make_decline_fn,
    _make_scoped_branch_fn,
    _ScopedBranchBuiltin,
    register_scope_guard_builtins,
)


LABELS_3 = ["meeting_invite", "document_request", "other"]
LABELS_2 = ["yes", "no"]

_STUB_MODEL = "anthropic:claude-sonnet-4"


def _make_namespace_with_tools(*tool_names: str) -> ns.Namespace:
    """Build a minimal namespace that contains stdlib builtins plus fake tools."""
    base = ns.Namespace.with_builtins()
    extra: dict[str, v.CaMeLValue] = {
        name: v.CaMeLBuiltin(name, lambda: None, Capabilities.camel(), (), False)
        for name in tool_names
    }
    return base.add_variables(extra)



class TestScopedBranchBuiltinWrapOutput:
    """The returned CaMeLStr must be trusted and have no tainted dependencies."""

    def _make_builtin(self) -> _ScopedBranchBuiltin:
        return _ScopedBranchBuiltin(
            name="scoped_branch",
            callable=lambda *a, **kw: "meeting_invite",
            metadata=Capabilities.camel(),
            dependencies=(),
        )

    def _make_untrusted_str(self) -> v.CaMeLStr:
        """A CaMeLStr carrying an untrusted Tool source."""
        from camel.capabilities.sources import Tool

        return v.CaMeLStr.from_raw(
            "some untrusted email body",
            Capabilities(frozenset({Tool("read_email")}), frozenset()),
            (),
        )

    def test_output_metadata_is_camel_trusted(self):
        """wrap_output must assign Capabilities.camel() regardless of args taint."""
        builtin = self._make_builtin()
        untrusted = self._make_untrusted_str()
        args = v.CaMeLTuple((untrusted,), Capabilities.camel(), ())
        kwargs = v.CaMeLDict({}, Capabilities.camel(), ())

        result = builtin.wrap_output("meeting_invite", args, kwargs, ns.Namespace.with_builtins())

        assert result._metadata == Capabilities.camel()

    def test_output_has_no_dependencies(self):
        """wrap_output must NOT propagate dependencies from the classifier input."""
        builtin = self._make_builtin()
        untrusted = self._make_untrusted_str()
        args = v.CaMeLTuple((untrusted,), Capabilities.camel(), ())
        kwargs = v.CaMeLDict({}, Capabilities.camel(), ())

        result = builtin.wrap_output("meeting_invite", args, kwargs, ns.Namespace.with_builtins())

        assert result._dependencies == ()

    def test_output_sources_are_only_camel(self):
        """get_all_sources on the output must contain only CaMeL (no Tool sources)."""
        builtin = self._make_builtin()
        untrusted = self._make_untrusted_str()
        args = v.CaMeLTuple((untrusted,), Capabilities.camel(), ())
        kwargs = v.CaMeLDict({}, Capabilities.camel(), ())

        result = builtin.wrap_output("meeting_invite", args, kwargs, ns.Namespace.with_builtins())

        all_sources, _ = get_all_sources(result)
        assert all_sources == frozenset({SourceEnum.CaMeL})

    def test_output_value_matches_label(self):
        result_str = _ScopedBranchBuiltin(
            name="scoped_branch",
            callable=lambda *a, **kw: "document_request",
            metadata=Capabilities.camel(),
            dependencies=(),
        ).wrap_output(
            "document_request",
            v.CaMeLTuple((), Capabilities.camel(), ()),
            v.CaMeLDict({}, Capabilities.camel(), ()),
            ns.Namespace.with_builtins(),
        )
        assert result_str.raw == "document_request"

    def test_nested_scoped_branch_output_is_also_trusted(self):
        """The output of a scoped_branch call used as input to another must still be trusted."""
        builtin = self._make_builtin()
        # Simulate the output of a first scoped_branch call (already trusted)
        trusted_inner = v.CaMeLStr.from_raw("meeting_invite", Capabilities.camel(), ())
        args = v.CaMeLTuple((trusted_inner,), Capabilities.camel(), ())
        kwargs = v.CaMeLDict({}, Capabilities.camel(), ())

        result = builtin.wrap_output("sub_action_a", args, kwargs, ns.Namespace.with_builtins())

        assert result._metadata == Capabilities.camel()
        assert result._dependencies == ()



class TestScopedBranchFn:
    """Tests for the raw Python function (before CaMeLBuiltin wrapping)."""

    def test_returns_valid_label_from_classifier(self):
        fn = _make_scoped_branch_fn(_STUB_MODEL)
        with patch("camel.scope_guard.classifier.classify", return_value="meeting_invite"):
            result = fn("email body", "Classify this email", LABELS_3)
        assert result == "meeting_invite"

    def test_returns_second_label_when_classifier_picks_it(self):
        fn = _make_scoped_branch_fn(_STUB_MODEL)
        with patch("camel.scope_guard.classifier.classify", return_value="document_request"):
            result = fn("email body", "Classify this email", LABELS_3)
        assert result == "document_request"

    def test_raises_on_fewer_than_two_labels(self):
        fn = _make_scoped_branch_fn(_STUB_MODEL)
        with pytest.raises(ValueError, match="at least 2 labels"):
            fn("email body", "Classify", ["only_one"])

    def test_raises_classifier_error_if_classify_raises(self):
        fn = _make_scoped_branch_fn(_STUB_MODEL)
        with patch(
            "camel.scope_guard.classifier.classify",
            side_effect=ScopedBranchClassifierError("evil", LABELS_3),
        ):
            with pytest.raises(ScopedBranchClassifierError):
                fn("email body", "Classify", LABELS_3)

    def test_allowed_tools_kwarg_is_accepted_and_ignored_at_runtime(self):
        """allowed_tools is optional metadata; passing it must not raise."""
        fn = _make_scoped_branch_fn(_STUB_MODEL)
        with patch("camel.scope_guard.classifier.classify", return_value="yes"):
            result = fn(
                "input",
                "yes or no?",
                LABELS_2,
                allowed_tools={"yes": ["tool_a"], "no": []},
            )
        assert result == "yes"

    def test_classifier_input_is_stringified(self):
        """Non-string classifier_input should be str()-converted before classify."""
        fn = _make_scoped_branch_fn(_STUB_MODEL)
        received_inputs: list[str] = []

        def _capture(model, inp, query, labels):
            received_inputs.append(inp)
            return labels[0]

        with patch("camel.scope_guard.classifier.classify", side_effect=_capture):
            fn(42, "Classify this number", LABELS_2)

        assert received_inputs == ["42"]



class TestDeclineFn:
    def test_decline_always_raises(self):
        fn = _make_decline_fn()
        with pytest.raises(ScopedBranchDeclineError):
            fn("task requires open-ended instructions")

    def test_decline_no_reason_still_raises(self):
        fn = _make_decline_fn()
        with pytest.raises(ScopedBranchDeclineError):
            fn()

    def test_decline_reason_preserved_in_exception(self):
        fn = _make_decline_fn()
        reason = "DDC-4: open-ended instructions-as-data"
        with pytest.raises(ScopedBranchDeclineError) as exc_info:
            fn(reason)
        assert exc_info.value.reason == reason



class TestRegisterScopeGuardBuiltins:
    def test_scoped_branch_is_added(self):
        base = ns.Namespace.with_builtins()
        extended = register_scope_guard_builtins(base, _STUB_MODEL)
        assert "scoped_branch" in extended.variables

    def test_decline_is_added(self):
        base = ns.Namespace.with_builtins()
        extended = register_scope_guard_builtins(base, _STUB_MODEL)
        assert "decline" in extended.variables

    def test_original_namespace_is_unmodified(self):
        base = ns.Namespace.with_builtins()
        register_scope_guard_builtins(base, _STUB_MODEL)
        assert "scoped_branch" not in base.variables

    def test_scoped_branch_is_correct_builtin_type(self):
        base = ns.Namespace.with_builtins()
        extended = register_scope_guard_builtins(base, _STUB_MODEL)
        assert isinstance(extended.variables["scoped_branch"], _ScopedBranchBuiltin)

    def test_existing_variables_are_preserved(self):
        base = ns.Namespace.with_builtins()
        # All stdlib names should survive registration
        extended = register_scope_guard_builtins(base, _STUB_MODEL)
        for name in ("len", "str", "int", "range"):
            assert name in extended.variables

    def test_scoped_branch_callable_returns_trusted_value(self):
        """End-to-end: calling the registered scoped_branch builtin returns a trusted str."""
        base = ns.Namespace.with_builtins()
        extended = register_scope_guard_builtins(base, _STUB_MODEL)

        sg_builtin = extended.variables["scoped_branch"]
        assert isinstance(sg_builtin, _ScopedBranchBuiltin)

        with patch("camel.scope_guard.classifier.classify", return_value="meeting_invite"):
            args = v.CaMeLTuple(
                (
                    v.CaMeLStr.from_raw("untrusted input", Capabilities.camel(), ()),
                    v.CaMeLStr.from_raw("Classify this", Capabilities.camel(), ()),
                    v.CaMeLList(
                        [v.CaMeLStr.from_raw(lbl, Capabilities.camel(), ()) for lbl in LABELS_3],
                        Capabilities.camel(),
                        (),
                    ),
                ),
                Capabilities.camel(),
                (),
            )
            kwargs = v.CaMeLDict({}, Capabilities.camel(), ())
            result, _ = sg_builtin.call(args, kwargs, extended)

        assert isinstance(result, v.CaMeLStr)
        assert result.raw == "meeting_invite"
        assert result._metadata == Capabilities.camel()
        assert result._dependencies == ()



class TestScopedBranchSpec:
    def test_valid_spec(self):
        spec = ScopedBranchSpec(
            classifier_query="What type?",
            branches=(
                BranchDefinition(label="a", allowed_tools=frozenset({"tool_x"})),
                BranchDefinition(label="b"),
            ),
        )
        assert spec.label_set == {"a", "b"}

    def test_raises_on_single_branch(self):
        with pytest.raises(Exception):
            ScopedBranchSpec(
                classifier_query="q",
                branches=(BranchDefinition(label="only"),),
            )

    def test_raises_on_duplicate_labels(self):
        with pytest.raises(Exception):
            ScopedBranchSpec(
                classifier_query="q",
                branches=(
                    BranchDefinition(label="dup"),
                    BranchDefinition(label="dup"),
                ),
            )

    def test_get_allowed_tools(self):
        spec = ScopedBranchSpec(
            classifier_query="q",
            branches=(
                BranchDefinition(label="a", allowed_tools=frozenset({"tool_a"})),
                BranchDefinition(label="b", allowed_tools=frozenset({"tool_b", "tool_c"})),
            ),
        )
        assert spec.get_allowed_tools("a") == frozenset({"tool_a"})
        assert spec.get_allowed_tools("b") == frozenset({"tool_b", "tool_c"})

    def test_get_allowed_tools_raises_on_unknown_label(self):
        spec = ScopedBranchSpec(
            classifier_query="q",
            branches=(BranchDefinition(label="a"), BranchDefinition(label="b")),
        )
        with pytest.raises(KeyError):
            spec.get_allowed_tools("unknown")

    def test_overlapping_allowed_tools_are_permitted(self):
        """The spec explicitly allows overlapping tool sets between branches."""
        spec = ScopedBranchSpec(
            classifier_query="q",
            branches=(
                BranchDefinition(label="a", allowed_tools=frozenset({"shared_tool", "tool_a"})),
                BranchDefinition(label="b", allowed_tools=frozenset({"shared_tool", "tool_b"})),
            ),
        )
        assert "shared_tool" in spec.get_allowed_tools("a")
        assert "shared_tool" in spec.get_allowed_tools("b")
