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

"""Tests for the Q-LLM classifier wrapper."""

from __future__ import annotations

from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

from camel.scope_guard.builtin import ScopedBranchClassifierError
from camel.scope_guard.classifier import classify, input_hash

_STUB_MODEL = "anthropic:claude-sonnet-4"
_LABELS = ["meeting_invite", "document_request", "other"]



def _make_enum_result(labels: list[str], chosen_index: int) -> object:
    """Build a mock that looks like a query_quarantined_llm Enum return value."""
    label_enum = Enum("_LabelEnum", {f"option_{i}": lbl for i, lbl in enumerate(labels)})
    member = list(label_enum)[chosen_index]
    return member



class TestClassify:
    def test_returns_first_label(self):
        mock_result = _make_enum_result(_LABELS, 0)
        with patch(
            "camel.scope_guard.classifier.query_quarantined_llm",
            return_value=mock_result,
        ):
            result = classify(_STUB_MODEL, "email body", "What type?", _LABELS)
        assert result == "meeting_invite"

    def test_returns_second_label(self):
        mock_result = _make_enum_result(_LABELS, 1)
        with patch(
            "camel.scope_guard.classifier.query_quarantined_llm",
            return_value=mock_result,
        ):
            result = classify(_STUB_MODEL, "a document request", "What type?", _LABELS)
        assert result == "document_request"

    def test_returns_last_label(self):
        mock_result = _make_enum_result(_LABELS, 2)
        with patch(
            "camel.scope_guard.classifier.query_quarantined_llm",
            return_value=mock_result,
        ):
            result = classify(_STUB_MODEL, "random text", "What type?", _LABELS)
        assert result == "other"

    def test_query_is_forwarded_to_q_llm(self):
        mock_result = _make_enum_result(_LABELS, 0)
        captured: list[str] = []

        def _capture(model, query, schema):
            captured.append(query)
            return mock_result

        with patch("camel.scope_guard.classifier.query_quarantined_llm", side_effect=_capture):
            classify(_STUB_MODEL, "email body", "My custom query", _LABELS)

        # The adversarial robustness note and the classifier_query should both appear
        assert len(captured) == 1
        assert "My custom query" in captured[0]
        assert "email body" in captured[0]
        assert _LABELS[0] in captured[0]

    def test_result_is_always_member_of_declared_labels(self):
        for i in range(len(_LABELS)):
            mock_result = _make_enum_result(_LABELS, i)
            with patch(
                "camel.scope_guard.classifier.query_quarantined_llm",
                return_value=mock_result,
            ):
                result = classify(_STUB_MODEL, "input", "q", _LABELS)
            assert result in _LABELS



class TestClassifyErrors:
    def test_raises_classifier_error_when_value_not_in_labels(self):
        """If the returned value is somehow not in labels, raise ScopedBranchClassifierError."""
        bad_result = MagicMock()
        bad_result.value = "evil_label_not_declared"

        with patch(
            "camel.scope_guard.classifier.query_quarantined_llm",
            return_value=bad_result,
        ):
            with pytest.raises(ScopedBranchClassifierError) as exc_info:
                classify(_STUB_MODEL, "input", "q", _LABELS)

        assert exc_info.value.returned == "evil_label_not_declared"
        assert exc_info.value.declared == _LABELS

    def test_falls_back_to_first_label_on_not_enough_information(self):
        """NotEnoughInformationError should log a warning and return labels[0]."""
        from camel.quarantined_llm import NotEnoughInformationError

        with patch(
            "camel.scope_guard.classifier.query_quarantined_llm",
            side_effect=NotEnoughInformationError,
        ):
            result = classify(_STUB_MODEL, "??", "q", _LABELS)

        assert result == _LABELS[0]

    def test_model_name_is_forwarded(self):
        """The q_llm_model argument must be passed through to query_quarantined_llm."""
        mock_result = _make_enum_result(_LABELS, 0)
        captured_models: list[object] = []

        def _capture(model, query, schema):
            captured_models.append(model)
            return mock_result

        custom_model = "google:gemini-2.5-pro"
        with patch("camel.scope_guard.classifier.query_quarantined_llm", side_effect=_capture):
            classify(custom_model, "input", "q", _LABELS)

        assert captured_models == [custom_model]



class TestInputHash:
    def test_returns_16_char_hex_string(self):
        h = input_hash("hello world")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_inputs_give_different_hashes(self):
        assert input_hash("abc") != input_hash("def")

    def test_same_input_gives_same_hash(self):
        assert input_hash("abc") == input_hash("abc")

    def test_pii_is_not_leaked(self):
        """The hash must not contain the original text."""
        sensitive = "user@example.com"
        h = input_hash(sensitive)
        assert sensitive not in h
