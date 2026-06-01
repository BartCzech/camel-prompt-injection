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

"""Standalone tests for the DDC-3 static analyser.

Run with:
    uv run python tests/test_ddc3_analyser.py

No pytest, no agentdojo — the analyser is pure stdlib (ast + logging) so it
can be loaded directly without triggering the camel.scope_guard package init.
"""

import importlib.util
import pathlib
import sys
import textwrap
import traceback

_ANALYSER_PATH = (
    pathlib.Path(__file__).parent.parent
    / "src/camel/scope_guard/ddc3_analyser.py"
)
_spec = importlib.util.spec_from_file_location("ddc3_analyser", _ANALYSER_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
analyse = _mod.analyse
TAINT_SOURCE = _mod.TAINT_SOURCE



def dedent(s: str) -> str:
    return textwrap.dedent(s).strip()


_PASSED: list[str] = []
_FAILED: list[str] = []


def _run(name: str, fn) -> None:
    try:
        fn()
        _PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception:
        _FAILED.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()



def t_direct_assignment_is_tainted():
    code = dedent("""
        details = query_ai_assistant("extract date", DateModel)
    """)
    _, taint = analyse(code)
    assert "details" in taint, f"expected 'details' in {taint}"


def t_attribute_propagation():
    code = dedent("""
        details = query_ai_assistant("q", Model)
        recipient = details.email
    """)
    _, taint = analyse(code)
    assert "details" in taint
    assert "recipient" in taint, f"expected 'recipient' in {taint}"


def t_subscript_propagation():
    code = dedent("""
        results = query_ai_assistant("q", Model)
        first = results[0]
    """)
    _, taint = analyse(code)
    assert "first" in taint


def t_plain_assignment_propagation():
    code = dedent("""
        raw = query_ai_assistant("q", Model)
        alias = raw
    """)
    _, taint = analyse(code)
    assert "alias" in taint


def t_untainted_not_marked():
    code = dedent("""
        clean = "hello@example.com"
        result = clean.upper()
    """)
    _, taint = analyse(code)
    assert "clean" not in taint
    assert "result" not in taint


def t_s6_pattern_create_calendar_event_rewritten():
    code = dedent("""
        lunch = query_ai_assistant("extract time", MeetingModel)
        create_calendar_event(start_time=lunch.start_time, end_time=lunch.end_time, title="Lunch")
    """)
    rewritten, _ = analyse(code)
    assert "decline(" in rewritten, f"expected decline() in:\n{rewritten}"
    # The sink name appears in the decline reason string but must not be called directly.
    # After ast.unparse the rewritten source will have exactly one line starting with "decline("
    call_lines = [l.strip() for l in rewritten.splitlines() if l.strip().startswith("create_calendar_event(")]
    assert call_lines == [], f"create_calendar_event() should not be called; rewritten:\n{rewritten}"


def t_s7_pattern_send_email_rewritten():
    code = dedent("""
        contact = query_ai_assistant("extract email address", ContactModel)
        send_email(recipients=[contact.email], subject="hello", body="world")
    """)
    rewritten, _ = analyse(code)
    assert "decline(" in rewritten
    call_lines = [l.strip() for l in rewritten.splitlines() if l.strip().startswith("send_email(")]
    assert call_lines == [], f"send_email() should not be called; rewritten:\n{rewritten}"


def t_recipient_list_containing_tainted_value():
    code = dedent("""
        info = query_ai_assistant("extract", Model)
        send_email(recipients=[info.address], subject="s", body="b")
    """)
    rewritten, _ = analyse(code)
    assert "decline(" in rewritten


def t_no_false_positive_on_clean_sink():
    code = dedent("""
        send_email(recipients=["alice@example.com"], subject="hi", body="hello")
    """)
    rewritten, _ = analyse(code)
    assert "decline(" not in rewritten, f"unexpected decline() in:\n{rewritten}"
    assert "send_email" in rewritten


def t_no_false_positive_on_ddc1_code():
    """scoped_branch result is not from query_ai_assistant → must not taint."""
    code = dedent("""
        label = scoped_branch(email_body, "classify", ["rsvp", "discount", "other"])
        if label == "rsvp":
            send_email(recipients=["organizer@example.com"], subject="Re: RSVP", body="Not interested.")
        elif label == "discount":
            create_calendar_event(title="Review discount", start_time="2024-05-27 09:00", end_time="2024-05-27 09:30")
    """)
    rewritten, taint = analyse(code)
    assert "decline(" not in rewritten, f"unexpected decline() in:\n{rewritten}"
    assert "label" not in taint


def t_cross_block_taint_accumulation():
    code1 = "details = query_ai_assistant('q', Model)"
    code2 = "send_email(recipients=[details.email], subject='hi', body='b')"

    _, taint_after_block1 = analyse(code1)
    assert "details" in taint_after_block1

    rewritten2, _ = analyse(code2, taint_after_block1)
    assert "decline(" in rewritten2, f"expected decline() in block2:\n{rewritten2}"


def t_syntax_error_passes_through():
    bad_code = "def foo(\n"
    rewritten, taint = analyse(bad_code)
    assert rewritten == bad_code
    assert taint == set()


def t_decline_reason_contains_sink_name():
    code = dedent("""
        info = query_ai_assistant("extract", Model)
        send_email(recipients=[info.email], subject="s", body="b")
    """)
    rewritten, _ = analyse(code)
    assert "send_email" in rewritten  # named in the decline reason string
    assert "DDC-3" in rewritten


def t_assignment_rhs_sink_rewritten():
    code = dedent("""
        info = query_ai_assistant("extract", Model)
        result = send_email(recipients=[info.email], subject="s", body="b")
    """)
    rewritten, _ = analyse(code)
    assert "decline(" in rewritten


def t_multi_taint_only_affected_sink_rewritten():
    code = dedent("""
        extracted = query_ai_assistant("extract email", Model)
        send_email(recipients=["safe@example.com"], subject="clean", body="hi")
        send_email(recipients=[extracted.address], subject="forwarding", body="msg")
    """)
    rewritten, _ = analyse(code)
    lines = [l.strip() for l in rewritten.splitlines() if l.strip()]
    send_lines = [l for l in lines if "send_email" in l]
    decline_lines = [l for l in lines if "decline(" in l]
    assert send_lines, f"clean send_email should survive in:\n{rewritten}"
    assert decline_lines, f"tainted send_email should be declined in:\n{rewritten}"


def t_augassign_propagation():
    code = dedent("""
        x = query_ai_assistant("q", Model)
        x += ".extra"
        send_email(recipients=[x], subject="s", body="b")
    """)
    rewritten, taint = analyse(code)
    assert "x" in taint
    assert "decline(" in rewritten


def t_ternary_taint_propagation():
    """S7_a1 pattern: ternary expression derived from tainted value must be caught."""
    code = dedent("""
        contact_info = query_ai_assistant("extract address", ContactModel)
        forward_address = contact_info.reply_to_address if contact_info.reply_to_address else contact_info.contact_address
        send_email(recipients=[forward_address], subject="Fwd", body="msg")
    """)
    rewritten, taint = analyse(code)
    assert "forward_address" in taint, f"forward_address should be tainted; taint={taint}"
    call_lines = [l.strip() for l in rewritten.splitlines() if l.strip().startswith("send_email(")]
    assert call_lines == [], f"send_email() should be declined; rewritten:\n{rewritten}"


def t_ternary_in_direct_arg():
    """Ternary used directly inside a sink call arg (no intermediate variable)."""
    code = dedent("""
        info = query_ai_assistant("q", Model)
        send_email(recipients=[info.a if info.a else info.b], subject="s", body="b")
    """)
    rewritten, _ = analyse(code)
    call_lines = [l.strip() for l in rewritten.splitlines() if l.strip().startswith("send_email(")]
    assert call_lines == [], f"send_email() should be declined; rewritten:\n{rewritten}"



TESTS = [
    t_direct_assignment_is_tainted,
    t_attribute_propagation,
    t_subscript_propagation,
    t_plain_assignment_propagation,
    t_untainted_not_marked,
    t_s6_pattern_create_calendar_event_rewritten,
    t_s7_pattern_send_email_rewritten,
    t_recipient_list_containing_tainted_value,
    t_no_false_positive_on_clean_sink,
    t_no_false_positive_on_ddc1_code,
    t_cross_block_taint_accumulation,
    t_syntax_error_passes_through,
    t_decline_reason_contains_sink_name,
    t_assignment_rhs_sink_rewritten,
    t_multi_taint_only_affected_sink_rewritten,
    t_augassign_propagation,
    t_ternary_taint_propagation,
    t_ternary_in_direct_arg,
]

print(f"Running {len(TESTS)} DDC-3 analyser tests…")
for test in TESTS:
    _run(test.__name__, test)

print()
print(f"{len(_PASSED)}/{len(TESTS)} passed, {len(_FAILED)} failed")

if _FAILED:
    print("FAILED:", _FAILED)
    sys.exit(1)
else:
    print("ALL PASS")
