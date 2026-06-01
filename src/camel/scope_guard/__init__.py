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

"""Scope-guarded branching extension for CaMeL."""

from camel.scope_guard.branch_spec import BranchDefinition, ScopedBranchSpec
from camel.scope_guard.builtin import (
    ScopedBranchClassifierError,
    ScopedBranchDeclineError,
    register_scope_guard_builtins,
    restrict_namespace,
)

# ScopedPrivilegedLLM and scoped_system_prompt_generator are NOT imported here
# because pipeline.py pulls in PrivilegedLLM → AgentDojo, which has a known
# circular-import at package init time.  Import them directly when needed:
#   from camel.scope_guard.pipeline import ScopedPrivilegedLLM

__all__ = [
    "BranchDefinition",
    "ScopedBranchSpec",
    "ScopedBranchClassifierError",
    "ScopedBranchDeclineError",
    "register_scope_guard_builtins",
    "restrict_namespace",
]
