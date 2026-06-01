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

"""Data models for the scope-guarded branching specification."""

from __future__ import annotations

import pydantic
from pydantic import field_validator


class BranchDefinition(pydantic.BaseModel):
    """Defines a single branch in a scope-guarded branch construct."""

    label: str
    """Branch label; must be unique within the ScopedBranchSpec."""

    allowed_tools: frozenset[str] = frozenset()
    """Tool names permitted inside this branch body (for tracing / post-hoc enforcement)."""

    description: str = ""
    """Human-readable description of when this branch should be selected."""

    model_config = pydantic.ConfigDict(frozen=True)

    @field_validator("label")
    @classmethod
    def label_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("label must not be empty or whitespace")
        return v


class ScopedBranchSpec(pydantic.BaseModel):
    """Full specification for a scope-guarded branch construct."""

    classifier_query: str
    """Prompt sent to the Q-LLM to select a branch."""

    branches: tuple[BranchDefinition, ...]
    """Ordered list of branches.  Must contain at least two entries with unique labels."""

    model_config = pydantic.ConfigDict(frozen=True)

    @field_validator("branches")
    @classmethod
    def at_least_two_branches(cls, v: tuple[BranchDefinition, ...]) -> tuple[BranchDefinition, ...]:
        if len(v) < 2:
            raise ValueError("ScopedBranchSpec requires at least two branches")
        return v

    @field_validator("branches")
    @classmethod
    def labels_unique(cls, v: tuple[BranchDefinition, ...]) -> tuple[BranchDefinition, ...]:
        labels = [b.label for b in v]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Branch labels must be unique; duplicates found in {labels}")
        return v

    @property
    def label_set(self) -> frozenset[str]:
        return frozenset(b.label for b in self.branches)

    def get_allowed_tools(self, label: str) -> frozenset[str]:
        for branch in self.branches:
            if branch.label == label:
                return branch.allowed_tools
        raise KeyError(f"No branch with label {label!r}")
