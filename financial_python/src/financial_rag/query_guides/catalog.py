from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "query_guides" / "v1.json"
)


class GuideDefinition(BaseModel):
    guide_id: str
    domain: str
    task: str
    examples: list[str] = Field(min_length=1)
    required_slots: list[str] = Field(default_factory=list)
    evidence_groups: list[str] = Field(default_factory=list)
    retrieval_policy: dict[str, Any] = Field(default_factory=dict)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    answer_format: str = "grounded_response"


class GuideCatalog(BaseModel):
    version: int = Field(ge=1)
    domains: dict[str, list[str]]
    tasks: dict[str, list[str]]
    analysis_dimensions: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    analysis_metric_groups: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    analysis_metric_profiles: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    guides: list[GuideDefinition]

    @model_validator(mode="after")
    def validate_catalog(self) -> "GuideCatalog":
        ids = [guide.guide_id for guide in self.guides]
        if len(ids) != len(set(ids)):
            raise ValueError("Query guide IDs must be unique")
        for guide in self.guides:
            if guide.domain not in self.domains:
                raise ValueError(f"Unknown domain in guide {guide.guide_id}: {guide.domain}")
            if guide.task not in self.tasks:
                raise ValueError(f"Unknown task in guide {guide.guide_id}: {guide.task}")
        return self

    def by_id(self) -> dict[str, GuideDefinition]:
        return {guide.guide_id: guide for guide in self.guides}

    def by_route(self) -> dict[tuple[str, str], GuideDefinition]:
        return {(guide.domain, guide.task): guide for guide in self.guides}


def load_guide_catalog(path: Path | None = None) -> GuideCatalog:
    source = path or DEFAULT_CATALOG_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    return GuideCatalog.model_validate(payload)
