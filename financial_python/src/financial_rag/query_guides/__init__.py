"""Experimental, opt-in query guide planning components.

This package is intentionally separate from the live API and ingestion path.
"""

from .matcher import RedisVLGuideMatcher, RuleBasedGuideMatcher
from .planner import QueryPlan, QueryPlanError, build_query_plan, validate_query_scope

__all__ = [
    "QueryPlan",
    "QueryPlanError",
    "RedisVLGuideMatcher",
    "RuleBasedGuideMatcher",
    "build_query_plan",
    "validate_query_scope",
]
