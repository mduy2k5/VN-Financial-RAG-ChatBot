"""Load the versioned report taxonomies and resolve source/query labels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Optional

from ..schemas import ChunkType, FinancialChunk, StatementType
from .source_groups import source_item_label


@dataclass(frozen=True)
class ReportSchemaNode:
    node_id: str
    canonical_name: str
    statement_type: StatementType
    parent_id: Optional[str]
    children: tuple[str, ...]
    kind: Optional[str]
    sector: Optional[str]
    aliases: tuple[str, ...]
    parent_contexts: tuple[str, ...]
    schema_version: str


_SCHEMA_BUNDLES = (
    (StatementType.BALANCE_SHEET, "financial_position_schema.json", "financial_position_aliases.json"),
    (StatementType.INCOME_STATEMENT, "income_statement_schema.json", "income_statement_aliases.json"),
    (StatementType.CASH_FLOW, "cash_flow_schema.json", "cash_flow_aliases.json"),
)

_BANK_OFF_BALANCE_SCHEMA = "off_balance_bank_schema.json"
_BANK_OFF_BALANCE_ALIASES = "off_balance_bank_aliases.json"


def _version_key(path: Path) -> tuple[int, ...]:
    match = re.search(r"_v(\d+(?:_\d+)*)\.json$", path.name)
    return tuple(int(part) for part in match.group(1).split("_")) if match else ()


def _latest_resource(schemas_dir: Path, base_name: str) -> Path:
    base_path = schemas_dir / base_name
    overlays = list((schemas_dir / "overlays").glob(f"{Path(base_name).stem}_v*.json"))
    if overlays:
        return max(overlays, key=_version_key)
    return base_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Schema resource must be a JSON object: {path}")
    return value


def _load_schema(path: Path, seen: Optional[set[Path]] = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = seen or set()
    if resolved in seen:
        raise ValueError(f"Schema overlay cycle at {resolved}")
    seen.add(resolved)
    data = _read_json(resolved)
    if "base_schema" not in data:
        return data

    base_path = (resolved.parent / str(data["base_schema"])).resolve()
    base = _load_schema(base_path, seen)
    nodes = {
        str(node["node_id"]): dict(node)
        for node in base.get("nodes", [])
        if isinstance(node, dict) and node.get("node_id")
    }
    for node_id, updates in data.get("node_updates", {}).items():
        nodes.setdefault(str(node_id), {"node_id": str(node_id)}).update(updates)
    for node in data.get("nodes_add", []):
        if isinstance(node, dict) and node.get("node_id"):
            nodes[str(node["node_id"])] = dict(node)
    merged = dict(base)
    merged.update({
        key: value
        for key, value in data.items()
        if key not in {"base_schema", "node_updates", "nodes_add", "formula_updates"}
    })
    merged["nodes"] = list(nodes.values())
    if "formula_updates" in data:
        formulas = dict(base.get("formulas", {}))
        formulas.update(data["formula_updates"])
        merged["formulas"] = formulas
    return merged


def _append_unique(target: list[Any], values: Iterable[Any]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _merge_alias_entry(base: Any, update: Any) -> Any:
    if isinstance(update, list):
        result = dict(base) if isinstance(base, dict) else {"aliases": list(base) if isinstance(base, list) else []}
        _append_unique(result.setdefault("aliases", []), update)
        return result
    if not isinstance(update, dict):
        return update

    result = dict(base) if isinstance(base, dict) else ({"aliases": list(base)} if isinstance(base, list) else {})
    if "add" in update and isinstance(update["add"], list):
        _append_unique(result.setdefault("aliases", []), update["add"])
    if "remove" in update and isinstance(update["remove"], list):
        result["aliases"] = [item for item in result.get("aliases", []) if item not in update["remove"]]

    for key, value in update.items():
        if key in {"add", "remove"}:
            continue
        if key.endswith("_add") and isinstance(value, list):
            field = key[:-4]
            _append_unique(result.setdefault(field, []), value)
        elif key.endswith("_remove") and isinstance(value, list):
            field = key[:-7]
            result[field] = [item for item in result.get(field, []) if item not in value]
        elif isinstance(value, dict):
            result[key] = _merge_alias_entry(result.get(key), value)
        else:
            result[key] = value
    return result


def _load_aliases(path: Path, seen: Optional[set[Path]] = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = seen or set()
    if resolved in seen:
        raise ValueError(f"Alias overlay cycle at {resolved}")
    seen.add(resolved)
    data = _read_json(resolved)
    if "base_aliases" in data:
        base = _load_aliases((resolved.parent / str(data["base_aliases"])).resolve(), seen)
        for node_id, update in data.get("entries_update", {}).items():
            base[str(node_id)] = _merge_alias_entry(base.get(str(node_id)), update)
        for node_id, entry in data.get("entries_add", {}).items():
            base[str(node_id)] = _merge_alias_entry(base.get(str(node_id)), entry)
        return base

    if isinstance(data.get("nodes"), dict):
        return {str(node_id): value for node_id, value in data["nodes"].items()}

    # Cash-flow aliases use a compact node_id -> aliases map rather than a wrapper.
    return {
        str(node_id): {"aliases": value if isinstance(value, list) else [str(value)]}
        for node_id, value in data.items()
        if str(node_id).startswith(("CASHFLOW.", "INCOME.", "ASSET.", "LIABILITY.", "EQUITY."))
    }


def _collect_aliases(value: Any) -> list[str]:
    aliases: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"exact", "aliases", "contains", "titles", "title_aliases"}:
                if isinstance(child, list):
                    aliases.extend(str(item) for item in child if item)
                elif isinstance(child, str) and child:
                    aliases.append(child)
            elif key in {"match", "contextual", "aliases_by_context", "by_context"}:
                aliases.extend(_collect_aliases(child))
    elif isinstance(value, list):
        aliases.extend(str(item) for item in value if isinstance(item, str) and item)
    elif isinstance(value, str):
        aliases.append(value)
    return aliases


def _collect_parent_contexts(value: Any) -> list[str]:
    contexts: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"parent_context", "parent_context_add"}:
                if isinstance(child, list):
                    contexts.extend(str(item) for item in child if item)
                elif isinstance(child, str) and child:
                    contexts.append(child)
            elif isinstance(child, dict):
                contexts.extend(_collect_parent_contexts(child))
    return contexts


class ReportSchemaCatalog:
    """Version-aware semantic catalog for balance-sheet, income, and cash-flow rows."""

    def __init__(self, schemas_dir: Optional[Path] = None):
        configured = os.getenv("FINANCIAL_SCHEMA_DIR", "").strip()
        if schemas_dir is None and configured:
            schemas_dir = Path(configured)
        if schemas_dir is None:
            schemas_dir = Path(__file__).resolve().parents[4] / "schemas"
        self.schemas_dir = schemas_dir
        self.nodes: dict[str, ReportSchemaNode] = {}
        self.children: dict[str, list[str]] = {}
        self.component_groups: dict[str, dict[str, Any]] = {}
        self.nodes_by_statement: dict[StatementType, list[ReportSchemaNode]] = {}
        self._row_label_index: dict[StatementType, dict[str, list[ReportSchemaNode]]] = {}
        self.available = False
        self._load()

    @staticmethod
    def normalize_label(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value)
        value = value.replace("**", " ").replace("__", " ").replace("`", " ")
        # Financial statements often append a formula/code to the row label,
        # e.g. "Doanh thu thuần ... (10 = 01 - 02)". It is not part of the
        # semantic label and otherwise prevents the source row from receiving
        # its schema node.
        value = re.sub(r"\s*\(\s*[0-9\s.=+:/-]+\s*\)\s*$", " ", value)
        value = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
        value = "".join(char for char in value if not unicodedata.combining(char))
        value = re.sub(r"^\s*(?:\(?\d+[a-z]?\)?|[a-z]|[ivx]+)\s*[.)\-:]\s*", "", value)
        return re.sub(r"[^a-z0-9]+", " ", value).strip()

    def _load(self) -> None:
        if not self.schemas_dir.is_dir():
            return
        for statement_type, schema_name, aliases_name in _SCHEMA_BUNDLES:
            try:
                schema_path = _latest_resource(self.schemas_dir, schema_name)
                alias_path = _latest_resource(self.schemas_dir, aliases_name)
                if not schema_path.is_file() or not alias_path.is_file():
                    continue
                schema = _load_schema(schema_path)
                aliases = _load_aliases(alias_path)
            except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
                continue
            version = str(schema.get("schema_version", "unknown"))
            definitions = [node for node in schema.get("nodes", []) if isinstance(node, dict)]
            known_ids = {str(node.get("node_id")) for node in definitions if node.get("node_id")}
            statement_nodes: list[ReportSchemaNode] = []
            for raw in definitions:
                node_id = str(raw.get("node_id", "")).strip()
                canonical_name = str(raw.get("canonical_name", "")).strip()
                if not node_id or not canonical_name:
                    continue
                alias_entry = aliases.get(node_id, {})
                phrases = [canonical_name, *_collect_aliases(alias_entry)]
                phrases = list(dict.fromkeys(phrase.strip() for phrase in phrases if phrase and phrase.strip()))
                explicit_parent = raw.get("parent_id")
                if explicit_parent is None:
                    parent_prefix = node_id.rsplit(".", 1)[0] if "." in node_id else None
                    explicit_parent = parent_prefix if parent_prefix in known_ids else None
                child_ids = list(raw.get("children", []))
                child_ids.extend(
                    str(candidate.get("node_id"))
                    for candidate in definitions
                    if candidate.get("parent_id") == node_id
                )
                node = ReportSchemaNode(
                    node_id=node_id,
                    canonical_name=canonical_name,
                    statement_type=statement_type,
                    parent_id=str(explicit_parent) if explicit_parent else None,
                    children=tuple(dict.fromkeys(str(child) for child in child_ids if child)),
                    # Older schema bundles omit ``kind`` on ordinary statement
                    # rows. Treat leaf rows as lines and rows with children as
                    # groups so query understanding can anchor direct lookups
                    # to the schema instead of falling back to ambiguous aliases.
                    kind=(
                        str(raw.get("kind"))
                        if raw.get("kind")
                        else "group" if child_ids else "line"
                    ),
                    sector=str(raw.get("sector")) if raw.get("sector") else None,
                    aliases=tuple(phrases),
                    parent_contexts=tuple(dict.fromkeys(_collect_parent_contexts(alias_entry))),
                    schema_version=version,
                )
                self.nodes[node_id] = node
                statement_nodes.append(node)
                self.children[node_id] = list(node.children)
            self.nodes_by_statement[statement_type] = statement_nodes
        self._load_bank_off_balance_nodes()
        self._load_note_aliases()
        self._load_component_groups()
        for statement_type, nodes in self.nodes_by_statement.items():
            label_index: dict[str, list[ReportSchemaNode]] = {}
            for node in nodes:
                for alias in node.aliases:
                    normalized = self.normalize_label(alias)
                    if normalized:
                        label_index.setdefault(normalized, []).append(node)
            self._row_label_index[statement_type] = label_index
        self.available = bool(self.nodes)

    def _load_component_groups(self) -> None:
        """Load explicit financial statement rollups used for composition answers."""
        path = self.schemas_dir / "financial_composition_groups.json"
        if not path.is_file():
            return
        try:
            data = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return

        groups = data.get("groups", {})
        if not isinstance(groups, dict):
            return
        for group_id, raw in groups.items():
            if not isinstance(raw, dict):
                continue
            group = dict(raw)
            self.component_groups[str(group_id)] = group
            # Schema-backed groups extend the taxonomy tree. Synthetic groups
            # (for example, funding structure) are resolved by the answerer.
            if group_id not in self.nodes:
                continue
            child_ids = list(group.get("component_node_ids", []))
            child_ids.extend(group.get("total_node_ids", []))
            if group.get("total_node_id"):
                child_ids.append(str(group["total_node_id"]))
            variants = group.get("variants", {})
            if isinstance(variants, dict):
                for variant in variants.values():
                    if isinstance(variant, dict):
                        child_ids.extend(variant.get("component_node_ids", []))
                        child_ids.extend(variant.get("total_node_ids", []))
                        if variant.get("total_node_id"):
                            child_ids.append(str(variant["total_node_id"]))
            known_children = [str(child) for child in child_ids if str(child) in self.nodes]
            self.children[group_id] = list(dict.fromkeys([
                *self.children.get(group_id, []), *known_children,
            ]))

    def _load_bank_off_balance_nodes(self) -> None:
        """Merge the bank-specific off-balance taxonomy and its aliases."""
        schema_path = self.schemas_dir / _BANK_OFF_BALANCE_SCHEMA
        alias_path = self.schemas_dir / _BANK_OFF_BALANCE_ALIASES
        if not schema_path.is_file() or not alias_path.is_file():
            return
        try:
            schema = _load_schema(schema_path)
            aliases = _load_aliases(alias_path)
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
            return

        nodes = self.nodes_by_statement.setdefault(StatementType.BALANCE_SHEET, [])
        node_positions = {node.node_id: index for index, node in enumerate(nodes)}
        schema_version = str(schema.get("schema_version", "unknown"))
        for raw in schema.get("nodes", []):
            if not isinstance(raw, dict) or not raw.get("node_id") or not raw.get("canonical_name"):
                continue
            node_id = str(raw["node_id"])
            existing = self.nodes.get(node_id)
            alias_entry = aliases.get(node_id, {})
            canonical_name = str(raw["canonical_name"])
            phrases = [
                *(existing.aliases if existing else ()),
                canonical_name,
                *_collect_aliases(alias_entry),
            ]
            children = raw.get("children") or (existing.children if existing else ())
            parent_id = raw.get("parent_id")
            if parent_id is None and existing is not None:
                parent_id = existing.parent_id
            node = ReportSchemaNode(
                node_id=node_id,
                canonical_name=canonical_name,
                statement_type=StatementType.BALANCE_SHEET,
                parent_id=str(parent_id) if parent_id else None,
                children=tuple(dict.fromkeys(str(child) for child in children if child)),
                kind=str(raw.get("kind") or (existing.kind if existing else "line")),
                sector=str(raw.get("sector") or "bank"),
                aliases=tuple(dict.fromkeys(phrase.strip() for phrase in phrases if phrase and phrase.strip())),
                parent_contexts=tuple(dict.fromkeys([
                    *(existing.parent_contexts if existing else ()),
                    *_collect_parent_contexts(alias_entry),
                ])),
                schema_version=schema_version,
            )
            self.nodes[node_id] = node
            self.children[node_id] = list(node.children)
            if node_id in node_positions:
                nodes[node_positions[node_id]] = node
            else:
                node_positions[node_id] = len(nodes)
                nodes.append(node)

    def _load_note_aliases(self) -> None:
        note_nodes: list[ReportSchemaNode] = []
        for path in sorted(self.schemas_dir.glob("notes_aliases_*.json")):
            try:
                data = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            profile = str(data.get("profile", "")).removesuffix("_v1")
            version = str(data.get("alias_schema_version", "unknown"))
            for section in data.get("sections", []):
                if not isinstance(section, dict) or not section.get("alias_id"):
                    continue
                section_id = str(section["alias_id"])
                section_phrases = [
                    str(section.get("canonical_name", "")),
                    *[str(item) for item in section.get("titles", [])],
                    *[str(item) for item in section.get("title_aliases", [])],
                ]
                row_nodes: list[ReportSchemaNode] = []
                for row in section.get("rows", []):
                    if not isinstance(row, dict) or not row.get("alias_id"):
                        continue
                    row_id = str(row["alias_id"])
                    aliases = [
                        str(row.get("canonical_name", "")),
                        *[str(item) for item in row.get("aliases", [])],
                    ]
                    row_node = ReportSchemaNode(
                        node_id=row_id,
                        canonical_name=str(row.get("canonical_name", row_id)),
                        statement_type=StatementType.NOTES,
                        parent_id=section_id,
                        children=(),
                        kind="note_row",
                        sector=profile or None,
                        aliases=tuple(dict.fromkeys(item for item in aliases if item)),
                        parent_contexts=tuple(dict.fromkeys(item for item in section_phrases if item)),
                        schema_version=version,
                    )
                    self.nodes[row_id] = row_node
                    self.children[row_id] = []
                    row_nodes.append(row_node)
                section_node = ReportSchemaNode(
                    node_id=section_id,
                    canonical_name=str(section.get("canonical_name", section_id)),
                    statement_type=StatementType.NOTES,
                    parent_id=None,
                    children=tuple(row.node_id for row in row_nodes),
                    kind="note_section",
                    sector=profile or None,
                    aliases=tuple(dict.fromkeys(item for item in section_phrases if item)),
                    parent_contexts=(),
                    schema_version=version,
                )
                self.nodes[section_id] = section_node
                self.children[section_id] = list(section_node.children)
                note_nodes.extend([section_node, *row_nodes])
        if note_nodes:
            self.nodes_by_statement[StatementType.NOTES] = note_nodes

    def resolve_row_label(
        self,
        label: Optional[str],
        statement_type: Optional[StatementType],
        parent_context: Optional[Iterable[str]] = None,
    ) -> Optional[ReportSchemaNode]:
        if not label or statement_type is None:
            return None
        normalized = self.normalize_label(label)
        if not normalized:
            return None
        candidates = list(self._row_label_index.get(statement_type, {}).get(normalized, []))
        candidates = list({node.node_id: node for node in candidates}.values())
        if len(candidates) == 1:
            return candidates[0]
        contexts = [self.normalize_label(value) for value in (parent_context or []) if value]
        if contexts:
            contextual = [
                node for node in candidates
                if any(
                    self.normalize_label(expected) in context or context in self.normalize_label(expected)
                    for expected in node.parent_contexts
                    for context in contexts
                )
            ]
            if len(contextual) == 1:
                return contextual[0]
            if contextual:
                candidates = contextual
        return candidates[0] if len(candidates) == 1 else None

    def resolve_source_row_label(
        self,
        label: Optional[str],
        statement_type: Optional[StatementType],
        parent_context: Optional[Iterable[str]] = None,
    ) -> Optional[ReportSchemaNode]:
        """Prefer a node's canonical source name when a broad alias is ambiguous."""
        if not label or statement_type is None:
            return None
        normalized = self.normalize_label(label)
        candidates = list(self._row_label_index.get(statement_type, {}).get(normalized, []))
        canonical_matches = [
            node for node in candidates
            if self.normalize_label(node.canonical_name) == normalized
        ]
        if len(canonical_matches) == 1:
            return canonical_matches[0]
        if canonical_matches:
            shallowest = min(node.node_id.count(".") for node in canonical_matches)
            canonical_matches = [
                node for node in canonical_matches
                if node.node_id.count(".") == shallowest
            ]
            if len(canonical_matches) == 1:
                return canonical_matches[0]
        return self.resolve_row_label(label, statement_type, parent_context)

    def is_node_within(self, node_id: str, ancestor_id: str) -> bool:
        """Check schema ancestry using parent links, not inferred row context."""
        current = self.nodes.get(node_id)
        visited: set[str] = set()
        while current and current.node_id not in visited:
            if current.node_id == ancestor_id:
                return True
            visited.add(current.node_id)
            current = self.nodes.get(current.parent_id or "")
        return False

    def resolve_query_node(
        self,
        query: str,
        statement_type: Optional[StatementType] = None,
        prefer_container: bool = False,
    ) -> Optional[ReportSchemaNode]:
        normalized_query = self.normalize_label(query)
        candidates: list[tuple[int, ReportSchemaNode]] = []
        node_list = (
            self.nodes_by_statement.get(statement_type, [])
            if statement_type
            else list(self.nodes.values())
        )
        for node in node_list:
            for alias in node.aliases:
                normalized_alias = self.normalize_label(alias)
                if len(normalized_alias) >= 3 and normalized_alias in normalized_query:
                    candidates.append((len(normalized_alias), node))
        if not candidates:
            return None
        longest = max(length for length, _ in candidates)
        matches = {node.node_id: node for length, node in candidates if length == longest}
        if prefer_container and len(matches) > 1:
            containers = {
                node_id: node
                for node_id, node in matches.items()
                if node.kind in {"section", "group", "root", "note_section"} or node.children
            }
            if len(containers) == 1:
                return next(iter(containers.values()))
        elif len(matches) > 1:
            # Một nhãn tổng quát có thể đồng thời là alias của node cha và
            # node chi tiết theo ngành (ví dụ HTM tổng quát và HTM trước dự
            # phòng của ngân hàng). Nếu không có ngữ cảnh ngành, truy vấn tên
            # gốc nên resolve về node tổng quát ở cấp cao hơn.
            universal = {
                node_id: node
                for node_id, node in matches.items()
                if node.sector in {None, "common"}
            }
            if universal:
                matches = universal
            else:
                shallowest = min(node.node_id.count(".") for node in matches.values())
                matches = {
                    node_id: node
                    for node_id, node in matches.items()
                    if node.node_id.count(".") == shallowest
                }
        return next(iter(matches.values())) if len(matches) == 1 else None

    def annotate_chunks(self, chunks: list[FinancialChunk]) -> None:
        for chunk in chunks:
            if chunk.chunk_type != ChunkType.FINANCIAL_ITEM:
                continue
            metadata = chunk.metadata
            contexts = [
                str(metadata.get("section_group", "")),
                str(metadata.get("table_title", chunk.table_title or "")),
            ]
            context_node = self.resolve_row_label(
                contexts[0], chunk.statement_type, parent_context=contexts[1:]
            ) or self.resolve_row_label(contexts[1], chunk.statement_type)
            existing_node_id = metadata.get("schema_node_id")
            existing_node = self.nodes.get(str(existing_node_id)) if existing_node_id else None
            # The extracted source row is authoritative during chunking. A
            # persisted schema ID or a registry's canonical metric can be stale
            # or overly broad, so resolve the raw row label first and use the
            # prior ID only when the source label cannot be mapped confidently.
            raw_label = source_item_label(chunk) or chunk.canonical_metric
            node = self.resolve_source_row_label(
                raw_label, chunk.statement_type, parent_context=contexts
            ) if raw_label else None
            node = node or existing_node
            if node is None:
                if context_node:
                    metadata["schema_context_node_id"] = context_node.node_id
                    metadata["schema_context_name"] = context_node.canonical_name
                continue

            # A section heading can leak across rows during extraction or
            # resolve to an ambiguous alias. Keep it only when the row's schema
            # node is actually inside that section's subtree.
            if context_node and self.is_node_within(node.node_id, context_node.node_id):
                metadata["schema_context_node_id"] = context_node.node_id
                metadata["schema_context_name"] = context_node.canonical_name
            else:
                metadata.pop("schema_context_node_id", None)
                metadata.pop("schema_context_name", None)

            metadata["schema_node_id"] = node.node_id
            metadata["schema_node_name"] = node.canonical_name
            metadata["schema_parent_id"] = node.parent_id
            metadata["schema_children"] = list(
                self.children.get(node.node_id, node.children)
            )
            group_config = self.component_groups.get(node.node_id, {})
            component_variants = group_config.get("variants", {})
            if isinstance(component_variants, dict) and component_variants:
                metadata["schema_component_variants"] = {
                    str(variant_name): [
                        str(node_id)
                        for node_id in variant.get("component_node_ids", [])
                        if str(node_id) in self.nodes
                    ]
                    for variant_name, variant in component_variants.items()
                    if isinstance(variant, dict)
                }
            else:
                metadata.pop("schema_component_variants", None)
            metadata["schema_node_kind"] = node.kind
            metadata["schema_sector"] = node.sector
            metadata["schema_version"] = node.schema_version


@lru_cache(maxsize=1)
def get_report_schema_catalog() -> ReportSchemaCatalog:
    return ReportSchemaCatalog()
