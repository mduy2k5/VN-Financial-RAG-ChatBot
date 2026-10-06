"""Shared source arithmetic for composition and contribution analysis."""
from decimal import Decimal, InvalidOperation
import re
import unicodedata

from ..schemas import FinancialChunk
from ..validation.numeric import NumericValidator


class SourceCompositionRows:
    @classmethod
    def _non_overlapping_composition_rows(
        cls,
        rows: list[dict[str, object]],
        total_row: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        """Keep source rows that do not overlap visible children or reconciled detail."""
        rows_by_node: dict[str, list[dict[str, object]]] = {}
        parent_nodes_with_visible_children: set[str] = set()
        for row in rows:
            chunk = row.get("chunk")
            if not isinstance(chunk, FinancialChunk):
                continue
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            node_id = str(metadata.get("schema_node_id") or "").strip()
            if node_id:
                rows_by_node.setdefault(node_id, []).append(row)

        visible_node_ids = set(rows_by_node)

        def numeric_values(row: dict[str, object]) -> dict[int, Decimal]:
            raw_values = row.get("values")
            if not isinstance(raw_values, dict):
                return {}
            values: dict[int, Decimal] = {}
            for year, raw_value in raw_values.items():
                normalized = NumericValidator.normalize_number(str(raw_value or ""))
                if normalized:
                    try:
                        values[int(year)] = Decimal(normalized)
                    except (InvalidOperation, TypeError, ValueError):
                        continue
            return values

        def node_is_composed(node_id: str) -> bool:
            parent_rows = rows_by_node.get(node_id, [])
            # Repeated source rows may encode net and gross presentations for
            # one schema node. Keep those until source arithmetic disambiguates.
            if len(parent_rows) != 1:
                return False
            parent = parent_rows[0]
            parent_chunk = parent.get("chunk")
            if not isinstance(parent_chunk, FinancialChunk):
                return False
            parent_metadata = parent_chunk.metadata if isinstance(parent_chunk.metadata, dict) else {}
            declared_children = {
                str(child_id).strip()
                for child_id in parent_metadata.get("schema_children", [])
                if str(child_id).strip()
            }
            direct_children: list[dict[str, object]] = []
            for row in rows:
                child_chunk = row.get("chunk")
                if not isinstance(child_chunk, FinancialChunk) or child_chunk.chunk_id == parent_chunk.chunk_id:
                    continue
                is_declared_child = (
                    str(child_chunk.metadata.get("schema_node_id") or "") in declared_children
                    if declared_children
                    else str(child_chunk.metadata.get("schema_parent_id") or "") == node_id
                )
                if is_declared_child:
                    direct_children.append(row)
            if len(direct_children) < 2:
                return False
            parent_values = numeric_values(parent)
            if not parent_values:
                return False
            parent_unit = NumericValidator._normalize_unit(str(parent.get("unit") or ""))
            if not parent_unit:
                return False
            child_chunks = [row.get("chunk") for row in direct_children]
            if any(
                not isinstance(child_chunk, FinancialChunk)
                or not cls._same_source_table(parent_chunk, child_chunk)
                or NumericValidator._normalize_unit(str(child.get("unit") or "")) != parent_unit
                for child, child_chunk in zip(direct_children, child_chunks)
            ):
                return False
            children_by_node: dict[str, dict[str, object]] = {}
            for child in direct_children:
                child_chunk = child.get("chunk")
                if isinstance(child_chunk, FinancialChunk):
                    child_id = str(child_chunk.metadata.get("schema_node_id") or "").strip()
                    if not child_id or child_id in children_by_node:
                        return False
                    children_by_node[child_id] = child
            child_values = [numeric_values(child) for child in direct_children]
            years = set(parent_values)
            if not years or any(not years.issubset(values) for values in child_values):
                return False
            return all(
                abs(
                    sum((values[year] for values in child_values), Decimal("0"))
                    - parent_values[year]
                ) <= max(Decimal("1"), abs(parent_values[year]) * Decimal("0.000001"))
                for year in years
            )

        for node_id in visible_node_ids:
            if node_is_composed(node_id):
                parent_nodes_with_visible_children.add(node_id)

        display_rows: list[dict[str, object]] = []
        for row in rows:
            if row is total_row:
                continue
            chunk = row.get("chunk")
            if isinstance(chunk, FinancialChunk):
                metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
                node_id = str(metadata.get("schema_node_id") or "").strip()
                if node_id in parent_nodes_with_visible_children:
                    continue
                if re.search(r"(?i)^\s*[-–—]?\s*trong\s+đó\s*:", str(row.get("label") or "")):
                    # "Trong đó" is a disclosure of an already reported line,
                    # not an additional balance-sheet component.
                    continue
            display_rows.append(row)
        return cls._remove_reconciled_composition_overlaps(display_rows, total_row)


    @classmethod
    def _remove_reconciled_composition_overlaps(
        cls,
        display_rows: list[dict[str, object]],
        total_row: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        """Use source arithmetic to remove duplicate gross/provision and subtotal rows.

        Some OCR tables contain the net balance, the gross balance, and its
        allowance as separate rows with the same label or without hierarchy
        IDs. When the source values reconcile, retain the net row and suppress
        the overlapping gross/provision detail. A second pass suppresses an
        untagged subtotal only when its value equals a consecutive set of
        following source rows in the same page and unit.
        """
        def row_values(row: dict[str, object]) -> dict[int, Decimal]:
            raw_values = row.get("values")
            if not isinstance(raw_values, dict):
                return {}
            parsed: dict[int, Decimal] = {}
            for year, raw_value in raw_values.items():
                normalized = NumericValidator.normalize_number(str(raw_value or ""))
                if normalized:
                    try:
                        parsed[int(year)] = Decimal(normalized)
                    except (InvalidOperation, TypeError, ValueError):
                        continue
            return parsed

        def same_unit(left: dict[str, object], right: dict[str, object]) -> bool:
            left_unit = NumericValidator._normalize_unit(str(left.get("unit") or ""))
            right_unit = NumericValidator._normalize_unit(str(right.get("unit") or ""))
            return bool(left_unit and left_unit == right_unit)

        def source_position(row: dict[str, object], fallback: int) -> tuple[int, int, int]:
            chunk = row.get("chunk")
            if not isinstance(chunk, FinancialChunk):
                return (0, fallback, fallback)
            item_match = re.search(r"_item_(\d+)(?:_|$)", chunk.chunk_id)
            return (
                chunk.page or 0,
                int(item_match.group(1)) if item_match else fallback,
                fallback,
            )

        ordered = sorted(
            enumerate(display_rows),
            key=lambda indexed: source_position(indexed[1], indexed[0]),
        )
        ordered_rows = [row for _, row in ordered]
        removed_ids: set[str] = set()

        # Identify repeated labels whose amount is the net of a nearby gross
        # amount plus a signed allowance (for example gross loans + provision).
        duplicate_groups: dict[str, list[dict[str, object]]] = {}
        for row in ordered_rows:
            label_key = cls._normalize_table_label(str(row.get("label") or ""))
            if label_key:
                duplicate_groups.setdefault(label_key, []).append(row)
        allowance_rows = [
            row for row in ordered_rows
            if any(token in cls._normalize_table_label(str(row.get("label") or ""))
                   for token in ("du phong", "allowance", "provision"))
        ]
        for duplicate_group in duplicate_groups.values():
            if len(duplicate_group) < 2:
                continue
            matched_duplicate_group = False
            duplicate_group.sort(key=lambda row: source_position(row, ordered_rows.index(row)))
            for left_index in range(len(duplicate_group)):
                for right_index in range(left_index + 1, len(duplicate_group)):
                    left = duplicate_group[left_index]
                    right = duplicate_group[right_index]
                    left_chunk = left.get("chunk")
                    right_chunk = right.get("chunk")
                    if not isinstance(left_chunk, FinancialChunk) or not isinstance(right_chunk, FinancialChunk):
                        continue
                    if not cls._same_source_table(left_chunk, right_chunk) or not same_unit(left, right):
                        continue
                    left_values = row_values(left)
                    right_values = row_values(right)
                    later_position = source_position(right, ordered_rows.index(right))
                    nearby_allowances = [
                        allowance for allowance in allowance_rows
                        if allowance is not left and allowance is not right
                        and isinstance(allowance.get("chunk"), FinancialChunk)
                        and cls._same_source_table(
                            left_chunk,
                            allowance["chunk"],  # type: ignore[arg-type]
                        )
                        and same_unit(left, allowance)
                        and 0 <= (
                            source_position(allowance, ordered_rows.index(allowance))[1]
                            - later_position[1]
                        ) <= 3
                    ]
                    nearby_allowances.sort(
                        key=lambda row: source_position(row, ordered_rows.index(row))
                    )
                    for allowance in nearby_allowances:
                        allowance_values = row_values(allowance)
                        comparable_years = sorted(
                            set(left_values) & set(right_values) & set(allowance_values)
                        )
                        if not comparable_years:
                            continue
                        left_plus_allowance = all(
                            abs(left_values[year] + allowance_values[year] - right_values[year])
                            <= max(Decimal("1"), abs(right_values[year]) * Decimal("0.000001"))
                            for year in comparable_years
                        )
                        right_plus_allowance = all(
                            abs(right_values[year] + allowance_values[year] - left_values[year])
                            <= max(Decimal("1"), abs(left_values[year]) * Decimal("0.000001"))
                            for year in comparable_years
                        )
                        if left_plus_allowance:
                            suppress = left
                        elif right_plus_allowance:
                            suppress = right
                        else:
                            continue
                        suppress_chunk = suppress.get("chunk")
                        allowance_chunk = allowance.get("chunk")
                        if isinstance(suppress_chunk, FinancialChunk):
                            removed_ids.add(suppress_chunk.chunk_id)
                        if isinstance(allowance_chunk, FinancialChunk):
                            removed_ids.add(allowance_chunk.chunk_id)
                        matched_duplicate_group = True
                        break
                    if matched_duplicate_group:
                        break
                if matched_duplicate_group:
                    break

        without_gross_detail: list[dict[str, object]] = []
        for row in ordered_rows:
            chunk = row.get("chunk")
            if not isinstance(chunk, FinancialChunk) or chunk.chunk_id not in removed_ids:
                without_gross_detail.append(row)

        # If an untagged source row exactly reconciles to the sum of the
        # following same-page rows, treat it as a subtotal and keep its leaves.
        subtotal_chunk_ids: set[str] = set()
        total_chunk = total_row.get("chunk") if total_row else None
        for parent_index, parent in enumerate(without_gross_detail):
            parent_chunk = parent.get("chunk")
            if not isinstance(parent_chunk, FinancialChunk):
                continue
            parent_values = row_values(parent)
            if not parent_values:
                continue
            # Explicit schema parents were already removed above. This pass is
            # for rows whose OCR did not preserve their parent/child IDs.
            followers: list[dict[str, object]] = []
            for child in without_gross_detail[parent_index + 1:parent_index + 16]:
                child_chunk = child.get("chunk")
                if not isinstance(child_chunk, FinancialChunk):
                    break
                if isinstance(total_chunk, FinancialChunk) and child_chunk.chunk_id == total_chunk.chunk_id:
                    break
                if (
                    child_chunk.page != parent_chunk.page
                    or not cls._same_source_table(parent_chunk, child_chunk)
                    or not same_unit(parent, child)
                ):
                    break
                followers.append(child)
            if len(followers) < 2:
                continue
            cumulative = {year: Decimal("0") for year in parent_values}
            child_count = 0
            for child in followers:
                child_chunk = child.get("chunk")
                if not isinstance(child_chunk, FinancialChunk):
                    break
                child_label = cls._normalize_table_label(str(child.get("label") or ""))
                if child_label.startswith("trong do"):
                    continue
                values = row_values(child)
                required_years = set(parent_values)
                if not required_years.issubset(values):
                    break
                for year in required_years:
                    cumulative[year] += values[year]
                child_count += 1
                if child_count < 2:
                    continue
                if all(
                    year in cumulative
                    and abs(cumulative[year] - parent_values[year])
                    <= max(Decimal("1"), abs(parent_values[year]) * Decimal("0.000001"))
                    for year in parent_values
                ):
                    subtotal_chunk_ids.add(parent_chunk.chunk_id)
                    break

        non_subtotals: list[dict[str, object]] = []
        for row in without_gross_detail:
            chunk = row.get("chunk")
            if not isinstance(chunk, FinancialChunk) or chunk.chunk_id not in subtotal_chunk_ids:
                non_subtotals.append(row)
        return non_subtotals


    @staticmethod
    def _same_source_table(left: FinancialChunk, right: FinancialChunk) -> bool:
        """Only derive composition shares when both facts share a source group."""
        if (
            (left.company_code or "").casefold() != (right.company_code or "").casefold()
            or (left.document_id or left.source_file) != (right.document_id or right.source_file)
            or left.report_type != right.report_type
            or left.statement_type != right.statement_type
        ):
            return False

        left_meta = left.metadata if isinstance(left.metadata, dict) else {}
        right_meta = right.metadata if isinstance(right.metadata, dict) else {}
        left_table_id = str(left_meta.get("table_id") or "").strip()
        right_table_id = str(right_meta.get("table_id") or "").strip()
        if left_table_id or right_table_id:
            return bool(left_table_id and right_table_id and left_table_id == right_table_id)

        left_title = " ".join(str(left.table_title or left_meta.get("table_title") or "").casefold().split())
        right_title = " ".join(str(right.table_title or right_meta.get("table_title") or "").casefold().split())
        if left_title or right_title:
            if left_title and right_title and left_title == right_title:
                return True

        left_node = str(left_meta.get("schema_node_id") or "")
        right_node = str(right_meta.get("schema_node_id") or "")
        left_parent = str(left_meta.get("schema_parent_id") or "")
        right_parent = str(right_meta.get("schema_parent_id") or "")
        left_children = {str(value) for value in left_meta.get("schema_children", [])}
        right_children = {str(value) for value in right_meta.get("schema_children", [])}
        if (
            (right_node and right_node in left_children)
            or (left_node and left_node in right_children)
            or (left_node and right_parent == left_node)
            or (right_node and left_parent == right_node)
        ):
            return True
        return left.page is not None and left.page == right.page


    @staticmethod
    def _normalize_table_label(value: str) -> str:
        text = re.sub(r"<[^>]*>", " ", value)
        text = text.replace("**", " ").replace("__", " ")
        text = unicodedata.normalize("NFKD", text.casefold())
        text = "".join(character for character in text if not unicodedata.combining(character))
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

