import re
import json
import html
from pathlib import Path
from typing import Optional, Dict, List

class CompanyExtractor:
    """
    Trích xuất mã chứng khoán (Ticker) và tên doanh nghiệp từ văn bản/tên file.
    Sử dụng từ điển `company_aliases.json` và regex bắt ticker 3 chữ cái.
    """

    COMMON_STOP_WORDS = {
        "B02", "B03", "B04", "B05", "VND", "USD", "EUR", "STT",
        "THU", "SAU", "BAO", "NAM", "TẠI", "CHO", "VAY", "KHO", "CỦA", "BCTC",
        "CTCP", "TNHH", "REU", "LUU", "HOA", "PHI", "BAN",
        "ROA", "ROE", "NIM", "CIR", "CAR", "EPS", "LCR", "LDR",
    }

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "company_aliases.json"

        self.aliases: Dict[str, List[str]] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.aliases = json.load(f)
        self._configured_aliases = {
            ticker: list(alias_list) for ticker, alias_list in self.aliases.items()
        }

    def set_dynamic_aliases(self, aliases: Dict[str, List[str]]) -> None:
        """Overlay legal names learned from documents already in the corpus."""
        merged = {
            ticker: list(alias_list)
            for ticker, alias_list in self._configured_aliases.items()
        }
        for ticker, alias_list in aliases.items():
            ticker = ticker.strip().upper()
            if not ticker:
                continue
            configured = merged.setdefault(ticker, [])
            seen = {alias.casefold() for alias in configured}
            for alias in [ticker, *alias_list]:
                alias = alias.strip()
                if alias.upper() == ticker and ticker in self.COMMON_STOP_WORDS:
                    continue
                if alias and alias.casefold() not in seen:
                    configured.append(alias)
                    seen.add(alias.casefold())
        self.aliases = merged

    @staticmethod
    def extract_legal_name_aliases(text: str) -> List[str]:
        """Read issuer-name variants from the opening pages of a report."""
        report_suffixes = re.compile(
            r"\b(?:báo cáo tài chính|financial statements|annual report|"
            r"báo cáo thường niên|thuyết minh báo cáo)\b",
            re.IGNORECASE,
        )
        legal_entity = re.compile(
            r"^(?:ngân hàng\b|công ty\b|tập đoàn\b|tổng công ty\b|"
            r"joint stock\b|securities\b|bank\b|corporation\b|company\b)",
            re.IGNORECASE,
        )
        legal_form = re.compile(
            r"\b(?:tmcp|thương mại cổ phần|cổ phần|tnhh|trách nhiệm hữu hạn|"
            r"joint stock|limited|incorporated|corporation|tập đoàn|tổng công ty)\b",
            re.IGNORECASE,
        )
        excluded = re.compile(
            r"\b(?:ngân hàng nhà nước|state bank of vietnam|kiểm toán|auditor|"
            r"địa chỉ|mã số doanh nghiệp|website|điện thoại)\b",
            re.IGNORECASE,
        )

        names: List[str] = []
        seen_names: set[str] = set()
        for raw_line in text[:20_000].splitlines()[:240]:
            line = html.unescape(re.sub(r"<[^>]*>", " ", raw_line))
            line = re.sub(r"^\s*(?:#+|[-•>]+)\s*", "", line)
            line = re.sub(r"[*_`]+", "", line).strip()
            line = re.sub(
                r"^(?:tên (?:công ty|doanh nghiệp)|company name|issuer|entity|"
                r"công ty|tổ chức phát hành)\s*[:：]\s*",
                "",
                line,
                flags=re.IGNORECASE,
            )
            suffix = report_suffixes.search(line)
            if suffix:
                line = line[:suffix.start()].strip(" -–—|:;")
            line = re.sub(r"\s*[\(\[].*$", "", line)
            line = re.sub(r"\s+", " ", line).strip(" -–—|:;.,")
            if not 8 <= len(line) <= 180:
                continue
            if (
                excluded.search(line)
                or not legal_entity.search(line)
                or not legal_form.search(line)
            ):
                continue
            normalized_name = line.casefold()
            if normalized_name not in seen_names:
                names.append(line)
                seen_names.add(normalized_name)
        return names

    @classmethod
    def extract_primary_legal_name(cls, text: str) -> Optional[str]:
        """Read the first issuer legal name, for callers needing one value."""
        aliases = cls.extract_legal_name_aliases(text)
        return aliases[0] if aliases else None

    def extract(self, text: str, fallback_file_name: str = "") -> Optional[str]:
        matches = self.extract_all(text, fallback_file_name=fallback_file_name)
        return matches[0] if matches else None

    def extract_all(self, text: str, fallback_file_name: str = "") -> List[str]:
        """Return every recognized company code in order of first appearance."""
        cleaned_fallback = fallback_file_name.replace("_", " ").replace("-", " ")
        source_text = f"{cleaned_fallback} {text}"
        text_upper = source_text.upper()

        positions: Dict[str, tuple[int, int]] = {}
        alias_matches: List[tuple[int, int, str]] = []

        # 1. Match tickers and full-name aliases case-insensitively.
        for ticker, alias_list in self.aliases.items():
            aliases = list(dict.fromkeys([ticker, *alias_list]))
            for alias in aliases:
                if alias.upper() == ticker and ticker in self.COMMON_STOP_WORDS:
                    continue
                alias_matches.extend(
                    (match.start(), match.end(), ticker)
                    for match in re.finditer(rf"\b{re.escape(alias.upper())}\b", text_upper)
                )

        # Prefer the longest alias when codes overlap (for example, ACB inside
        # the company name "ACB Securities"), then suppress its shorter span.
        selected_aliases: List[tuple[int, int, str]] = []
        for start, end, ticker in sorted(
            alias_matches, key=lambda item: (item[0], -(item[1] - item[0]), item[2])
        ):
            if any(start < selected_end and end > selected_start
                   for selected_start, selected_end, _ in selected_aliases):
                continue
            selected_aliases.append((start, end, ticker))
            candidate = (start, -(end - start))
            positions[ticker] = min(positions.get(ticker, candidate), candidate)

        # 2. Regex tìm ticker 3 ký tự in hoa đứng độc lập
        # Scan the original casing so ordinary Vietnamese words such as
        # "thu" and "sau" are not mistaken for ticker symbols.
        matches = re.finditer(r"(?<!\w)([A-Z]{3})(?!\w)", source_text)
        for match in matches:
            ticker = match.group(1)
            overlaps_alias = any(
                match.start() < end and match.end() > start
                for start, end, _ in selected_aliases
            )
            if ticker not in self.COMMON_STOP_WORDS and not overlaps_alias:
                candidate = (match.start(), -(match.end() - match.start()))
                positions[ticker] = min(positions.get(ticker, candidate), candidate)

        return [
            ticker
            for ticker, _ in sorted(
                positions.items(), key=lambda item: (item[1][0], item[1][1], item[0])
            )
        ]
