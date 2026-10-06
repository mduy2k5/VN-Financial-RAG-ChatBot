"""Build a content-hashed manifest for the shared OCR annual-report corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "dataset" / "ocr_annual_bench"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "corpus_manifest.json"
NAME_PATTERN = re.compile(
    r"^(?P<issuer>[A-Z]+)_.*?_(?P<year>20\d{2})_Kiemtoan_"
    r"(?P<basis>Congtyme|Hopnhat)_extracted\.txt$",
    re.IGNORECASE,
)
PAGE_PATTERN = re.compile(rb"(?m)^===== PAGE (\d+) =====[ \t]*\r?$")


def describe(path: Path, corpus: Path) -> dict[str, object]:
    raw = path.read_bytes()
    match = NAME_PATTERN.match(path.name)
    if match:
        issuer = match.group("issuer").upper()
        fiscal_year: int | None = int(match.group("year"))
        basis = "separate" if match.group("basis").casefold() == "congtyme" else "consolidated"
    elif path.name.casefold().startswith("vpbank 31.12.2023"):
        issuer = "VPB"
        fiscal_year = 2023
        basis = "separate"
    else:
        # Keep outlier source files in the manifest; metadata needs human review.
        issuer = None
        fiscal_year = None
        basis = None
    pages = [int(value) for value in PAGE_PATTERN.findall(raw)]
    return {
        "file_name": path.name,
        "relative_path": path.relative_to(ROOT).as_posix(),
        "issuer": issuer,
        "fiscal_year": fiscal_year,
        "report_basis": basis,
        "metadata_review_required": match is None,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "page_markers": len(pages),
        "first_page_marker": min(pages) if pages else None,
        "last_page_marker": max(pages) if pages else None,
        "utf8_chars": len(raw.decode("utf-8-sig", errors="replace")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        parser.error(f"Corpus directory does not exist: {corpus}")
    files = sorted(path for path in corpus.glob("*.txt") if path.is_file())
    if not files:
        parser.error(f"No top-level .txt files found in {corpus}")

    entries = [describe(path, corpus) for path in files]
    summary: dict[str, object] = {
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "issuers": sorted({str(item["issuer"]) for item in entries if item["issuer"]}),
        "years": sorted({int(item["fiscal_year"]) for item in entries if item["fiscal_year"]}),
        "report_bases": {
            basis: sum(item["report_basis"] == basis for item in entries)
            for basis in ("separate", "consolidated")
        },
        "files_requiring_metadata_review": [
            item["file_name"] for item in entries if item["metadata_review_required"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "corpus_id": "ocr_annual_bench",
                "summary": summary,
                "files": entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
