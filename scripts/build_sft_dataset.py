"""Convert labeled_excel_rows.json → Gold SFT dataset (train/dev/test).

v2: Uses original PDF/docx passages as input instead of reverse-engineered text.
    Fixes: risk dedup, stratified split, think chain quality, 1-to-N support.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import random

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GDRIVE = Path(
    "/Users/kaitaocao/Library/CloudStorage/"
    "GoogleDrive-barrientosangie599@gmail.com/My Drive/"
    "AtlasNERDataset/国家电网文件"
)
LABELED_ROWS = GDRIVE / "清洗输出" / "labeled_excel_rows.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"

# PDF paths for passage extraction
PDF_89 = (
    GDRIVE / "模版及案例" / "规章制度文件"
    / "1-1-2-7.国家电网有限公司关于进一步加强生产现场作业风险管控工作的通知"
    "（国家电网设备〔2022〕89号）.pdf"
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

COLUMN_MAP: dict[str, str] = {
    "设备": "equipment",
    "设备类型": "equipment",
    "操作对象": "equipment",
    "工序": "process",
    "操作行为": "process",
    "工作内容": "process",
    "主要风险点": "risk",
    "基本风险点": "risk",
    "存在的主要风险": "risk",
    "风险类型": "risk",
    "作业风险类型": "risk",
    "风险可能导致的后果": "risk",
    "风险等级": "risk_level",
    "工序风险库等级": "risk_level",
    "风险防范措施": "prevention",
    "预控措施": "prevention",
    "对应防范措施": "prevention",
    "工艺管控措施": "control",
    "质量管控措施": "control",
}

IGNORED_COLUMNS = {"序号", "违章库目录", "措施类型"}

FIELD_ORDER = ["equipment", "process", "risk", "risk_level", "prevention", "control"]

FIELD_CN = {
    "equipment": "设备",
    "process": "工序",
    "risk": "风险描述",
    "risk_level": "风险等级",
    "prevention": "防范措施",
    "control": "管控措施",
}

SYSTEM_PROMPT = (
    "你是电力领域结构化信息抽取专家。给定电力文档段落，提取风险管控记录。"
    "仅输出文档中确实存在的字段，不臆造信息。输出严格JSON格式。\n\n"
    "可提取的字段：\n"
    "- equipment: 设备名称\n"
    "- process: 工序/作业内容\n"
    "- risk: 风险描述（风险类型或可能导致的后果）\n"
    "- risk_level: 风险等级\n"
    "- prevention: 防范措施\n"
    "- control: 管控措施"
)

LONG_TEXT_FIELDS = {"prevention", "control"}

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_value(v: object, is_long_text: bool = False) -> str | None:
    """Return stripped string or None if empty/nan."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    if not is_long_text:
        s = re.sub(r"(?<!\n)\n(?!\n)", "", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = s.strip()
    else:
        s = re.sub(r"(?<!\n)\n(?!\n)", "", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" ?\n ?", "\n", s)
        s = s.strip()
    return s


# ---------------------------------------------------------------------------
# Field extraction with risk dedup
# ---------------------------------------------------------------------------


def _merge_risk_values(existing: str, new: str) -> str:
    """Merge risk values with deduplication."""
    sep_pattern = r"[、；，\n]+"
    existing_parts = [p.strip() for p in re.split(sep_pattern, existing) if p.strip()]
    new_parts = [p.strip() for p in re.split(sep_pattern, new) if p.strip()]
    seen: set[str] = set()
    merged: list[str] = []
    for p in existing_parts + new_parts:
        if p not in seen:
            seen.add(p)
            merged.append(p)
    return "、".join(merged)


def extract_unified_fields(row: dict) -> dict[str, str]:
    """Map raw Excel columns to the unified 6-field schema."""
    unified: dict[str, str] = {}
    for raw_col, unified_key in COLUMN_MAP.items():
        is_long = unified_key in LONG_TEXT_FIELDS
        val = clean_value(row.get(raw_col), is_long_text=is_long)
        if val is None:
            continue
        if unified_key in unified:
            if unified_key == "risk":
                unified[unified_key] = _merge_risk_values(unified[unified_key], val)
            elif unified_key in LONG_TEXT_FIELDS:
                existing = unified[unified_key]
                if val not in existing:
                    unified[unified_key] = existing + "\n" + val
            else:
                existing = unified[unified_key]
                if val != existing and val not in existing:
                    unified[unified_key] = existing + "、" + val
        else:
            unified[unified_key] = val
    return unified


# ---------------------------------------------------------------------------
# PDF passage extraction
# ---------------------------------------------------------------------------


def load_pdf_pages(pdf_path: Path) -> list[str]:
    """Load all pages from a PDF as text strings."""
    import PyPDF2

    pages: list[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


# Repeated table header pattern to strip from page text
_TABLE_HEADER_RE = re.compile(
    r"序号\s*设备\s*工序\s*风险可能导\s*致的后果\s*工序风险\s*库等级\s*风险防范措施\s*工艺管控措施"
)


def _strip_header(text: str) -> str:
    return _TABLE_HEADER_RE.sub("", text).strip()


def _normalize_for_match(text: str) -> str:
    """Remove all whitespace for fuzzy substring matching."""
    return re.sub(r"\s+", "", text)


def build_pdf_passage_index(
    pages: list[str],
    window_chars: int = 2000,
) -> list[dict]:
    """Build overlapping passage windows from PDF pages.

    Returns list of {text, start_page, end_page, clean_text}.
    """
    cleaned_pages = []
    for i, page_text in enumerate(pages):
        cleaned = _strip_header(page_text)
        if cleaned:
            cleaned_pages.append({"page": i, "text": cleaned})

    passages: list[dict] = []
    i = 0
    while i < len(cleaned_pages):
        combined = cleaned_pages[i]["text"]
        start_page = cleaned_pages[i]["page"]
        end_page = start_page
        j = i + 1
        while j < len(cleaned_pages) and len(combined) < window_chars:
            combined += "\n" + cleaned_pages[j]["text"]
            end_page = cleaned_pages[j]["page"]
            j += 1
        passages.append({
            "text": combined[:window_chars + 500],
            "start_page": start_page,
            "end_page": end_page,
            "normalized": _normalize_for_match(combined[:window_chars + 500]),
        })
        i += 1

    return passages


def find_passage_for_row(
    row: dict,
    fields: dict[str, str],
    passages: list[dict],
) -> str | None:
    """Find the best PDF passage containing this row's content.

    Searches using the most distinctive field value (prevention > control > risk).
    Returns the raw passage text or None.
    """
    search_keys: list[str] = []
    for field in ["prevention", "control", "risk", "process", "equipment"]:
        val = fields.get(field, "")
        if len(val) >= 15:
            search_keys.append(_normalize_for_match(val[:60]))

    if not search_keys:
        return None

    for key in search_keys:
        for passage in passages:
            if key in passage["normalized"]:
                return passage["text"]
    return None


# ---------------------------------------------------------------------------
# Fallback: build document-style passage from fields
# ---------------------------------------------------------------------------


def build_document_passage(fields: dict[str, str]) -> str:
    """Build a document-like passage from unified fields (fallback)."""
    parts: list[str] = []

    header_parts: list[str] = []
    if "equipment" in fields:
        header_parts.append(fields["equipment"])
    if "process" in fields:
        header_parts.append(fields["process"])
    if header_parts:
        parts.append(" ".join(header_parts))

    if "risk" in fields:
        parts.append(f"主要风险：{fields['risk']}")

    if "risk_level" in fields:
        parts.append(f"风险等级：{fields['risk_level']}")

    if "prevention" in fields:
        parts.append(f"风险防范措施：\n{fields['prevention']}")

    if "control" in fields:
        parts.append(f"管控措施：\n{fields['control']}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Think chain generation
# ---------------------------------------------------------------------------


def count_measure_items(text: str) -> int:
    """Count numbered items in a measure list."""
    return len(re.findall(r"(?:^|\n)\s*\d+[\.\、．]\s*", text))


def build_think_chain(fields: dict[str, str], passage: str) -> str:
    """Generate a think chain grounded in the passage text."""
    lines: list[str] = ["分析这段文档的结构和内容："]
    step = 1

    for field_key in FIELD_ORDER:
        cn_name = FIELD_CN[field_key]
        if field_key in fields:
            val = fields[field_key]
            if field_key in ("prevention", "control"):
                n_items = count_measure_items(val)
                if n_items > 1:
                    lines.append(
                        f"{step}. [{cn_name}] 文档中包含{n_items}条编号措施。"
                        f" → 完整提取"
                    )
                else:
                    preview = val[:40].replace("\n", " ")
                    lines.append(
                        f'{step}. [{cn_name}] 找到措施内容：'
                        f'"{preview}..." → 完整提取'
                    )
            elif len(val) > 30:
                lines.append(
                    f'{step}. [{cn_name}] 文档中提到'
                    f'"{val[:25]}..." → {field_key} = "{val[:25]}..."'
                )
            else:
                lines.append(
                    f'{step}. [{cn_name}] 文档中明确标注'
                    f' → {field_key} = "{val}"'
                )
        else:
            lines.append(f"{step}. [{cn_name}] 文档中未找到相关信息，跳过")
        step += 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SFT sample builder
# ---------------------------------------------------------------------------


def build_sft_sample(
    row: dict,
    fields: dict[str, str],
    passage: str,
    idx: int,
    passage_source: str = "pdf",
) -> dict:
    """Build one SFT conversation sample."""
    think = build_think_chain(fields, passage)

    output_json = json.dumps(fields, ensure_ascii=False, indent=2)
    assistant_content = f"<think>\n{think}\n</think>\n\n```json\n{output_json}\n```"

    source_file = row.get("_source", "unknown")
    source_sheet = row.get("_sheet", "unknown")
    source_row = row.get("_row", 0)

    sample_id = f"sgcc_{idx:06d}"

    return {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请从以下文档段落中提取结构化风险管控记录：\n\n"
                    f"<document>\n{passage}\n</document>"
                ),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "quality": "gold",
            "source_file": source_file,
            "source_sheet": source_sheet,
            "source_row": source_row,
            "tier": row.get("_tier", "unknown"),
            "field_count": len(fields),
            "input_char_count": len(passage),
            "output_char_count": len(output_json),
            "passage_source": passage_source,
        },
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def _field_count_bucket(fc: int) -> str:
    if fc <= 2:
        return "low"
    if fc <= 4:
        return "mid"
    return "high"


def assign_splits(
    samples: list[dict],
    dev_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Stratified split: each stratum (source_file x field_count_bucket)
    is independently split 80/10/10. Each sample is treated individually
    (no contiguity grouping) to ensure balanced distribution."""
    rng = random.Random(seed)

    # Assign each sample a stratum key
    strata: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        m = s["metadata"]
        fc_bucket = _field_count_bucket(m["field_count"])
        stratum_key = f"{m['source_file']}_{fc_bucket}"
        strata[stratum_key].append(s)

    # Within each stratum, shuffle then split by ratio
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    for stratum_key, stratum_samples in sorted(strata.items()):
        rng.shuffle(stratum_samples)
        n = len(stratum_samples)
        n_test = max(1, round(n * test_ratio))
        n_dev = max(1, round(n * dev_ratio))
        splits["test"].extend(stratum_samples[:n_test])
        splits["dev"].extend(stratum_samples[n_test : n_test + n_dev])
        splits["train"].extend(stratum_samples[n_test + n_dev :])

    for split_name, split_samples in splits.items():
        for s in split_samples:
            s["metadata"]["split"] = split_name

    return splits


# ---------------------------------------------------------------------------
# 1-to-N: merge contiguous rows into multi-record samples
# ---------------------------------------------------------------------------


def build_multi_record_samples(
    samples: list[dict],
    merge_count: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Build 1-to-N training samples by merging N contiguous single-record
    samples into one multi-record sample. Only applied to train split."""
    rng = random.Random(seed)

    by_sheet: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        m = s["metadata"]
        if m.get("split") != "train":
            continue
        key = f"{m['source_file']}_{m['source_sheet']}"
        by_sheet[key].append(s)

    multi_samples: list[dict] = []
    idx = 0

    for key, sheet_samples in by_sheet.items():
        sheet_samples.sort(key=lambda s: s["metadata"]["source_row"])

        for i in range(0, len(sheet_samples) - merge_count + 1, merge_count):
            group = sheet_samples[i : i + merge_count]

            passages: list[str] = []
            all_fields: list[dict[str, str]] = []
            for s in group:
                user_msg = s["messages"][1]["content"]
                doc_start = user_msg.index("<document>\n") + len("<document>\n")
                doc_end = user_msg.index("\n</document>")
                passages.append(user_msg[doc_start:doc_end])

                assistant_msg = s["messages"][2]["content"]
                json_str = assistant_msg.split("```json\n")[1].split("\n```")[0]
                all_fields.append(json.loads(json_str))

            combined_passage = "\n\n---\n\n".join(passages)

            think_lines = [
                f"这段文档包含{len(all_fields)}条独立的风险管控记录，"
                "需要分别提取："
            ]
            for ri, fields in enumerate(all_fields, 1):
                present = [FIELD_CN[k] for k in FIELD_ORDER if k in fields]
                think_lines.append(
                    f"记录{ri}：包含{'/'.join(present)}共{len(fields)}个字段"
                )
            think = "\n".join(think_lines)

            output_json = json.dumps(all_fields, ensure_ascii=False, indent=2)
            assistant_content = (
                f"<think>\n{think}\n</think>\n\n```json\n{output_json}\n```"
            )

            sample_id = f"sgcc_multi_{idx:06d}"
            multi_samples.append({
                "id": sample_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "请从以下文档段落中提取所有结构化风险管控记录"
                            "（可能包含多条记录）：\n\n"
                            f"<document>\n{combined_passage}\n</document>"
                        ),
                    },
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "quality": "gold",
                    "source_file": group[0]["metadata"]["source_file"],
                    "source_sheet": group[0]["metadata"]["source_sheet"],
                    "source_rows": [s["metadata"]["source_row"] for s in group],
                    "record_count": len(all_fields),
                    "field_count": sum(len(f) for f in all_fields),
                    "input_char_count": len(combined_passage),
                    "output_char_count": len(output_json),
                    "split": "train",
                    "passage_source": "merged",
                },
            })
            idx += 1

    rng.shuffle(multi_samples)
    return multi_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading labeled rows from {LABELED_ROWS}")
    with LABELED_ROWS.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    print(f"  Total rows: {len(rows)}")

    # --- Load PDF for passage extraction ---
    print(f"\nLoading PDF: {PDF_89.name}")
    pdf_pages = load_pdf_pages(PDF_89)
    print(f"  Pages: {len(pdf_pages)}")
    passages = build_pdf_passage_index(pdf_pages, window_chars=2000)
    print(f"  Passage windows: {len(passages)}")

    # --- Convert to unified schema ---
    samples: list[dict] = []
    skipped = 0
    pdf_matched = 0
    fallback_used = 0

    for idx, row in enumerate(rows):
        fields = extract_unified_fields(row)
        if not fields:
            skipped += 1
            continue

        passage = find_passage_for_row(row, fields, passages)
        if passage:
            pdf_matched += 1
            source = "pdf"
        else:
            passage = build_document_passage(fields)
            fallback_used += 1
            source = "fallback"

        sample = build_sft_sample(row, fields, passage, idx, passage_source=source)
        samples.append(sample)

    print(f"\n  Converted: {len(samples)}, Skipped (empty): {skipped}")
    print(f"  PDF-matched: {pdf_matched}, Fallback: {fallback_used}")

    # --- Statistics ---
    field_counts = Counter()
    field_presence = Counter()
    for s in samples:
        fc = s["metadata"]["field_count"]
        field_counts[fc] += 1
        msg = s["messages"][2]["content"]
        parsed = json.loads(msg.split("```json\n")[1].split("\n```")[0])
        for k in parsed:
            field_presence[k] += 1

    print("\n=== Field count distribution ===")
    for fc in sorted(field_counts):
        print(f"  {fc} fields: {field_counts[fc]} samples")

    print("\n=== Field presence ===")
    for field in FIELD_ORDER:
        print(f"  {field}: {field_presence.get(field, 0)} samples")

    # --- Stratified split ---
    splits = assign_splits(samples)
    print(f"\n=== Splits ===")
    for split_name in ("train", "dev", "test"):
        n = len(splits[split_name])
        fc_dist = Counter(
            _field_count_bucket(s["metadata"]["field_count"])
            for s in splits[split_name]
        )
        src_dist = Counter(
            s["metadata"]["source_file"] for s in splits[split_name]
        )
        print(f"  {split_name}: {n} samples")
        print(f"    field_count_buckets: {dict(fc_dist)}")
        print(f"    sources: {dict(src_dist)}")

    # --- Build multi-record samples (train only) ---
    all_single = []
    for split_samples in splits.values():
        all_single.extend(split_samples)
    multi_samples = build_multi_record_samples(all_single, merge_count=2)
    print(f"\n=== Multi-record samples (1-to-N) ===")
    print(f"  Generated: {len(multi_samples)} merged samples")

    # --- Write output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_samples in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        write_samples = split_samples[:]
        if split_name == "train":
            write_samples.extend(multi_samples)
        with out_path.open("w", encoding="utf-8") as f:
            for s in write_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Wrote {out_path} ({len(write_samples)} samples)")

    # --- Write manifest ---
    reports_dir = OUTPUT_DIR.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total_single_samples": len(samples),
        "total_multi_samples": len(multi_samples),
        "splits": {k: len(v) for k, v in splits.items()},
        "field_presence": dict(field_presence),
        "field_count_distribution": {
            str(k): v for k, v in sorted(field_counts.items())
        },
        "pdf_matched": pdf_matched,
        "fallback_used": fallback_used,
    }
    # Per-split distribution check
    for split_name, split_samples in splits.items():
        fc_pct = Counter()
        for s in split_samples:
            fc_pct[_field_count_bucket(s["metadata"]["field_count"])] += 1
        total = len(split_samples)
        manifest[f"{split_name}_distribution"] = {
            k: f"{v}/{total} ({v/total*100:.1f}%)" for k, v in fc_pct.items()
        }

    with (reports_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote {reports_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
