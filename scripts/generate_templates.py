"""Generate annotation templates for human annotators.

For each matched Excel row, produces a JSON template with:
- Pre-filled output fields (from Excel data)
- Blank passage field (human transcribes from PDF image)
- Blank think chain field (human writes reasoning)
- Reference info (matched pages, source file)

The annotator workflow:
1. Look at rendered page images
2. Write the input passage by transcribing relevant text from the image
3. Write a think chain explaining the extraction reasoning
4. Review and adjust the pre-filled output JSON if needed
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GDRIVE = Path(
    "/Users/kaitaocao/Library/CloudStorage/"
    "GoogleDrive-barrientosangie599@gmail.com/My Drive/"
    "AtlasNERDataset/国家电网文件"
)
LABELED_ROWS = GDRIVE / "清洗输出" / "labeled_excel_rows.json"
ANNOTATION_DIR = Path(__file__).resolve().parent.parent / "data" / "annotation"
MANIFEST_PATH = ANNOTATION_DIR / "annotation_manifest.json"
TEMPLATES_DIR = ANNOTATION_DIR / "templates"

# ---------------------------------------------------------------------------
# Schema (shared with build_sft_dataset.py)
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

FIELD_ORDER = ["equipment", "process", "risk", "risk_level", "prevention", "control"]

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
# Helpers
# ---------------------------------------------------------------------------


def clean_value(v: object, is_long_text: bool = False) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    if not is_long_text:
        s = re.sub(r"\s+", "", s)
    else:
        s = re.sub(r"(?<!\n)\n(?!\n)", "", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" ?\n ?", "\n", s)
        s = s.strip()
    return s


def _merge_risk_values(existing: str, new: str) -> str:
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
    unified: dict[str, str] = {}
    for raw_col, unified_key in COLUMN_MAP.items():
        is_long = unified_key in LONG_TEXT_FIELDS
        val = clean_value(row.get(raw_col), is_long_text=is_long)
        if val is None:
            continue
        if unified_key in unified:
            existing = unified[unified_key]
            if unified_key == "risk":
                unified[unified_key] = _merge_risk_values(existing, val)
            elif val not in existing:
                unified[unified_key] = existing + "；" + val
        else:
            unified[unified_key] = val
    return unified


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def build_template(row: dict, manifest_entry: dict, row_index: int) -> dict:
    """Build an annotation template for one row."""
    fields = extract_unified_fields(row)

    # Pre-fill output JSON (annotator can edit)
    output_json = {}
    for k in FIELD_ORDER:
        if k in fields:
            output_json[k] = fields[k]

    # Get page references
    page_refs = []
    for doc_match in manifest_entry.get("doc_matches", []):
        for page in doc_match.get("pages", [])[:3]:
            page_refs.append({
                "doc_name": doc_match["doc_name"],
                "page_num": page["page_num"],
                "match_score": page["score"],
            })

    # Get rendered image paths from render_index if available
    image_paths = []
    render_index_path = ANNOTATION_DIR / "render_index.json"
    if render_index_path.exists():
        render_index = json.loads(render_index_path.read_text())
        for ri in render_index:
            if ri["row_index"] == row_index and ri.get("image_path"):
                image_paths.append(ri["image_path"])

    return {
        "id": f"sgcc_{row_index:06d}",
        "status": "pending",
        "source": {
            "excel_file": row.get("_source", "unknown"),
            "sheet": row.get("_sheet", "unknown"),
            "row": row.get("_row", 0),
            "tier": row.get("_tier", "unknown"),
        },
        "reference": {
            "page_refs": page_refs,
            "image_paths": image_paths,
        },
        "annotation": {
            "passage": "",
            "think_chain": "",
            "output_json": output_json,
            "annotator_notes": "",
        },
        "sft_format": {
            "system": SYSTEM_PROMPT,
            "user_template": "请从以下文档段落中提取结构化风险管控记录：\n\n<document>\n{passage}\n</document>",
            "assistant_template": "<think>\n{think_chain}\n</think>\n\n```json\n{output_json}\n```",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(row_indices: list[int] | None = None) -> None:
    print(f"Loading data...")
    with LABELED_ROWS.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest["entries"]
    print(f"  Rows: {len(rows)}, Manifest entries: {len(entries)}")

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    # Filter
    if row_indices is not None:
        target = set(row_indices)
        entries_to_process = [e for e in entries if e["row_index"] in target]
    else:
        entries_to_process = [e for e in entries if e["status"] == "matched"]

    print(f"  Generating {len(entries_to_process)} templates...")

    templates = []
    for entry in entries_to_process:
        idx = entry["row_index"]
        row = rows[idx]
        template = build_template(row, entry, idx)
        templates.append(template)

    # Write individual templates (for annotator use)
    for t in templates:
        path = TEMPLATES_DIR / f"{t['id']}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=2)

    # Write batch index
    index_path = TEMPLATES_DIR / "_index.json"
    index = {
        "total": len(templates),
        "templates": [
            {
                "id": t["id"],
                "status": t["status"],
                "equipment": t["annotation"]["output_json"].get("equipment", ""),
                "process": t["annotation"]["output_json"].get("process", ""),
                "field_count": len(t["annotation"]["output_json"]),
                "page_refs": len(t["reference"]["page_refs"]),
            }
            for t in templates
        ],
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n=== Results ===")
    print(f"  Templates: {len(templates)}")
    print(f"  Dir: {TEMPLATES_DIR}")
    print(f"  Index: {index_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="*", type=int, help="Specific row indices")
    args = parser.parse_args()

    main(row_indices=args.rows)
