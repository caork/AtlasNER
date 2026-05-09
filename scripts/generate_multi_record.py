"""Generate 1-to-N multi-record TEST samples from completed annotations.

Takes completed single-record annotations and groups those from adjacent
pages of the same document. Combines their passages and outputs to create
multi-record test samples where one passage yields multiple extraction records.

These are TEST-ONLY samples — not used for training.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ANNOTATION_DIR = Path(__file__).resolve().parent.parent / "data" / "annotation"
TEMPLATES_DIR = ANNOTATION_DIR / "templates"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

FIELD_ORDER = ["equipment", "process", "risk", "risk_level", "prevention", "control"]

# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def load_completed_templates() -> list[dict]:
    """Load all completed annotation templates."""
    templates = []
    for tf in sorted(TEMPLATES_DIR.glob("sgcc_*.json")):
        t = json.loads(tf.read_text())
        if t.get("status") == "completed":
            ann = t.get("annotation", {})
            if ann.get("passage") and ann.get("think_chain") and ann.get("output_json"):
                templates.append(t)
    return templates


def get_primary_page(template: dict) -> tuple[str, int]:
    """Get (doc_name, page_num) of the top-matched page."""
    refs = template.get("reference", {}).get("page_refs", [])
    if refs:
        return refs[0]["doc_name"], refs[0]["page_num"]
    return "unknown", 0


def group_by_proximity(
    templates: list[dict],
    max_page_gap: int = 3,
) -> list[list[dict]]:
    """Group templates that share the same document and nearby pages.

    Groups templates where the primary matched page is within max_page_gap
    pages of each other in the same document.
    """
    # Group by document
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for t in templates:
        doc_name, page_num = get_primary_page(t)
        by_doc[doc_name].append((page_num, t))

    groups: list[list[dict]] = []
    for doc_name, items in by_doc.items():
        if doc_name == "unknown":
            continue

        items.sort(key=lambda x: x[0])

        current_group: list[dict] = [items[0][1]]
        current_max_page = items[0][0]

        for page_num, t in items[1:]:
            if page_num - current_max_page <= max_page_gap:
                current_group.append(t)
                current_max_page = page_num
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [t]
                current_max_page = page_num

        if len(current_group) >= 2:
            groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# Multi-record sample building
# ---------------------------------------------------------------------------


def build_multi_record_sample(
    group: list[dict],
    sample_idx: int,
) -> dict:
    """Build a 1-to-N SFT sample from a group of adjacent single annotations."""
    # Combine passages
    passages = []
    records = []
    think_parts = []

    for t in group:
        ann = t["annotation"]
        passages.append(ann["passage"].strip())

        ordered = {}
        for k in FIELD_ORDER:
            if k in ann["output_json"]:
                ordered[k] = ann["output_json"][k]
        records.append(ordered)

        think_parts.append(ann["think_chain"].strip())

    combined_passage = "\n\n".join(passages)

    # Build combined think chain
    combined_think = "文档包含多条风险管控记录，逐一分析：\n\n"
    for i, (think, record) in enumerate(zip(think_parts, records), 1):
        equip = record.get("equipment", "未知设备")
        proc = record.get("process", "未知工序")
        combined_think += f"【记录{i}】{equip} - {proc}\n{think}\n\n"

    source_files = list({t["source"]["excel_file"] for t in group})
    source_ids = [t["id"] for t in group]

    return {
        "id": f"sgcc_multi_{sample_idx:04d}",
        "system": SYSTEM_PROMPT,
        "input": combined_passage,
        "think": combined_think.strip(),
        "output": records,
        "metadata": {
            "quality": "gold",
            "type": "multi_record",
            "record_count": len(records),
            "source_ids": source_ids,
            "source_files": source_files,
            "input_char_count": len(combined_passage),
            "output_char_count": len(json.dumps(records, ensure_ascii=False)),
            "split": "test",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    max_records_per_sample: int = 5,
    max_samples: int = 100,
    seed: int = 42,
) -> None:
    print(f"Loading completed templates from {TEMPLATES_DIR}")
    templates = load_completed_templates()
    print(f"  Completed templates: {len(templates)}")

    if not templates:
        print("  No completed templates. Run annotation first.")
        return

    # Group by proximity
    groups = group_by_proximity(templates)
    print(f"  Proximity groups (2+ templates): {len(groups)}")

    # Generate multi-record samples
    rng = random.Random(seed)
    samples: list[dict] = []

    for group in groups:
        # Generate samples of different sizes (2, 3, 4, ...)
        n = len(group)
        for size in range(2, min(n + 1, max_records_per_sample + 1)):
            if len(samples) >= max_samples:
                break
            # Take consecutive sub-groups
            for start in range(0, n - size + 1):
                if len(samples) >= max_samples:
                    break
                sub = group[start : start + size]
                sample = build_multi_record_sample(sub, len(samples))
                samples.append(sample)

    rng.shuffle(samples)
    if len(samples) > max_samples:
        samples = samples[:max_samples]

    print(f"\n=== Generated {len(samples)} multi-record test samples ===")

    # Stats
    from collections import Counter
    size_dist = Counter(s["metadata"]["record_count"] for s in samples)
    for size in sorted(size_dist):
        print(f"  {size} records: {size_dist[size]} samples")

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "test_multi_record.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        max_records_per_sample=args.max_records,
        max_samples=args.max_samples,
        seed=args.seed,
    )
