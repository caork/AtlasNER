"""Assemble human-annotated templates into final SFT dataset.

Reads completed annotation templates, validates them, assembles into
the SFT conversation format, and applies stratified train/dev/test split.
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

ANNOTATION_DIR = Path(__file__).resolve().parent.parent / "data" / "annotation"
TEMPLATES_DIR = ANNOTATION_DIR / "templates"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"

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
# Validation
# ---------------------------------------------------------------------------


class ValidationError:
    def __init__(self, template_id: str, field: str, message: str):
        self.template_id = template_id
        self.field = field
        self.message = message

    def __repr__(self) -> str:
        return f"{self.template_id}: [{self.field}] {self.message}"


def validate_template(template: dict) -> list[ValidationError]:
    """Validate a completed annotation template."""
    errors: list[ValidationError] = []
    tid = template.get("id", "unknown")
    ann = template.get("annotation", {})

    passage = ann.get("passage", "").strip()
    if not passage:
        errors.append(ValidationError(tid, "passage", "Empty passage"))
    elif len(passage) < 20:
        errors.append(ValidationError(tid, "passage", f"Passage too short ({len(passage)} chars)"))

    think = ann.get("think_chain", "").strip()
    if not think:
        errors.append(ValidationError(tid, "think_chain", "Empty think chain"))
    elif len(think) < 10:
        errors.append(ValidationError(tid, "think_chain", f"Think chain too short ({len(think)} chars)"))

    output = ann.get("output_json", {})
    # Only consider standard fields for validation
    standard_output = {k: v for k, v in output.items() if k in FIELD_ORDER}
    if not standard_output:
        errors.append(ValidationError(tid, "output_json", "No standard fields in output JSON"))
    else:
        for key, val in standard_output.items():
            if not isinstance(val, str) or not val.strip():
                errors.append(ValidationError(tid, "output_json", f"Empty value for {key}"))

    return errors


# ---------------------------------------------------------------------------
# SFT assembly
# ---------------------------------------------------------------------------


def build_sft_sample(template: dict) -> dict:
    """Convert a validated annotation template to flat KV SFT format.

    Stores system, input, think, output as separate keys so downstream
    training scripts can compose them into any model input format.
    """
    ann = template["annotation"]
    source = template["source"]

    passage = ann["passage"].strip()
    think_chain = ann["think_chain"].strip()
    output_json = ann["output_json"]

    ordered_output = {}
    for k in FIELD_ORDER:
        if k in output_json:
            ordered_output[k] = output_json[k]

    return {
        "id": template["id"],
        "system": SYSTEM_PROMPT,
        "input": passage,
        "think": think_chain,
        "output": ordered_output,
        "metadata": {
            "quality": "gold",
            "source_file": source.get("excel_file", "unknown"),
            "source_sheet": source.get("sheet", "unknown"),
            "source_row": source.get("row", 0),
            "tier": source.get("tier", "unknown"),
            "field_count": len(ordered_output),
            "input_char_count": len(passage),
            "output_char_count": len(json.dumps(ordered_output, ensure_ascii=False)),
        },
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def _field_count_bucket(fc: int) -> str:
    if fc <= 2:
        return "few"
    elif fc <= 4:
        return "mid"
    else:
        return "full"


def assign_splits(
    samples: list[dict],
    dev_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Stratified split by (source_file × field_count_bucket)."""
    rng = random.Random(seed)

    strata: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        m = s["metadata"]
        fc_bucket = _field_count_bucket(m["field_count"])
        stratum_key = f"{m['source_file']}_{fc_bucket}"
        strata[stratum_key].append(s)

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    for stratum_key, stratum_samples in sorted(strata.items()):
        rng.shuffle(stratum_samples)
        n = len(stratum_samples)
        n_test = max(1, round(n * test_ratio))
        n_dev = max(1, round(n * dev_ratio))

        if n_test + n_dev >= n:
            n_test = max(1, n // 3)
            n_dev = max(1, n // 3)

        splits["test"].extend(stratum_samples[:n_test])
        splits["dev"].extend(stratum_samples[n_test : n_test + n_dev])
        splits["train"].extend(stratum_samples[n_test + n_dev :])

    for split_name, split_samples in splits.items():
        for s in split_samples:
            s["metadata"]["split"] = split_name

    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading templates from {TEMPLATES_DIR}")

    template_files = sorted(TEMPLATES_DIR.glob("sgcc_*.json"))
    print(f"  Found {len(template_files)} template files")

    templates: list[dict] = []
    skipped_pending = 0
    all_errors: list[ValidationError] = []

    for tf in template_files:
        t = json.loads(tf.read_text())
        if t.get("status") == "pending":
            skipped_pending += 1
            continue
        if t.get("status") != "completed":
            continue

        errors = validate_template(t)
        if errors:
            all_errors.extend(errors)
            continue

        templates.append(t)

    print(f"  Valid completed: {len(templates)}")
    print(f"  Pending (skipped): {skipped_pending}")
    print(f"  Validation errors: {len(all_errors)}")

    if all_errors:
        print("\n  Errors:")
        for e in all_errors[:10]:
            print(f"    {e}")
        if len(all_errors) > 10:
            print(f"    ... and {len(all_errors) - 10} more")

    if not templates:
        print("\n  No completed templates to assemble. Run annotation first.")
        return

    # Build SFT samples
    samples = [build_sft_sample(t) for t in templates]

    # Statistics
    field_counts: Counter = Counter()
    field_presence: Counter = Counter()
    for s in samples:
        field_counts[s["metadata"]["field_count"]] += 1
        for k in s["output"]:
            field_presence[k] += 1

    print(f"\n=== Field count distribution ===")
    for fc in sorted(field_counts):
        print(f"  {fc} fields: {field_counts[fc]} samples")

    print(f"\n=== Field presence ===")
    for field in FIELD_ORDER:
        print(f"  {field}: {field_presence.get(field, 0)} samples")

    # Split
    splits = assign_splits(samples)
    print(f"\n=== Splits ===")
    for split_name in ("train", "dev", "test"):
        print(f"  {split_name}: {len(splits[split_name])} samples")

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_samples in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Wrote {out_path} ({len(split_samples)} samples)")

    # Write manifest
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total_samples": len(samples),
        "splits": {k: len(v) for k, v in splits.items()},
        "field_presence": dict(field_presence),
        "field_count_distribution": {str(k): v for k, v in sorted(field_counts.items())},
    }
    manifest_path = REPORTS_DIR / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote {manifest_path}")


if __name__ == "__main__":
    main()
