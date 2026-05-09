"""Batch generate annotation templates using DeepSeek V4 Flash API.

Uses DeepSeek V4 Flash with thinking mode (reasoning_effort=high) to generate:
- Cleaned passage from raw PDF page text
- Chinese think chain (思维链) explaining extraction reasoning
- Verified output_json with OCR corrections

Usage:
    python scripts/deepseek_batch_generate.py --key <api_key> [--concurrency 5] [--start 0] [--limit 100]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
BATCH_INPUT = ROOT / "data" / "annotation" / "batch_input.jsonl"
TEMPLATES_DIR = ROOT / "data" / "annotation" / "templates"
PROGRESS_FILE = ROOT / "data" / "annotation" / "deepseek_progress.json"

SYSTEM_PROMPT_SFT = (
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
# Prompt for DeepSeek
# ---------------------------------------------------------------------------

def build_prompt(entry: dict) -> str:
    """Build the annotation prompt for one entry."""
    fields = entry["excel_fields"]
    field_count = len(fields)
    field_list = "\n".join(f"  - {k}: {v}" for k, v in fields.items())

    source = entry["source"]
    page_num = entry["best_page"]
    score = entry["best_score"]

    return f"""你是电力文档标注专家。请根据以下PDF页面原文和Excel预提取字段，完成标注工作。

## PDF页面原文（第{page_num}页，匹配分{score}）：
{entry["page_text"]}

## Excel预提取字段（{field_count}个）：
{field_list}

## 来源信息：
- Excel文件: {source["excel_file"]}
- Sheet: {source["sheet"]}
- 行号: {source["row"]}

## 你的任务：
1. **passage**: 从PDF原文中提取与这条记录相关的段落，清理格式（去掉页码页眉、修复断行、保留完整的序号/设备/工序/风险/措施结构）。如果PDF原文与Excel字段不匹配（如best_score<0.5），则根据Excel字段构造一段合理的passage。

2. **think_chain**: 写150-350字的中文分析推理，说明：
   - 这段来自哪个文档哪一页
   - 有哪些字段可提取，为什么
   - 是否有OCR错误需要修正（如"固棒"→"固件"、"条棒"→"条件"等常见PDF OCR问题）
   - 为什么某些字段不存在（如果字段不全的话）

3. **output_json**: 验证并修正Excel预提取字段，输出最终的结构化JSON。修正明显的OCR错误。

## 输出格式（严格JSON）：
```json
{{
  "passage": "清理后的段落文本",
  "think_chain": "中文分析推理150-350字",
  "output_json": {{按需包含的字段}},
  "annotator_notes": "简短备注"
}}
```

请直接输出JSON，不要输出其他内容。"""


# ---------------------------------------------------------------------------
# Async API caller
# ---------------------------------------------------------------------------

async def call_deepseek(
    client,
    entry: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict | None:
    """Call DeepSeek V4 Flash API for one entry."""
    row_idx = entry["row_index"]
    prompt = build_prompt(entry)

    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="deepseek-v4-flash",
                    messages=[
                        {"role": "system", "content": "你是电力文档标注专家，负责生成高质量的结构化抽取训练数据。请严格按要求输出JSON格式。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.3,
                    extra_body={"thinking": {"type": "enabled", "reasoning_effort": "high"}},
                    response_format={"type": "json_object"},
                )

            content = resp.choices[0].message.content
            if not content:
                print(f"  [row {row_idx}] Empty content, attempt {attempt+1}")
                continue

            # Parse JSON response
            result = json.loads(content)

            # Validate required fields
            if not result.get("passage") or not result.get("think_chain") or not result.get("output_json"):
                print(f"  [row {row_idx}] Missing fields, attempt {attempt+1}")
                continue

            return result

        except json.JSONDecodeError as e:
            print(f"  [row {row_idx}] JSON parse error: {e}, attempt {attempt+1}")
            # Try to extract JSON from content
            if content:
                m = re.search(r'\{[\s\S]*\}', content)
                if m:
                    try:
                        result = json.loads(m.group())
                        if result.get("passage") and result.get("think_chain") and result.get("output_json"):
                            return result
                    except json.JSONDecodeError:
                        pass
            continue

        except Exception as e:
            err_str = str(e)
            if "rate" in err_str.lower() or "429" in err_str:
                wait = min(30 * (attempt + 1), 120)
                print(f"  [row {row_idx}] Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            elif "500" in err_str or "502" in err_str or "503" in err_str:
                wait = 10 * (attempt + 1)
                print(f"  [row {row_idx}] Server error, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  [row {row_idx}] Error: {e}, attempt {attempt+1}")
                await asyncio.sleep(5)

    print(f"  [row {row_idx}] FAILED after {max_retries} attempts")
    return None


def build_template(entry: dict, annotation: dict) -> dict:
    """Build a complete annotation template."""
    row_idx = entry["row_index"]
    source = entry["source"]

    # Order output_json fields
    output_json = {}
    raw_output = annotation.get("output_json", {})
    for k in FIELD_ORDER:
        if k in raw_output:
            output_json[k] = raw_output[k]
    # Include any extra fields not in FIELD_ORDER
    for k, v in raw_output.items():
        if k not in output_json:
            output_json[k] = v

    return {
        "id": f"sgcc_{row_idx:06d}",
        "status": "completed",
        "source": {
            "excel_file": source.get("excel_file", "unknown"),
            "sheet": source.get("sheet", "unknown"),
            "row": source.get("row", 0),
            "tier": source.get("tier", "unknown"),
        },
        "reference": {
            "page_refs": [{
                "doc_name": entry.get("best_doc", ""),
                "page_num": entry.get("best_page", 0),
                "match_score": entry.get("best_score", 0),
            }],
            "image_paths": [],
        },
        "annotation": {
            "passage": annotation.get("passage", ""),
            "think_chain": annotation.get("think_chain", ""),
            "output_json": output_json,
            "annotator_notes": annotation.get("annotator_notes", "auto-generated by DeepSeek V4 Flash"),
        },
        "sft_format": {
            "system": SYSTEM_PROMPT_SFT,
            "user_template": "请从以下文档段落中提取结构化风险管控记录：\n\n<document>\n{passage}\n</document>",
            "assistant_template": "<think>\n{think_chain}\n</think>\n\n```json\n{output_json}\n```",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def process_batch(
    entries: list[dict],
    api_key: str,
    concurrency: int = 5,
) -> tuple[int, int]:
    """Process a batch of entries. Returns (success, fail) counts."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    semaphore = asyncio.Semaphore(concurrency)

    success = 0
    fail = 0

    # Process in chunks to show progress
    chunk_size = concurrency * 2
    for i in range(0, len(entries), chunk_size):
        chunk = entries[i:i + chunk_size]
        tasks = [call_deepseek(client, e, semaphore) for e in chunk]
        results = await asyncio.gather(*tasks)

        for entry, result in zip(chunk, results):
            if result is not None:
                template = build_template(entry, result)
                path = TEMPLATES_DIR / f"{template['id']}.json"
                with path.open("w", encoding="utf-8") as f:
                    json.dump(template, f, ensure_ascii=False, indent=2)
                success += 1
            else:
                fail += 1

        total_done = i + len(chunk)
        print(f"  Progress: {total_done}/{len(entries)} | Success: {success} | Fail: {fail}")

    return success, fail


def main():
    parser = argparse.ArgumentParser(description="Generate annotations with DeepSeek V4 Flash")
    parser.add_argument("--key", required=True, help="DeepSeek API key")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--start", type=int, default=0, help="Start index in missing list")
    parser.add_argument("--limit", type=int, default=0, help="Max entries to process (0=all)")
    args = parser.parse_args()

    # Load all entries from batch_input
    print("Loading batch_input.jsonl...")
    all_entries = {}
    with BATCH_INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            all_entries[entry["row_index"]] = entry

    # Find which entries still need templates
    completed = set()
    for f_name in os.listdir(TEMPLATES_DIR):
        if f_name.startswith("sgcc_") and f_name.endswith(".json"):
            try:
                completed.add(int(f_name[5:11]))
            except ValueError:
                pass

    missing = sorted(set(all_entries.keys()) - completed)
    print(f"Total entries: {len(all_entries)}")
    print(f"Completed: {len(completed)}")
    print(f"Missing: {len(missing)}")

    # Apply start/limit
    if args.start > 0:
        missing = missing[args.start:]
    if args.limit > 0:
        missing = missing[:args.limit]

    if not missing:
        print("Nothing to do!")
        return

    entries_to_process = [all_entries[idx] for idx in missing]
    print(f"\nProcessing {len(entries_to_process)} entries with concurrency={args.concurrency}...")
    print(f"Estimated cost: ~${len(entries_to_process) * 0.001:.2f} (very rough)")

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    success, fail = asyncio.run(
        process_batch(entries_to_process, args.key, args.concurrency)
    )
    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Success: {success}")
    print(f"Failed: {fail}")
    print(f"Rate: {success/elapsed*60:.1f} entries/min")

    # Update progress
    progress = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processed": success + fail,
        "success": success,
        "fail": fail,
        "elapsed_seconds": elapsed,
        "total_completed": len(completed) + success,
        "total_target": len(all_entries),
    }
    with PROGRESS_FILE.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
