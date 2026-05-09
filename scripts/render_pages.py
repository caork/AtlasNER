"""Render matched PDF/DOCX pages as images for human annotation reference.

Reads annotation_manifest.json and renders the top-matched pages for
each row as PNG images. For PDFs, uses PyMuPDF to render pages directly.
For DOCXs, extracts text blocks (no rendering engine available).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GDRIVE = Path(
    "/Users/kaitaocao/Library/CloudStorage/"
    "GoogleDrive-barrientosangie599@gmail.com/My Drive/"
    "AtlasNERDataset/国家电网文件"
)
ANNOTATION_DIR = Path(__file__).resolve().parent.parent / "data" / "annotation"
MANIFEST_PATH = ANNOTATION_DIR / "annotation_manifest.json"
PAGES_DIR = ANNOTATION_DIR / "pages"

DOC_SEARCH_DIRS = [
    GDRIVE / "模版及案例",
    GDRIVE / "补充来源文件",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_doc_path(doc_name: str) -> Path | None:
    for search_dir in DOC_SEARCH_DIRS:
        for p in search_dir.rglob(doc_name):
            return p
    return None


def render_pdf_page(pdf_path: Path, page_num: int, output_path: Path, dpi: int = 200) -> None:
    """Render a single PDF page to PNG."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(output_path))
    doc.close()


def render_pages_for_entry(
    entry: dict,
    max_pages: int = 3,
    dpi: int = 200,
) -> list[dict]:
    """Render the top pages for a manifest entry. Returns list of rendered page info."""
    rendered = []

    for doc_match in entry.get("doc_matches", []):
        doc_name = doc_match["doc_name"]
        doc_path = resolve_doc_path(doc_name)
        if not doc_path:
            continue

        doc_type = doc_match["doc_type"]
        pages = doc_match["pages"][:max_pages]

        for page_info in pages:
            page_num = page_info["page_num"]
            row_idx = entry["row_index"]

            # Sanitize doc name for filename
            safe_doc = doc_name.replace("/", "_").replace(" ", "_")
            if len(safe_doc) > 60:
                safe_doc = safe_doc[:60]

            img_name = f"row{row_idx:04d}_{safe_doc}_p{page_num:04d}.png"
            img_path = PAGES_DIR / img_name

            if doc_type == ".pdf":
                render_pdf_page(doc_path, page_num, img_path, dpi=dpi)
                rendered.append({
                    "row_index": row_idx,
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "score": page_info["score"],
                    "image_path": str(img_path.relative_to(ANNOTATION_DIR)),
                    "rendered": True,
                })
            else:
                # DOCX: can't render without LibreOffice; mark as text-only
                rendered.append({
                    "row_index": row_idx,
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "score": page_info["score"],
                    "image_path": None,
                    "rendered": False,
                    "note": "DOCX rendering requires LibreOffice; use text extraction",
                })

        # Only render from the top-scoring document
        break

    return rendered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    row_indices: list[int] | None = None,
    max_pages: int = 3,
    dpi: int = 200,
) -> None:
    """Render pages for specified rows (or all matched rows if None)."""
    print(f"Loading manifest from {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest["entries"]
    print(f"  Total entries: {len(entries)}")

    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to specific rows if requested
    if row_indices is not None:
        target = {i for i in row_indices}
        entries = [e for e in entries if e["row_index"] in target]
        print(f"  Rendering {len(entries)} selected rows")
    else:
        entries = [e for e in entries if e["status"] == "matched"]
        print(f"  Rendering all {len(entries)} matched rows")

    all_rendered: list[dict] = []
    pdf_cache: dict[str, fitz.Document] = {}

    for i, entry in enumerate(entries):
        rendered = render_pages_for_entry(entry, max_pages=max_pages, dpi=dpi)
        all_rendered.extend(rendered)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(entries)}...")

    # Write render index
    render_index_path = ANNOTATION_DIR / "render_index.json"
    with render_index_path.open("w", encoding="utf-8") as f:
        json.dump(all_rendered, f, ensure_ascii=False, indent=2)

    actual_rendered = sum(1 for r in all_rendered if r["rendered"])
    print(f"\n=== Results ===")
    print(f"  Total pages: {len(all_rendered)}")
    print(f"  Rendered as images: {actual_rendered}")
    print(f"  Text-only (DOCX): {len(all_rendered) - actual_rendered}")
    print(f"  Pages dir: {PAGES_DIR}")
    print(f"  Index: {render_index_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="*", type=int, help="Specific row indices to render")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages per document")
    parser.add_argument("--dpi", type=int, default=200, help="Render resolution")
    args = parser.parse_args()

    main(row_indices=args.rows, max_pages=args.max_pages, dpi=args.dpi)
