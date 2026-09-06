"""Copy a reproducible, focused subset of the Zotero UST collection."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZOTERO_DIR = Path.home() / "Zotero"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "papers"

PREFERRED_TITLES = [
    ("V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning", "vjepa2.pdf"),
    ("V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning", "vjepa2_1.pdf"),
    ("Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", "jepa_images.pdf"),
    ("VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model", "vla_jepa.pdf"),
    ("DINOv3", "dinov3.pdf"),
    ("World Action Models: The Next Frontier in Embodied AI", "world_action_models.pdf"),
    ("Fast-WAM: Do World Action Models Need Test-time Future Imagination?", "fast_wam.pdf"),
    ("LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies", "lawam.pdf"),
    ("Patch Policy: Efficient Embodied Control via Dense Visual Representations", "patch_policy.pdf"),
    ("Causal World Modeling for Robot Control", "causal_world_modeling.pdf"),
]


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path))}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=1)


def _find_database(zotero_dir: Path) -> tuple[sqlite3.Connection, Path]:
    candidates = [
        zotero_dir / "zotero.sqlite",
        zotero_dir / "zotero.sqlite.bak",
        zotero_dir / "zotero.sqlite.1.bak",
    ]
    errors: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            connection = _connect_read_only(path)
            connection.execute("select 1").fetchone()
            return connection, path
        except sqlite3.Error as exc:
            errors.append(f"{path}: {exc}")
    detail = "; ".join(errors) if errors else "no zotero.sqlite file found"
    raise FileNotFoundError(f"Could not open Zotero database: {detail}")


def _load_ust_items(connection: sqlite3.Connection, zotero_dir: Path) -> list[dict[str, Any]]:
    collection = connection.execute(
        "select collectionID from collections where collectionName = ?", ("UST",)
    ).fetchone()
    if collection is None:
        raise ValueError("Zotero collection UST was not found.")

    query = """
        select ci.orderIndex, p.itemID, p.key, title.value, a.key, ia.path
        from collectionItems ci
        join items p on p.itemID = ci.itemID
        join itemData title_data on title_data.itemID = p.itemID
        join fieldsCombined title_field
          on title_field.fieldID = title_data.fieldID and title_field.fieldName = 'title'
        join itemDataValues title on title.valueID = title_data.valueID
        join itemAttachments ia on ia.parentItemID = p.itemID and ia.contentType = 'application/pdf'
        join items a on a.itemID = ia.itemID
        where ci.collectionID = ?
        order by ci.orderIndex
    """
    rows = connection.execute(query, (int(collection[0]),)).fetchall()
    items: list[dict[str, Any]] = []
    for order_index, item_id, key, title, attachment_key, attachment_path in rows:
        if not attachment_path or not str(attachment_path).startswith("storage:"):
            continue
        relative_name = str(attachment_path).split(":", 1)[1]
        source_path = zotero_dir / "storage" / str(attachment_key) / relative_name
        if source_path.is_file():
            items.append(
                {
                    "order_index": order_index,
                    "item_id": item_id,
                    "zotero_key": key,
                    "title": str(title),
                    "attachment_key": attachment_key,
                    "source_path": str(source_path),
                }
            )
    return items


def _select(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_title = {item["title"]: item for item in items}
    missing = [title for title, _ in PREFERRED_TITLES if title not in by_title]
    if missing:
        raise ValueError("Required UST papers are missing: " + "; ".join(missing))
    if count < len(PREFERRED_TITLES):
        raise ValueError(f"count must be at least {len(PREFERRED_TITLES)} to include required JEPA/DINO papers")
    selected = []
    selected_titles = set()
    for title, filename in PREFERRED_TITLES:
        item = dict(by_title[title])
        item["filename"] = filename
        selected.append(item)
        selected_titles.add(title)
    if len(selected) < count:
        for item in items:
            if item["title"] in selected_titles:
                continue
            item = dict(item)
            item["filename"] = f"ust_{len(selected) + 1:02d}.pdf"
            selected.append(item)
            if len(selected) >= count:
                break
    return selected


def import_ust(
    zotero_dir: str | Path = DEFAULT_ZOTERO_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    count: int = 10,
) -> dict[str, Any]:
    """Copy selected UST PDFs and write their provenance manifest."""

    source_dir = Path(zotero_dir).expanduser()
    target_dir = Path(output_dir)
    connection, database_path = _find_database(source_dir)
    try:
        items = _load_ust_items(connection, source_dir)
    finally:
        connection.close()
    selected = _select(items, count)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for item in selected:
        destination = target_dir / item["filename"]
        shutil.copy2(item["source_path"], destination)
        manifest.append(
            {
                "filename": item["filename"],
                "title": item["title"],
                "zotero_key": item["zotero_key"],
                "attachment_key": item["attachment_key"],
                "collection": "UST",
                "source_path": item["source_path"],
                "database": str(database_path),
            }
        )
        print(f"[Zotero] {item['title']} -> {destination.name}")
    manifest_path = target_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[Zotero] Imported {len(manifest)} PDFs from UST into {target_dir}")
    return {"count": len(manifest), "output_dir": str(target_dir), "manifest": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import selected papers from Zotero collection UST")
    parser.add_argument("--zotero-dir", default=str(DEFAULT_ZOTERO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--count", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import_ust(args.zotero_dir, args.output_dir, args.count)
    except (FileNotFoundError, ValueError, sqlite3.Error, OSError) as exc:
        print(f"ToolError: Zotero import failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
