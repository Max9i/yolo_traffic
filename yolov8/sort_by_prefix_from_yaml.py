#!/usr/bin/env python3
from pathlib import Path
import shutil
import yaml
import re
import argparse
from typing import List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def load_class_names(yaml_path: Path) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # `names` can be a list or a {id: name} dict
    names = data.get("names")
    if isinstance(names, dict):
        # ensure order by numeric key
        names = [v for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    if not isinstance(names, list):
        raise ValueError("YAML does not contain a valid `names` list or dict.")
    return names

def parse_prefix(stem: str) -> Optional[int]:
    """
    Extract leading integer prefix before '_' or '-' (e.g., '001_xxx' -> 1 or 0 depending on base).
    Returns the integer as written (e.g., 1 for '001'), without base adjustment.
    """
    m = re.match(r"^(\d+)[-_].*", stem)
    if not m:
        return None
    return int(m.group(1))

def decide_base(prefix_num: int, num_classes: int) -> Optional[int]:
    """
    Decide whether filenames are 0-based or 1-based.
    Returns base 0 or 1, or None if undecidable.
    """
    if 0 <= prefix_num < num_classes:
        return 0
    if 1 <= prefix_num <= num_classes:
        return 1
    return None

def class_id_from_stem(stem: str, num_classes: int, known_base: Optional[int]) -> Optional[int]:
    p = parse_prefix(stem)
    if p is None:
        return None
    if known_base is None:
        base = decide_base(p, num_classes)
        if base is None:
            return None
    else:
        base = known_base
    cid = p - base
    if 0 <= cid < num_classes:
        return cid
    return None

def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p

def sort_split(split_dir: Path, out_dir: Path, class_names: List[str], move: bool, base_hint: Optional[int] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prepare class directories
    class_dirs = []
    for name in class_names:
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        class_dirs.append(d)

    # First pass: try to infer base if not provided
    inferred_base = base_hint
    if inferred_base is None:
        for img in iter_images(split_dir):
            cid0 = class_id_from_stem(img.stem, len(class_names), None)
            if cid0 is not None:
                # `class_id_from_stem` internally decides base -> we can back out base:
                p = parse_prefix(img.stem)
                inferred_base = p - cid0 if p is not None else None
                break
        if inferred_base is None:
            # Fall back to 0-based assumption
            inferred_base = 0

    moved, skipped = 0, 0
    for img in iter_images(split_dir):
        cid = class_id_from_stem(img.stem, len(class_names), inferred_base)
        if cid is None:
            skipped += 1
            continue
        dest = class_dirs[cid] / img.name
        if move:
            shutil.move(str(img), str(dest))
        else:
            shutil.copy2(str(img), str(dest))
        moved += 1

    return moved, skipped, inferred_base

def main():
    ap = argparse.ArgumentParser(description="Sort mixed train/val images into class folders using filename prefixes and YAML names.")
    ap.add_argument("--yaml", required=True, type=Path, help="Path to the dataset YAML (contains `names`).")
    ap.add_argument("--train", required=True, type=Path, help="Path to the mixed train images folder.")
    ap.add_argument("--val", required=True, type=Path, help="Path to the mixed val images folder.")
    ap.add_argument("--out", required=True, type=Path, help="Output root directory to place sorted datasets.")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying.")
    ap.add_argument("--base", choices=["0", "1"], help="Force base for prefixes: 0 for 0-based (000_), 1 for 1-based (001_). If omitted, auto-detect.")
    args = ap.parse_args()

    class_names = load_class_names(args.yaml)
    base_hint = int(args.base) if args.base is not None else None

    out_train = args.out / "train"
    out_val = args.out / "val"

    moved_tr, skipped_tr, base_tr = sort_split(args.train, out_train, class_names, move=args.move, base_hint=base_hint)
    moved_vl, skipped_vl, base_vl = sort_split(args.val, out_val, class_names, move=args.move, base_hint=base_hint)

    print(f"[train] moved/copied: {moved_tr}, skipped (no valid prefix): {skipped_tr}, base={base_tr}")
    print(f"[val]   moved/copied: {moved_vl}, skipped (no valid prefix): {skipped_vl}, base={base_vl}")
    if base_hint is None and base_tr != base_vl:
        print("Warning: inferred base differs between train and val. You may want to set --base explicitly.")

if __name__ == "__main__":
    main()
