#!/usr/bin/env python3
"""Group-aware train/valid/test split for the NumbersRec dataset.

Frames are crops from video, so a RANDOM per-image split leaks near-duplicate
frames of the same player/match across splits and inflates the metrics. This
tool splits by GROUP (whole matches, or whole tracks) so no group spans two
splits — the valid/test scores then reflect real generalization to new footage.

Group key is taken from the filename. Our miner names crops
    <job_id>_p<pid>_f<frame>.jpg
and Roboflow keeps that stem (it only appends a ".rf.<hash>" suffix), so:
    --by match  -> group = job_id            (default; holds out whole games)
    --by track  -> group = job_id + "_p<pid>"

Input: any folder tree containing images/ + labels/ (a Roboflow YOLOv8 export,
or this project's webapp/training_dataset). Output: train/ valid/ test/ each with
images/ + labels/, plus data.yaml — a Roboflow-ready YOLOv8 dataset.

Usage:
    python3 roboflow_split.py <input_dir> <output_dir> [--by match|track]
        [--ratios 0.8 0.1 0.1] [--seed 42] [--dry-run] [--selftest]
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}
NUM_CLASSES = [str(i) for i in range(10)]


def group_key(stem, by):
    """Derive the group id from an image stem. Robust to Roboflow's
    '.rf.<hash>' suffix (strip it) and to a leading path."""
    name = stem.split(".rf.")[0]              # drop Roboflow hash suffix
    if by == "track":
        m = re.match(r"(.+?_p\d+)_f", name)   # job_id + _p<pid>
        return m.group(1) if m else name
    # by match: everything before the first _p<pid>
    m = re.match(r"(.+?)_p\d+", name)
    return m.group(1) if m else name


def find_images(root):
    """All images under root, paired with their label (searched next to the
    image, or under a sibling 'labels' dir mirroring 'images')."""
    pairs = []
    for img in root.rglob("*"):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            # images/<name>.jpg -> labels/<name>.txt
            parts = list(img.parts)
            if "images" in parts:
                parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
                cand = Path(*parts).with_suffix(".txt")
                if cand.exists():
                    lbl = cand
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs


def split_groups(groups, ratios, seed):
    """Assign whole groups to test/valid/train so image counts approximate the
    ratios. Bigger groups first keeps the small splits from overshooting."""
    train_r, val_r, test_r = ratios
    total = sum(len(v) for v in groups.values())
    want_test = total * test_r
    want_val = total * val_r
    keys = list(groups)
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    assign = {}
    n_test = n_val = 0
    for k in keys:
        sz = len(groups[k])
        if n_test + sz <= want_test or n_test == 0 and want_test > 0:
            assign[k] = "test"; n_test += sz
        elif n_val + sz <= want_val or n_val == 0 and want_val > 0:
            assign[k] = "valid"; n_val += sz
        else:
            assign[k] = "train"
    return assign


def run(inp, outp, by, ratios, seed, dry):
    pairs = find_images(inp)
    if not pairs:
        raise SystemExit(f"no images found under {inp}")
    groups = defaultdict(list)
    for img, lbl in pairs:
        groups[group_key(img.stem, by)].append((img, lbl))

    assign = split_groups(groups, ratios, seed)
    counts = defaultdict(int)
    for k, split in assign.items():
        counts[split] += len(groups[k])

    print(f"images: {len(pairs)} | groups ({by}): {len(groups)}")
    for s in ("train", "valid", "test"):
        g = sum(1 for v in assign.values() if v == s)
        print(f"  {s:5}: {counts[s]:5d} images  across {g} {by}s")
    # class coverage check in valid/test
    for s in ("valid", "test"):
        seen = set()
        for k, split in assign.items():
            if split != s:
                continue
            for _img, lbl in groups[k]:
                if lbl:
                    for line in lbl.read_text().split("\n"):
                        if line.strip():
                            seen.add(line.split()[0])
        missing = [c for c in NUM_CLASSES if c not in seen]
        if missing:
            print(f"  ! {s}: нет примеров цифр {missing} — метрика по ним будет пустой")

    if dry:
        print("dry-run: файлы не скопированы")
        return

    outp = Path(outp)
    for s in ("train", "valid", "test"):
        (outp / s / "images").mkdir(parents=True, exist_ok=True)
        (outp / s / "labels").mkdir(parents=True, exist_ok=True)
    for k, split in assign.items():
        for img, lbl in groups[k]:
            shutil.copy2(img, outp / split / "images" / img.name)
            if lbl:
                shutil.copy2(lbl, outp / split / "labels" / (img.stem + ".txt"))
    names = "[" + ", ".join(f"'{c}'" for c in NUM_CLASSES) + "]"
    (outp / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
        f"nc: {len(NUM_CLASSES)}\nnames: {names}\n")
    print(f"готово -> {outp}")


def _selftest():
    # 3 matches, several frames each; whole matches must stay in one split
    groups = {f"game{g}": list(range(10)) for g in range(3)}
    a = split_groups(groups, (0.34, 0.33, 0.33), seed=1)
    assert set(a.values()) <= {"train", "valid", "test"}
    assert len(set(a.values())) == 3, a  # each split got a whole game
    assert group_key("abc123_p7_f42", "match") == "abc123"
    assert group_key("abc123_p7_f42", "track") == "abc123_p7"
    assert group_key("abc123_p7_f42.rf.deadbeef", "match") == "abc123"
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--by", choices=["match", "track"], default="match")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        if not args.input or not args.output:
            ap.error("input and output required (or use --selftest)")
        run(Path(args.input), args.output, args.by, args.ratios, args.seed, args.dry_run)
