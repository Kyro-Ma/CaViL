"""
step1_build_item2meta_and_category.py

From a raw category JSONL and Amazon-Reviews-2023 meta_*.jsonl.gz files:
  1) Create LaViC category splits:
       data/<category>/{train.jsonl, valid.jsonl, test.jsonl}
  2) Create global item meta files (NO LLaVA FIELDS YET):
       data/item2meta_train.json   (dict keyed by ASIN, with LaViC-style fields)
       data/item2meta_valid.jsonl  (JSONL: {title, image_name, image})

If your ASINs span multiple meta files, pass multiple --meta_gz arguments.
"""

import argparse, gzip, json, random, re
from pathlib import Path
from typing import Dict, Any, Iterable, List, Set, Tuple

ASIN_RE = re.compile(r'^[A-Z0-9]{10}$')
RNG = random.Random(42)

# Default LaViC-like fields to keep in item2meta_train.json
DEFAULT_KEEP_FIELDS = [
    "main_category","title","average_rating","rating_number","features",
    "description","price","images","videos","store","categories",
    "details","parent_asin","bought_together"
]
# Some subtrees can be extremely large; optionally drop them
DEFAULT_DROP_FIELDS = []  # e.g., ["details.Best Sellers Rank"]


# ---------- I/O ----------
def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def read_jsonl_gz(path: str) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


REQ_KEYS = ["context","gt_items"]

def normalize_raw_row(rec: dict) -> dict:
    for k in REQ_KEYS:
        if k not in rec:
            raise ValueError(f"Missing required key: {k}")
    # ensure candidates lists exist
    for cf in ("candidates_st","candidates_gpt_large"):
        if cf not in rec or not isinstance(rec[cf], list):
            rec[cf] = []
    # save: ensure ground-truth appears in at least one candidate list if available
    gts = rec.get("gt_items") or []
    for cf in ("candidates_st","candidates_gpt_large"):
        cands = rec.get(cf) or []
        if cands and not any(g in set(cands) for g in gts):
            # force-insert first gt in slot 0
            cands = list(cands)
            if gts:
                if cands:
                    cands[0] = gts[0]
                else:
                    cands = [gts[0]]
            rec[cf] = cands
    return rec

def split_rows(rows: List[dict], train_ratio=0.8, valid_ratio=0.1) -> Tuple[List[dict],List[dict],List[dict]]:
    n = len(rows)
    n_tr = int(n * train_ratio)
    n_va = int(n * valid_ratio)
    return rows[:n_tr], rows[n_tr:n_tr+n_va], rows[n_tr+n_va:]

# ---------- ASIN collection ----------
def collect_target_asins(raw_path: str) -> Set[str]:
    targets: Set[str] = set()
    for rec in read_jsonl(raw_path):
        for k in ("gt_items","candidates_st","candidates_gpt_large","context_items"):
            arr = rec.get(k) or []
            if isinstance(arr, list):
                for a in arr:
                    if isinstance(a, str):
                        a = a.strip()
                        if ASIN_RE.match(a):
                            targets.add(a)
    return targets

# ---------- Image normalization & merging ----------
def normalize_images_from_meta(rec: dict) -> List[dict]:
    """
    Build a rich list of image dicts with possible keys:
      thumb, large, variant, hi_res
    Falls back to imageURLHighRes/imageURLs/imUrl when needed.
    Caps at 10.
    """
    out = []
    imgs = rec.get("images")
    if isinstance(imgs, list) and imgs:
        for it in imgs:
            if not isinstance(it, dict):
                continue
            d = {}
            for k in ("thumb","large","variant","hi_res"):
                v = it.get(k)
                if isinstance(v, str) and v:
                    d[k] = v
            if "large" not in d:
                v = it.get("hi_res") or it.get("thumb")
                if isinstance(v, str) and v:
                    d["large"] = v
            if d:
                out.append(d)

    if not out:
        urls = []
        if isinstance(rec.get("imageURLHighRes"), list):
            urls += [u for u in rec["imageURLHighRes"] if isinstance(u,str) and u.startswith("http")]
        if isinstance(rec.get("imageURLs"), list):
            urls += [u for u in rec["imageURLs"] if isinstance(u,str) and u.startswith("http")]
        if isinstance(rec.get("imUrl"), str) and rec["imUrl"].startswith("http"):
            urls.append(rec["imUrl"])

        seen = set()
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append({"large": u})

    return out[:10]

def merge_images(existing: List[dict], incoming: List[dict]) -> List[dict]:
    def url_key(d):
        return d.get("large") or d.get("hi_res") or d.get("thumb")
    by_url = {}
    for it in existing or []:
        u = url_key(it)
        if isinstance(u, str):
            by_url[u] = dict(it)
    for it in incoming or []:
        u = url_key(it)
        if not isinstance(u, str):
            continue
        base = by_url.get(u, {})
        for k in ("large","thumb","variant","hi_res"):
            v = it.get(k)
            if isinstance(v, str) and v and not base.get(k):
                base[k] = v
        by_url[u] = base
    return list(by_url.values())[:10]

# ---------- Build item2meta ----------
def to_valid_lines(asin_key: str, title: str, images: List[dict], k: int) -> List[dict]:
    lines = []
    urls = []
    for im in images:
        u = im.get("hi_res") or im.get("large") or im.get("thumb")
        if isinstance(u, str) and u.startswith("http"):
            urls.append(u)
    for idx, u in enumerate(urls[:max(1,k)]):
        lines.append({
            "title": title or "",
            "image_name": f"{asin_key}_{idx}.jpg",
            "image": u
        })
    return lines

def minify_record(rec: dict, keep_fields: List[str], drop_fields: List[str]) -> dict:
    out = {}
    for k in keep_fields:
        if k in rec:
            out[k] = rec[k]
    # Drop subfields like details
    for dotted in drop_fields:
        cur = out
        parts = dotted.split(".")
        for i, p in enumerate(parts):
            if p not in cur:
                break
            if i == len(parts)-1:
                try:
                    del cur[p]
                except Exception:
                    pass
            else:
                if isinstance(cur[p], dict):
                    cur = cur[p]
                else:
                    break
    return out

def build_item2meta(
    raw_path: str,
    meta_gz_paths: List[str],
    train_out: Path,
    valid_out: Path,
    valid_images_per_asin: int,
    keep_fields: List[str],
    drop_fields: List[str],
    stop_when_complete: bool
) -> Tuple[int,int,int]:
    targets = collect_target_asins(raw_path)
    if not targets:
        raise SystemExit("No ASINs discovered in raw file.")
    pending = set(targets)
    train_dict: Dict[str, dict] = {}
    valid_lines: List[dict] = []
    found = 0
    with_images = 0

    for gz in meta_gz_paths:
        for rec in read_jsonl_gz(gz):
            asin = rec.get("asin")
            parent = rec.get("parent_asin")
            key = None
            if isinstance(asin, str) and asin in pending:
                key = asin
            elif isinstance(parent, str) and parent in pending:
                key = parent

            if key is None:
                # If a sibling child appears after parent captured, use it to augment images.
                if isinstance(parent, str) and parent in train_dict:
                    imgs = normalize_images_from_meta(rec)
                    if imgs:
                        train_dict[parent]["images"] = merge_images(train_dict[parent].get("images", []), imgs)
                continue

            imgs = normalize_images_from_meta(rec)

            # Keep a LaViC-like minimal record (to keep file size sane)
            entry = minify_record(rec, keep_fields, drop_fields)
            entry["images"] = merge_images(entry.get("images", []), imgs)

            train_dict[key] = entry
            found += 1
            if entry["images"]:
                with_images += 1
                valid_lines.extend(to_valid_lines(key, entry.get("title",""), entry["images"], valid_images_per_asin))

            if key in pending:
                pending.remove(key)
            if stop_when_complete and not pending:
                break
        if stop_when_complete and not pending:
            break

    # write outputs
    train_out.parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_dict, f, ensure_ascii=False, sort_keys=True)

    valid_out.parent.mkdir(parents=True, exist_ok=True)
    with open(valid_out, "w", encoding="utf-8") as f:
        for row in valid_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(targets), found, with_images

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Raw category JSONL (with gt_items/candidates...)")
    ap.add_argument("--meta_gz", required=True, nargs="+", help="One or more Amazon meta_*.jsonl.gz files")
    ap.add_argument("--category", required=True, help="Category folder name under data/")
    ap.add_argument("--data_root", default='../../data', help="Where to write data/")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--valid_images_per_asin", type=int, default=1)
    ap.add_argument("--stop_when_complete", action="store_true")

    ap.add_argument("--keep_fields", nargs="*", default=DEFAULT_KEEP_FIELDS)
    ap.add_argument("--drop_fields", nargs="*", default=DEFAULT_DROP_FIELDS)

    args = ap.parse_args()
    data_root = Path(args.data_root)
    cat_dir = data_root / args.category

    # 1) Split category into train/valid/test
    rows = []
    for rec in read_jsonl(args.raw):
        try:
            rows.append(normalize_raw_row(rec))
        except Exception as e:
            print(f"[SKIP] bad record: {e}")
    RNG.shuffle(rows)
    tr, va, te = split_rows(rows, args.train_ratio, args.valid_ratio)
    write_jsonl(cat_dir / "train.jsonl", tr)
    write_jsonl(cat_dir / "valid.jsonl", va)
    write_jsonl(cat_dir / "test.jsonl",  te)
    print(f"[CATEGORY SPLIT] Wrote {cat_dir}/train({len(tr)}), valid({len(va)}), test({len(te)})")

    # 2) Build item2meta files (NO LLaVA YET)
    train_out = data_root / f"item2meta_train_{args.category}.json"
    valid_out = data_root / f"item2meta_valid_{args.category}.jsonl"
    total, found, with_images = build_item2meta(
        raw_path=args.raw,
        meta_gz_paths=args.meta_gz,
        train_out=train_out,
        valid_out=valid_out,
        valid_images_per_asin=max(1, args.valid_images_per_asin),
        keep_fields=args.keep_fields,
        drop_fields=args.drop_fields,
        stop_when_complete=args.stop_when_complete
    )
    print(f"[ITEM2META] Raw ASINs: {total} | Found: {found} | With images: {with_images}")
    print(f"[ITEM2META] Wrote:\n - {train_out}\n - {valid_out}")

if __name__ == "__main__":
    main()
