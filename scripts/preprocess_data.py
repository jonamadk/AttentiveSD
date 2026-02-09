#!/usr/bin/env python3
import argparse
from pathlib import Path

from attentivesd.data.cnnsplice import build_raw_paths, load_raw_split, save_npz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data")
    parser.add_argument("--out-root", default="data/processed")
    parser.add_argument("--dataset", choices=["balanced", "imbalanced"], default="balanced")
    parser.add_argument(
        "--organism", choices=["hs", "at", "d_mel", "c_elegans", "oriza"], default="hs"
    )
    parser.add_argument("--site", choices=["donor", "acceptor"], default="donor")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    raw_x, raw_y = build_raw_paths(
        raw_root=Path(args.raw_root),
        dataset=args.dataset,
        organism=args.organism,
        split=args.split,
        site=args.site,
    )

    x, y = load_raw_split(raw_x, raw_y, max_samples=args.max_samples)
    out_dir = Path(args.out_root) / args.dataset / args.organism
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_{args.site}_{args.organism}.npz"
    save_npz(out_path, x, y)

    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
