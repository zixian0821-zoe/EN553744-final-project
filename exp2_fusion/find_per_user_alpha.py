\
\
\
\
\
\
\
\
   
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PIPELINE_DIR = _Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_PIPELINE_DIR))

import os
import sys
from pathlib import Path

ROOTS = [
    Path("/content/drive/MyDrive/Experiment2"),
]

KEYWORDS = [
    "per_user",
    "peruser",
    "user_alpha",
    "alpha_per",
    "learned_alpha",
    "alpha",
]

EXTS = {".npy", ".npz", ".pt", ".ckpt", ".json", ".csv", ".pkl", ".txt"}

def scan(root: Path) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    if not root.exists():
        return hits
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            fn_lower = fn.lower()
            ext = os.path.splitext(fn_lower)[1]
            if ext not in EXTS:
                continue
            matched = next((kw for kw in KEYWORDS if kw in fn_lower), None)
            if matched is None:
                parent_lower = os.path.basename(dirpath).lower()
                matched = next((kw for kw in KEYWORDS if kw in parent_lower), None)
            if matched is None:
                continue
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            hits.append((p, size, matched))
    return hits

def human(n: int) -> str:
    if n < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def main() -> int:
    any_root_ok = False
    all_hits: list[tuple[Path, int, str]] = []
    for r in ROOTS:
        exists = r.exists()
        print(f"[root] {r}  ->  {'EXISTS' if exists else 'missing'}")
        if exists:
            any_root_ok = True
            all_hits.extend(scan(r))
    if not any_root_ok:
        print("No Drive root found. Make sure drive.mount('/content/drive') ran.")
        return 1

    seen: set[Path] = set()
    uniq: list[tuple[Path, int, str]] = []
    for p, s, kw in all_hits:
        if p in seen:
            continue
        seen.add(p)
        uniq.append((p, s, kw))

    if not uniq:
        print("\nNo alpha-related artefacts found under the Drive roots above.")
        print("=> per-user-alpha model output was NOT saved. Skip that plot.")
        return 0

    kw_rank = {kw: i for i, kw in enumerate(KEYWORDS)}
    uniq.sort(key=lambda t: (kw_rank.get(t[2], 99), str(t[0])))

    print(f"\nFound {len(uniq)} candidate file(s):\n")
    print(f"{'KW':<14} {'SIZE':>9}  PATH")
    print("-" * 100)
    for p, s, kw in uniq:
        print(f"{kw:<14} {human(s):>9}  {p}")

    per_user_hits = [h for h in uniq if h[2] in ("per_user", "peruser", "user_alpha", "alpha_per")]
    print("\n--- summary ---")
    if per_user_hits:
        print(f"PER-USER alpha artefacts: {len(per_user_hits)} file(s). Can plot per-user histogram.")
        print("Key files to inspect:")
        for p, s, _ in per_user_hits[:10]:
            print(f"  - {p}")
    else:
        print("PER-USER alpha artefacts: NONE.")
        print("=> Drop the per-user-alpha histogram from the plot plan.")
        print("=> Remaining 3 plots (alpha-sweep / val-vs-test scatter / best_epoch vs alpha) are unaffected.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
