"""Rebuild deploy/demo_cache for the CURRENT track set.

Cached case files are keyed by a hash of NUMERIC_TRACKS, so any change to the
track list (adding the SP4 drug tracks, for example) orphans every previously
bundled file: the deployed image would miss its cache and try to reach VitalDB,
which is exactly what bundling was meant to avoid. Run this after changing tracks.
"""
import json
import shutil

import twin.config as c
from twin.data.vitaldb_loader import load_numeric_frame, _tracks_tag

DEMO_CACHE = c.ROOT / "deploy" / "demo_cache" if hasattr(c, "ROOT") else None


def main():
    tag = _tracks_tag()
    demo_path = c.RESULTS_DIR / "demo_cases.json"
    caseids = [d["caseid"] for d in json.loads(demo_path.read_text())]
    out_dir = (c.DATA_CACHE.parent / "deploy" / "demo_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop files from an older track set; they can never be hit again.
    stale = [p for p in out_dir.glob("case_*.parquet") if tag not in p.name]
    for p in stale:
        p.unlink()
    print(f"tag={tag} | removed {len(stale)} stale demo files", flush=True)

    copied = 0
    for cid in caseids:
        src = c.DATA_CACHE / f"case_{cid}_{tag}.parquet"
        if not src.exists():
            try:
                load_numeric_frame(cid)          # populates the dev cache
            except Exception as exc:
                print(f"  skip {cid}: {exc}", flush=True)
                continue
        if src.exists():
            shutil.copy2(src, out_dir / src.name)
            copied += 1
    print(f"demo cache rebuilt: {copied}/{len(caseids)} cases -> {out_dir}")


if __name__ == "__main__":
    main()
