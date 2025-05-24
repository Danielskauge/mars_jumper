# To run this script: pip install urdfpy
# Unless IsaacLab already has it and you're uysing IsaacLab

from pathlib import Path
import argparse
import re
import math
from urdfpy import URDF, Joint

knee_lower = (math.pi/180)*20
knee_upper = (math.pi/180)*170

# ── 1.  regex → (lower, upper) radians  ─────────────────────────
LIMIT_PATTERNS = [
    (re.compile(r"_HFE$"), (       0.0, -math.pi)),
    (re.compile(r"_KFE$"), (knee_lower,  knee_upper)),
    (re.compile(r"_HAA$"), (-math.pi/2,  math.pi/2)),
]

def find_limits(joint_name: str):
    """Return (lower, upper) if name matches a pattern, else None."""
    for regex, limits in LIMIT_PATTERNS:
        if regex.search(joint_name):
            return limits
    return None

# ── 2.  CLI  ─────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Add joint limits to URDF.")
    p.add_argument("input",  help="Path to original URDF",  type=Path)
    p.add_argument("output", nargs="?", help="Path to save limited URDF", type=Path)
    return p.parse_args()

# ── 3.  Main logic  ──────────────────────────────────────────────
def main():
    args = parse_args()
    in_path  = args.input.expanduser().resolve()
    out_path = (args.output.expanduser().resolve()
                if args.output else in_path.with_stem(in_path.stem + "_limited"))
    if not in_path.is_file():
        raise FileNotFoundError(f"Input URDF not found: {in_path}")

    robot = URDF.load(in_path)

    changed = []        # track what we modify for a helpful log
    for j in robot.joints:
        limits = find_limits(j.name)
        if j.type == "continuous" and limits:
            lower, upper = limits
            j.type = "revolute"
            if j.limit is None:
                j.limit = Joint.Limit(lower, upper, effort=10.0, velocity=2.0)
            else:
                j.limit.lower, j.limit.upper = lower, upper
            changed.append(j.name)

    robot.save(out_path)
    print(f"✓ Saved  →  {out_path}")
    if changed:
        print("  Modified joints:")
        for name in changed:
            print("   •", name)
    else:
        print("  (No joints matched patterns.)")

if __name__ == "__main__":
    main()