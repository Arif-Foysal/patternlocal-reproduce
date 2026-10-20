"""Compare the four completed XOR sweeps against the paper's claims."""
import glob
import json
import re


def load():
    rows = {}
    for k in ("xor-white-sweep-1", "xor-white-sweep-5",
              "xor-corr-sweep-1", "xor-corr-sweep-5"):
        for f in glob.glob(f"/tmp/sweeps/{k}/*_results.json"):
            for _, r in json.load(open(f)).items():
                rows[r["config"]] = r
    return rows


def sort_key(c):
    m = re.search(r"XOR_(\w+)_alpha([0-9.]+)\.npz", c)
    return (m.group(1), float(m.group(2)))


def main():
    rows = load()
    methods = ["lime", "patternlocal", "analytical_ig", "analytical_grad"]
    print(f"{'config':24s} {'acc':>6s} | "
          + " ".join(f"{m[:7]:>7s}" for m in methods)
          + f" | {'PL<LIME':>7s}")
    print("-" * 78)
    for c in sorted(rows, key=sort_key):
        r = rows[c]
        if r.get("skipped"):
            print(f"{c:24s} SKIPPED (acc={r['test_accuracy']:.3f})")
            continue
        tag = c.replace("XOR_", "").replace(".npz", "")
        vals = " ".join(f"{r[m]['emd_mean']:7.4f}" for m in methods)
        win = "YES" if r["patternlocal"]["emd_mean"] < r["lime"]["emd_mean"] else "no"
        print(f"{tag:24s} {r['test_accuracy']:6.3f} | {vals} | {win:>7s}")

    print("\nIME (lower = less false-positive mass):")
    print(f"{'config':24s} " + " ".join(f"{m[:7]:>7s}" for m in methods))
    for c in sorted(rows, key=sort_key):
        r = rows[c]
        if r.get("skipped"):
            continue
        tag = c.replace("XOR_", "").replace(".npz", "")
        vals = " ".join(f"{r[m]['ime_mean']:7.4f}" for m in methods)
        print(f"{tag:24s} {vals}")


if __name__ == "__main__":
    main()
