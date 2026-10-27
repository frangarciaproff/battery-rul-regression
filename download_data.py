"""Download the processed HNEI dataset, pinned to a fixed commit, and check its hash.

The source repository has no licence file, so the CSV is not committed here.

Usage:
    python download_data.py [output_path]
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

SOURCE_REPO = "https://github.com/ignavinuales/Battery_RUL_Prediction"
COMMIT = "10320b0acc6c77d9aeee1a4d3d8136d2fccfb0fa"
URL = (
    "https://raw.githubusercontent.com/ignavinuales/Battery_RUL_Prediction/"
    f"{COMMIT}/Datasets/HNEI_Processed/Final%20Database.csv"
)
SHA256 = "36f4269f1e4b8191c061e717cc9768b19e5d0fce8bcaf0bb5756812b6f88342e"
DEFAULT_PATH = Path("data") / "hnei_final_database.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(out_path):
    if out_path.exists() and sha256(out_path) == SHA256:
        print(f"{out_path} already present, hash OK")
        return
    out_path.parent.mkdir(parents = True, exist_ok = True)
    print(f"Downloading from {SOURCE_REPO} @ {COMMIT[:7]}")
    urllib.request.urlretrieve(URL, out_path)
    digest = sha256(out_path)
    if digest != SHA256:
        out_path.unlink()
        raise SystemExit(f"Hash mismatch: got {digest}, expected {SHA256}")
    print(f"Saved {out_path}, hash OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH)
