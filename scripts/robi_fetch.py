"""Fetch selected files of one ROBI scene without downloading the whole 3.7 GB archive.

ROBI (Yang et al., IROS 2021, CC BY-NC-SA 4.0) is hosted as one zip per object on Google Drive.
Full downloads often fail with "quota exceeded", but HTTP range requests work, so we read the
zip directory remotely and extract only the members we need.

RealSense D415 data exists only in scenes 4, 5 (full bin) and 8, 9 (low bin).
Usage: python scripts/robi_fetch.py --object Zigzag --scene 4 --out data/robi
"""
import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

DRIVE_IDS = {
    "Zigzag": "1lc2yNZo2W8cnd8oUe6Vb5cdUnR1t8YTK",
    "Chrome_screw": "1qqG0KVW6_Upxc95Wuk9HecR0JrxGZSd4",
    "Gear": "1qWcz7AD_9JwwFGskUfDFUirw0KOiGJ75",
    "Eye_bolt": "1LCUF0xis1PtsU6TRZPuqtS_r0om0VMUh",
    "Tube_fitting": "17HwGES3oqTo4ASOwoFxX4Xz6rdLI0RJt",
    "DIN_connector": "1NfHuroh76xL3AuvZOGCDR89RlNufsj9b",
    "DSub_connector": "1O914_sp1fnTQ8iN1bj0x_H--ZBhFf7OC",
}
CAD_ID = "1WkL4DHY-HpwsOIJ3f0Ef4EZg2V4fLpT2"
TOOLKIT = "https://raw.githubusercontent.com/junyang224/ROBI_Toolkit/main/"
UA = {"User-Agent": "Mozilla/5.0"}


def fetch_small(url: str, target: Path):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read())


def drive_url(file_id: str) -> str:
    base = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    html = urllib.request.urlopen(urllib.request.Request(base, headers=UA)).read().decode("utf-8", "ignore")
    m = re.search(r'name="uuid" value="([^"]+)"', html)
    return base + "&confirm=t" + (f"&uuid={m.group(1)}" if m else "")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        if data[:15].lower().startswith(b"<!doctype html") or data[:5] == b"<html":
            raise RuntimeError("Google Drive returned an HTML page (quota?) instead of data")
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def wanted(name: str, obj: str, scene: int) -> bool:
    prefix = f"{obj}/Scene_{scene}/"
    if not name.startswith(prefix) or name.endswith("/"):
        return False
    rest = name[len(prefix):]
    return rest == "GT_world2obj.json" or rest.startswith(("RealSense/Depth/", "RealSense/list.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="Zigzag", choices=DRIVE_IDS)
    ap.add_argument("--scene", type=int, default=4)
    ap.add_argument("--out", default="data/robi")
    ap.add_argument("--list", action="store_true", help="only list archive members of the scene")
    args = ap.parse_args()

    out = Path(args.out)
    # Camera intrinsics (toolkit) and CAD models (small, downloaded whole).
    for name in ("RealSense_DEPTH.yml", "RealSense_LEFT.yml", "RealSense_RIGHT.yml"):
        fetch_small(TOOLKIT + "cam/" + name, out / "toolkit" / "cam" / name)
    if not (out / "_CAD_models").exists():
        fetch_small(drive_url(CAD_ID), out / "CAD_models.zip")
        zipfile.ZipFile(out / "CAD_models.zip").extractall(out)

    f = io.BufferedReader(HttpRangeFile(drive_url(DRIVE_IDS[args.object])), buffer_size=1 << 20)
    with zipfile.ZipFile(f) as z:
        names = [n for n in z.namelist() if n.startswith(f"{args.object}/Scene_{args.scene}/")]
        if args.list:
            folders = sorted({n.rsplit("/", 1)[0] for n in names})
            for d in folders:
                print(d, sum(n.startswith(d + "/") for n in names))
            return
        todo = [n for n in names if wanted(n, args.object, args.scene)]
        for i, n in enumerate(todo):
            target = out / n
            if target.exists() and target.stat().st_size == z.getinfo(n).file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(n))
            if i % 20 == 0:
                print(f"{i + 1}/{len(todo)} {n}")
        print(f"done: {len(todo)} files in {out / args.object / f'Scene_{args.scene}'}")


if __name__ == "__main__":
    main()
