"""
Unpacks the merged map + demand once after cloning:
    python setup_map.py
combined.net.xml (33 MB) and combined.rou.xml (5 MB) are committed gzipped
to keep the repo small; every script here reads the plain .xml by relative
path (ET.parse / combined.sumocfg), so run this before anything else.
"""
import gzip
import os
import shutil

for name in ("combined.net.xml", "combined.rou.xml"):
    if os.path.exists(name):
        print(f"{name} already present")
        continue
    with gzip.open(name + ".gz", "rb") as src, open(name, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"unpacked {name}")
