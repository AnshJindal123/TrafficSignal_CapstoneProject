"""
Demand calibration: keeps an evenly spaced fraction of the vehicles in
combined.rou.xml (randomTrips, period 0.325 -> 11,077 trips/hour), so the
departure-time profile and OD mix are unchanged, only the volume.

The full demand gridlocks this map: under fixed time only ~38% of trips
finish and ~88% of teleports happen at unsignalised priority junctions,
out of any signal agent's reach. bigmap/ was thinned until fixed time
finished ~61%; this does the same here.

  python thin_demand.py 0.5          -> demand_0.5.rou.xml
  python run_baseline.py demand_0.5.rou.xml 0.5    (fixed-time check)
"""
import os
import random
import sys
import xml.etree.ElementTree as ET


def thin(frac, src="combined.rou.xml"):
    root = ET.parse(src).getroot()
    vehicles = [el for el in root if el.tag in ("vehicle", "trip")]
    kept = 0
    for i, el in enumerate(vehicles):
        # keep vehicle i iff floor((i+1)*frac) > floor(i*frac): exactly
        # round(n*frac) vehicles, evenly spread over the hour
        if int((i + 1) * frac) > int(i * frac):
            kept += 1
        else:
            root.remove(el)
    out = f"demand_{frac}.rou.xml"
    ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)
    print(f"{out}: kept {kept}/{len(vehicles)} vehicles")
    return out


def random_subset(frac, seed, src="combined.rou.xml"):
    """Training demand for one seed: a RANDOM subset (not evenly spaced like
    thin()) of src, ~frac of its vehicles, so each seed gives a different OD
    mix at the same volume. Uses its own RNG, so it never disturbs the
    training RNG. An existing file is reused.
      -> demand_train_s<seed>.rou.xml"""
    out = f"demand_train_s{seed}.rou.xml"
    if os.path.exists(out):
        return out
    rng = random.Random(seed)
    root = ET.parse(src).getroot()
    new = ET.Element(root.tag, root.attrib)
    for el in root:   # original order, so departures stay sorted
        if el.tag not in ("vehicle", "trip") or rng.random() < frac:
            new.append(el)
    ET.ElementTree(new).write(out, encoding="UTF-8", xml_declaration=True)
    kept = sum(1 for el in new if el.tag in ("vehicle", "trip"))
    print(f"{out}: {kept} vehicles", flush=True)
    return out


if __name__ == "__main__":
    thin(float(sys.argv[1]))
