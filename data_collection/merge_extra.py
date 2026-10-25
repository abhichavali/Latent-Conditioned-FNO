"""Merge dataset B (K0 ICs per equation) with its supplement (generate.py --supplement) in place:
B/eq_X/{u,static}.npy = concat along the IC axis, spec["samples"] += supplement samples.json; a supplement with
K1 windows (equation reselected from scratch) replaces the stored ones instead.
Resumable and idempotent: each file is checked by its IC count, written to a tmp file and renamed; an equation's
supplement dir is removed once all its files are merged (keeps peak disk ~ final size).

  python merge_extra.py /workspace/data/B /workspace/data/B_extra 18 30
"""
import json
import os
import shutil
import sys

import numpy as np

B, X, K0, K1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])


def put(path, write):
    with open(path + ".tmp", "wb" if path.endswith(".npy") else "w") as f:
        write(f)
    os.rename(path + ".tmp", path)


ids = sorted(d for d in os.listdir(X) if d.startswith("eq_"))
for n, eid in enumerate(ids):
    b, x = os.path.join(B, eid), os.path.join(X, eid)
    xs = json.load(open(os.path.join(x, "samples.json")))
    full = len(xs) == K1
    for name in ("u.npy", "static.npy"):
        if os.path.exists(os.path.join(x, name)):
            cur = np.load(os.path.join(b, name), mmap_mode="r")
            if cur.shape[0] == K0:
                new = np.load(os.path.join(x, name)) if full else np.concatenate([cur, np.load(os.path.join(x, name))])
                put(os.path.join(b, name), lambda f: np.save(f, new))
            assert np.load(os.path.join(b, name), mmap_mode="r").shape[0] == K1, (eid, name)
    spec = json.load(open(os.path.join(b, "spec.json")))
    if len(spec["samples"]) == K0:
        spec["samples"] = xs if full else spec["samples"] + xs
        put(os.path.join(b, "spec.json"), lambda f: json.dump(spec, f))
    assert len(spec["samples"]) == K1, eid
    shutil.rmtree(x)
    if n % 256 == 0:
        print(n, eid, flush=True)
print("merged", len(ids), flush=True)
