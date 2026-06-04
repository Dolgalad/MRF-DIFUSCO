#!/bin/bash

export DATA_ROOT=/users/schulz/DIFUSCO/data/mis_er_100_5000

python - <<'PY'
import os
import pickle
from pathlib import Path

DATA_ROOT = Path(os.environ["DATA_ROOT"])

for split in ["train", "val", "test"]:
    graph_dir = DATA_ROOT / split
    ann_dir = DATA_ROOT / f"{split}_annotations"

    if not graph_dir.exists():
        print(f"Skipping missing split dir: {graph_dir}")
        continue

    if not ann_dir.exists():
        print(f"Skipping missing annotation dir: {ann_dir}")
        continue

    for gpath in sorted(graph_dir.glob("*.gpickle")):
        # ER_700_800_0.15_0.gpickle
        # -> ER_700_800_0.15_0_unweighted.result
        apath = ann_dir / f"{gpath.stem}_unweighted.result"

        if not apath.exists():
            raise FileNotFoundError(f"Missing annotation for {gpath}: {apath}")

        with open(apath) as f:
            label = [int(x) for x in f.read().strip().split()]

        with open(gpath, "rb") as f:
            G = pickle.load(f)

        if len(label) != G.number_of_nodes():
            raise ValueError(
                f"{gpath.name}: annotation length {len(label)} "
                f"!= num_nodes {G.number_of_nodes()}"
            )

        for i, y in enumerate(label):
            G.nodes[i]["label"] = int(y)

        G.graph["label"] = label
        G.graph["label_sum"] = int(sum(label))

        with open(gpath, "wb") as f:
            pickle.dump(G, f)

        print(f"{split}: wrote label of size {len(label)} to {gpath.name}")

print("Done.")
PY
