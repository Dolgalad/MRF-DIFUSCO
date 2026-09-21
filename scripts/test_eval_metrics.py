import sys
sys.path.append("/data1/schulz/MRF-DIFUSCO")

import torch

from difusco.utils.eval_metrics import mis_state_metrics


def check(state, edge_index):
    state = torch.tensor(state)
    edge_index = torch.tensor(edge_index)
    return mis_state_metrics(state, edge_index)


# One undirected edge represented twice.
edge = [
    [0, 1],
    [1, 0],
]

m = check([0, 0], edge)
assert int(m["cost"]) == 0
assert int(m["violations"]) == 0
assert bool(m["feasible"])

m = check([1, 0], edge)
assert int(m["cost"]) == 1
assert int(m["violations"]) == 0
assert bool(m["feasible"])

m = check([1, 1], edge)
assert int(m["cost"]) == 2
assert int(m["violations"]) == 1
assert int(m["violating_vertices"]) == 2
assert not bool(m["feasible"])


# Triangle: each undirected edge represented twice.
triangle = [
    [0, 1, 1, 2, 2, 0],
    [1, 0, 2, 1, 0, 2],
]

m = check([1, 1, 1], triangle)
assert int(m["cost"]) == 3
assert int(m["violations"]) == 3
assert int(m["violating_vertices"]) == 3
assert not bool(m["feasible"])

print("All MIS metric tests passed.")
