import torch


def binary_state_from_xt(xt, threshold=0.5):
    """
    Convert a diffusion state to a binary MIS assignment.

    Integer states are returned as long tensors.
    Floating-point states are thresholded at `threshold`.
    """
    if torch.is_floating_point(xt):
        return (xt >= threshold).long()
    return xt.long()


def canonical_undirected_edges(edge_index):
    """
    Keep every undirected non-self-loop edge exactly once.

    Assumes that if both (u,v) and (v,u) exist, using u < v
    gives one representative.
    """
    u = edge_index[0].long()
    v = edge_index[1].long()

    mask = u < v
    return u[mask], v[mask]


def mis_state_metrics(state, edge_index):
    """
    Compute metrics for one binary MIS assignment.

    Returns tensors/scalars on the same device.
    """
    x = binary_state_from_xt(state)

    u, v = canonical_undirected_edges(edge_index)

    selected = x.bool()
    violating_edge_mask = selected[u] & selected[v]

    violations = violating_edge_mask.sum()

    violating_vertices_mask = torch.zeros_like(
        selected,
        dtype=torch.bool,
    )

    if violating_edge_mask.any():
        violating_u = u[violating_edge_mask]
        violating_v = v[violating_edge_mask]

        violating_vertices_mask[violating_u] = True
        violating_vertices_mask[violating_v] = True

    violating_vertices = violating_vertices_mask.sum()

    cost = selected.sum()
    feasible = violations == 0

    num_edges = u.numel()
    num_selected = cost

    violation_rate = (
        violations.float() / max(num_edges, 1)
    )

    selected_violation_rate = (
        violating_vertices.float()
        / torch.clamp(num_selected.float(), min=1.0)
    )

    selected_fraction = (
        cost.float() / max(x.numel(), 1)
    )

    return {
        "cost": cost,
        "feasible": feasible,
        "violations": violations,
        "violating_vertices": violating_vertices,
        "violation_rate": violation_rate,
        "selected_violation_rate": selected_violation_rate,
        "selected_fraction": selected_fraction,
        "violating_edge_mask": violating_edge_mask,
    }
