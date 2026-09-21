import torch
import numpy as np

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

def split_parallel_states(
        xt,
        num_nodes,
        parallel_sampling,
    ):
      return xt.reshape(
          parallel_sampling,
          num_nodes,
      )
def compute_parallel_metrics(
    xt,
    base_edge_index,
    num_nodes,
    num_samples,
):
    states = xt.reshape(
        num_samples,
        num_nodes,
    )

    return [
        mis_state_metrics(
            states[s],
            base_edge_index,
        )
        for s in range(num_samples)
    ]

def mis_violation_mask(
    state,
    edge_index,
):
    x = binary_state_from_xt(state)
    u, v = canonical_undirected_edges(edge_index)

    return x[u].bool() & x[v].bool()

def append_parallel_metrics_to_trajectory(
    trajectory,
    metrics_list,
):
    fields = {
        "raw_cost": "cost",
        "feasible": "feasible",
        "violations": "violations",
        "violating_vertices": "violating_vertices",
        "violation_rate": "violation_rate",
        "selected_violation_rate": "selected_violation_rate",
        "selected_fraction": "selected_fraction",
    }

    for trajectory_key, metric_key in fields.items():
        trajectory[trajectory_key].append(
            torch.tensor([
                float(m[metric_key].item())
                for m in metrics_list
            ]).cpu().numpy()
        )

def finalize_trajectory(
    sequential_trajectories,
):
    result = {}

    state_fields = [
        "raw_cost",
        "feasible",
        "violations",
        "violating_vertices",
        "violation_rate",
        "selected_violation_rate",
        "selected_fraction",
    ]

    transition_fields = [
        "created_violations",
        "resolved_violations",
    ]

    for key in state_fields:
        per_seq = []

        for traj in sequential_trajectories:
            # [T+1, P] -> [P, T+1]
            arr = np.stack(
                traj[key],
                axis=0,
            ).T

            per_seq.append(arr)

        result[key] = np.concatenate(
            per_seq,
            axis=0,
        )

    for key in transition_fields:
        per_seq = []

        for traj in sequential_trajectories:
            # [T, P] -> [P, T]
            arr = np.stack(
                traj[key],
                axis=0,
            ).T

            per_seq.append(arr)

        result[key] = np.concatenate(
            per_seq,
            axis=0,
        )

    # Step time is shared across parallel samples.
    # Keep one [sequential_sampling, T] array.
    result["step_time"] = np.stack(
        [
            np.asarray(
                traj["step_time"],
                dtype=np.float64,
            )
            for traj
            in sequential_trajectories
        ],
        axis=0,
    )

    return result
