import numpy as np


UNRESOLVED = 0
SELECTED = 1
EXCLUDED = -1


def _complete_mis_unary_np(
    solution,
    predictions,
    adj_matrix,
):
  """
  Complete a partially constructed MIS using the original
  vertex-wise greedy rule.

  solution:
      0  = unresolved
      1  = selected
      -1 = excluded
  """
  csr_adj_matrix = adj_matrix.tocsr()
  sorted_predict_labels = np.argsort(-predictions)

  for i in sorted_predict_labels:
    if solution[i] != UNRESOLVED:
      continue

    solution[
        csr_adj_matrix[i].nonzero()[1]
    ] = EXCLUDED

    solution[i] = SELECTED

  return solution


def _mis_decode_unary_np(predictions, adj_matrix):
  """Original vertex-wise greedy MIS decoder."""

  solution = np.zeros(
      predictions.shape[0],
      dtype=np.int64,
  )

  solution = _complete_mis_unary_np(
      solution,
      predictions,
      adj_matrix,
  )

  return (solution == SELECTED).astype(int)


def _mis_decode_static_pairwise_np(
    predictions,
    adj_matrix,
    pair_probs,
    pair_edges,
):
  """
  Greedy MIS decoder using static-matching joint predictions first,
  followed by the original unary greedy completion.

  pair_probs has shape [num_pairs, 3], with states:

      0 -> 00
      1 -> 01
      2 -> 10

  pair_edges has shape [2, num_pairs].
  """

  if pair_probs is None or pair_edges is None:
    raise ValueError(
        "static_pairwise decoding requires "
        "pair_probs and pair_edges"
    )

  pair_probs = np.asarray(pair_probs)
  pair_edges = np.asarray(pair_edges)

  if pair_probs.ndim != 2 or pair_probs.shape[1] != 3:
    raise ValueError(
        f"Expected pair_probs shape [num_pairs, 3], "
        f"got {pair_probs.shape}"
    )

  if pair_edges.ndim != 2 or pair_edges.shape[0] != 2:
    raise ValueError(
        f"Expected pair_edges shape [2, num_pairs], "
        f"got {pair_edges.shape}"
    )

  if pair_probs.shape[0] != pair_edges.shape[1]:
    raise ValueError(
        "pair_probs and pair_edges contain different "
        "numbers of pairs"
    )

  csr_adj_matrix = adj_matrix.tocsr()

  solution = np.zeros(
      predictions.shape[0],
      dtype=np.int64,
  )

  #
  # Decide which pair predictions to consider first.
  #
  # The confidence of a joint prediction is simply the probability
  # of its most likely joint state.
  #
  pair_states = np.argmax(pair_probs, axis=1)
  pair_scores = np.max(pair_probs, axis=1)

  sorted_pairs = np.argsort(-pair_scores)

  #
  # Pairwise phase.
  #
  for pair_idx in sorted_pairs:
    u = int(pair_edges[0, pair_idx])
    v = int(pair_edges[1, pair_idx])

    #
    # The pair remains a candidate only while both vertices are
    # unresolved.
    #
    if (
        solution[u] != UNRESOLVED
        or solution[v] != UNRESOLVED
    ):
      continue

    state = pair_states[pair_idx]

    #
    # 00:
    #
    # The pairwise model does not want either endpoint.
    # Do not permanently exclude them here: simply leave them
    # unresolved and let unary decoding consider them later.
    #
    if state == 0:
      continue

    #
    # 01 -> select v
    # 10 -> select u
    #
    if state == 1:
      selected_node = v
    elif state == 2:
      selected_node = u
    else:
      raise RuntimeError(
          f"Unexpected pair state {state}"
      )

    solution[
        csr_adj_matrix[selected_node].nonzero()[1]
    ] = EXCLUDED

    solution[selected_node] = SELECTED

  #
  # Unary fallback.
  #
  # This is exactly the old greedy decoder, except that vertices
  # already selected/excluded by the pairwise phase are skipped.
  #
  solution = _complete_mis_unary_np(
      solution,
      predictions,
      adj_matrix,
  )

  return (solution == SELECTED).astype(int)


def mis_decode_np(
    predictions,
    adj_matrix,
    selection_mode="unary",
    pair_probs=None,
    pair_edges=None,
):
  if selection_mode == "unary":
    return _mis_decode_unary_np(
        predictions,
        adj_matrix,
    )

  if selection_mode == "static_pairwise":
    return _mis_decode_static_pairwise_np(
        predictions,
        adj_matrix,
        pair_probs,
        pair_edges,
    )

  raise ValueError(
      f"Unknown MIS decode selection mode: "
      f"{selection_mode}"
  )
