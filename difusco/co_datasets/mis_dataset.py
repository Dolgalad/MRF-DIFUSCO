"""MIS (Maximal Independent Set) dataset."""

import glob
import os
import pickle5 as pickle

import numpy as np
import torch

from torch_geometric.data import Data as GraphData


def greedy_static_matching(edges, num_nodes):
    """Return indices of a deterministic maximal matching.

    edges: [E, 2] array of undirected graph edges.
    """
    used = np.zeros(num_nodes, dtype=bool)
    matching_edge_indices = []

    # Sort candidates deterministically without changing the actual edge order.
    candidates = sorted(
        range(len(edges)),
        key=lambda k: (
            min(int(edges[k, 0]), int(edges[k, 1])),
            max(int(edges[k, 0]), int(edges[k, 1])),
        ),
    )

    for edge_idx in candidates:
        u, v = edges[edge_idx]

        if not used[u] and not used[v]:
            matching_edge_indices.append(edge_idx)
            used[u] = True
            used[v] = True

    return np.asarray(matching_edge_indices, dtype=np.int64)

class MISDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, data_label_dir=None):
    self.data_file = data_file
    self.file_lines = glob.glob(data_file)
    self.data_label_dir = data_label_dir
    print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    with open(self.file_lines[idx], "rb") as f:
      graph = pickle.load(f)

    num_nodes = graph.number_of_nodes()

    if self.data_label_dir is None:
      node_labels = [_[1] for _ in graph.nodes(data='label')]
      if node_labels is not None and node_labels[0] is not None:
        node_labels = np.array(node_labels, dtype=np.int64)
      else:
        node_labels = np.zeros(num_nodes, dtype=np.int64)
    else:
      base_label_file = os.path.basename(self.file_lines[idx]).replace('.gpickle', '_unweighted.result')
      node_label_file = os.path.join(self.data_label_dir, base_label_file)
      with open(node_label_file, 'r') as f:
        node_labels = [int(_) for _ in f.read().splitlines()]
      node_labels = np.array(node_labels, dtype=np.int64)
      assert node_labels.shape[0] == num_nodes

    undirected_edges = np.array(graph.edges, dtype=np.int64)
    
    matching_edge_indices = greedy_static_matching(
        undirected_edges,
        num_nodes,
    )
    
    num_undirected_edges = undirected_edges.shape[0]
    
    edges = np.concatenate(
        [undirected_edges, undirected_edges[:, ::-1]],
        axis=0,
    )
    # add self loop
    self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
    edges = np.concatenate([edges, self_loop], axis=0)

    pair_mask = np.zeros(edges.shape[0], dtype=bool)
    pair_mask[matching_edge_indices] = True

    pair_edges = edges[pair_mask]

    matched_nodes = np.unique(pair_edges.reshape(-1))
    num_matched_nodes = len(matched_nodes)
    num_unmatched_nodes = num_nodes - num_matched_nodes
    
    #print(
    #  f"[matching] nodes={num_nodes}, "
    #  f"pairs={pair_edges.shape[0]}, "
    #  f"covered={num_matched_nodes}, "
    #  f"uncovered={num_unmatched_nodes}"
    #)

    if pair_edges.shape[0] > 0:
        flat_nodes = pair_edges.reshape(-1)
        assert len(np.unique(flat_nodes)) == len(flat_nodes)
        assert np.all(pair_edges[:,0] != pair_edges[:,1])

    edges = edges.T



    return num_nodes, node_labels, edges, pair_mask

  def __getitem__(self, idx):
    num_nodes, node_labels, edge_index, pair_mask = self.get_example(idx)
    graph_data = GraphData(x=torch.from_numpy(node_labels),
                           edge_index=torch.from_numpy(edge_index),
                           pair_mask=torch.from_numpy(pair_mask),
                           )

    point_indicator = np.array([num_nodes], dtype=np.int64)
    return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        graph_data,
        torch.from_numpy(point_indicator).long(),
    )
