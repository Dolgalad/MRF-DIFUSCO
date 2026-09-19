"""MIS (Maximal Independent Set) dataset."""

import glob
import os
import pickle5 as pickle

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import Data as GraphData

from matchings import (
    balanced_matching_family,
    greedy_balanced_matching_family,
)

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

def matching_family_to_membership(
    undirected_edges,
    family,
    num_total_edges,
):
  """Convert a family of edge-pair lists to [E_total, K] masks.

  Only the first orientation of each undirected edge is activated,
  matching the convention currently used by pair_mask.
  """
  num_matchings = len(family)

  membership = np.zeros(
      (num_total_edges, num_matchings),
      dtype=bool,
  )

  edge_to_idx = {
      tuple(sorted((int(u), int(v)))): i
      for i, (u, v) in enumerate(undirected_edges)
  }

  for matching_idx, matching in enumerate(family):
    for u, v in matching:
      edge = tuple(sorted((int(u), int(v))))
      membership[edge_to_idx[edge], matching_idx] = True

  return membership

def save_matching_family(path, family):
  """Save the raw matching family, independent of edge ordering."""
  os.makedirs(os.path.dirname(path), exist_ok=True)

  family_array = np.asarray(family, dtype=np.int64)

  tmp_path = path + ".tmp.npz"

  np.savez_compressed(
      tmp_path,
      family=family_array,
  )

  os.replace(tmp_path, path)

class MISDataset(torch.utils.data.Dataset):
  def __init__(self, 
               data_file, 
               data_label_dir=None,
               prediction_type="unary",
               num_pairwise_matchings=32,
               matching_seed=0,
               matching_vertex_balance=0.05,
               matching_cache_dir=None,
               matching_generator="greedy",
               ):
    self.data_file = data_file
    self.file_lines = glob.glob(data_file)
    self.data_label_dir = data_label_dir

    self.prediction_type = prediction_type
    self.num_pairwise_matchings = num_pairwise_matchings
    self.matching_seed = matching_seed
    self.matching_vertex_balance = matching_vertex_balance
    self.matching_cache_dir = matching_cache_dir
    self.matching_generator = matching_generator

    print(
        f'Loaded "{data_file}" with '
        f'{len(self.file_lines)} examples'
    )

    if self.prediction_type == "random_pairwise":
      if self.matching_cache_dir is None:
        raise ValueError(
            "matching_cache_dir is required for random_pairwise"
        )

      os.makedirs(
          self.matching_cache_dir,
          exist_ok=True,
      )

      self.precompute_matching_cache()
  def _matching_cache_path(self, idx):
    graph_file = self.file_lines[idx]

    base = os.path.basename(graph_file)
    base = os.path.splitext(base)[0]

    return os.path.join(
        self.matching_cache_dir,
        f"{base}.npz",
    )
  def _load_or_create_matching_family(
      self,
      idx,
      graph,
  ):
    cache_path = self._matching_cache_path(idx)
  
    if os.path.exists(cache_path):
      with np.load(cache_path) as data:
        family_array = data["family"]
  
      return [
          [
              tuple(map(int, edge))
              for edge in matching
          ]
          for matching in family_array
      ]
  
    #family = balanced_matching_family(
    #    graph,
    #    num_matchings=self.num_pairwise_matchings,
    #    seed=self.matching_seed,
    #    vertex_balance=self.matching_vertex_balance,
    #)

    if self.matching_generator == "balanced":
      family = balanced_matching_family(
          graph,
          num_matchings=self.num_pairwise_matchings,
          seed=self.matching_seed,
          vertex_balance=self.matching_vertex_balance,
      )
    
    elif self.matching_generator == "greedy":
      family = greedy_balanced_matching_family(
          graph,
          num_matchings=self.num_pairwise_matchings,
          seed=self.matching_seed,
          vertex_balance=self.matching_vertex_balance,
      )
    
    else:
      raise ValueError(
          f"Unknown matching generator: "
          f"{self.matching_generator}"
      )
  
    save_matching_family(
        cache_path,
        family,
    )
  
    return family

  def precompute_matching_cache(self):
    num_graphs = len(self.file_lines)
  
    print(
        f"Preparing balanced matching cache: "
        f"{num_graphs} graphs, "
        f"{self.num_pairwise_matchings} matchings/graph"
    )
  
    num_created = 0
  
    for idx in tqdm(
        range(num_graphs),
        desc=(
            f"Matching cache "
            f"({self.matching_generator}, "
            f"K={self.num_pairwise_matchings})"
        ),
        unit="graph",
    ):
      cache_path = self._matching_cache_path(idx)
  
      if os.path.exists(cache_path):
        continue
  
      with open(self.file_lines[idx], "rb") as f:
        graph = pickle.load(f)
  
      self._load_or_create_matching_family(
          idx,
          graph,
      )
  
      num_created += 1
  
      if num_created % 100 == 0:
        print(
            f"[matching cache] created "
            f"{num_created} new files"
        )
  
    print(
        f"[matching cache] done: "
        f"{num_created} new files"
    )

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

    pair_membership = None

    if self.prediction_type == "random_pairwise":
      family = self._load_or_create_matching_family(
          idx,
          graph,
      )
    
      pair_membership = matching_family_to_membership(
          undirected_edges=undirected_edges,
          family=family,
          num_total_edges=edges.shape[0],
      )

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

    return (
        num_nodes,
        node_labels,
        edges,
        pair_mask,
        pair_membership,
    )

  #def __getitem__(self, idx):
  #  num_nodes, node_labels, edge_index, pair_mask = self.get_example(idx)
  #  graph_data = GraphData(x=torch.from_numpy(node_labels),
  #                         edge_index=torch.from_numpy(edge_index),
  #                         pair_mask=torch.from_numpy(pair_mask),
  #                         )

  #  point_indicator = np.array([num_nodes], dtype=np.int64)
  #  return (
  #      torch.LongTensor(np.array([idx], dtype=np.int64)),
  #      graph_data,
  #      torch.from_numpy(point_indicator).long(),
  #  )
  def __getitem__(self, idx):
    (
        num_nodes,
        node_labels,
        edge_index,
        pair_mask,
        pair_membership,
    ) = self.get_example(idx)
  
    graph_kwargs = {
        "x": torch.from_numpy(node_labels),
        "edge_index": torch.from_numpy(edge_index),
        "pair_mask": torch.from_numpy(pair_mask),
    }
  
    if pair_membership is not None:
      graph_kwargs["pair_membership"] = (
          torch.from_numpy(pair_membership)
      )
  
    graph_data = GraphData(**graph_kwargs)
  
    point_indicator = np.array(
        [num_nodes],
        dtype=np.int64,
    )
  
    return (
        torch.LongTensor(
            np.array([idx], dtype=np.int64)
        ),
        graph_data,
        torch.from_numpy(point_indicator).long(),
    )
