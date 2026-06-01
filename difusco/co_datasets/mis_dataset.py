"""MIS (Maximal Independent Set) dataset."""

import glob
import os
import json
import time
import hashlib
import fcntl
from contextlib import contextmanager
import pickle5 as pickle

import numpy as np
import torch

from torch_geometric.data import Data as GraphData

from bpropy.co.mis import MIS

@contextmanager
def file_lock(lock_path):
    """Simple process-level file lock for cache writes."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class MISDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, 
                 data_label_dir=None, 
                 input_representation="original",
                 bipartite_cache_dir=None,
                 bipartite_cache_version="v1",
                 bipartite_cache_refresh=False,
                 bipartite_with_lbp=False,
                 bipartite_with_nmf=False,
                 bipartite_with_ss=False,
                 bipartite_return_factor_graph=False,
                 bipartite_return_vc_graph=False,
      ):
        self.data_file = data_file
        self.file_lines = glob.glob(data_file)
        self.data_label_dir = data_label_dir
        self.input_representation = input_representation
  
        self.bipartite_cache_dir = bipartite_cache_dir
        self.bipartite_cache_version = bipartite_cache_version
        self.bipartite_cache_refresh = bipartite_cache_refresh
        self.bipartite_with_lbp = bipartite_with_lbp
        self.bipartite_with_nmf = bipartite_with_nmf
        self.bipartite_with_ss = bipartite_with_ss
        self.bipartite_return_factor_graph = bipartite_return_factor_graph
        self.bipartite_return_vc_graph = bipartite_return_vc_graph
  
        if self.input_representation not in ("original", "bipartite"):
            raise ValueError(
                    f"Unknown input_representation={self.input_representation!r}. "
                    "Expected 'original' or 'bipartite'."
                    )
  
        if self.input_representation == "bipartite":
            self._init_bipartite_cache_dir()
        print(
                f'Loaded "{data_file}" with {len(self.file_lines)} examples '
                f'using input_representation="{self.input_representation}"'
        )

    def _init_bipartite_cache_dir(self):
        """Initialize the directory used to store cached bipartite examples."""
        if self.bipartite_cache_dir is None:
          # Put cache next to the graph files by default.
          # self.data_file may be a glob like /path/train/*gpickle.
          data_root = os.path.dirname(os.path.dirname(os.path.abspath(self.data_file)))
          self.bipartite_cache_dir = os.path.join(
              data_root,
              f"bipartite_cache_{self.bipartite_cache_version}",
          )

        os.makedirs(self.bipartite_cache_dir, exist_ok=True)
    def _bipartite_cache_key(self, idx):
        graph_path = self.file_lines[idx]
        st = os.stat(graph_path)
  
        payload = {
            "idx": int(idx),
            "source_path": os.path.abspath(graph_path),
            "source_mtime_ns": int(st.st_mtime_ns),
            "source_size": int(st.st_size),
            "cache_version": self.bipartite_cache_version,
            "with_lbp": bool(self.bipartite_with_lbp),
            "with_nmf": bool(self.bipartite_with_nmf),
            "with_ss": bool(self.bipartite_with_ss),
            "return_factor_graph": bool(self.bipartite_return_factor_graph),
            "return_vc_graph": bool(self.bipartite_return_vc_graph),
        }
  
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest(), payload
    def _bipartite_cache_paths(self, idx):
        graph_path = self.file_lines[idx]
        graph_name = os.path.basename(graph_path).replace(".gpickle", "")
        cache_key, payload = self._bipartite_cache_key(idx)

        fname = f"{idx:06d}__{graph_name}__{cache_key}"

        paths = {
            "data": os.path.join(self.bipartite_cache_dir, fname + ".pt"),
            "meta": os.path.join(self.bipartite_cache_dir, fname + ".meta.json"),
            "factor_graph": os.path.join(self.bipartite_cache_dir, fname + ".fg.pt"),
            "vc_graph": os.path.join(self.bipartite_cache_dir, fname + ".vcg.pt"),
            "lbp_edges": os.path.join(self.bipartite_cache_dir, fname + ".lbp.pt"),
            "nmf_edges": os.path.join(self.bipartite_cache_dir, fname + ".nmf.pt"),
            "lock": os.path.join(self.bipartite_cache_dir, fname + ".lock"),
        }

        return paths, cache_key, payload
    def _load_bipartite_cache(self, paths, cache_key):
        if not (os.path.exists(paths["data"]) and os.path.exists(paths["meta"])):
            return None

        try:
            with open(paths["meta"], "r") as f:
                meta = json.load(f)

            if meta.get("cache_key") != cache_key:
                return None

            required_sidecars = []

            if self.bipartite_return_factor_graph:
                required_sidecars.append("factor_graph")
            if self.bipartite_return_vc_graph:
                required_sidecars.append("vc_graph")
            if self.bipartite_with_lbp:
                required_sidecars.append("lbp_edges")
            if self.bipartite_with_nmf:
                required_sidecars.append("nmf_edges")

            for key in required_sidecars:
                if not os.path.exists(paths[key]):
                    return None

            data = torch.load(paths["data"], weights_only=False)

            if self.bipartite_return_factor_graph:
                data.factor_graph = torch.load(paths["factor_graph"], weights_only=False)

            if self.bipartite_return_vc_graph:
                data.vc_graph = torch.load(paths["vc_graph"], weights_only=False)

            if self.bipartite_with_lbp:
                data.lbp_edges = torch.load(paths["lbp_edges"], weights_only=False)

            if self.bipartite_with_nmf:
                data.nmf_edges = torch.load(paths["nmf_edges"], weights_only=False)

            data.preprocess_cached = True
            data.preprocess_time_s = float(meta.get("preprocess_time_s", 0.0))
            return data

        except Exception as exc:
            print(f"Failed to load bipartite cache {paths['data']}: {exc}")
            return None
    def _save_bipartite_cache(
        self,
        paths,
        data,
        meta,
        factor_graph=None,
        vc_graph=None,
        lbp_edges=None,
        nmf_edges=None,
):
        tmp_paths = {}

        objects = {
            "data": data,
            "factor_graph": factor_graph,
            "vc_graph": vc_graph,
            "lbp_edges": lbp_edges,
            "nmf_edges": nmf_edges,
        }

        for key, obj in objects.items():
            if obj is None:
                continue
            tmp_path = paths[key] + ".tmp"
            torch.save(obj, tmp_path)
            tmp_paths[key] = tmp_path

        tmp_meta = paths["meta"] + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2)

        for key, tmp_path in tmp_paths.items():
            os.replace(tmp_path, paths[key])

        os.replace(tmp_meta, paths["meta"])
  
    def __len__(self):
        return len(self.file_lines)
  
    def get_example(self, idx):
        if self.input_representation == "original":
            return self.get_original_example(idx)
        if self.input_representation == "bipartite":
            return self.get_bipartite_example(idx)
        raise RuntimeError("Unreachable input_representation branch.")
  
    def get_original_example(self, idx):
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
  
        edges = np.array(graph.edges, dtype=np.int64)
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        # add self loop
        self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
        edges = np.concatenate([edges, self_loop], axis=0)
        edges = edges.T
  
        return num_nodes, node_labels, edges
    
    def get_bipartite_example(self, idx):
        paths, cache_key, cache_payload = self._bipartite_cache_paths(idx)
    
        if not self.bipartite_cache_refresh:
            cached = self._load_bipartite_cache(paths, cache_key)
            if cached is not None:
                return cached
    
        with file_lock(paths["lock"]):
            if not self.bipartite_cache_refresh:
                cached = self._load_bipartite_cache(paths, cache_key)
                if cached is not None:
                    return cached
    
            t0 = time.time()
            data, sidecars = self._build_bipartite_example(idx)
            preprocess_time_s = time.time() - t0
    
            meta = {
                "cache_key": cache_key,
                "preprocess_time_s": preprocess_time_s,
                "created": time.time(),
                "has_factor_graph": sidecars.get("factor_graph") is not None,
                "has_vc_graph": sidecars.get("vc_graph") is not None,
                "has_lbp_edges": sidecars.get("lbp_edges") is not None,
                "has_nmf_edges": sidecars.get("nmf_edges") is not None,
                **cache_payload,
            }
    
            self._save_bipartite_cache(
                paths,
                data,
                meta,
                factor_graph=sidecars.get("factor_graph"),
                vc_graph=sidecars.get("vc_graph"),
                lbp_edges=sidecars.get("lbp_edges"),
                nmf_edges=sidecars.get("nmf_edges"),
            )
    
            data.preprocess_cached = False
            data.preprocess_time_s = preprocess_time_s
    
            return data  

    def _load_node_labels(self, idx, graph):
        num_nodes = graph.number_of_nodes()

        if self.data_label_dir is None:
            node_labels = [_[1] for _ in graph.nodes(data="label")]
            if node_labels is not None and node_labels[0] is not None:
                node_labels = np.array(node_labels, dtype=np.int64)
            else:
                node_labels = np.zeros(num_nodes, dtype=np.int64)
        else:
            base_label_file = os.path.basename(self.file_lines[idx]).replace(
                ".gpickle", "_unweighted.result"
            )
            node_label_file = os.path.join(self.data_label_dir, base_label_file)
            with open(node_label_file, "r") as f:
                node_labels = [int(_) for _ in f.read().splitlines()]
            node_labels = np.array(node_labels, dtype=np.int64)

        assert node_labels.shape[0] == num_nodes
        return node_labels

    def mis_to_bipartite_data(
        self,
        pb,
        *,
        y=None,
        with_lbp=False,
        with_nmf=False,
        with_ss=False,
        return_graphs=False,
    ):
        """Lightweight MIS -> homogeneous VC PyG Data conversion.
      
        This intentionally avoids bpropy.adapters.vc_to_data.problem_to_data because
        that path computes LP/handcrafted features, PE features, and hetero graphs.
        For DIFUSCO bipartite input, node features are only current assignment values.
        """
        from torch_geometric.data import Data
        from bpropy.graphs.vc_graph import VCGraph
        from bpropy.utils.graph_utils import get_edge_index
        from bpropy.mrf.factor_graph import FactorGraph
        from bpropy.mrf.lbp_edges import LBPEdges
        from bpropy.mrf.nmf_edges import NMFEdges
        from bpropy.mrf.sufficient_statistics import x2ss
      
        vcg = VCGraph(pb)
      
        edge_index = get_edge_index(vcg)
      
        num_vc_nodes = vcg.number_of_nodes()
        bipartite = torch.zeros(num_vc_nodes, dtype=torch.long)
        factor_sizes = torch.ones(num_vc_nodes, dtype=torch.long)
        neighbors = torch.zeros(num_vc_nodes, dtype=torch.long)
      
        for n, d in vcg.nodes(data=True):
            is_constraint = int(d["bipartite"])
            bipartite[n] = is_constraint
            degree = len(list(vcg.neighbors(n)))
            neighbors[n] = degree
            if is_constraint:
                factor_sizes[n] = degree
      
        # Current assignment values only.
        # Variable nodes get y, constraint nodes get 0.
        x = torch.zeros((num_vc_nodes, 1), dtype=torch.float32)
        if y is not None:
            if isinstance(y, np.ndarray):
                y_tensor = torch.from_numpy(y)
            else:
                y_tensor = torch.as_tensor(y)
            y_tensor = y_tensor.to(torch.float32).view(-1, 1)
            x[: y_tensor.shape[0]] = y_tensor
      
        data = Data(
            x=x,
            edge_index=edge_index,
            bipartite=bipartite,
            factor_sizes=factor_sizes,
            neighbors=neighbors,
        )
      
        data.problem_type_name = "MIS"
        data.problem_type_code = torch.tensor([0], dtype=torch.long)
      
        # Keep both labels: one generic PyG-style y and one explicit original label.
        if y is not None:
            data.y = y_tensor.view(-1).to(torch.long)
            data.original_node_labels = data.y.clone()
      
        fg = None
        if return_graphs or with_lbp or with_nmf or (with_ss and y is not None):
            fg = FactorGraph(pb)
      
        if with_lbp:
            data.lbp_edges = LBPEdges(fg)
      
        if with_nmf:
            data.nmf_edges = NMFEdges(fg)
      
        if with_ss and y is not None:
            data.y_ss = torch.from_numpy(
                x2ss(data.y.cpu().numpy().astype(int), fg)
            ).to(torch.float32)

        if return_graphs:
            return data, fg, vcg
      
        return data

    def _build_bipartite_example(self, idx):
        with open(self.file_lines[idx], "rb") as f:
            graph = pickle.load(f)

        node_labels = self._load_node_labels(idx, graph)

        #pb = self._networkx_mis_to_bpropy_mis(graph)
        pb = MIS(graph)

        try:
            from bpropy.adapters.vc_to_data import problem_to_data
        except ImportError as exc:
            raise ImportError(
                "input_representation='bipartite' requires bpropy to be installed "
                "and importable in the active environment."
            ) from exc

        y = torch.from_numpy(node_labels).to(torch.float32)

        need_graphs = (
            self.bipartite_return_factor_graph
            or self.bipartite_return_vc_graph
        )

        out = self.mis_to_bipartite_data(
            pb,
            y=y.numpy(),
            with_lbp=self.bipartite_with_lbp,
            with_nmf=self.bipartite_with_nmf,
            with_ss=self.bipartite_with_ss,
            return_graphs=need_graphs,
        )

        if need_graphs:
            data, factor_graph, vc_graph = out
        else:
            data = out
            factor_graph = None
            vc_graph = None

        lbp_edges = getattr(data, "lbp_edges", None)
        nmf_edges = getattr(data, "nmf_edges", None)

        if hasattr(data, "lbp_edges"):
            del data.lbp_edges
        if hasattr(data, "nmf_edges"):
            del data.nmf_edges

        data.target = y
        data.variable_mask = data.bipartite == 0
        data.constraint_mask = data.bipartite == 1
        data.original_num_nodes = torch.tensor([graph.number_of_nodes()], dtype=torch.long)

        # Current assignment values as the minimal feature signal.
        # Variables receive their assignment value; constraints receive zero.
        assignment_feature = torch.zeros((data.num_nodes, 1), dtype=torch.float32)
        assignment_feature[data.variable_mask] = y.view(-1, 1)
        data.x = assignment_feature

        sidecars = {
            "factor_graph": factor_graph if self.bipartite_return_factor_graph else None,
            "vc_graph": vc_graph if self.bipartite_return_vc_graph else None,
            "lbp_edges": lbp_edges if self.bipartite_with_lbp else None,
            "nmf_edges": nmf_edges if self.bipartite_with_nmf else None,
        }

        return data, sidecars

    def __getitem__(self, idx):
        example = self.get_example(idx)
  
        if self.input_representation == "bipartite":
            point_indicator = torch.tensor(
                [int(example.original_num_nodes.item())],
                dtype=torch.long,
            )
            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                example,
                point_indicator,
            )
  
        num_nodes, node_labels, edge_index = example
        graph_data = GraphData(
            x=torch.from_numpy(node_labels),
            edge_index=torch.from_numpy(edge_index),
        )
        point_indicator = np.array([num_nodes], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
        )

