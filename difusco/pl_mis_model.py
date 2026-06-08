"""Lightning module for training the DIFUSCO MIS model."""

import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from co_datasets.mis_dataset import MISDataset
from utils.diffusion_schedulers import InferenceSchedule
from pl_meta_model import COMetaModel
from utils.mis_utils import mis_decode_np


class MISModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(MISModel, self).__init__(param_args=param_args, node_feature_only=True)

    data_label_dir = None
    if self.args.training_split_label_dir is not None:
      data_label_dir = os.path.join(self.args.storage_path, self.args.training_split_label_dir)

    self.train_dataset = MISDataset(
        data_file=os.path.join(self.args.storage_path, self.args.training_split),
        data_label_dir=data_label_dir,
        input_representation=self.args.input_representation,
        bipartite_cache_dir=self.args.bipartite_cache_dir,
        bipartite_cache_version=self.args.bipartite_cache_version,
        bipartite_cache_refresh=self.args.bipartite_cache_refresh,
        bipartite_with_lbp=self.args.bipartite_with_lbp,
        bipartite_with_nmf=self.args.bipartite_with_nmf,
        bipartite_with_ss=self.args.bipartite_with_ss,
        bipartite_return_factor_graph=self.args.bipartite_return_factor_graph,
        bipartite_return_vc_graph=self.args.bipartite_return_vc_graph,
    )

    self.test_dataset = MISDataset(
        data_file=os.path.join(self.args.storage_path, self.args.test_split),
        input_representation=self.args.input_representation,
        bipartite_cache_dir=self.args.bipartite_cache_dir,
        bipartite_cache_version=self.args.bipartite_cache_version,
        bipartite_cache_refresh=self.args.bipartite_cache_refresh,
        bipartite_with_lbp=self.args.bipartite_with_lbp,
        bipartite_with_nmf=self.args.bipartite_with_nmf,
        bipartite_with_ss=self.args.bipartite_with_ss,
        bipartite_return_factor_graph=self.args.bipartite_return_factor_graph,
        bipartite_return_vc_graph=self.args.bipartite_return_vc_graph,
    )

    self.validation_dataset = MISDataset(
        data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        input_representation=self.args.input_representation,
        bipartite_cache_dir=self.args.bipartite_cache_dir,
        bipartite_cache_version=self.args.bipartite_cache_version,
        bipartite_cache_refresh=self.args.bipartite_cache_refresh,
        bipartite_with_lbp=self.args.bipartite_with_lbp,
        bipartite_with_nmf=self.args.bipartite_with_nmf,
        bipartite_with_ss=self.args.bipartite_with_ss,
        bipartite_return_factor_graph=self.args.bipartite_return_factor_graph,
        bipartite_return_vc_graph=self.args.bipartite_return_vc_graph,
    )
  def _use_mrf_inference(self):
    return getattr(self.args, "mrf_inference", "none") not in (None, "none", "off", "")


  def forward_mrf_marginals(
      self,
      x,
      t,
      edge_index,
      graph_data,
      variable_mask,
  ):
      raw_params = self.forward(x, t, edge_index)
  
      # Only variable nodes correspond to MIS variables.
      variable_params = raw_params[variable_mask]
  
      if self.args.mrf_inference == "lbp":
          if not hasattr(graph_data, "lbp_edges"):
              raise AttributeError(
                  "MRF inference requested with mrf_inference='lbp', "
                  "but graph_data has no lbp_edges. "
                  "Enable the matching dataset/cache flag."
              )
          return self._run_lbp_marginals(variable_params, graph_data)
  
      if self.args.mrf_inference == "nmf":
          if not hasattr(graph_data, "nmf_edges"):
              raise AttributeError(
                  "MRF inference requested with mrf_inference='nmf', "
                  "but graph_data has no nmf_edges. "
                  "Enable the matching dataset/cache flag."
              )
          return self._run_nmf_marginals(variable_params, graph_data)
  
      raise ValueError(f"Unknown mrf_inference={self.args.mrf_inference!r}")


  def forward(self, x, t, edge_index, graph_data=None):
    if getattr(self.model, "use_mrf_inference", False):
      return self.model(
          x,
          t,
          edge_index=edge_index,
          graph_data=graph_data,
      )

    return self.model(x, t, edge_index=edge_index)

  def categorical_training_step(self, batch, batch_idx):
      _, graph_data, point_indicator = batch
  
      device = graph_data.x.device
      edge_index = graph_data.edge_index.to(device).reshape(2, -1)
  
      is_bipartite = (
          hasattr(graph_data, "variable_mask")
          and hasattr(graph_data, "constraint_mask")
      )
  
      if is_bipartite:
          variable_mask = graph_data.variable_mask.bool().to(device)
          constraint_mask = graph_data.constraint_mask.bool().to(device)
  
          if not hasattr(graph_data, "batch"):
              raise AttributeError(
                  "Batched bipartite graph_data is missing graph_data.batch. "
                  "This is needed to assign one diffusion timestep per graph."
              )
  
          graph_batch = graph_data.batch.to(device)
          variable_batch = graph_batch[variable_mask]
          batch_size = int(graph_batch.max().item()) + 1
  
          # Labels are only variable-node labels.
          node_labels = graph_data.x[variable_mask].reshape(-1).long().to(device)
  
          # One diffusion timestep per graph, expanded to variable nodes.
          point_indicator = torch.bincount(
              variable_batch,
              minlength=batch_size,
          )
  
          t_per_graph_np = np.random.randint(
              1,
              self.diffusion.T + 1,
              batch_size,
          ).astype(int)
  
          t_variables_np = np.repeat(
              t_per_graph_np,
              point_indicator.reshape(-1).cpu().numpy(),
          )
  
          node_labels_onehot = F.one_hot(
              node_labels.long(),
              num_classes=2,
          ).float()
  
          node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)
  
          # xt_variables is binary, shape [num_variable_nodes].
          xt_variables = self.diffusion.sample(
              node_labels_onehot,
              t_variables_np,
          )
  
          xt_variables = xt_variables.reshape(-1).long().to(device)
  
          x_features = self._build_bipartite_node_features(
              x_template=graph_data.x,
              variable_values=xt_variables.float(),
              variable_mask=variable_mask,
              constraint_mask=constraint_mask,
              edge_index=edge_index,
          )
  
          # Give every node in a graph the same timestep, including constraints.
          t_per_graph = torch.from_numpy(t_per_graph_np).float().to(device)
          t_full = t_per_graph[graph_batch].reshape(-1)
  
          x0_pred_full = self.forward(
              x_features.float(),
              t_full.float(),
              edge_index,
              graph_data=graph_data,
          )
  
          x0_pred_variables = x0_pred_full[variable_mask].reshape(-1, 2)
  
          loss = F.cross_entropy(
              x0_pred_variables,
              node_labels,
          )
  
      else:
          t = np.random.randint(
              1,
              self.diffusion.T + 1,
              point_indicator.shape[0],
          ).astype(int)
  
          node_labels = graph_data.x.reshape(-1).long().to(device)
  
          node_labels_onehot = F.one_hot(
              node_labels.long(),
              num_classes=2,
          ).float()
  
          node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)
  
          t = torch.from_numpy(t).long()
          t = t.repeat_interleave(
              point_indicator.reshape(-1).cpu(),
              dim=0,
          ).numpy()
  
          xt = self.diffusion.sample(node_labels_onehot, t)
          xt = xt.reshape(-1).long().to(device)
  
          t = torch.from_numpy(t).float().to(device).reshape(-1)
  
          x0_pred = self.forward(
              xt.float(),
              t.float(),
              edge_index,
              graph_data=graph_data,
          )
  
          loss = F.cross_entropy(
              x0_pred.reshape(-1, 2),
              node_labels,
          )
  
      self.log("train/loss", loss)
      return loss
  def _get_mis_variable_labels(self, graph_data):
    is_bipartite = (
      hasattr(graph_data, "variable_mask")
      and hasattr(graph_data, "constraint_mask")
    )

    if is_bipartite:
      variable_mask = graph_data.variable_mask.bool()
      labels = graph_data.x[variable_mask].reshape(-1)
      return labels, variable_mask

    labels = graph_data.x.reshape(-1)
    return labels, None

  #def _fill_bipartite_constraint_values(
  #    self,
  #    assignment_values,
  #    edge_index,
  #    variable_mask,
  #    constraint_mask,
  #):
  #    """Set constraint-node assignment values to the sum of neighboring variables.
  #
  #    Assumes a bipartite variable-constraint graph.
  #
  #    assignment_values:
  #      shape [num_nodes]
  #      contains current assignment/noisy assignment values for variable nodes.
  #
  #    Returns:
  #      shape [num_nodes]
  #      same as assignment_values, but constraint nodes are overwritten with
  #      the sum of adjacent variable assignment values.
  #    """
  #    assignment_values = assignment_values.reshape(-1)
  #    variable_mask = variable_mask.bool().reshape(-1).to(assignment_values.device)
  #    constraint_mask = constraint_mask.bool().reshape(-1).to(assignment_values.device)
  #    edge_index = edge_index.long().to(assignment_values.device)
  #
  #    src, dst = edge_index[0], edge_index[1]
  #
  #    src_is_var = variable_mask[src]
  #    dst_is_var = variable_mask[dst]
  #    src_is_con = constraint_mask[src]
  #    dst_is_con = constraint_mask[dst]
  #
  #    # Directed messages from variable nodes to constraint nodes.
  #    var_to_con = src_is_var & dst_is_con
  #    con_to_var = src_is_con & dst_is_var
  #
  #    variable_nodes = torch.cat([src[var_to_con], dst[con_to_var]], dim=0)
  #    constraint_nodes = torch.cat([dst[var_to_con], src[con_to_var]], dim=0)
  #
  #    constraint_values = torch.zeros_like(assignment_values)
  #    constraint_values.index_add_(
  #        0,
  #        constraint_nodes,
  #        assignment_values[variable_nodes],
  #    )
  #
  #    out = assignment_values.clone()
  #    out[constraint_mask] = constraint_values[constraint_mask]
  #    return out

  def _fill_bipartite_constraint_values(
    self,
    assignment_values,
    edge_index,
    variable_mask,
    constraint_mask,
  ):
    assignment_values = assignment_values.reshape(-1)
    device = assignment_values.device

    variable_mask = variable_mask.bool().reshape(-1).to(device)
    constraint_mask = constraint_mask.bool().reshape(-1).to(device)
    edge_index = edge_index.long().to(device)

    src, dst = edge_index[0], edge_index[1]

    # Keep only edges oriented variable -> constraint.
    keep = variable_mask[src] & constraint_mask[dst]

    variable_nodes = src[keep]
    constraint_nodes = dst[keep]

    # If edge_index was only constraint -> variable, fall back to that orientation.
    if variable_nodes.numel() == 0:
        keep = constraint_mask[src] & variable_mask[dst]
        variable_nodes = dst[keep]
        constraint_nodes = src[keep]

    constraint_values = torch.zeros_like(assignment_values)
    constraint_values.index_add_(
        0,
        constraint_nodes,
        assignment_values[variable_nodes],
    )

    out = assignment_values.clone()
    out[constraint_mask] = constraint_values[constraint_mask]
    return out

  def _build_bipartite_node_features(
      self,
      x_template,
      variable_values,
      variable_mask,
      constraint_mask,
      edge_index,
  ):
      """Build full node features for a bipartite MIS graph.
  
      variable_values contains the current noisy/diffused values only for
      variable nodes. Constraint-node values are reconstructed from the
      neighboring variable values.
  
      Feature layout:
        x[:, 0] = assignment-like value
        x[:, 1] = node type, 0 for variables and 1 for constraints
      """
      device = variable_values.device
  
      variable_mask = variable_mask.bool().reshape(-1).to(device)
      constraint_mask = constraint_mask.bool().reshape(-1).to(device)
      edge_index = edge_index.long().to(device)
  
      x_full_assignment = x_template.clone().float().reshape(-1).to(device)
      x_full_assignment[variable_mask] = variable_values.float().reshape(-1)
  
      x_full_assignment = self._fill_bipartite_constraint_values(
          assignment_values=x_full_assignment,
          edge_index=edge_index,
          variable_mask=variable_mask,
          constraint_mask=constraint_mask,
      )
  
      node_type = constraint_mask.float().reshape(-1)
  
      x_features = torch.stack(
          [x_full_assignment, node_type],
          dim=-1,
      )
  
      return x_features

  def gaussian_training_step(self, batch, batch_idx):
    _, graph_data, point_indicator = batch
  
    edge_index = graph_data.edge_index
    device = graph_data.x.device
  
    is_bipartite = (
        hasattr(graph_data, "variable_mask")
        and hasattr(graph_data, "constraint_mask")
    )
  
    #if is_bipartite:
    #  variable_mask = graph_data.variable_mask.bool().to(device)
  
    #  # Diffusion labels are only the variable-node labels.
    #  node_labels = graph_data.x[variable_mask].reshape(-1).to(device)
  
    #  # For now assume batch_size=1 in the smoke test.
    #  point_indicator = variable_mask.sum().view(1)
    if is_bipartite:
      variable_mask = graph_data.variable_mask.bool().to(device)

      node_labels = graph_data.x[variable_mask].reshape(-1).to(device)

      if not hasattr(graph_data, "batch"):
        raise AttributeError(
            "Batched bipartite graph_data is missing graph_data.batch. "
            "This is needed to assign one diffusion timestep per graph."
        )

      graph_batch = graph_data.batch.to(device)
      variable_batch = graph_batch[variable_mask]

      batch_size = int(graph_batch.max().item()) + 1

      # Number of variable nodes per graph. This replaces point_indicator
      # for the bipartite representation.
      point_indicator = torch.bincount(
          variable_batch,
          minlength=batch_size,
      )
    else:
      variable_mask = None
      node_labels = graph_data.x.reshape(-1).to(device)
  
    t = np.random.randint(
        1,
        self.diffusion.T + 1,
        point_indicator.shape[0],
    ).astype(int)
  
    node_labels = node_labels.float() * 2 - 1
    node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
    node_labels = node_labels.unsqueeze(1).unsqueeze(1)
  
    t = torch.from_numpy(t).long()
    t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()
  
    xt, epsilon = self.diffusion.sample(node_labels, t)
  
    edge_index = edge_index.to(device).reshape(2, -1)
  
    if is_bipartite:
      t_tensor = torch.from_numpy(t).float().to(device).reshape(-1)

      t_per_graph = torch.zeros(
          batch_size,
          device=device,
          dtype=t_tensor.dtype,
      )
      t_per_graph.scatter_(
          0,
          variable_batch,
          t_tensor,
      )
      
      xt = xt.reshape(-1).to(device=device, dtype=graph_data.x.dtype)
      epsilon = epsilon.reshape(-1).to(device=device, dtype=graph_data.x.dtype)

      xt_full_assignment = graph_data.x.clone().float().reshape(-1).to(device)
      xt_full_assignment[variable_mask] = xt

      xt_full_assignment = self._fill_bipartite_constraint_values(
              assignment_values=xt_full_assignment,
              edge_index=graph_data.edge_index,
              variable_mask=graph_data.variable_mask,
              constraint_mask=graph_data.constraint_mask,
      )

      
      node_type = graph_data.constraint_mask.float().reshape(-1).to(device)
      
      # Bipartite node feature layout:
      #   x[:, 0] = assignment-like diffusion value
      #   x[:, 1] = node type indicator, 0 for variables and 1 for constraints
      x_features = torch.stack(
          [xt_full_assignment, node_type],
          dim=-1,
      )
      
      t_full = t_per_graph[graph_batch]

      num_nodes = x_features.shape[0]
      
      ei = edge_index.long().to(device)
      if ei.numel() == 0:
          raise RuntimeError("Empty edge_index in bipartite batch.")
      
      bad = (
          ei.min().item() < 0
          or ei.max().item() >= num_nodes
      )
      if bad:
          raise RuntimeError(
              "Invalid bipartite edge_index before forward: "
              f"num_nodes={num_nodes}, "
              f"edge_min={ei.min().item()}, "
              f"edge_max={ei.max().item()}, "
              f"x_features_shape={tuple(x_features.shape)}, "
              f"xt_full_assignment_shape={tuple(xt_full_assignment.shape)}, "
              f"graph_data.x_shape={tuple(graph_data.x.shape)}, "
              f"batch_size={batch_size}"
          )      
      epsilon_pred_full = self.forward(
          x_features.float(),
          t_full.float(),
          edge_index,
          graph_data=graph_data,
      )


      epsilon_pred = epsilon_pred_full[variable_mask].reshape(-1)

      loss = F.mse_loss(epsilon_pred, epsilon.float())  
    else:
      xt = xt.reshape(-1).to(device)
      epsilon = epsilon.reshape(-1).to(device)
      t_tensor = torch.from_numpy(t).float().to(device).reshape(-1)
  
      epsilon_pred = self.forward(
          xt.float(),
          t_tensor.float(),
          edge_index,
          graph_data=graph_data,
      ).reshape(-1)
  
      loss = F.mse_loss(epsilon_pred, epsilon.float())
  
    self.log("train/loss", loss)
    return loss
  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None, graph_data=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
          graph_data=graph_data,
      )
      x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  def gaussian_denoise_step(self, xt, t, device, edge_index=None, target_t=None, graph_data=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
          graph_data=graph_data,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def _is_bipartite_graph_data(self, graph_data):
    return (
        hasattr(graph_data, "variable_mask")
        and hasattr(graph_data, "constraint_mask")
    )

  def test_step(self, batch, batch_idx, draw=False, split='test'):
    real_batch_idx, graph_data, point_indicator = batch

    if self._is_bipartite_graph_data(graph_data):
      return self.bipartite_test_step(batch, batch_idx, draw=draw, split=split)

    device = graph_data.x.device
    node_labels = graph_data.x
    edge_index = graph_data.edge_index

    stacked_predict_labels = []
    edge_index = edge_index.to(device).reshape(2, -1)
    edge_index_np = edge_index.cpu().numpy()
    adj_mat = scipy.sparse.coo_matrix(
        (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
    )

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(node_labels.float())

      if self.args.parallel_sampling > 1:
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      xt = xt.reshape(-1)

      cur_edge_index = edge_index
      if self.args.parallel_sampling > 1:
        cur_edge_index = self.duplicate_edge_index(
            edge_index,
            node_labels.shape[0],
            device,
        )

      batch_size = 1
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(
          inference_schedule=self.args.inference_schedule,
          T=self.diffusion.T,
          inference_T=steps,
      )

      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
        t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

        if self.diffusion_type == 'gaussian':
          xt = self.gaussian_denoise_step(
              xt, t1, device, cur_edge_index, target_t=t2)
        else:
          xt = self.categorical_denoise_step(
              xt, t1, device, cur_edge_index, target_t=t2)

      if self.diffusion_type == 'gaussian':
        predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
      else:
        predict_labels = xt.float().cpu().detach().numpy() + 1e-6

      stacked_predict_labels.append(predict_labels)

    predict_labels = np.concatenate(stacked_predict_labels, axis=0)
    all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

    splitted_predict_labels = np.split(predict_labels, all_sampling)
    solved_solutions = [
        mis_decode_np(predict_labels, adj_mat)
        for predict_labels in splitted_predict_labels
    ]
    solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
    best_solved_cost = np.max(solved_costs)

    gt_cost = node_labels.cpu().numpy().sum()
    metrics = {
        f"{split}/gt_cost": gt_cost,
    }

    for k, v in metrics.items():
      self.log(k, float(v), on_epoch=True, sync_dist=True)

    print()
    print("best_solved_cost = ", best_solved_cost)
    print()

    self.log(
        f"{split}/solved_cost",
        float(best_solved_cost),
        prog_bar=True,
        on_epoch=True,
        sync_dist=True,
    )

    return metrics

  def bipartite_test_step(self, batch, batch_idx, draw=False, split='test'):
    real_batch_idx, graph_data, point_indicator = batch
    device = graph_data.x.device

    if not hasattr(graph_data, "graph_edge_index"):
      raise AttributeError(
          "Bipartite graph_data is missing graph_edge_index. "
          "Store the original MIS graph edges as graph_data.graph_edge_index."
      )

    variable_mask = graph_data.variable_mask.bool().to(device)
    constraint_mask = graph_data.constraint_mask.bool().to(device)
    node_labels = graph_data.x[variable_mask].reshape(-1).to(device)

    model_edge_index = graph_data.edge_index.to(device).reshape(2, -1)
    graph_edge_index = graph_data.graph_edge_index.cpu().long().reshape(2, -1)

    if graph_edge_index.numel() > 0:
      assert graph_edge_index.min().item() >= 0
      assert graph_edge_index.max().item() < node_labels.shape[0], (
          "graph_edge_index must use variable-local ids in [0, num_variables)."
      )

    edge_index_np = graph_edge_index.numpy()
    adj_mat = scipy.sparse.coo_matrix(
        (
            np.ones_like(edge_index_np[0]),
            (edge_index_np[0], edge_index_np[1]),
        ),
        shape=(node_labels.shape[0], node_labels.shape[0]),
    )

    stacked_predict_labels = []
    num_total_nodes = graph_data.x.shape[0]
    num_variable_nodes = node_labels.shape[0]

    for _ in range(self.args.sequential_sampling):
      if self.diffusion_type == "gaussian":
        xt = torch.randn_like(node_labels.float()).reshape(-1)
        xt.requires_grad = True
      else:
        xt = torch.randn_like(node_labels.float()).reshape(-1)
        xt = (xt > 0).long()

      cur_model_edge_index = model_edge_index
      cur_variable_mask = variable_mask
      cur_x_template = graph_data.x.float().reshape(-1).to(device)

      #if self.args.parallel_sampling > 1:
      #  xt = xt.repeat(self.args.parallel_sampling)
      #  xt = torch.randn_like(xt)
      #  xt.requires_grad = True

      #  cur_model_edge_index = self.duplicate_edge_index(
      #      model_edge_index,
      #      num_total_nodes,
      #      device,
      #  )
      #  cur_variable_mask = variable_mask.repeat(self.args.parallel_sampling)
      #  cur_x_template = cur_x_template.repeat(self.args.parallel_sampling)
      if self.args.parallel_sampling > 1:
        if self.diffusion_type == "gaussian":
          xt = xt.repeat(self.args.parallel_sampling)
          xt = torch.randn_like(xt)
          xt.requires_grad = True
        else:
          xt = xt.repeat(self.args.parallel_sampling).long()
    
        cur_model_edge_index = self.duplicate_edge_index(
            model_edge_index,
            num_total_nodes,
            device,
        )
    
        cur_variable_mask = variable_mask.repeat(self.args.parallel_sampling)
        cur_constraint_mask = constraint_mask.repeat(self.args.parallel_sampling)
        cur_x_template = cur_x_template.repeat(self.args.parallel_sampling)
      else:
        cur_constraint_mask = constraint_mask

      batch_size = 1
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(
          inference_schedule=self.args.inference_schedule,
          T=self.diffusion.T,
          inference_T=steps,
      )

      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
        t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

        if self.diffusion_type == "gaussian":
          xt = self.bipartite_gaussian_denoise_step(
              xt_variables=xt,
              t=t1,
              device=device,
              x_template=cur_x_template,
              variable_mask=cur_variable_mask,
              edge_index=cur_model_edge_index,
              target_t=t2,
              graph_data=graph_data,
          )
        else:
          xt = self.bipartite_categorical_denoise_step(
              xt_variables=xt,
              t=t1,
              device=device,
              x_template=cur_x_template,
              variable_mask=cur_variable_mask,
              constraint_mask=constraint_mask,
              edge_index=cur_model_edge_index,
              target_t=t2,
              graph_data=graph_data,
          )

      if self.diffusion_type == "gaussian":
        predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
      else:
        predict_labels = xt.float().cpu().detach().numpy() + 1e-6
      stacked_predict_labels.append(predict_labels)

    print(xt)

    predict_labels = np.concatenate(stacked_predict_labels, axis=0)
    all_sampling = self.args.sequential_sampling * self.args.parallel_sampling
    splitted_predict_labels = np.split(predict_labels, all_sampling)

    for sample in splitted_predict_labels:
      assert sample.shape[0] == num_variable_nodes, (
          f"Expected prediction length {num_variable_nodes}, got {sample.shape[0]}"
      )

    print(predict_labels)
    solved_solutions = [
        mis_decode_np(predict_labels, adj_mat)
        for predict_labels in splitted_predict_labels
    ]
    solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
    best_solved_cost = np.max(solved_costs)

    gt_cost = node_labels.cpu().numpy().sum()
    metrics = {
        f"{split}/gt_cost": gt_cost,
    }

    for k, v in metrics.items():
      self.log(k, float(v), on_epoch=True, sync_dist=True)

    print()
    print(best_solved_cost)
    print()

    self.log(
        f"{split}/solved_cost",
        float(best_solved_cost),
        prog_bar=True,
        on_epoch=True,
        sync_dist=True,
    )

    return metrics

  def bipartite_gaussian_denoise_step(
      self,
      xt_variables,
      t,
      device,
      x_template,
      variable_mask,
      edge_index,
      target_t=None,
  ):
    with torch.no_grad():
      # Keep this identical in spirit to gaussian_denoise_step:
      # gaussian_posterior expects a CPU tensor / numpy-compatible timestep.
      t_cpu = torch.from_numpy(t).view(1)

      # The model input timestep still needs to live on the model/device.
      t_model = t_cpu.float().to(device)

      xt_full_assignment = x_template.clone().float().reshape(-1).to(device)
      xt_full_assignment[variable_mask] = xt_variables.float().reshape(-1).to(device)

      constraint_mask = ~variable_mask

      xt_full_assignment = self._fill_bipartite_constraint_values(
          assignment_values=xt_full_assignment,
          edge_index=edge_index,
          variable_mask=variable_mask,
          constraint_mask=constraint_mask,
      )
      
      node_type = (~variable_mask).float().reshape(-1).to(device)
      
      # Bipartite node feature layout:
      #   x[:, 0] = assignment-like diffusion value
      #   x[:, 1] = node type indicator, 0 for variables and 1 for constraints
      x_features = torch.stack(
          [xt_full_assignment, node_type],
          dim=-1,
      )
      
      t_full = torch.zeros(
          xt_full_assignment.shape[0],
          device=device,
          dtype=t_model.dtype,
      )
      t_full[variable_mask] = t_model
      
      pred_full = self.forward(
          x_features.float(),
          t_full.float(),
          edge_index.long().to(device),
          graph_data=graph_data,
      )

      pred_variables = pred_full[variable_mask].reshape(-1)

      xt_variables = self.gaussian_posterior(
          target_t,
          t_cpu,
          pred_variables,
          xt_variables,
      )

      return xt_variables
  #def test_step(self, batch, batch_idx, draw=False, split='test'):
  #  device = batch[-1].device

  #  real_batch_idx, graph_data, point_indicator = batch
  #  node_labels = graph_data.x
  #  edge_index = graph_data.edge_index

  #  stacked_predict_labels = []
  #  
  #  if self._is_bipartite_graph_data(graph_data):
  #    variable_mask = graph_data.variable_mask.bool().to(device)
  #    node_labels = graph_data.x[variable_mask].reshape(-1)
  #    model_edge_index = graph_data.edge_index.to(device).reshape(2,-1)
  #    graph_edge_index = graph_data.graph_edge_index.cpu().long().reshape(2,-1)
  #    edge_index_np = graph_edge_index.numpy()
  #    adj_mat = scipy.sparse.coo_matrix(
  #            (
  #                np.ones_like(edge_index_np[0]),
  #                (edge_index_np[0], edge_index_np[1]),
  #            ),
  #            shape=(node_labels.shape[0], node_labels.shape[0]),
  #    )
  #  edge_index = edge_index.to(node_labels.device).reshape(2, -1)
  #  edge_index_np = edge_index.cpu().numpy()
  #  adj_mat = scipy.sparse.coo_matrix(
  #      (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
  #  )

  #  for _ in range(self.args.sequential_sampling):
  #    xt = torch.randn_like(node_labels.float())
  #    if self.args.parallel_sampling > 1:
  #      xt = xt.repeat(self.args.parallel_sampling, 1, 1)
  #      xt = torch.randn_like(xt)

  #    if self.diffusion_type == 'gaussian':
  #      xt.requires_grad = True
  #    else:
  #      xt = (xt > 0).long()
  #    xt = xt.reshape(-1)

  #    if self.args.parallel_sampling > 1:
  #      edge_index = self.duplicate_edge_index(edge_index, node_labels.shape[0], device)

  #    batch_size = 1
  #    steps = self.args.inference_diffusion_steps
  #    time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
  #                                      T=self.diffusion.T, inference_T=steps)

  #    for i in range(steps):
  #      t1, t2 = time_schedule(i)
  #      t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
  #      t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

  #      if self.diffusion_type == 'gaussian':
  #        xt = self.gaussian_denoise_step(
  #            xt, t1, device, edge_index, target_t=t2)
  #      else:
  #        xt = self.categorical_denoise_step(
  #            xt, t1, device, edge_index, target_t=t2)

  #    if self.diffusion_type == 'gaussian':
  #      predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
  #    else:
  #      predict_labels = xt.float().cpu().detach().numpy() + 1e-6
  #    stacked_predict_labels.append(predict_labels)

  #  predict_labels = np.concatenate(stacked_predict_labels, axis=0)
  #  all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

  #  splitted_predict_labels = np.split(predict_labels, all_sampling)
  #  solved_solutions = [mis_decode_np(predict_labels, adj_mat) for predict_labels in splitted_predict_labels]
  #  solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
  #  best_solved_cost = np.max(solved_costs)

  #  gt_cost = node_labels.cpu().numpy().sum()
  #  metrics = {
  #      f"{split}/gt_cost": gt_cost,
  #  }
  #  for k, v in metrics.items():
  #    self.log(k, v, on_epoch=True, sync_dist=True)
  #  self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
  #  return metrics

  def bipartite_categorical_denoise_step(
      self,
      xt_variables,
      t,
      device,
      x_template,
      variable_mask,
      constraint_mask,
      edge_index,
      target_t=None,
      graph_data=None,
  ):
      with torch.no_grad():
          t_cpu = torch.from_numpy(t).view(1)
          t_model = t_cpu.float().to(device)
  
          xt_variables = xt_variables.reshape(-1).long().to(device)
  
          x_features = self._build_bipartite_node_features(
              x_template=x_template,
              variable_values=xt_variables.float(),
              variable_mask=variable_mask,
              constraint_mask=constraint_mask,
              edge_index=edge_index,
          )
  
          # Same timestep for all nodes in this sampled graph copy.
          t_full = torch.full(
              (x_features.shape[0],),
              float(t_model.item()),
              device=device,
              dtype=torch.float32,
          )
  
          x0_pred_full = self.forward(
              x_features.float(),
              t_full.float(),
              edge_index.long().to(device),
              graph_data=graph_data,
          )
  
          x0_pred_variables = x0_pred_full[variable_mask].reshape(-1, 2)
  
          x0_pred_prob = x0_pred_variables.reshape(
              (1, xt_variables.shape[0], -1, 2)
          ).softmax(dim=-1)
  
          xt_variables = self.categorical_posterior(
              target_t,
              t_cpu,
              x0_pred_prob,
              xt_variables,
          )
  
          return xt_variables.reshape(-1)
  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
