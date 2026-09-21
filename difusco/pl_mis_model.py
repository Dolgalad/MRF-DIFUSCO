"""Lightning module for training the DIFUSCO MIS model."""

import os
import time
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
from utils.eval_metrics import (
    binary_state_from_xt,
    mis_state_metrics,
    compute_parallel_metrics,
    mis_violation_mask,
    append_parallel_metrics_to_trajectory,
    finalize_trajectory,
)

class MISModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(MISModel, self).__init__(param_args=param_args, node_feature_only=True)

    train_label_dir = None
    val_label_dir = None
    test_label_dir = None

    self._eval_records = {
            "val": [],
            "test": [],
            }
    self._eval_wall_start = {
            "val": None,
            "test": None,
            }
    
    if self.args.training_split_label_dir is not None:
      train_label_dir = os.path.join(
          self.args.storage_path,
          self.args.training_split_label_dir,
      )
    
    if self.args.validation_split_label_dir is not None:
      val_label_dir = os.path.join(
          self.args.storage_path,
          self.args.validation_split_label_dir,
      )
    
    if self.args.test_split_label_dir is not None:
      test_label_dir = os.path.join(
          self.args.storage_path,
          self.args.test_split_label_dir,
      )

    matching_config_name = (
        f"{self.args.matching_generator}"
        f"_K{self.args.num_pairwise_matchings}"
        f"_seed{self.args.matching_seed}"
        f"_vb{self.args.matching_vertex_balance}"
    )
    
    matching_cache_root = os.path.join(
        self.args.storage_path,
        self.args.matching_cache_dir,
        matching_config_name,
    )
    
    self.train_dataset = MISDataset(
        data_file=os.path.join(
            self.args.storage_path,
            self.args.training_split,
        ),
        data_label_dir=train_label_dir,
        prediction_type=self.args.prediction_type,
        num_pairwise_matchings=self.args.num_pairwise_matchings,
        matching_seed=self.args.matching_seed,
        matching_vertex_balance=self.args.matching_vertex_balance,
        matching_cache_dir=os.path.join(
            matching_cache_root,
            "train",
        ),
        matching_generator=self.args.matching_generator,
    )    
    self.validation_dataset = MISDataset(
        data_file=os.path.join(
            self.args.storage_path,
            self.args.validation_split,
        ),
        data_label_dir=val_label_dir,
        prediction_type=self.args.prediction_type,
        num_pairwise_matchings=self.args.num_pairwise_matchings,
        matching_seed=self.args.matching_seed,
        matching_vertex_balance=self.args.matching_vertex_balance,
        matching_cache_dir=os.path.join(
            matching_cache_root,
            "val",
        ),
        matching_generator=self.args.matching_generator,
    )    
    self.test_dataset = MISDataset(
        data_file=os.path.join(
            self.args.storage_path,
            self.args.test_split,
        ),
        data_label_dir=test_label_dir,
        prediction_type=self.args.prediction_type,
        num_pairwise_matchings=self.args.num_pairwise_matchings,
        matching_seed=self.args.matching_seed,
        matching_vertex_balance=self.args.matching_vertex_balance,
        matching_cache_dir=os.path.join(
            matching_cache_root,
            "test",
        ),
        matching_generator=self.args.matching_generator,
    )

  def forward(self, x, t, edge_index):
    return self.model(x, t, edge_index=edge_index)

  def select_random_pair_mask(
      self,
      graph_data,
      edge_index,
  ):
    membership = (
        graph_data.pair_membership
        .bool()
        .to(edge_index.device)
    )
  
    num_matchings = membership.shape[1]
  
    # Determine which graph owns each edge.
    edge_batch = graph_data.batch[
        edge_index[0].long()
    ]
  
    num_graphs = int(graph_data.num_graphs)
  
    # Pick one cached matching independently for every graph.
    selected_matching = torch.randint(
        low=0,
        high=num_matchings,
        size=(num_graphs,),
        device=edge_index.device,
    )
  
    selected_for_edge = selected_matching[
        edge_batch
    ]
  
    edge_ids = torch.arange(
        membership.shape[0],
        device=edge_index.device,
    )
  
    return membership[
        edge_ids,
        selected_for_edge,
    ]

  def static_pairwise_categorical_loss(
      self,
      node_pred,
      edge_pred,
      node_labels,
      edge_index,
      pair_mask,
  ):
    assert pair_mask.dtype == torch.bool
    assert pair_mask.shape[0] == edge_pred.shape[0]
    assert edge_pred.shape[1] == 3
    assert node_pred.shape[0] == node_labels.shape[0]
    assert node_pred.shape[1] == 2
  
    edge_u = edge_index[0]
    edge_v = edge_index[1]

    non_self_loop = edge_u != edge_v

    pair_u = edge_u[non_self_loop]
    pair_v = edge_v[non_self_loop]
  
    # Pair states:
    # 00 -> 0
    # 01 -> 1
    # 10 -> 2
    # 11 -> invalid for an MIS solution
    pair_targets = (
        2 * node_labels[pair_u].long()
        + node_labels[pair_v].long()
    )

    bad = pair_targets == 3

    #if bad.any():
    #  bad_idx = torch.nonzero(bad, as_tuple=False).flatten()
    #
    #  print("FOUND INVALID MIS EDGES")
    #  print("num invalid directed edges:", bad_idx.numel())
    #
    #  for k in bad_idx[:20]:
    #    u = int(edge_index[0, k])
    #    v = int(edge_index[1, k])
    #    yu = int(node_labels[u])
    #    yv = int(node_labels[v])
    #
    #    print(
    #        f"edge_idx={int(k)} "
    #        f"edge=({u},{v}) "
    #        f"labels=({yu},{yv})"
    #    )
    #
    #  raise ValueError("Invalid MIS target: graph edge has state 11.")
  
    if torch.any(pair_targets == 3):
      raise ValueError(
          "Invalid MIS target: graph edge has state 11."
      )
  
    pair_loss = F.cross_entropy(
        edge_pred[non_self_loop],
        pair_targets,
        reduction="sum",
    )
  
    unary_loss = F.cross_entropy(
        node_pred,
        node_labels.long(),
        reduction="sum",
    )
  
    num_blocks = (
        edge_pred.shape[0]
        + node_labels.numel()
    )
  
    loss = (pair_loss + unary_loss) / num_blocks

    unary_loss_mean = unary_loss / node_labels.numel()
    pair_loss_mean = pair_loss / edge_pred.shape[0]

    return loss, unary_loss_mean, pair_loss_mean

  def static_pairwise_to_unary_probs(
      self,
      node_pred,
      edge_pred,
      edge_index,
      pair_mask,
  ):
    node_prob = node_pred.softmax(dim=-1)

    pair_prob = edge_pred[pair_mask].softmax(dim=-1)
    pair_edges = edge_index[:, pair_mask]

    pair_u = pair_edges[0]
    pair_v = pair_edges[1]

    # pair_prob ordering:
    # 0 = 00
    # 1 = 01
    # 2 = 10

    p00 = pair_prob[:, 0]
    p01 = pair_prob[:, 1]
    p10 = pair_prob[:, 2]

    unary_prob = node_prob.clone()

    unary_prob[pair_u, 0] = p00 + p01
    unary_prob[pair_u, 1] = p10

    unary_prob[pair_v, 0] = p00 + p10
    unary_prob[pair_v, 1] = p01

    return unary_prob

  def categorical_training_step(self, batch, batch_idx):
    _, graph_data, point_indicator = batch
    t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
    node_labels = graph_data.x
    edge_index = graph_data.edge_index

    # Sample from diffusion
    node_labels_onehot = F.one_hot(node_labels.long(), num_classes=2).float()
    node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)

    t = torch.from_numpy(t).long()
    t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()

    xt = self.diffusion.sample(node_labels_onehot, t)
    xt = xt * 2 - 1
    xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

    t = torch.from_numpy(t).float()
    t = t.reshape(-1)
    xt = xt.reshape(-1)
    edge_index = edge_index.to(node_labels.device).reshape(2, -1)

    # Denoise
    pair_mask = graph_data.pair_mask.bool()

    prediction = self.forward(
        xt.float().to(node_labels.device),
        t.float().to(node_labels.device),
        edge_index,
    )


    #loss_func = nn.CrossEntropyLoss()
    #loss = loss_func(x0_pred, node_labels)

    if self.args.prediction_type == "unary":
        loss = F.cross_entropy(prediction, node_labels)
        self.log("train/unary_loss", loss)
    else:
        node_pred, edge_pred = prediction
    
        loss, unary_loss_mean, pair_loss_mean = self.static_pairwise_categorical_loss(
            node_pred=node_pred,
            edge_pred=edge_pred,
            node_labels=node_labels,
            edge_index=edge_index,
            pair_mask=pair_mask,
        )
        self.log("train/unary_loss", unary_loss_mean)
        self.log("train/pairwise_loss", pair_loss_mean)

    self.log("train/loss", loss)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    _, graph_data, point_indicator = batch
    t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
    node_labels = graph_data.x
    edge_index = graph_data.edge_index
    device = node_labels.device

    # Sample from diffusion
    node_labels = node_labels.float() * 2 - 1
    node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
    node_labels = node_labels.unsqueeze(1).unsqueeze(1)

    t = torch.from_numpy(t).long()
    t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()
    xt, epsilon = self.diffusion.sample(node_labels, t)

    t = torch.from_numpy(t).float()
    t = t.reshape(-1)
    xt = xt.reshape(-1)
    edge_index = edge_index.to(device).reshape(2, -1)
    epsilon = epsilon.reshape(-1)

    # Denoise
    epsilon_pred = self.forward(
        xt.float().to(device),
        t.float().to(device),
        edge_index,
    )
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def static_pairwise_block_posterior(
      self,
      target_t,
      t,
      pair_x0_prob,
      xt,
      pair_u,
      pair_v,
  ):
    """
    Compute p(z_target | z_source) for matched pair blocks.
  
    pair_x0_prob:
        [M, 3] clean-state probabilities over
        {00, 01, 10}
  
    returns:
        [M, 4] posterior probabilities over
        {00, 01, 10, 11}
    """
  
    device = pair_x0_prob.device
    diffusion = self.diffusion
  
    source_t = int(t.item())
  
    if target_t is None:
      target_t_value = source_t - 1
    else:
      target_t_value = int(np.asarray(target_t).item())
  
    #
    # Existing binary cumulative transition matrices.
    #
  
    Qbar_source_binary = torch.from_numpy(
        diffusion.Q_bar[source_t]
    ).float().to(device)
  
    Qbar_target_binary = torch.from_numpy(
        diffusion.Q_bar[target_t_value]
    ).float().to(device)
  
    #
    # Convert to 4-state pair transitions.
    #
    # State order:
    # 00, 01, 10, 11
    #
  
    Qbar_source_pair = torch.kron(
        Qbar_source_binary,
        Qbar_source_binary,
    )
  
    Qbar_target_pair = torch.kron(
        Qbar_target_binary,
        Qbar_target_binary,
    )
  
    #
    # Same construction as categorical_posterior():
    #
    # Q_target->source =
    # inv(Qbar_target) @ Qbar_source
    #
  
    Q_pair = torch.linalg.solve(
        Qbar_target_pair,
        Qbar_source_pair,
    )
  
    #
    # Current noisy state z_source.
    #
  
    z_source = (
        2 * xt[pair_u].long()
        + xt[pair_v].long()
    )
  
    num_pairs = z_source.shape[0]
  
    posterior = torch.zeros(
        num_pairs,
        4,
        dtype=pair_x0_prob.dtype,
        device=device,
    )
  
    #
    # Sum over the three possible clean states.
    #
  
    for z0 in range(3):
  
      # p(z_target | z_source, z0)
      #
      # This mirrors categorical_posterior().
  
      source_onehot = F.one_hot(
          z_source,
          num_classes=4,
      ).float()
  
      part_1 = torch.matmul(
          source_onehot,
          Q_pair.T,
      )
  
      part_2 = Qbar_target_pair[z0]
  
      part_3 = (
          Qbar_source_pair[z0]
          * source_onehot
      ).sum(
          dim=-1,
          keepdim=True,
      )
  
      conditional = (
          part_1 * part_2
      ) / part_3.clamp_min(1e-12)
  
      posterior += (
          conditional
          * pair_x0_prob[:, z0].unsqueeze(-1)
      )
  
    posterior = posterior / posterior.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
  

    assert torch.isfinite(posterior).all(), (
      "Non-finite values in pair posterior"
    )
    
    assert (posterior >= -1e-6).all(), (
      f"Negative pair posterior values: min={posterior.min().item()}"
    )
    
    assert torch.allclose(
        posterior.sum(dim=-1),
        torch.ones(
            posterior.shape[0],
            device=posterior.device,
            dtype=posterior.dtype,
        ),
        atol=1e-5,
    ), (
        f"Pair posterior does not sum to 1: "
        f"{posterior.sum(dim=-1)}"
    )

    return posterior

  def static_pairwise_block_denoise_step(
      self,
      node_pred,
      edge_pred,
      edge_index,
      pair_mask,
      xt,
      t,
      target_t,
  ):
    """
    Denoising step using true joint diffusion for matched pair blocks.
  
    Clean pair states:
        0 = 00
        1 = 01
        2 = 10
  
    Noisy pair states:
        0 = 00
        1 = 01
        2 = 10
        3 = 11
    """
  
    device = node_pred.device
  
    #
    # 1. First compute the ordinary unary posterior for ALL nodes.
    #    These values will remain untouched for unmatched nodes.
    #
  
    node_prob = node_pred.softmax(dim=-1)
  
    unary_x0_prob = node_prob.reshape(
        (1, xt.shape[0], -1, 2)
    )
  
    xt_next = self.categorical_posterior(
        target_t,
        t,
        unary_x0_prob,
        xt,
    )
  
    #
    # 2. Retrieve the fixed matching.
    #
  
    pair_edges = edge_index[:, pair_mask]
  
    if pair_edges.shape[1] == 0:
      return xt_next
  
    pair_u = pair_edges[0]
    pair_v = pair_edges[1]
  
    #
    # 3. Model prediction of the CLEAN pair state.
    #
    # Shape: [num_pairs, 3]
    #
  
    pair_x0_prob = edge_pred[pair_mask].softmax(dim=-1)
  
    #
    # 4. Build the pair-block posterior.
    #
  
    pair_posterior = self.static_pairwise_block_posterior(
        target_t=target_t,
        t=t,
        pair_x0_prob=pair_x0_prob,
        xt=xt,
        pair_u=pair_u,
        pair_v=pair_v,
    )
  
    #
    # 5. Sample/update the pair jointly.
    #
  
    if target_t is None:
      target_t_value = int(t.item()) - 1
    else:
      target_t_value = int(np.asarray(target_t).item())
  
    if target_t_value > 0:
  
      pair_state = torch.multinomial(
          pair_posterior,
          num_samples=1,
      ).squeeze(-1)
  
      # state encoding:
      #
      # 0 -> 00
      # 1 -> 01
      # 2 -> 10
      # 3 -> 11
  
      pair_u_next = pair_state // 2
      pair_v_next = pair_state % 2
  
      xt_next[pair_u] = pair_u_next.to(xt_next.dtype)
      xt_next[pair_v] = pair_v_next.to(xt_next.dtype)
  
    else:
  
      # Final step: retain continuous probabilities just like
      # categorical_posterior() does at target_t == 0.
  
      p_u_1 = (
          pair_posterior[:, 2]
          + pair_posterior[:, 3]
      )
  
      p_v_1 = (
          pair_posterior[:, 1]
          + pair_posterior[:, 3]
      )
  
      xt_next[pair_u] = p_u_1
      xt_next[pair_v] = p_v_1
  
    return xt_next

  def categorical_denoise_step(
          self, 
          xt, 
          t, 
          device, 
          edge_index=None, 
          pair_mask=None,
          target_t=None
  ):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      #x0_pred = self.forward(
      #    xt.float().to(device),
      #    t.float().to(device),
      #    edge_index.long().to(device) if edge_index is not None else None,
      #)
      #x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
      prediction = self.forward(
        xt.float().to(device),
        t.float().to(device),
        edge_index.long().to(device),
      )
      
      if self.args.prediction_type in {"static_pairwise", "random_pairwise"}:
        node_pred, edge_pred = prediction
        assert pair_mask is not None
        assert pair_mask.dtype == torch.bool
        assert pair_mask.shape[0] == edge_index.shape[1]

        if self.args.pairwise_diffusion_mode == "marginal":
          x0_pred_prob = self.static_pairwise_to_unary_probs(
            node_pred,
            edge_pred,
            edge_index,
            pair_mask,
          )

          x0_pred_prob = x0_pred_prob.reshape(
              (1, xt.shape[0], -1, 2)
          )

          return self.categorical_posterior(
              target_t,
              t,
              x0_pred_prob,
              xt,
          )
        elif self.args.pairwise_diffusion_mode == "block":
          return self.static_pairwise_block_denoise_step(
              node_pred,
              edge_pred,
              edge_index,
              pair_mask,
              xt,
              t,
              target_t,
          )
      else:
        x0_pred_prob = prediction.softmax(dim=-1)

      x0_pred_prob = x0_pred_prob.reshape(
        (1, xt.shape[0], -1, 2)
      )
      
      xt = self.categorical_posterior(
        target_t,
        t,
        x0_pred_prob,
        xt,
      )
      
      return xt 
      #xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      #return xt

  def gaussian_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt


  def test_step(self, batch, batch_idx, draw=False, split='test'):
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    
    sampling_wall_time = time.perf_counter()
    real_batch_idx, graph_data, point_indicator = batch

    device = batch[-1].device

    static_pair_mask = None
    
    if self.args.prediction_type == "static_pairwise":
      static_pair_mask = (
          graph_data.pair_mask
          .bool()
          .to(device)
      )
    
    node_labels = graph_data.x
    
    base_edge_index = (
        graph_data.edge_index
        .to(node_labels.device)
        .reshape(2, -1)
    )
    
    stacked_predict_labels = []
    
    edge_index_np = base_edge_index.cpu().numpy()

    adj_mat = scipy.sparse.coo_matrix(
        (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
    )

    sequential_trajectories = []

    batch_size = 1
    steps = self.args.inference_diffusion_steps

    schedule_t1 = []
    schedule_t2 = []

    for i in range(steps):
      t1, t2 = time_schedule(i)
      schedule_t1.append(int(t1))
      schedule_t2.append(int(t2))


    for _ in range(self.args.sequential_sampling):
      edge_index = base_edge_index
      xt = torch.randn_like(node_labels.float())
      if self.args.parallel_sampling > 1:
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()
      xt = xt.reshape(-1)

      if self.args.parallel_sampling > 1:
        edge_index = self.duplicate_edge_index(
            base_edge_index,
            node_labels.shape[0],
            device,
        )

      #batch_size = 1
      #steps = self.args.inference_diffusion_steps
      #time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
      #                                  T=self.diffusion.T, inference_T=steps)

      trajectory = {
          "raw_cost": [],
          "feasible": [],
          "violations": [],
          "violating_vertices": [],
          "violation_rate": [],
          "selected_violation_rate": [],
          "selected_fraction": [],
          "created_violations": [],
          "resolved_violations": [],
          "step_time": [],
      }

      current_metrics = compute_parallel_metrics(
          xt,
          base_edge_index,
          node_labels.shape[0],
          self.args.parallel_sampling,
      )

      append_parallel_metrics_to_trajectory(trajectory, current_metrics)

      for i in range(steps):
        t1 = schedule_t1[i]
        t2 = schedule_t2[i]
    
        t1 = np.array(
            [t1 for _ in range(batch_size)]
        ).astype(int)
    
        t2 = np.array(
            [t2 for _ in range(batch_size)]
        ).astype(int)
    
        pair_mask = None
    
        if self.args.prediction_type == "static_pairwise":
          pair_mask = static_pair_mask
    
        elif self.args.prediction_type == "random_pairwise":
          pair_mask = self.select_random_pair_mask(
              graph_data,
              base_edge_index,
          )
    
        if (
            pair_mask is not None
            and self.args.parallel_sampling > 1
        ):
          pair_mask = pair_mask.repeat(
              self.args.parallel_sampling
          )

        prev_states = binary_state_from_xt(
            xt
        ).reshape(
            self.args.parallel_sampling,
            node_labels.shape[0],
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_start = time.perf_counter()
    
        if self.diffusion_type == "gaussian":
          xt = self.gaussian_denoise_step(
              xt,
              t1,
              device,
              edge_index,
              target_t=t2,
          )
        else:
          xt = self.categorical_denoise_step(
              xt,
              t1,
              device,
              edge_index,
              pair_mask=pair_mask,
              target_t=t2,
          )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_elapsed = time.perf_counter() - step_start
        trajectory["step_time"].append(step_elapsed)

        new_states = binary_state_from_xt(
            xt
        ).reshape(
            self.args.parallel_sampling,
            node_labels.shape[0],
        )

        current_metrics = compute_parallel_metrics(
                xt,
                base_edge_index,
                node_labels.shape[0],
                self.args.parallel_sampling,
        )
        append_parallel_metrics_to_trajectory(trajectory, current_metrics)

        created_this_step = []
        resolved_this_step = []

        for sample_idx in range(
            self.args.parallel_sampling
        ):
          prev_mask = mis_violation_mask(
              prev_states[sample_idx],
              base_edge_index,
          )
        
          new_mask = mis_violation_mask(
              new_states[sample_idx],
              base_edge_index,
          )
        
          created = (
              (~prev_mask) & new_mask
          ).sum()
        
          resolved = (
              prev_mask & (~new_mask)
          ).sum()

          created_this_step.append(float(created.item()))
          resolved_this_step.append(float(resolved.item()))
        trajectory["created_violations"].append(
            np.asarray(created_this_step)
        )
        
        trajectory["resolved_violations"].append(
            np.asarray(resolved_this_step)
        )

      if self.diffusion_type == 'gaussian':
        predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
      else:
        predict_labels = xt.float().cpu().detach().numpy() + 1e-6
      stacked_predict_labels.append(predict_labels)

      sequential_trajectories.append(trajectory)

    trajectory = finalize_trajectory(sequential_trajectories)

    num_samples = (
        self.args.sequential_sampling
        * self.args.parallel_sampling
    )

    assert trajectory["raw_cost"].shape == (
        num_samples,
        steps + 1,
    )

    assert trajectory["violations"].shape == (
        num_samples,
        steps + 1,
    )

    assert trajectory["created_violations"].shape == (
        num_samples,
        steps,
    )

    assert trajectory["resolved_violations"].shape == (
        num_samples,
        steps,
    )

    model_diffusion_time = sum(
        sum(t["step_time"])
        for t in sequential_trajectories
    )

    predict_labels = np.concatenate(stacked_predict_labels, axis=0)
    all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

    if torch.cuda.is_available():
      torch.cuda.synchronize()
    
    sampling_wall_time = (
        time.perf_counter()
        - sampling_wall_time
    )

    splitted_predict_labels = np.split(predict_labels, all_sampling)

    raw_solutions = [
        (sample >= 0.5).astype(np.int64)
        for sample in splitted_predict_labels
    ]

    raw_metric_list = []

    for raw_solution in raw_solutions:
      raw_tensor = torch.from_numpy(
          raw_solution
      ).to(base_edge_index.device)
    
      raw_metrics = mis_state_metrics(
          raw_tensor,
          base_edge_index,
      )
    
      raw_metric_list.append({
          k: (
              float(v.item())
              if torch.is_tensor(v) and v.numel() == 1
              else v
          )
          for k, v in raw_metrics.items()
          if k != "violating_edge_mask"
      })

    final_trajectory_cost = trajectory[
        "raw_cost"
    ][:, -1]

    final_trajectory_violations = trajectory[
        "violations"
    ][:, -1]

    final_raw_cost = np.asarray([
        m["cost"]
        for m in raw_metric_list
    ])

    final_raw_violations = np.asarray([
        m["violations"]
        for m in raw_metric_list
    ])

    assert np.allclose(
        final_trajectory_cost,
        final_raw_cost,
    ), (
        "Final trajectory raw cost does not match "
        "final raw solution cost."
    )

    assert np.allclose(
        final_trajectory_violations,
        final_raw_violations,
    ), (
        "Final trajectory violation count does not match "
        "final raw solution violation count."
    )

    mean_raw_cost = np.mean([
        m["cost"] for m in raw_metric_list
    ])
    
    mean_raw_feasible = np.mean([
        m["feasible"] for m in raw_metric_list
    ])
    
    mean_raw_violations = np.mean([
        m["violations"] for m in raw_metric_list
    ])
    
    mean_raw_violating_vertices = np.mean([
        m["violating_vertices"] for m in raw_metric_list
    ])
    
    mean_raw_violation_rate = np.mean([
        m["violation_rate"] for m in raw_metric_list
    ])
    
    mean_raw_selected_violation_rate = np.mean([
        m["selected_violation_rate"]
        for m in raw_metric_list
    ])
    
    # Decode MIS solution
    postprocess_start = time.perf_counter()

    solved_solutions = [
            mis_decode_np(sample, adj_mat) 
            for sample in splitted_predict_labels
    ]
    postprocess_time = (
        time.perf_counter()
        - postprocess_start
    )
    solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
    best_solved_cost = np.max(solved_costs)
    mean_solved_cost = np.mean(solved_costs)
    std_solved_cost = np.std(solved_costs)
    gt_cost = node_labels.cpu().numpy().sum()

    # postprocessing metrics
    postprocess_stats = []

    for raw, solved in zip(
        raw_solutions,
        solved_solutions,
    ):
      raw = raw.astype(bool)
      solved = solved.astype(bool)
    
      removed = np.logical_and(
          raw,
          np.logical_not(solved),
      ).sum()
    
      added = np.logical_and(
          np.logical_not(raw),
          solved,
      ).sum()
    
      flips = np.not_equal(
          raw,
          solved,
      ).sum()
    
      postprocess_stats.append({
          "cost_gain": solved.sum() - raw.sum(),
          "removed_vertices": removed,
          "added_vertices": added,
          "total_flips": flips,
          "changed_fraction":
              flips / max(raw.size, 1),
      })


    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/raw_cost": mean_raw_cost,
        f"{split}/raw_feasible": mean_raw_feasible,
        f"{split}/raw_violations": mean_raw_violations,
        f"{split}/raw_violating_vertices": mean_raw_violating_vertices,
        f"{split}/raw_violation_rate": mean_raw_violation_rate,
        f"{split}/raw_selected_violation_rate": mean_raw_selected_violation_rate,
        f"{split}/best_solved_cost": best_solved_cost,
        f"{split}/solved_cost_mean": mean_solved_cost,
        f"{split}/solved_cost_std": std_solved_cost,
    }
    metrics.update({
        f"{split}/postprocess_cost_gain":
            np.mean([m["cost_gain"] for m in postprocess_stats]),
        f"{split}/postprocess_removed_vertices":
            np.mean([m["removed_vertices"] for m in postprocess_stats]),
        f"{split}/postprocess_added_vertices":
            np.mean([m["added_vertices"] for m in postprocess_stats]),
        f"{split}/postprocess_total_flips":
            np.mean([m["total_flips"] for m in postprocess_stats]),
        f"{split}/postprocess_changed_fraction":
            np.mean([m["changed_fraction"] for m in postprocess_stats]),
        f"{split}/sampling_wall_time":
            sampling_wall_time,
        f"{split}/model_diffusion_time":
            model_diffusion_time,
        f"{split}/postprocess_time":
            postprocess_time,
    })
    for k, v in metrics.items():
      self.log(k, float(v), on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", float(best_solved_cost), prog_bar=True, on_epoch=True, sync_dist=True)

    record = {
        "graph_id": int(batch_idx),
        "batch_idx": int(batch_idx),
        "gt_cost": float(gt_cost),
    
        "raw_cost": np.asarray([
            m["cost"]
            for m in raw_metric_list
        ], dtype=np.float32),
    
        "raw_feasible": np.asarray([
            m["feasible"]
            for m in raw_metric_list
        ], dtype=np.float32),
    
        "raw_violations": np.asarray([
            m["violations"]
            for m in raw_metric_list
        ], dtype=np.float32),
    
        "raw_violating_vertices": np.asarray([
            m["violating_vertices"]
            for m in raw_metric_list
        ], dtype=np.float32),

        "raw_violation_rate": np.asarray([
            m["violation_rate"]
            for m in raw_metric_list
        ], dtype=np.float32),

        "raw_selected_violation_rate": np.asarray([
            m["selected_violation_rate"]
            for m in raw_metric_list
        ], dtype=np.float32),

        "raw_selected_fraction": np.asarray([
            m["selected_fraction"]
            for m in raw_metric_list
        ], dtype=np.float32),
    
        "solved_cost": np.asarray(
            solved_costs,
            dtype=np.float32,
        ),

        "postprocess_cost_gain": np.asarray([
            m["cost_gain"]
            for m in postprocess_stats
        ], dtype=np.float32),

        "postprocess_removed_vertices": np.asarray([
            m["removed_vertices"]
            for m in postprocess_stats
        ], dtype=np.float32),

        "postprocess_added_vertices": np.asarray([
            m["added_vertices"]
            for m in postprocess_stats
        ], dtype=np.float32),

        "postprocess_total_flips": np.asarray([
            m["total_flips"]
            for m in postprocess_stats
        ], dtype=np.float32),

        "postprocess_changed_fraction": np.asarray([
            m["changed_fraction"]
            for m in postprocess_stats
        ], dtype=np.float32),
    
        "raw_solutions": np.stack(
            raw_solutions,
            axis=0,
        ),
    
        "solved_solutions": np.stack(
            solved_solutions,
            axis=0,
        ),
    
        "trajectory": trajectory,
    
        "schedule_t1": np.asarray(
            schedule_t1,
            dtype=np.int64,
        ),
    
        "schedule_t2": np.asarray(
            schedule_t2,
            dtype=np.int64,
        ),
    
        "sampling_wall_time": float(
            sampling_wall_time
        ),

        "model_diffusion_time": float(
            model_diffusion_time
        ),    

        "postprocess_time": float(
            postprocess_time
        ),
    }

    self._eval_records[split].append(record)
    
    return {
        "summary": metrics,
        "schedule_t1": schedule_t1,
        "schedule_t2": schedule_t2,
        "sampling_wall_time": sampling_wall_time,
        "model_diffusion_time": model_diffusion_time,
        "postprocess_time": postprocess_time,
        "trajectory": trajectory,
        "solved_solutions": solved_solutions,
        "raw_solutions": raw_solutions,
        "graph_id": int(batch_idx),
    }
  def on_validation_epoch_start(self):
    self._eval_records["val"] = []
    self._eval_wall_start["val"] = (
        time.perf_counter()
    )

    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats()
  
  def on_test_epoch_start(self):
    self._eval_records["test"] = []
    self._eval_wall_start["test"] = (
        time.perf_counter()
    )
    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats()

  def _eval_output_path(
      self,
      split,
      extension,
  ):
    rank = getattr(
        self.trainer,
        "global_rank",
        0,
    )
  
    logs_dir = os.path.join(
        self.args.storage_path,
        "logs",
    )
  
    os.makedirs(
        logs_dir,
        exist_ok=True,
    )
  
    return os.path.join(
        logs_dir,
        f"{split}_metrics_rank{rank}.{extension}",
    )
  def on_test_epoch_end(self):
    wall_time = (
        time.perf_counter()
        - self._eval_wall_start["test"]
    )

    peak_allocated_mb = None
    peak_reserved_mb = None

    if torch.cuda.is_available():
      peak_allocated_mb = (
          torch.cuda.max_memory_allocated()
          / 1024**2
      )
      peak_reserved_mb = (
          torch.cuda.max_memory_reserved()
          / 1024**2
      )

    self._save_eval_records(
        "test",
        wall_time=wall_time,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
    )

  def on_validation_epoch_end(self):
    wall_time = (
        time.perf_counter()
        - self._eval_wall_start["val"]
    )

    peak_allocated_mb = None
    peak_reserved_mb = None

    if torch.cuda.is_available():
      peak_allocated_mb = (
          torch.cuda.max_memory_allocated()
          / 1024**2
      )
      peak_reserved_mb = (
          torch.cuda.max_memory_reserved()
          / 1024**2
      )

    self._save_eval_records(
        "val",
        wall_time=wall_time,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,

    )

  def _save_eval_records(self, split, wall_time, peak_allocated_mb=None, peak_reserved_mb=None):
    pass


  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
