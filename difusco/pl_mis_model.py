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

    train_label_dir = None
    val_label_dir = None
    test_label_dir = None
    
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
    
    self.train_dataset = MISDataset(
        data_file=os.path.join(
            self.args.storage_path,
            self.args.training_split,
        ),
        data_label_dir=train_label_dir,
    )
    
    self.validation_dataset = MISDataset(
        data_file=os.path.join(
            self.args.storage_path,
            self.args.validation_split,
        ),
        data_label_dir=val_label_dir,
    )
    
    self.test_dataset = MISDataset(
        data_file=os.path.join(
            self.args.storage_path,
            self.args.test_split,
        ),
        data_label_dir=test_label_dir,
    )


  def forward(self, x, t, edge_index):
    return self.model(x, t, edge_index=edge_index)

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
      
      if self.args.prediction_type == "static_pairwise":
        node_pred, edge_pred = prediction
      
        x0_pred_prob = self.static_pairwise_to_unary_probs(
          node_pred,
          edge_pred,
          edge_index,
          pair_mask,
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
    real_batch_idx, graph_data, point_indicator = batch

    device = batch[-1].device

    pair_mask = None
    if self.args.prediction_type == "static_pairwise":
      pair_mask = graph_data.pair_mask.bool().to(device)

    node_labels = graph_data.x
    edge_index = graph_data.edge_index

    stacked_predict_labels = []
    edge_index = edge_index.to(node_labels.device).reshape(2, -1)
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

      if self.args.parallel_sampling > 1:
        edge_index = self.duplicate_edge_index(edge_index, node_labels.shape[0], device)
        if pair_mask is not None:
          pair_mask = pair_mask.repeat(
              self.args.parallel_sampling
          )

      batch_size = 1
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
        t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

        if self.diffusion_type == 'gaussian':
          xt = self.gaussian_denoise_step(
              xt, t1, device, edge_index, target_t=t2)
        else:
          xt = self.categorical_denoise_step(
            xt,
            t1,
            device,
            edge_index,
            pair_mask=pair_mask,
            target_t=t2,
          )

      if self.diffusion_type == 'gaussian':
        predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
      else:
        predict_labels = xt.float().cpu().detach().numpy() + 1e-6
      stacked_predict_labels.append(predict_labels)

    predict_labels = np.concatenate(stacked_predict_labels, axis=0)
    all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

    splitted_predict_labels = np.split(predict_labels, all_sampling)
    solved_solutions = [mis_decode_np(predict_labels, adj_mat) for predict_labels in splitted_predict_labels]
    solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
    best_solved_cost = np.max(solved_costs)

    gt_cost = node_labels.cpu().numpy().sum()
    metrics = {
        f"{split}/gt_cost": gt_cost,
    }
    for k, v in metrics.items():
      self.log(k, float(v), on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", float(best_solved_cost), prog_bar=True, on_epoch=True, sync_dist=True)
    return metrics

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
