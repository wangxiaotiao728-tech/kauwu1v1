#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
D401 replica model.

Compared with the baseline model, this version:
1. Actually uses LSTM for train/inference.
2. Splits the baseline feature into hero/tower groups and encodes them before fusion.
3. Keeps 6 official action heads.
4. Uses a naive target-attention head for the target branch.
5. Adds value clipping and optional dual-clip PPO in compute_loss.
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

from agent_ppo.conf.conf import DimConfig, Config, GameConfig


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate

        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])
        self.legal_action_dim = int(np.sum(Config.LEGAL_ACTION_SIZE_LIST))
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        # Baseline feature split: hero(3) + enemy tower/organ(7).
        # If the baseline feature process is expanded later, update these slices explicitly.
        self.hero_feature_dim = min(3, self.feature_dim)
        self.organ_feature_dim = max(0, self.feature_dim - self.hero_feature_dim)

        self.hero_encoder = MLP([self.hero_feature_dim, 64, 128], "hero_encoder", non_linearity_last=True)
        self.organ_encoder = MLP([max(1, self.organ_feature_dim), 64, 128], "organ_encoder", non_linearity_last=True)
        self.fusion_mlp = MLP([256, 512, self.lstm_unit_size], "fusion_mlp", non_linearity_last=True)

        self.lstm = nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        # First five action branches are standard MLP heads.
        self.label_mlp = ModuleDict(
            {
                "hero_label{0}_mlp".format(label_index): MLP(
                    [self.lstm_unit_size, 256, self.label_size_list[label_index]],
                    "hero_label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )

        # Naive target attention: query from LSTM output, key per official target index.
        self.lstm_tar_embed_mlp = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)
        self.target_unit_embedding = nn.Embedding(self.label_size_list[-1], self.target_embed_dim)
        nn.init.orthogonal_(self.target_unit_embedding.weight)

        self.value_mlp = MLP([self.lstm_unit_size, 256, 1], "hero_value_mlp")
        # D401 multi-critic style value heads. In this drop-in replica branch the
        # sample still stores one global return, so group heads are trained as
        # auxiliary value estimates against the same target. If later group
        # returns are serialized, compute_loss can be extended without changing
        # the model interface.
        self.reward_groups = list(getattr(GameConfig, "REWARD_GROUPS", {}).keys())
        self.value_group_mlps = ModuleDict(
            {
                group: MLP([self.lstm_unit_size, 128, 1], f"hero_value_{group}_mlp")
                for group in self.reward_groups
            }
        )

    def _encode_feature(self, feature_vec: torch.Tensor) -> torch.Tensor:
        hero_feat = feature_vec[:, : self.hero_feature_dim]
        organ_feat = feature_vec[:, self.hero_feature_dim :]
        if self.organ_feature_dim <= 0:
            organ_feat = torch.zeros((feature_vec.shape[0], 1), device=feature_vec.device, dtype=feature_vec.dtype)
        hero_emb = self.hero_encoder(hero_feat)
        organ_emb = self.organ_encoder(organ_feat)
        return self.fusion_mlp(torch.cat([hero_emb, organ_emb], dim=1))

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        batch_size = lstm_hidden_init.shape[0]
        total_frames = feature_vec.shape[0]
        time_steps = max(1, total_frames // max(1, batch_size))

        encoded = self._encode_feature(feature_vec)
        lstm_input = encoded.reshape(batch_size, time_steps, self.lstm_unit_size)

        h0 = lstm_hidden_init.unsqueeze(0).contiguous()
        c0 = lstm_cell_init.unsqueeze(0).contiguous()
        lstm_out, (h_n, c_n) = self.lstm(lstm_input, (h0, c0))
        lstm_flat = lstm_out.reshape(-1, self.lstm_unit_size)

        self.lstm_hidden_output = h_n
        self.lstm_cell_output = c_n

        result_list = []
        for label_index in range(len(self.label_size_list) - 1):
            result_list.append(self.label_mlp[f"hero_label{label_index}_mlp"](lstm_flat))

        target_query = self.lstm_tar_embed_mlp(lstm_flat)
        target_keys = self.target_unit_embedding.weight
        target_logits = torch.matmul(target_query, target_keys.t()) / math.sqrt(float(self.target_embed_dim))
        result_list.append(target_logits)

        value_result = self.value_mlp(lstm_flat)
        group_values = [self.value_group_mlps[group](lstm_flat) for group in self.reward_groups]
        result_list.append(value_result)
        result_list.extend(group_values)

        logits = torch.flatten(torch.cat(result_list[: len(self.label_size_list)], 1), start_dim=1)
        if inference:
            if len(group_values) > 0:
                group_value_tensor = torch.cat(group_values, dim=1)
            else:
                group_value_tensor = torch.zeros((value_result.shape[0], 0), device=value_result.device)
            return [logits, value_result, group_value_tensor, self.lstm_cell_output, self.lstm_hidden_output]
        return result_list

    def compute_loss(self, data_list, rst_list):
        label_num = len(self.label_size_list)
        group_num = int(getattr(Config, "REWARD_GROUP_NUM", 0))

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])

        group_return_start = 3
        group_adv_start = group_return_start + group_num
        action_start = group_adv_start + group_num
        old_prob_start = action_start + label_num
        weight_start = old_prob_start + label_num
        old_value_idx = weight_start + label_num
        old_group_value_start = old_value_idx + 1
        is_train_idx = old_group_value_start + group_num

        group_returns = []
        group_advantages = []
        for g in range(group_num):
            group_returns.append(data_list[group_return_start + g].reshape(-1, self.data_split_shape[group_return_start + g]).squeeze(dim=1))
            group_advantages.append(data_list[group_adv_start + g].reshape(-1, self.data_split_shape[group_adv_start + g]).squeeze(dim=1))

        old_value = data_list[old_value_idx].reshape(-1, self.data_split_shape[old_value_idx]).squeeze(dim=1)
        old_group_values = []
        for g in range(group_num):
            idx = old_group_value_start + g
            old_group_values.append(data_list[idx].reshape(-1, self.data_split_shape[idx]).squeeze(dim=1))

        usq_is_train = data_list[is_train_idx].reshape(-1, self.data_split_shape[is_train_idx])

        usq_label_list = data_list[action_start : action_start + label_num]
        for shape_index in range(label_num):
            idx = action_start + shape_index
            usq_label_list[shape_index] = usq_label_list[shape_index].reshape(-1, self.data_split_shape[idx]).long()

        old_label_probability_list = data_list[old_prob_start : old_prob_start + label_num]
        for shape_index in range(label_num):
            idx = old_prob_start + shape_index
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(-1, self.data_split_shape[idx])

        usq_weight_list = data_list[weight_start : weight_start + label_num]
        for shape_index in range(label_num):
            idx = weight_start + shape_index
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(-1, self.data_split_shape[idx])

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        if group_num > 0 and len(group_advantages) == group_num:
            weights = torch.tensor(Config.REWARD_GROUP_ADV_WEIGHTS, dtype=advantage.dtype, device=advantage.device)
            advantage = torch.stack(group_advantages, dim=1).matmul(weights)

        frame_is_train = usq_is_train.squeeze(dim=1)
        valid_mask = frame_is_train > 0.5
        if Config.ADV_NORM and torch.sum(valid_mask) > 1:
            adv_valid = advantage[valid_mask]
            advantage = (advantage - adv_valid.mean()) / (adv_valid.std(unbiased=False) + 1e-5)

        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]

        label_result = rst_list[:label_num]
        value_result = rst_list[label_num]
        group_value_results = rst_list[label_num + 1 : label_num + 1 + group_num]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        # Global value loss.
        value_pred = value_result.squeeze(dim=1)
        if Config.USE_VALUE_CLIP:
            value_pred_clipped = old_value + torch.clamp(
                value_pred - old_value, -Config.VALUE_CLIP_PARAM, Config.VALUE_CLIP_PARAM
            )
            value_loss_unclipped = torch.square(reward - value_pred)
            value_loss_clipped = torch.square(reward - value_pred_clipped)
            value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped)
        else:
            value_loss = torch.square(reward - value_pred)
        value_denom = torch.maximum(torch.sum(frame_is_train), torch.tensor(1.0, device=frame_is_train.device))
        global_value_cost = 0.5 * torch.sum(value_loss * frame_is_train) / value_denom

        # Full D401 multi-critic loss: one clipped value loss per reward group.
        group_value_cost = torch.tensor(0.0, device=seri_vec.device)
        for g, group_value in enumerate(group_value_results):
            group_pred = group_value.squeeze(dim=1)
            target = group_returns[g]
            old_group_value = old_group_values[g]
            if Config.USE_VALUE_CLIP:
                pred_clipped = old_group_value + torch.clamp(
                    group_pred - old_group_value, -Config.VALUE_CLIP_PARAM, Config.VALUE_CLIP_PARAM
                )
                loss_unclipped = torch.square(target - group_pred)
                loss_clipped = torch.square(target - pred_clipped)
                loss = torch.maximum(loss_unclipped, loss_clipped)
            else:
                loss = torch.square(target - group_pred)
            group_value_cost = group_value_cost + 0.5 * torch.sum(loss * frame_is_train) / value_denom

        self.value_cost = Config.VALUE_LOSS_COEF * global_value_cost
        if group_num > 0:
            self.value_cost = self.value_cost + Config.GROUP_VALUE_LOSS_COEF * group_value_cost

        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0, device=seri_vec.device)
        approx_kl_sum = torch.tensor(0.0, device=seri_vec.device)
        clip_frac_sum = torch.tensor(0.0, device=seri_vec.device)
        active_task_cnt = 0

        for task_index in range(label_num):
            if not self.is_reinforce_task_list[task_index]:
                continue
            active_task_cnt += 1
            one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
            boundary = torch.tensor(1e20, device=seri_vec.device)
            legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

            label_logits_subtract_max = torch.clamp(
                label_result[task_index]
                - torch.max(label_result[task_index] - legal_action_flag_list_max_mask, dim=1, keepdim=True).values,
                -boundary,
                1,
            )
            label_exp_logits = legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
            label_probability = label_exp_logits / label_exp_logits.sum(1, keepdim=True)
            label_probability_list.append(label_probability)

            policy_p = (one_hot_actions * label_probability).sum(1)
            policy_log_p = torch.log(policy_p + epsilon)
            old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
            old_policy_log_p = torch.log(old_policy_p)
            log_ratio = policy_log_p - old_policy_log_p
            ratio = torch.exp(log_ratio).clamp(0.0, 10.0)
            clipped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)

            surr1 = ratio * advantage
            surr2 = clipped_ratio * advantage
            base_obj = torch.minimum(surr1, surr2)
            if Config.USE_DUAL_CLIP:
                dual_obj = torch.maximum(base_obj, Config.DUAL_CLIP_C * advantage)
                ppo_obj = torch.where(advantage < 0, dual_obj, base_obj)
            else:
                ppo_obj = base_obj

            denom = torch.maximum(
                torch.sum(weight_list[task_index].float() * frame_is_train),
                torch.tensor(1.0, device=seri_vec.device),
            )
            temp_policy_loss = -torch.sum(ppo_obj * weight_list[task_index].float() * frame_is_train) / denom
            self.policy_cost = self.policy_cost + temp_policy_loss

            with torch.no_grad():
                approx_kl = torch.sum((old_policy_log_p - policy_log_p) * weight_list[task_index].float() * frame_is_train) / denom
                clip_frac = torch.sum(
                    ((torch.abs(ratio - 1.0) > self.clip_param).float())
                    * weight_list[task_index].float()
                    * frame_is_train
                ) / denom
                approx_kl_sum = approx_kl_sum + approx_kl
                clip_frac_sum = clip_frac_sum + clip_frac

        entropy_loss_list = []
        current_entropy_loss_index = 0
        for task_index in range(label_num):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )
                temp_entropy_loss = -torch.sum(temp_entropy_loss * weight_list[task_index].float() * frame_is_train) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train),
                    torch.tensor(1.0, device=seri_vec.device),
                )
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0, device=seri_vec.device))

        self.entropy_cost = torch.stack(entropy_loss_list).sum()
        self.entropy_cost_list = entropy_loss_list
        self.approx_kl = approx_kl_sum / max(1, active_task_cnt)
        self.clip_fraction = clip_frac_sum / max(1, active_task_cnt)
        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost, self.approx_kl, self.clip_fraction],
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def feature_vec_device(x):
    return x.device


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
