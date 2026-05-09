#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO LSTM actor-critic model.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config, DimConfig


class MLP(nn.Module):
    def __init__(self, dims: List[int], act=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PPOActorCriticLSTM(nn.Module):
    def __init__(self, feature_dim=512, hidden_size=256, label_sizes=None, num_layers=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.label_sizes = label_sizes or [12, 16, 16, 16, 16, 9]

        self.in_ln = nn.LayerNorm(feature_dim)
        self.frame_encoder = MLP([feature_dim, 512, 256])
        self.lstm = nn.LSTM(256, hidden_size, num_layers=num_layers, batch_first=True)
        self.policy_encoder = MLP([hidden_size, 256, 256])
        self.value_encoder = MLP([hidden_size, 256, 256])
        self.action_heads = nn.ModuleList([nn.Linear(256, n) for n in self.label_sizes])
        self.value_head = nn.Linear(256, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        for head in getattr(self, "action_heads", []):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, 0.0)
        if hasattr(self, "value_head"):
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.constant_(self.value_head.bias, 0.0)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward(self, features, hidden=None):
        # features: 推理 [B, 512] 或 [B, 1, 512]；训练 [B, T, 512]。
        if features.dim() == 2:
            features = features.unsqueeze(1)
        batch_size, _, dim = features.shape
        if dim != self.feature_dim:
            raise ValueError(f"feature dim must be {self.feature_dim}, got {dim}")

        x = self.in_ln(features)
        x = self.frame_encoder(x)
        if hidden is None:
            hidden = self.init_hidden(batch_size, features.device)
        lstm_out, next_hidden = self.lstm(x, hidden)
        pi_x = self.policy_encoder(lstm_out)
        v_x = self.value_encoder(lstm_out)
        logits = [head(pi_x) for head in self.action_heads]
        value = self.value_head(v_x).squeeze(-1)
        return logits, value, next_hidden


class Model(PPOActorCriticLSTM):
    def __init__(self):
        super().__init__(
            feature_dim=DimConfig.FEATURE_DIM,
            hidden_size=Config.LSTM_UNIT_SIZE,
            label_sizes=Config.LABEL_SIZE_LIST,
        )
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.var_beta = Config.BETA_START
        self.learning_rate = Config.INIT_LEARNING_RATE_START
        self.ppo_epoch = Config.PPO_EPOCH
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.restore_list = []

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        if feature_vec.dim() == 1:
            feature_vec = feature_vec.reshape(-1, self.feature_dim)

        if lstm_hidden_init.dim() == 1:
            lstm_hidden_init = lstm_hidden_init.reshape(-1, self.lstm_unit_size)
        if lstm_cell_init.dim() == 1:
            lstm_cell_init = lstm_cell_init.reshape(-1, self.lstm_unit_size)

        batch_size = lstm_hidden_init.shape[0]
        time_steps = 1 if inference else self.lstm_time_steps
        expected = batch_size * time_steps
        if feature_vec.shape[0] != expected:
            time_steps = max(1, feature_vec.shape[0] // max(1, batch_size))
        features = feature_vec.reshape(batch_size, time_steps, self.feature_dim)
        hidden = (lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0))

        logits_seq, value_seq, (next_hidden, next_cell) = super().forward(features, hidden)
        self.lstm_hidden_output = next_hidden
        self.lstm_cell_output = next_cell

        if inference:
            logits = torch.cat([head[:, -1, :] for head in logits_seq], dim=1)
            value = value_seq[:, -1:].reshape(batch_size, 1)
            return [logits, value, next_cell, next_hidden]

        flat_logits = [head.reshape(-1, head.shape[-1]) for head in logits_seq]
        flat_value = value_seq.reshape(-1, 1)
        return flat_logits + [flat_value]

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        device = seri_vec.device
        reward = data_list[1].reshape(-1, self.data_split_shape[1]).squeeze(1)
        advantage = data_list[2].reshape(-1, self.data_split_shape[2]).squeeze(1)
        frame_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3]).squeeze(1)

        label_list = []
        for i in range(len(self.label_size_list)):
            label = data_list[3 + i].reshape(-1, self.data_split_shape[3 + i]).long().squeeze(1)
            label_list.append(label)

        old_prob_list = []
        old_prob_start = 3 + len(self.label_size_list)
        for i in range(len(self.label_size_list)):
            old_prob = data_list[old_prob_start + i].reshape(
                -1, self.data_split_shape[old_prob_start + i]
            )
            old_prob_list.append(old_prob)

        rule_bias_list = []
        rule_bias_start = old_prob_start + len(self.label_size_list)
        for i in range(len(self.label_size_list)):
            rule_bias = data_list[rule_bias_start + i].reshape(
                -1, self.data_split_shape[rule_bias_start + i]
            )
            rule_bias_list.append(rule_bias)

        weight_list = []
        weight_start = 3 + 3 * len(self.label_size_list)
        for i in range(len(self.label_size_list)):
            weight = data_list[weight_start + i].reshape(-1, self.data_split_shape[weight_start + i]).squeeze(1)
            weight_list.append(weight)

        _, split_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        legal_action_list = torch.split(split_legal_action, self.label_size_list, dim=1)
        label_result = rst_list[:-1]
        value_result = rst_list[-1].squeeze(1)

        valid_count = torch.maximum(frame_is_train.sum(), torch.tensor(1.0, device=device))
        self.value_cost = 0.5 * ((value_result - reward) ** 2 * frame_is_train).sum() / valid_count

        self.policy_cost = torch.tensor(0.0, device=device)
        entropy_total = torch.tensor(0.0, device=device)
        clip_fraction_total = torch.tensor(0.0, device=device)
        approx_kl_total = torch.tensor(0.0, device=device)
        reinforce_heads = 0
        eps = 1e-6

        for i, reinforce in enumerate(self.is_reinforce_task_list):
            if not reinforce:
                continue
            reinforce_heads += 1
            logits = label_result[i] + rule_bias_list[i].to(label_result[i].device)
            legal = legal_action_list[i].float()
            legal = torch.where(legal.sum(dim=1, keepdim=True) > 0, legal, torch.ones_like(legal))
            masked_logits = logits.masked_fill(legal <= 0, -1e9)
            prob = torch.softmax(masked_logits, dim=1)
            dist = torch.distributions.Categorical(probs=prob)

            action = label_list[i].clamp(0, self.label_size_list[i] - 1)
            new_logprob = dist.log_prob(action)
            old_action_prob = (nn.functional.one_hot(action, self.label_size_list[i]).float() * old_prob_list[i]).sum(1)
            old_logprob = torch.log(old_action_prob + eps)

            ratio = torch.exp(new_logprob - old_logprob).clamp(0.0, 3.0)
            surr1 = ratio * advantage
            surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            head_mask = weight_list[i].float() * frame_is_train
            denom = torch.maximum(head_mask.sum(), torch.tensor(1.0, device=device))
            self.policy_cost = self.policy_cost - (torch.minimum(surr1, surr2) * head_mask).sum() / denom
            entropy_total = entropy_total + (dist.entropy() * head_mask).sum() / denom

            with torch.no_grad():
                log_ratio = new_logprob - old_logprob
                approx_kl_total = approx_kl_total + (((torch.exp(log_ratio) - 1.0) - log_ratio) * head_mask).sum() / denom
                clip_fraction_total = clip_fraction_total + (((ratio - 1.0).abs() > self.clip_param).float() * head_mask).sum() / denom

        normalizer = max(1, reinforce_heads)
        self.entropy_cost = -(entropy_total / normalizer)
        self.approx_kl = approx_kl_total / normalizer
        self.clip_fraction = clip_fraction_total / normalizer
        self.loss = Config.VALUE_LOSS_COEF * self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
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
