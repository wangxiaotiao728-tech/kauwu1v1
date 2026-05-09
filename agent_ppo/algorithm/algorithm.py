#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""D401 replica PPO learner wrapper."""

import os
import time
import torch
import numpy as np

from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0
        self.logger = logger
        self.monitor = monitor
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.last_report_monitor_time = 0

    def _format_inputs(self, data_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, _ = seri_vec.split(
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])], dim=1
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)
        return [feature_vec, lstm_hidden_state, lstm_cell_state]

    def learn(self, list_sample_data):
        _input_datas = torch.stack([sample.sample for sample in list_sample_data]).to(self.device)
        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data_list[i] = data.reshape(-1).float()

        self.model.set_train_mode()
        last_total_loss = None
        last_info_list = None

        for _ in range(max(1, int(getattr(Config, "PPO_EPOCH", 1)))):
            self.optimizer.zero_grad()
            format_inputs = self._format_inputs(data_list)
            rst_list = self.model(format_inputs)
            total_loss, info_list = self.model.compute_loss(data_list, rst_list)
            total_loss.backward()
            if Config.USE_GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
            self.optimizer.step()
            last_total_loss = total_loss
            last_info_list = info_list

        self.train_step += 1
        # LambdaLR expects epoch/step, ReduceLROnPlateau expects metric. Support both.
        try:
            self.scheduler.step(self.train_step)
        except TypeError:
            self.scheduler.step(float(last_total_loss.item()))

        # Optional model-level dynamic schedules.
        if hasattr(self.model, "var_beta"):
            progress = min(1.0, self.train_step / max(1, Config.TARGET_STEP))
            self.model.var_beta = Config.TARGET_BETA + (Config.BETA_START - Config.TARGET_BETA) * (1.0 - progress)
        if hasattr(self.model, "clip_param"):
            progress = min(1.0, self.train_step / max(1, Config.TARGET_STEP))
            self.model.clip_param = Config.TARGET_CLIP_PARAM + (Config.CLIP_PARAM - Config.TARGET_CLIP_PARAM) * (
                1.0 - progress
            )

        info_flat = []
        for info in last_info_list:
            if isinstance(info, list):
                info_flat.append([i.item() for i in info])
            else:
                info_flat.append(info.item())

        results = {"total_loss": round(float(last_total_loss.item()), 4)}
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, metric_list = info_flat
            value_loss, policy_loss, entropy_loss = metric_list[:3]
            approx_kl = metric_list[3] if len(metric_list) > 3 else 0.0
            clip_fraction = metric_list[4] if len(metric_list) > 4 else 0.0
            lr = self.optimizer.param_groups[0].get("lr", 0.0)
            results.update(
                {
                    "value_loss": round(value_loss, 4),
                    "policy_loss": round(policy_loss, 4),
                    "entropy_loss": round(entropy_loss, 4),
                    "approx_kl": round(approx_kl, 6),
                    "clip_fraction": round(clip_fraction, 4),
                    "lr": round(lr, 8),
                    "entropy_beta": round(float(getattr(self.model, "var_beta", 0.0)), 6),
                    "ppo_clip": round(float(getattr(self.model, "clip_param", 0.0)), 4),
                }
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now
        return results
