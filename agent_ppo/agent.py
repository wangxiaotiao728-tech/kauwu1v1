#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""PPO agent with LSTM, fixed features and rule-aware sampling."""

import os

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config, CurriculumConfig, GameConfig
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.feature_schema import get_feature_schema_hash
from agent_ppo.feature.reward_process import GameRewardManager
from agent_ppo.model.model import Model
from agent_ppo.rule_controller import RuleController
from kaiwudrl.interface.agent import BaseAgent


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        self.model = Model()
        if self.device is not None:
            self.model = self.model.to(self.device)

        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.rule_override = False
        self.tactical_state = 0
        self.feature_processes = None
        self.rule_controller = RuleController(Config)
        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)
        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        if step > self.target_step:
            return self.target_lr / self.lr
        return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        my_heroes = config_data.get("my_heroes", [])
        return {hero_id: 80115 for hero_id in my_heroes}

    def reset(self, observation):
        self.hero_camp = observation.get("player_camp", observation.get("camp", 0))
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.rule_override = False
        self.tactical_state = 0
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp)

    def predict(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data], use_max=False)[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, True)

    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data], use_max=True)[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, False)

    def observation_process(self, observation):
        feature = self.feature_processes.process_feature(observation)
        return ObsData(
            feature=feature,
            legal_action=observation["legal_action"],
            lstm_cell=self.lstm_cell,
            lstm_hidden=self.lstm_hidden,
            aux=getattr(self.feature_processes, "last_aux", {}),
            observation=observation,
        )

    def _model_inference(self, list_obs_data, use_max=False):
        features = np.asarray([obs.feature for obs in list_obs_data], dtype=np.float32)
        lstm_hidden = np.asarray([obs.lstm_hidden for obs in list_obs_data], dtype=np.float32)
        lstm_cell = np.asarray([obs.lstm_cell for obs in list_obs_data], dtype=np.float32)
        torch_inputs = [
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(lstm_hidden, dtype=torch.float32),
            torch.as_tensor(lstm_cell, dtype=torch.float32),
        ]
        if self.device is not None:
            torch_inputs = [tensor.to(self.device) for tensor in torch_inputs]

        feature_vec = torch_inputs[0].reshape(-1, self.seri_vec_split_shape[0][0])
        hidden_state = torch_inputs[1].reshape(-1, self.lstm_unit_size)
        cell_state = torch_inputs[2].reshape(-1, self.lstm_unit_size)

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model([feature_vec, hidden_state, cell_state], inference=True)

        logits, value, next_lstm_cell, next_lstm_hidden = [output.detach().cpu().numpy() for output in output_list[:4]]
        next_lstm_cell = next_lstm_cell.squeeze(axis=0)
        next_lstm_hidden = next_lstm_hidden.squeeze(axis=0)

        list_act_data = []
        for i, obs_data in enumerate(list_obs_data):
            sample = self._sample_masked_action(
                logits[i],
                obs_data.legal_action,
                obs_data.feature,
                obs_data.aux or {},
                use_max=use_max,
            )
            list_act_data.append(
                ActData(
                    action=sample["action"],
                    d_action=sample["d_action"],
                    prob=[sample["prob"]],
                    d_prob=[sample["d_prob"]],
                    value=value[i],
                    lstm_cell=next_lstm_cell[i],
                    lstm_hidden=next_lstm_hidden[i],
                    final_legal_action=sample["final_legal_action"],
                    rule_bias=sample["rule_bias"],
                    rule_state=sample["rule_state"],
                )
            )
        return list_act_data

    def action_process(self, observation, act_data, is_stochastic):
        action = act_data.action if is_stochastic else act_data.d_action
        action = [int(value) for value in action]
        self.rule_override = False
        self.tactical_state = int(getattr(act_data, "rule_state", 0) or 0)
        if self.feature_processes is not None:
            self.feature_processes.update_after_action(observation, action, observation.get("reward", {}))
        return action

    def _sample_masked_action(self, logits, legal_action, feature, aux, use_max=False):
        logits_split = np.split(np.asarray(logits, dtype=np.float64), np.cumsum(self.label_size_list)[:-1])
        official_masks = self._split_official_masks(legal_action)
        rule_output = self.rule_controller.compute(feature, aux, official_masks)

        actions, d_actions = [], []
        prob_parts, d_prob_parts = [], []
        final_masks, rule_biases = [], []

        for head in range(len(self.label_size_list) - 1):
            final_mask = self._compose_final_mask(official_masks[head], rule_output.hard_mask[head])
            probs = self._legal_softmax(logits_split[head], final_mask, rule_output.logit_bias[head])
            action = self._sample_from_probs(probs, use_max=use_max)
            d_action = self._sample_from_probs(probs, use_max=True)
            actions.append(action)
            d_actions.append(d_action)
            prob_parts.extend(probs.tolist())
            d_prob_parts.extend(probs.tolist())
            final_masks.append(final_mask)
            rule_biases.append(rule_output.logit_bias[head])

        target_mask = self._target_mask_for_button(legal_action, actions[0])
        target_final = self._compose_final_mask(target_mask, rule_output.hard_mask[-1])
        target_probs = self._legal_softmax(logits_split[-1], target_final, rule_output.logit_bias[-1])
        target_action = self._sample_from_probs(target_probs, use_max=use_max)

        d_target_mask = self._target_mask_for_button(legal_action, d_actions[0])
        d_target_final = self._compose_final_mask(d_target_mask, rule_output.hard_mask[-1])
        d_target_probs = self._legal_softmax(logits_split[-1], d_target_final, rule_output.logit_bias[-1])
        d_target_action = self._sample_from_probs(d_target_probs, use_max=True)

        actions.append(target_action)
        d_actions.append(d_target_action)
        prob_parts.extend(target_probs.tolist())
        d_prob_parts.extend(d_target_probs.tolist())
        final_masks.append(target_final)
        rule_biases.append(rule_output.logit_bias[-1])

        return {
            "action": actions,
            "d_action": d_actions,
            "prob": prob_parts,
            "d_prob": d_prob_parts,
            "final_legal_action": np.concatenate(final_masks).astype(np.float32),
            "rule_bias": np.concatenate(rule_biases).astype(np.float32),
            "rule_state": rule_output.state_id,
        }

    def _split_official_masks(self, legal_action):
        legal_action = np.asarray(legal_action, dtype=np.float32)
        compact_size = sum(self.label_size_list)
        if legal_action.size == compact_size:
            return list(np.split(legal_action, np.cumsum(self.label_size_list)[:-1]))
        fix_size = sum(self.label_size_list[:-1])
        masks = list(np.split(legal_action[:fix_size], np.cumsum(self.label_size_list[:-1])[:-1]))
        masks.append(np.ones(self.label_size_list[-1], dtype=np.float32))
        return masks

    def _target_mask_for_button(self, legal_action, button):
        legal_action = np.asarray(legal_action, dtype=np.float32)
        compact_size = sum(self.label_size_list)
        if legal_action.size == compact_size:
            return np.split(legal_action, np.cumsum(self.label_size_list)[:-1])[-1]
        target_size = self.label_size_list[-1]
        top_size = self.label_size_list[0]
        target_part = legal_action[-target_size * top_size :].reshape(top_size, target_size)
        button = int(np.clip(button, 0, top_size - 1))
        return target_part[button]

    def _compose_final_mask(self, official_mask, rule_forbid_mask):
        official_mask = np.asarray(official_mask, dtype=np.float32)
        rule_forbid_mask = np.asarray(rule_forbid_mask, dtype=np.float32)
        final_mask = official_mask * (1.0 - rule_forbid_mask)
        if final_mask.sum() <= 0:
            final_mask = np.zeros_like(official_mask, dtype=np.float32)
            final_mask[0] = 1.0
            self.rule_controller.mask_fallback_count += 1
        return final_mask

    def _legal_softmax(self, logits, legal_mask, logit_bias=None):
        logits = np.asarray(logits, dtype=np.float64)
        legal_mask = np.asarray(legal_mask, dtype=np.float64)
        if logit_bias is not None:
            logits = logits + np.asarray(logit_bias, dtype=np.float64)
        if legal_mask.size <= 0:
            legal_mask = np.ones_like(logits, dtype=np.float64)
        if legal_mask.sum() <= 0:
            legal_mask = np.ones_like(legal_mask, dtype=np.float64)
        masked_logits = np.where(legal_mask > 0, logits, -1e9)
        masked_logits = masked_logits - np.max(masked_logits)
        probs = np.exp(np.clip(masked_logits, -60.0, 60.0)) * legal_mask
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = probs.sum()
        return probs / total if total > 0 else legal_mask / max(1.0, legal_mask.sum())

    def _sample_from_probs(self, probs, use_max=False):
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.maximum(np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        if use_max:
            return int(np.argmax(probs))
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        return int(np.searchsorted(cdf, np.random.random(), side="right"))

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "ckpt_meta": self._ckpt_meta(),
            },
            model_file_path,
        )
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1", load_optimizer=False):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            if self.logger:
                self.logger.info(f"current model is {model_file_path}, so skip load model")
            return
        state = torch.load(model_file_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            self._validate_ckpt_meta(state.get("ckpt_meta", {}))
            self._load_state_dict_compatible(state["model_state_dict"])
            if load_optimizer:
                if "optimizer_state_dict" in state:
                    self.optimizer.load_state_dict(state["optimizer_state_dict"])
                if "scheduler_state" in state:
                    self.scheduler.load_state_dict(state["scheduler_state"])
        else:
            self._load_state_dict_compatible(state)
        self.cur_model_name = model_file_path
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _ckpt_meta(self):
        return {
            "feature_dim": Config.FEATURE_DIM,
            "feature_schema_hash": get_feature_schema_hash(),
            "lstm_hidden_size": Config.LSTM_UNIT_SIZE,
            "lstm_time_steps": Config.LSTM_TIME_STEPS,
            "label_size_list": Config.LABEL_SIZE_LIST,
            "model_version": GameConfig.MODEL_VERSION,
            "curriculum_stage": os.environ.get("HOK_CURRICULUM_STAGE", CurriculumConfig.CURRENT_STAGE),
        }

    def _validate_ckpt_meta(self, meta):
        if not meta:
            return
        expected = self._ckpt_meta()
        for key in ("feature_dim", "feature_schema_hash", "lstm_hidden_size", "lstm_time_steps", "label_size_list"):
            if meta.get(key) != expected.get(key):
                raise ValueError(f"checkpoint meta mismatch: {key} {meta.get(key)} != {expected.get(key)}")

    def _load_state_dict_compatible(self, state_dict):
        model_state = self.model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(key)
        model_state.update(compatible)
        self.model.load_state_dict(model_state, strict=False)
        if skipped and self.logger:
            self.logger.info(f"compatible model load skipped keys: {skipped[:20]}")
