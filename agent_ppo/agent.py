#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO agent for HOK 1v1.

The rule controller and short action memory are intentionally kept in this
existing file so the package layout stays unchanged.
"""

import os

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config, RuleConfig
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.definition import *
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.reward_process import GameRewardManager
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


SUMMONER_SKILL_IDS = [80102, 80103, 80104, 80105, 80107, 80108, 80109, 80110, 80115, 80121]

STATE_MODEL = 0
STATE_RETREAT_TOWER_AGGRO = 1
STATE_FINISH_TOWER = 2
STATE_DEATH_WINDOW_PUSH = 3
STATE_ENHANCED_POKE_TOWER = 4
STATE_PUSH_WITH_MINION = 5
STATE_PICK_BLOOD_PACK = 6
STATE_DEFEND_TOWER = 7
STATE_CLEAR_WAVE = 8
STATE_HARASS_OR_TRADE = 9
STATE_FARM_MONSTER = 10

BUTTON_MOVE = 2
BUTTON_ATTACK = 3
BUTTON_SKILL_1 = 4
BUTTON_SKILL_2 = 5
BUTTON_SKILL_3 = 6
BUTTON_CHOSEN_SKILL = 8

TARGET_NONE = 0
TARGET_ENEMY = 1
TARGET_SELF = 2
TARGET_SOLDIER_0 = 3
TARGET_TOWER = 7
TARGET_MONSTER = 8


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        self.model = Model()
        if self.device is not None:
            self.model = self.model.to(self.device)

        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None

        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.rule_override = False
        self.tactical_state = STATE_MODEL
        self.feature_processes = None
        self.reset_rule_memory()

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)
        super().__init__(agent_type, device, logger, monitor)

    def reset_rule_memory(self):
        self.common_attack_count = 0
        self.luban_attack_count = 0
        self.direnjie_attack_mod2 = 0
        self.recent_skill_frames = 0
        self.last_rule_state = STATE_MODEL
        self.state_hold_steps = 0

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
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.rule_override = False
        self.tactical_state = STATE_MODEL
        self.reset_rule_memory()
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp)

    def _model_inference(self, list_obs_data):
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        if self.device is not None:
            torch_inputs = [tensor.to(self.device) for tensor in torch_inputs]
        for i, data in enumerate(torch_inputs):
            torch_inputs[i] = data.reshape(-1).float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model([feature_vec, lstm_hidden_state, lstm_cell_state], inference=True)

        logits, value, next_lstm_cell, next_lstm_hidden = [output.detach().cpu().numpy() for output in output_list[:4]]
        next_lstm_cell = next_lstm_cell.squeeze(axis=0)
        next_lstm_hidden = next_lstm_hidden.squeeze(axis=0)

        list_act_data = []
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value,
                    lstm_cell=next_lstm_cell[i],
                    lstm_hidden=next_lstm_hidden[i],
                )
            )
        return list_act_data

    def predict(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, True)

    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, False)

    def observation_process(self, observation):
        feature = self.feature_processes.process_feature(observation)
        return ObsData(
            feature=feature,
            legal_action=observation["legal_action"],
            lstm_cell=self.lstm_cell,
            lstm_hidden=self.lstm_hidden,
        )

    def action_process(self, observation, act_data, is_stochastic):
        self.rule_override = False
        raw_action = act_data.action if is_stochastic else act_data.d_action
        raw_action = [int(value) for value in raw_action]
        final_action, override, state_id = self._rule_control_action(observation, raw_action)
        self.rule_override = override
        self.tactical_state = state_id
        if override:
            if is_stochastic:
                act_data.action = final_action
            else:
                act_data.d_action = final_action
        self._update_rule_memory(observation, final_action, state_id)
        return final_action

    def _rule_control_action(self, observation, raw_action):
        frame_state = observation.get("frame_state", {})
        my_hero, enemy_hero = self._split_heroes(frame_state, observation.get("player_id"), observation.get("player_camp", observation.get("camp", self.hero_camp)))
        my_camp = my_hero.get("camp") if my_hero else observation.get("player_camp", observation.get("camp", self.hero_camp))
        own_tower, enemy_tower, friendly_minions, enemy_minions, monsters = self._split_npcs(frame_state, my_camp)
        legal_action = observation.get("legal_action", [])

        if self._tower_targets_self(my_hero, enemy_tower) and self._button_legal(legal_action, BUTTON_MOVE):
            return self._move_to(my_hero, self._pos(own_tower)), True, STATE_RETREAT_TOWER_AGGRO

        if self._can_finish_tower(my_hero, enemy_tower) and self._button_legal(legal_action, BUTTON_ATTACK):
            return self._attack_tower_action(), True, STATE_FINISH_TOWER

        if enemy_hero and not self._alive(enemy_hero) and enemy_tower:
            if self._can_attack_tower(my_hero, enemy_tower) and self._button_legal(legal_action, BUTTON_ATTACK):
                return self._attack_tower_action(), True, STATE_DEATH_WINDOW_PUSH
            if self._button_legal(legal_action, BUTTON_MOVE):
                return self._move_to(my_hero, self._pos(enemy_tower)), True, STATE_DEATH_WINDOW_PUSH

        if self._enhanced_attack_ready(my_hero) and self._can_attack_tower(my_hero, enemy_tower) and self._button_legal(legal_action, BUTTON_ATTACK):
            return self._attack_tower_action(), True, STATE_ENHANCED_POKE_TOWER

        if self._friendly_minion_tanking(enemy_tower, friendly_minions):
            if self._can_attack_tower(my_hero, enemy_tower) and self._button_legal(legal_action, BUTTON_ATTACK):
                return self._attack_tower_action(), True, STATE_PUSH_WITH_MINION

        blood_action = self._blood_pack_action(frame_state, my_hero, enemy_hero, enemy_tower, legal_action)
        if blood_action is not None:
            return blood_action, True, STATE_PICK_BLOOD_PACK

        state_id = self._soft_tactical_state(my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions, monsters)
        return raw_action, False, state_id

    def _soft_tactical_state(self, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions, monsters):
        if own_tower and self._hp_ratio(own_tower) < RuleConfig.LOW_HP_RISK_RATIO and enemy_minions:
            return STATE_DEFEND_TOWER
        if len(enemy_minions) > len(friendly_minions) + 1:
            return STATE_CLEAR_WAVE
        if enemy_hero and self._hp_ratio(my_hero) > self._hp_ratio(enemy_hero) + RuleConfig.TRADE_HP_ADVANTAGE:
            return STATE_HARASS_OR_TRADE
        if monsters and self._hp_ratio(my_hero) > RuleConfig.FARM_MONSTER_HP_RATIO:
            return STATE_FARM_MONSTER
        return STATE_MODEL

    def _blood_pack_action(self, frame_state, my_hero, enemy_hero, enemy_tower, legal_action):
        if not my_hero or not self._button_legal(legal_action, BUTTON_MOVE) or self._hp_ratio(my_hero) >= RuleConfig.BLOOD_PACK_HP_RATIO:
            return None
        cakes = frame_state.get("cakes", []) or []
        if not cakes:
            return None
        my_pos = self._pos(my_hero)
        cake = min(cakes, key=lambda item: self._dist(my_pos, self._cake_pos(item)))
        cake_pos = self._cake_pos(cake)
        dist = self._dist(my_pos, cake_pos)
        if dist > RuleConfig.BLOOD_PACK_MAX_DIST:
            return None
        if self._tower_risk(my_hero, enemy_tower) > RuleConfig.BLOOD_PACK_TOWER_RISK_LIMIT:
            return None
        if enemy_hero and self._alive(enemy_hero) and self._dist(self._pos(enemy_hero), my_pos) < RuleConfig.ENEMY_NEAR_DIST:
            if self._hp_ratio(my_hero) < self._hp_ratio(enemy_hero):
                return None
        if enemy_tower and self._hp_ratio(enemy_tower) < RuleConfig.FINISH_TOWER_HP_RATIO and self._hp_ratio(my_hero) > RuleConfig.FINISH_TOWER_SELF_HP_RATIO:
            return None
        return self._move_to(my_hero, cake_pos)

    def _update_rule_memory(self, observation, action, state_id):
        if state_id == self.last_rule_state:
            self.state_hold_steps += 1
        else:
            self.state_hold_steps = 0
        self.last_rule_state = state_id

        if self.recent_skill_frames > 0:
            self.recent_skill_frames -= 1

        button = action[0] if action else 0
        frame_state = observation.get("frame_state", {})
        my_hero, _ = self._split_heroes(frame_state, observation.get("player_id"), observation.get("player_camp", observation.get("camp", self.hero_camp)))
        config_id = (my_hero or {}).get("config_id")
        if button == BUTTON_ATTACK:
            self.common_attack_count += 1
            if config_id == 112:
                self.luban_attack_count = (self.luban_attack_count + 1) % 5
            if config_id == 133:
                self.direnjie_attack_mod2 = (self.direnjie_attack_mod2 + 1) % 2
        elif button in (BUTTON_SKILL_1, BUTTON_SKILL_2, BUTTON_SKILL_3, BUTTON_CHOSEN_SKILL):
            self.recent_skill_frames = 5
            if config_id == 112:
                self.luban_attack_count = 4

    def _split_heroes(self, frame_state, player_id, player_camp=None):
        heroes = frame_state.get("hero_states", []) or []
        my_hero, enemy_hero = None, None
        for hero in heroes:
            if hero.get("runtime_id") == player_id:
                my_hero = hero
                break
        if my_hero is None and player_camp is not None:
            for hero in heroes:
                if self._same_camp(hero.get("camp"), player_camp):
                    my_hero = hero
                    break
        my_camp = my_hero.get("camp") if my_hero else player_camp
        for hero in heroes:
            if hero is not my_hero and not self._same_camp(hero.get("camp"), my_camp):
                enemy_hero = hero
                break
        return my_hero, enemy_hero

    def _split_npcs(self, frame_state, my_camp):
        own_tower, enemy_tower = None, None
        friendly_minions, enemy_minions, monsters = [], [], []
        for npc in frame_state.get("npc_states", []) or []:
            npc_camp = npc.get("camp")
            if self._is_tower(npc):
                if self._same_camp(npc_camp, my_camp):
                    own_tower = npc
                else:
                    enemy_tower = npc
            elif self._is_monster(npc, my_camp):
                monsters.append(npc)
            elif self._same_camp(npc_camp, my_camp):
                friendly_minions.append(npc)
            elif npc_camp not in (0, None, "0"):
                enemy_minions.append(npc)
        return own_tower, enemy_tower, friendly_minions, enemy_minions, monsters

    def _is_tower(self, npc):
        sub_type = npc.get("sub_type")
        return sub_type == 21 or "TOWER" in str(sub_type).upper()

    def _is_monster(self, npc, my_camp):
        actor_type = str(npc.get("actor_type", "")).upper()
        sub_type = str(npc.get("sub_type", "")).upper()
        camp = self._camp_value(npc.get("camp"))
        return "MONSTER" in actor_type or "MONSTER" in sub_type or camp in (0, None, "0")

    def _tower_targets_self(self, my_hero, enemy_tower):
        return bool(my_hero and enemy_tower and enemy_tower.get("attack_target") == my_hero.get("runtime_id"))

    def _friendly_minion_tanking(self, enemy_tower, friendly_minions):
        target = (enemy_tower or {}).get("attack_target")
        return target is not None and target in {unit.get("runtime_id") for unit in friendly_minions}

    def _can_finish_tower(self, my_hero, enemy_tower):
        return bool(
            my_hero
            and enemy_tower
            and self._hp_ratio(enemy_tower) < RuleConfig.FINISH_TOWER_HP_RATIO
            and self._hp_ratio(my_hero) > RuleConfig.FINISH_TOWER_SELF_HP_RATIO
            and self._can_attack_tower(my_hero, enemy_tower)
        )

    def _can_attack_tower(self, my_hero, enemy_tower):
        if not my_hero or not enemy_tower:
            return False
        return self._dist(self._pos(my_hero), self._pos(enemy_tower)) <= my_hero.get("attack_range", 0) + RuleConfig.ATTACK_TOWER_EXTRA_RANGE

    def _enhanced_attack_ready(self, my_hero):
        if not my_hero:
            return False
        config_id = my_hero.get("config_id")
        if config_id == 112:
            return self.luban_attack_count >= 4 or self.recent_skill_frames > 0
        if config_id == 133:
            marks = (((my_hero.get("buff_state") or {}).get("buff_marks", []) or []))
            max_layer = max((mark.get("layer", 0) for mark in marks), default=0)
            return self.direnjie_attack_mod2 == 0 or max_layer > 0
        return False

    def _button_legal(self, legal_action, button):
        try:
            return bool(legal_action[button])
        except (TypeError, IndexError):
            return True

    def _attack_tower_action(self):
        return [BUTTON_ATTACK, 15, 15, 15, 15, TARGET_TOWER]

    def _move_to(self, my_hero, pos):
        my_pos = self._pos(my_hero)
        return [BUTTON_MOVE, self._dir_bin(pos[0] - my_pos[0]), self._dir_bin(pos[1] - my_pos[1]), 15, 15, TARGET_NONE]

    def _dir_bin(self, value):
        value = max(-1.0, min(1.0, self._safe_div(value, RuleConfig.MOVE_DIR_SCALE)))
        return int(max(0, min(15, round((value + 1.0) * 7.5))))

    def _tower_risk(self, my_hero, enemy_tower):
        if not my_hero or not enemy_tower:
            return 0.0
        dist = self._dist(self._pos(my_hero), self._pos(enemy_tower))
        tower_range = enemy_tower.get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE)
        smooth = self._sigmoid((tower_range - dist) / RuleConfig.TOWER_RISK_SIGMOID_SCALE)
        target_self = 1.0 if enemy_tower.get("attack_target") == my_hero.get("runtime_id") else 0.0
        low_hp = self._sigmoid((RuleConfig.LOW_HP_RISK_RATIO - self._hp_ratio(my_hero)) / 0.08)
        return min(1.0, smooth * (0.35 + 0.45 * target_self + 0.20 * low_hp))

    def _hp_ratio(self, obj):
        return max(0.0, min(1.0, self._safe_div((obj or {}).get("hp", 0), (obj or {}).get("max_hp", 0))))

    def _alive(self, obj):
        return bool(obj and obj.get("hp", 0) > 0)

    def _pos(self, obj):
        loc = obj.get("location", {}) if obj else {}
        return loc.get("x", 100000), loc.get("z", 100000)

    def _cake_pos(self, cake):
        loc = (cake.get("collider", {}) or {}).get("location", {})
        return loc.get("x", 100000), loc.get("z", 100000)

    def _dist(self, a, b):
        ax, az = a
        bx, bz = b
        if 100000 in (ax, az, bx, bz):
            return 100000.0
        return ((ax - bx) ** 2 + (az - bz) ** 2) ** 0.5

    def _same_camp(self, left, right):
        return self._camp_value(left) == self._camp_value(right)

    def _camp_value(self, camp):
        if isinstance(camp, str):
            if camp.endswith("_1") or camp == "1":
                return 1
            if camp.endswith("_2") or camp == "2":
                return 2
        return camp

    def _sigmoid(self, value):
        value = max(-30.0, min(30.0, value))
        return 1.0 / (1.0 + np.exp(-value))

    def _safe_div(self, num, den):
        try:
            return num / den if den else 0.0
        except (TypeError, ZeroDivisionError):
            return 0.0

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return
        state_dict = torch.load(model_file_path, map_location=self.device)
        self._load_state_dict_compatible(state_dict)
        self.cur_model_name = model_file_path
        self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _load_state_dict_compatible(self, state_dict):
        model_state = self.model.state_dict()
        aliases = {
            "concat_mlp.fc_layers.concat_mlp_fc1.weight": "feature_fc.weight",
            "concat_mlp.fc_layers.concat_mlp_fc1.bias": "feature_fc.bias",
        }
        compatible_state = {}
        migrated = []
        skipped = []
        for key, value in state_dict.items():
            target_key = aliases.get(key, key)
            if target_key not in model_state:
                skipped.append(key)
                continue
            target = model_state[target_key]
            if target.shape == value.shape:
                compatible_state[target_key] = value
                continue
            if len(target.shape) == len(value.shape):
                new_value = target.clone()
                slices = tuple(slice(0, min(target.shape[i], value.shape[i])) for i in range(len(target.shape)))
                new_value[slices] = value[slices]
                if len(target.shape) == 2 and target.shape[1] > value.shape[1]:
                    new_value[:, value.shape[1] :] = 0
                compatible_state[target_key] = new_value
                migrated.append(f"{key}->{target_key}")
            else:
                skipped.append(key)

        model_state.update(compatible_state)
        self.model.load_state_dict(model_state, strict=False)
        if self.logger:
            if migrated:
                self.logger.info(f"compatible model load migrated layers: {migrated}")
            if skipped:
                self.logger.info(f"compatible model load skipped keys: {skipped}")

    def _sample_masked_action(self, logits, legal_action):
        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            action_list.append(self._legal_sample(probs, use_max=False))
            d_action_list.append(self._legal_sample(probs, use_max=True))

        index = len(self.label_size_list) - 1
        if len(legal_actions[index]) == self.label_size_list[-1]:
            target_legal_action = legal_actions[index]
            target_legal_action_d = legal_actions[index]
        else:
            target_legal_action_o = np.reshape(
                legal_actions[index],
                [
                    self.legal_action_size[0],
                    self.legal_action_size[-1] // self.legal_action_size[0],
                ],
            )
            one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]].reshape([self.label_size_list[0], 1])
            target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

            one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]].reshape([self.label_size_list[0], 1])
            target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        action_list.append(self._legal_sample(probs, use_max=False))

        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)
        d_action_list.append(self._legal_sample(probs, use_max=True))

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        legal_action = np.asarray(legal_action, dtype=np.float64)
        if legal_action.size <= 0:
            return np.ones_like(input_hidden, dtype=np.float64) / len(input_hidden)
        if np.sum(legal_action) <= 0:
            legal_action = np.ones_like(legal_action, dtype=np.float64)
        tmp = np.asarray(input_hidden, dtype=np.float64)
        tmp = tmp - np.max(tmp * legal_action - 1e20 * (1.0 - legal_action), keepdims=True)
        tmp = np.clip(tmp, -60.0, 60.0)
        probs = np.exp(tmp) * legal_action
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        prob_sum = np.sum(probs, dtype=np.float64)
        if prob_sum <= 0:
            probs = legal_action / np.sum(legal_action, dtype=np.float64)
        else:
            probs = probs / prob_sum
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.maximum(probs, 0.0)
        prob_sum = np.sum(probs, dtype=np.float64)
        if prob_sum <= 0:
            probs = np.ones_like(probs, dtype=np.float64) / len(probs)
        else:
            probs = probs / prob_sum

        if use_max:
            return int(np.argmax(probs))

        cdf = np.cumsum(probs, dtype=np.float64)
        cdf[-1] = 1.0
        return int(np.searchsorted(cdf, np.random.random(), side="right"))
