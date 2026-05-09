#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Full D401 replica sample definition.

This version extends the baseline sequence sample protocol with full multi-critic
support: per-reward-group rewards, returns, advantages, and old values are
serialized and used by the learner.
"""

from common_python.utils.common_func import create_cls, Frame
import numpy as np
import collections
import random
from agent_ppo.conf.conf import Config
import itertools


def _lineup_iterator_shuffle_cycle(camps):
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


def lineup_iterator_roundrobin_camp_heroes(camp_heroes=None):
    if not camp_heroes:
        raise Exception("camp_heroes is empty")
    camps = []
    for lineups in itertools.product(camp_heroes, camp_heroes):
        camps.append(list(lineups))
    return _lineup_iterator_shuffle_cycle(camps)


ObsData = create_cls("ObsData", feature=None, legal_action=None, lstm_cell=None, lstm_hidden=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    d_prob=None,
    value=None,
    value_groups=None,
    lstm_cell=None,
    lstm_hidden=None,
)

SampleData = create_cls("SampleData", sample=sum([shape[0] for shape in Config.data_shapes]))

NONE_ACTION = [0, 15, 15, 15, 15, 0]


def sample_process(collector):
    return collector.sample_process()


def _ordered_group_vector(reward_dict):
    groups = reward_dict.get("reward_groups", {}) if isinstance(reward_dict, dict) else {}
    return np.array([float(groups.get(name, 0.0)) for name in Config.REWARD_GROUP_NAMES], dtype=np.float32)


def build_frame(agent, observation):
    obs_data, act_data = agent.obs_data, agent.act_data
    is_train = False
    frame_state = observation["frame_state"]
    hero_list = frame_state["hero_states"]
    frame_no = frame_state["frame_no"]
    for hero in hero_list:
        hero_camp = hero.get("camp") if isinstance(hero, dict) else getattr(hero, "camp", None)
        hero_hp = hero.get("hp") if isinstance(hero, dict) else getattr(hero, "hp", 0)
        if hero_camp == agent.hero_camp:
            is_train = True if hero_hp > 0 else False

    if obs_data.feature is not None:
        feature_vec = np.array(obs_data.feature, dtype=np.float32)
    else:
        feature_vec = np.array(observation["observation"], dtype=np.float32)

    reward_info = observation.get("reward", {})
    reward = float(reward_info.get("reward_sum", 0.0))
    group_reward = _ordered_group_vector(reward_info)

    sub_action_mask = observation["sub_action_mask"]
    prob, value, action = act_data.prob, act_data.value, act_data.action
    value_groups = act_data.value_groups
    if value_groups is None:
        value_groups = np.zeros([Config.REWARD_GROUP_NUM], dtype=np.float32)
    value_groups = np.array(value_groups, dtype=np.float32).reshape([-1])[: Config.REWARD_GROUP_NUM]
    if value_groups.shape[0] < Config.REWARD_GROUP_NUM:
        value_groups = np.pad(value_groups, [0, Config.REWARD_GROUP_NUM - value_groups.shape[0]])

    lstm_cell, lstm_hidden = act_data.lstm_cell, act_data.lstm_hidden
    legal_action = _update_legal_action(observation["legal_action"], action)

    frame = Frame(
        frame_no=frame_no,
        feature=feature_vec.reshape([-1]),
        legal_action=legal_action.reshape([-1]),
        action=action,
        reward=reward,
        reward_sum=0.0,
        value=float(np.array(value).reshape([-1])[0]),
        next_value=0.0,
        advantage=0.0,
        prob=prob,
        sub_action=sub_action_mask[str(action[0])],
        lstm_info=np.concatenate([lstm_cell.flatten(), lstm_hidden.flatten()]).reshape([-1]),
        is_train=False if action[0] < 0 else is_train,
        group_reward=group_reward,
        group_reward_sum=np.zeros([Config.REWARD_GROUP_NUM], dtype=np.float32),
        group_value=value_groups,
        next_group_value=np.zeros([Config.REWARD_GROUP_NUM], dtype=np.float32),
        group_advantage=np.zeros([Config.REWARD_GROUP_NUM], dtype=np.float32),
    )
    return frame


def _update_legal_action(original_la, action):
    target_size = Config.LABEL_SIZE_LIST[-1]
    top_size = Config.LABEL_SIZE_LIST[0]
    original_la = np.array(original_la)
    fix_part = original_la[: -target_size * top_size]
    target_la = original_la[-target_size * top_size :]
    target_la = target_la.reshape([top_size, target_size])[action[0]]
    return np.concatenate([fix_part, target_la], axis=0)


class FrameCollector:
    def __init__(self, num_agents):
        self._data_shapes = Config.data_shapes
        self._LSTM_FRAME = Config.LSTM_TIME_STEPS
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA
        self.group_adv_weights = np.array(Config.REWARD_GROUP_ADV_WEIGHTS, dtype=np.float32)

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def save_frame(self, rl_data_info, agent_id):
        reward = self._clip_reward(float(rl_data_info.reward))
        group_reward = self._clip_group_reward(getattr(rl_data_info, "group_reward", np.zeros([Config.REWARD_GROUP_NUM])))

        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = rl_data_info.value
            last_rl_data_info.next_group_value = np.array(
                getattr(rl_data_info, "group_value", np.zeros([Config.REWARD_GROUP_NUM])), dtype=np.float32
            )
            last_rl_data_info.reward = reward
            last_rl_data_info.group_reward = group_reward

        rl_data_info.reward = 0.0
        rl_data_info.group_reward = np.zeros([Config.REWARD_GROUP_NUM], dtype=np.float32)
        self.rl_data_map[agent_id][rl_data_info.frame_no] = rl_data_info

    def save_last_frame(self, reward, agent_id):
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = 0.0
            last_rl_data_info.next_group_value = np.zeros([Config.REWARD_GROUP_NUM], dtype=np.float32)
            last_rl_data_info.reward = self._clip_reward(float(reward))
            # Terminal scalar rewards are distributed into the no_decay group if present.
            terminal_groups = np.zeros([Config.REWARD_GROUP_NUM], dtype=np.float32)
            if "no_decay" in Config.REWARD_GROUP_NAMES:
                terminal_groups[Config.REWARD_GROUP_NAMES.index("no_decay")] = float(reward)
            last_rl_data_info.group_reward = self._clip_group_reward(terminal_groups)

    def sample_process(self):
        self._calc_reward()
        self._format_data()
        return self.m_replay_buffer

    def _calc_reward(self):
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae = 0.0
            group_gae = np.zeros([Config.REWARD_GROUP_NUM], dtype=np.float32)
            for j in reversed_keys:
                rl_info = self.rl_data_map[i][j]

                group_reward = np.array(getattr(rl_info, "group_reward", np.zeros([Config.REWARD_GROUP_NUM])), dtype=np.float32)
                group_value = np.array(getattr(rl_info, "group_value", np.zeros([Config.REWARD_GROUP_NUM])), dtype=np.float32)
                next_group_value = np.array(
                    getattr(rl_info, "next_group_value", np.zeros([Config.REWARD_GROUP_NUM])), dtype=np.float32
                )
                group_delta = group_reward + self.gamma * next_group_value - group_value
                group_gae = group_gae * self.gamma * self.lamda + group_delta
                rl_info.group_advantage = group_gae.copy()
                rl_info.group_reward_sum = group_gae + group_value

                # Full multi-critic policy advantage: weighted sum of group advantages.
                rl_info.advantage = float(np.sum(rl_info.group_advantage * self.group_adv_weights))
                rl_info.reward_sum = float(np.sum(rl_info.group_reward_sum * self.group_adv_weights))

                # Also keep scalar GAE for compatibility / global value stabilization.
                delta = -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                gae = gae * self.gamma * self.lamda + delta
                # Prefer grouped returns when reward groups are available; fallback to scalar if all zero.
                if Config.REWARD_GROUP_NUM <= 0 or np.allclose(group_reward, 0.0):
                    rl_info.advantage = gae
                    rl_info.reward_sum = gae + rl_info.value

    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)], dtype=np.float32)
        idx, s_idx = 0, 0
        sample[-sample_lstm.shape[0] :] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx : s_idx + split_shape[0]] = sample_batch[:, idx : idx + one_shape].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]
        return sample.astype(np.float32)

    def _format_data(self):
        sample_one_size = np.sum(self._data_shapes[:-2]) // self._LSTM_FRAME
        sample_lstm_size = np.sum(self._data_shapes[-2:])
        sample_batch = np.zeros([self._LSTM_FRAME, sample_one_size], dtype=np.float32)

        for i in range(self.num_agents):
            sample_lstm = np.zeros([sample_lstm_size], dtype=np.float32)
            cnt = 0
            for j in self.rl_data_map[i]:
                rl_info = self.rl_data_map[i][j]
                idx = 0

                dlen = rl_info.feature.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.feature
                idx += dlen

                dlen = rl_info.legal_action.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.legal_action
                idx += dlen

                sample_batch[cnt, idx] = rl_info.reward_sum
                idx += 1
                sample_batch[cnt, idx] = rl_info.advantage
                idx += 1

                group_reward_sum = np.array(getattr(rl_info, "group_reward_sum", np.zeros([Config.REWARD_GROUP_NUM])))
                sample_batch[cnt, idx : idx + Config.REWARD_GROUP_NUM] = group_reward_sum
                idx += Config.REWARD_GROUP_NUM

                group_advantage = np.array(getattr(rl_info, "group_advantage", np.zeros([Config.REWARD_GROUP_NUM])))
                sample_batch[cnt, idx : idx + Config.REWARD_GROUP_NUM] = group_advantage
                idx += Config.REWARD_GROUP_NUM

                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.action
                idx += dlen

                for p in rl_info.prob:
                    dlen = len(p)
                    sample_batch[cnt, idx : idx + dlen] = p
                    idx += dlen

                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.sub_action
                idx += dlen

                sample_batch[cnt, idx] = rl_info.value
                idx += 1

                group_value = np.array(getattr(rl_info, "group_value", np.zeros([Config.REWARD_GROUP_NUM])))
                sample_batch[cnt, idx : idx + Config.REWARD_GROUP_NUM] = group_value
                idx += Config.REWARD_GROUP_NUM

                sample_batch[cnt, idx] = rl_info.is_train
                idx += 1

                assert idx == sample_one_size, "Sample check failed, {}/{}".format(idx, sample_one_size)

                cnt += 1
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    sample_array = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                    self.m_replay_buffer[i].append(SampleData(sample=sample_array))
                    sample_lstm = rl_info.lstm_info
                    sample_batch.fill(0.0)

    def _clip_reward(self, reward, max=100, min=-100):
        return max if reward > max else min if reward < min else reward

    def _clip_group_reward(self, group_reward, max=100, min=-100):
        arr = np.array(group_reward, dtype=np.float32).reshape([-1])
        if arr.shape[0] < Config.REWARD_GROUP_NUM:
            arr = np.pad(arr, [0, Config.REWARD_GROUP_NUM - arr.shape[0]])
        arr = arr[: Config.REWARD_GROUP_NUM]
        return np.clip(arr, min, max)

    def __len__(self):
        return max([len(agent_samples) for agent_samples in self.rl_data_map])
