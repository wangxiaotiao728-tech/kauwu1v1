#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import time
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_from_pairs,
)
from agent_ppo.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


# Metrics used by monitor_builder.py.
# 说明：
# 1. 旧 D401 指标保留。
# 2. 新增指标只追加。
# 3. learner-only 指标不兜底上报 0，避免 value_loss/policy_loss 显示假 0。
OLD_D401_KEYS = [
    "hero_hurt",
    "total_damage",
    "hero_damage",
    "crit",
    "skill_hit",
    "no_ops",
    "in_grass",
    "under_tower_behavior",
    "passive_skills",
]

LEARNER_ONLY_KEYS = {
    "total_loss",
    "value_loss",
    "policy_loss",
    "entropy_loss",
    "approx_kl",
    "clip_fraction",
    "learning_rate",
    "entropy_beta",
    "ppo_clip",
    "grad_norm",
    "hidden_norm",
    "feature_nan_count",
}

MATCHUP_PAIRS = [(112, 112), (112, 133), (133, 112), (133, 133)]
MATCHUP_MONITOR_KEYS = []
for _my_hero_id, _enemy_hero_id in MATCHUP_PAIRS:
    _prefix = f"matchup_{_my_hero_id}_vs_{_enemy_hero_id}"
    MATCHUP_MONITOR_KEYS.extend([
        f"{_prefix}_game",
        f"{_prefix}_win",
        f"{_prefix}_win_rate",
        f"{_prefix}_game_total",
    ])

NEW_MONITOR_KEYS = [
    # environment
    "reward", "episode_cnt", "frame_no", "win",
    "my_hp", "enemy_hp", "own_tower_hp_ratio", "enemy_tower_hp_ratio",
    "kill_count", "death_count", "cake_count", "neutral_count",

    # ppo / learner
    "total_loss", "value_loss", "policy_loss", "entropy_loss",
    "approx_kl", "clip_fraction", "learning_rate", "entropy_beta", "ppo_clip",
    "grad_norm", "hidden_norm", "feature_nan_count",

    # reward groups
    "reward_objective", "reward_growth_combat", "reward_behavior_safety",

    # reward items
    "tower_hp_point", "kill", "death",
    "money", "exp", "last_hit",
    "hp_point", "hero_hurt", "hero_damage", "total_damage",
    "lane_clear", "defense", "cake", "forward",
    "tower_risk", "stuck", "no_ops", "grass_behavior",
    "skill_hit", "crit", "in_grass", "under_tower_behavior", "passive_skills",

    # behavior diagnostics from frame/action/history
    "enemy_soldier_count", "friendly_soldier_count",
    "enemy_soldier_near_own_tower_count", "own_tower_target_enemy_soldier_count",
    "defense_emergency_count",
    "target_soldier_rate", "target_enemy_rate", "target_tower_rate", "target_monster_rate",
    "button_move_rate", "button_attack_rate", "button_none_rate",
    "button_skill1_rate", "button_skill2_rate", "button_skill3_rate",
    "skill_target_enemy_rate", "skill_target_soldier_rate", "skill_target_tower_rate", "skill_center_rate",
    "stuck_count", "grass_long_stay_count", "grass_no_effective_count", "unsafe_tower_entry_count",
    "own_cake_pick_count", "low_hp_own_cake_approach_count",
] + MATCHUP_MONITOR_KEYS


def _dedup_keys(keys):
    out = []
    for key in keys:
        if key not in out:
            out.append(key)
    return out


MONITOR_KEYS = _dedup_keys(OLD_D401_KEYS + NEW_MONITOR_KEYS)

# These keys should be averaged over the episode if emitted per decision frame.
AVG_MONITOR_KEYS = {
    "my_hp", "enemy_hp", "own_tower_hp_ratio", "enemy_tower_hp_ratio",
    "in_grass", "under_tower_behavior",
    "enemy_soldier_count", "friendly_soldier_count",
    "enemy_soldier_near_own_tower_count",
    "target_soldier_rate", "target_enemy_rate", "target_tower_rate", "target_monster_rate",
    "button_move_rate", "button_attack_rate", "button_none_rate",
    "button_skill1_rate", "button_skill2_rate", "button_skill3_rate",
    "skill_target_enemy_rate", "skill_target_soldier_rate", "skill_target_tower_rate", "skill_center_rate",
}

REWARD_GROUPS_FOR_MONITOR = {
    "reward_objective": ["tower_hp_point", "kill", "death"],
    "reward_growth_combat": [
        "hp_point", "money", "exp", "last_hit", "hero_hurt",
        "total_damage", "hero_damage", "skill_hit",
    ],
    "reward_behavior_safety": [
        "lane_clear", "defense", "cake", "forward", "tower_risk", "stuck",
        "no_ops", "grass_behavior",
    ],
}

def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Lineup iterator (112:Luban, 133:DiRenjie)
    # 阵容迭代器 (112:鲁班， 133:狄仁杰)
    lineup_mode = getattr(GameConfig, "TRAIN_LINEUP_MODE", "all_matchups")
    lineup_pairs = getattr(GameConfig, "TRAIN_LINEUP_PAIRS", {}).get(lineup_mode)
    if not lineup_pairs:
        lineup_pairs = [(112, 112), (112, 133), (133, 112), (133, 133)]
    lineup_iterator = lineup_iterator_from_pairs(lineup_pairs)
    if logger:
        logger.info(f"lineup_mode={lineup_mode}, lineup_pairs={lineup_pairs}")

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_training_metric_log_time = 0
        self.latest_training_metrics = {}
        self.matchup_stats = {
            f"matchup_{my_hero_id}_vs_{enemy_hero_id}": {"game": 0.0, "win": 0.0}
            for my_hero_id, enemy_hero_id in MATCHUP_PAIRS
        }

    @staticmethod
    def _safe_float(value):
        try:
            if value is None:
                return 0.0
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if hasattr(value, "item"):
                return float(value.item())
            if isinstance(value, (list, tuple)):
                return float(value[0]) if len(value) > 0 else 0.0
            return float(value)
        except Exception:
            return 0.0

    @classmethod
    def _flatten_training_metrics(cls, metrics):
        """
        Flatten learner metrics and normalize common aliases.

        Some learner implementations return nested dicts or slightly different key names.
        This function keeps original keys and also writes normalized monitor keys.
        """
        flat = {}
        if not isinstance(metrics, dict):
            return flat

        def put(k, v):
            flat[str(k)] = cls._safe_float(v)

        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    put(sub_key, sub_value)
            else:
                put(key, value)

        # Common aliases -> monitor keys. Keep this defensive; unknown keys remain untouched.
        alias_map = {
            "lr": "learning_rate",
            "learn_rate": "learning_rate",
            "clip": "ppo_clip",
            "clip_param": "ppo_clip",
            "beta": "entropy_beta",
            "var_beta": "entropy_beta",
            "kl": "approx_kl",
            "approximate_kl": "approx_kl",
            "clip_frac": "clip_fraction",
            "clip_ratio": "clip_fraction",
        }
        for src, dst in alias_map.items():
            if src in flat and dst not in flat:
                flat[dst] = flat[src]

        return flat

    @staticmethod
    def _new_monitor_acc():
        return {
            "sum": {key: 0.0 for key in MONITOR_KEYS},
            "cnt": {key: 0 for key in MONITOR_KEYS},
        }

    @classmethod
    def _accumulate_monitor_items(cls, acc, data):
        if not isinstance(data, dict):
            return
        for key in MONITOR_KEYS:
            if key in data:
                acc["sum"][key] += cls._safe_float(data.get(key, 0.0))
                acc["cnt"][key] += 1

    @classmethod
    def _finalize_monitor_items(cls, acc):
        out = {}
        for key in MONITOR_KEYS:
            if key in AVG_MONITOR_KEYS:
                cnt = max(1, acc["cnt"].get(key, 0))
                out[key] = acc["sum"].get(key, 0.0) / cnt
            else:
                out[key] = acc["sum"].get(key, 0.0)
        # If reward process did not directly emit grouped rewards, build them from item sums.
        for group_key, item_keys in REWARD_GROUPS_FOR_MONITOR.items():
            if abs(out.get(group_key, 0.0)) < 1e-12:
                out[group_key] = sum(out.get(item_key, 0.0) for item_key in item_keys)
        return out

    @classmethod
    def _episode_win_value(cls, observation, report_side, terminated, truncated, is_train_test_timeout):
        """Use env win when available; use 0.5 for timeout-like exits."""
        if truncated or is_train_test_timeout:
            return 0.5
        try:
            side_obs = observation.get(str(report_side), {}) if isinstance(observation, dict) else {}
            if "win" in side_obs:
                return 1.0 if cls._safe_float(side_obs.get("win", 0.0)) > 0.5 else 0.0
        except Exception:
            pass
        if terminated:
            return 0.0
        return 0.5

    @staticmethod
    def _matchup_monitor_items(lineup, report_side, win_value):
        data = {}
        try:
            if not isinstance(lineup, (list, tuple)) or len(lineup) < 2:
                return data
            side = int(report_side)
            if side not in (0, 1):
                side = 0
            my_hero = int(lineup[side])
            enemy_hero = int(lineup[1 - side])
            prefix = f"matchup_{my_hero}_vs_{enemy_hero}"
            data[f"{prefix}_game"] = 1.0
            data[f"{prefix}_win"] = float(win_value)
        except Exception:
            pass
        return data

    def _record_matchup_result(self, lineup, report_side, win_value):
        try:
            if not isinstance(lineup, (list, tuple)) or len(lineup) < 2:
                return
            side = int(report_side)
            if side not in (0, 1):
                side = 0
            prefix = f"matchup_{int(lineup[side])}_vs_{int(lineup[1 - side])}"
            if prefix not in self.matchup_stats:
                self.matchup_stats[prefix] = {"game": 0.0, "win": 0.0}
            self.matchup_stats[prefix]["game"] += 1.0
            self.matchup_stats[prefix]["win"] += float(win_value)
        except Exception:
            pass

    def _matchup_rate_monitor_items(self):
        data = {}
        for my_hero_id, enemy_hero_id in MATCHUP_PAIRS:
            prefix = f"matchup_{my_hero_id}_vs_{enemy_hero_id}"
            stat = self.matchup_stats.get(prefix, {"game": 0.0, "win": 0.0})
            game = float(stat.get("game", 0.0))
            win = float(stat.get("win", 0.0))
            data[f"{prefix}_game_total"] = game
            data[f"{prefix}_win_rate"] = win / game if game > 0 else 0.0
        return data

    def _extract_env_monitor_items(self, obs):
        """Extract low-cost environment metrics for panels."""
        data = {}
        try:
            if not isinstance(obs, dict):
                return data
            frame_state = obs.get("frame_state", {}) or {}
            hero_states = frame_state.get("hero_states", [])
            npc_states = frame_state.get("npc_states", [])
            my_hero = self._get_main_hero_from_obs(obs)
            if my_hero is None:
                return data

            obs_camp = obs.get("camp", obs.get("player_camp", None))
            main_camp = obs_camp if obs_camp is not None else my_hero.get("camp", None)
            if main_camp is not None and my_hero.get("camp", None) != main_camp:
                for hero in hero_states:
                    if hero.get("camp", None) == main_camp:
                        my_hero = hero
                        break
            enemy_hero = None
            for hero in hero_states:
                if hero is not my_hero and hero.get("camp") != main_camp:
                    enemy_hero = hero
                    break
            if my_hero:
                data["my_hp"] = self._safe_float(my_hero.get("hp", 0)) / max(1.0, self._safe_float(my_hero.get("max_hp", 1)))
                data["kill_count"] = self._safe_float(my_hero.get("kill_cnt", 0))
                data["death_count"] = self._safe_float(my_hero.get("dead_cnt", 0))
                data["in_grass"] = 1.0 if bool(my_hero.get("is_in_grass", False)) else 0.0
            if enemy_hero:
                data["enemy_hp"] = self._safe_float(enemy_hero.get("hp", 0)) / max(1.0, self._safe_float(enemy_hero.get("max_hp", 1)))
            own_tower = None
            enemy_tower = None
            friendly_soldier_ids = set()
            enemy_soldier_ids = set()
            enemy_soldiers_near_own_tower = 0
            friendly_soldiers = 0
            enemy_soldiers = 0
            neutral_count = 0
            for npc in npc_states:
                sub_type = int(npc.get("sub_type", -1))
                config_id = int(self._safe_float(npc.get("config_id", npc.get("configId", -1))))
                hp = self._safe_float(npc.get("hp", 0))
                if hp <= 0:
                    continue
                if sub_type == 21:
                    if npc.get("camp") == main_camp:
                        own_tower = npc
                    else:
                        enemy_tower = npc
                elif sub_type == 11:
                    if npc.get("camp") == main_camp:
                        friendly_soldiers += 1
                        friendly_soldier_ids.add(npc.get("runtime_id"))
                    else:
                        enemy_soldiers += 1
                        enemy_soldier_ids.add(npc.get("runtime_id"))
                elif config_id in getattr(GameConfig, "MONSTER_CONFIG_IDS", set()):
                    neutral_count += 1
            if own_tower:
                data["own_tower_hp_ratio"] = self._safe_float(own_tower.get("hp", 0)) / max(1.0, self._safe_float(own_tower.get("max_hp", 1)))
            if enemy_tower:
                data["enemy_tower_hp_ratio"] = self._safe_float(enemy_tower.get("hp", 0)) / max(1.0, self._safe_float(enemy_tower.get("max_hp", 1)))
            if own_tower:
                tower_loc = own_tower.get("location", {}) or {}
                tx = self._safe_float(tower_loc.get("x", 0))
                tz = self._safe_float(tower_loc.get("z", 0))
                tower_range = self._safe_float(own_tower.get("attack_range", 8800)) * 1.15
                for npc in npc_states:
                    if npc.get("runtime_id") not in enemy_soldier_ids:
                        continue
                    loc = npc.get("location", {}) or {}
                    dx = self._safe_float(loc.get("x", 0)) - tx
                    dz = self._safe_float(loc.get("z", 0)) - tz
                    if (dx * dx + dz * dz) ** 0.5 <= tower_range:
                        enemy_soldiers_near_own_tower += 1

            own_tower_target = own_tower.get("attack_target", 0) if own_tower else 0
            enemy_tower_target = enemy_tower.get("attack_target", 0) if enemy_tower else 0
            own_tower_target_enemy_soldier = 1.0 if own_tower_target in enemy_soldier_ids else 0.0
            defense_emergency = 1.0 if own_tower_target_enemy_soldier > 0 or enemy_soldiers_near_own_tower > 0 else 0.0

            unsafe_tower_entry = 0.0
            under_tower_behavior = 0.0
            if enemy_tower:
                hero_loc = my_hero.get("location", {}) or {}
                tower_loc = enemy_tower.get("location", {}) or {}
                dx = self._safe_float(hero_loc.get("x", 0)) - self._safe_float(tower_loc.get("x", 0))
                dz = self._safe_float(hero_loc.get("z", 0)) - self._safe_float(tower_loc.get("z", 0))
                enemy_tower_range = self._safe_float(enemy_tower.get("attack_range", 8800))
                in_enemy_tower = (dx * dx + dz * dz) ** 0.5 <= enemy_tower_range
                friendly_tanking = enemy_tower_target in friendly_soldier_ids
                tower_target_me = enemy_tower_target == my_hero.get("runtime_id")
                unsafe_tower_entry = 1.0 if (in_enemy_tower and not friendly_tanking) or tower_target_me else 0.0
                if tower_target_me:
                    under_tower_behavior = -2.0
                elif in_enemy_tower and not friendly_tanking:
                    under_tower_behavior = -1.0
                elif in_enemy_tower and friendly_tanking:
                    under_tower_behavior = 0.5

            data["friendly_soldier_count"] = friendly_soldiers
            data["enemy_soldier_count"] = enemy_soldiers
            data["enemy_soldier_near_own_tower_count"] = enemy_soldiers_near_own_tower
            data["own_tower_target_enemy_soldier_count"] = own_tower_target_enemy_soldier
            data["defense_emergency_count"] = defense_emergency
            data["unsafe_tower_entry_count"] = unsafe_tower_entry
            data["under_tower_behavior"] = under_tower_behavior
            data["cake_count"] = len(frame_state.get("cakes", []) or [])
            data["neutral_count"] = neutral_count
        except Exception:
            pass
        return data


    # ============================================================
    # Action / history / cake diagnosis monitors
    # ============================================================

    @classmethod
    def _flatten_action(cls, action):
        """Robustly flatten action to [button, move_x, move_z, skill_x, skill_z, target]."""
        try:
            if action is None:
                return []

            if hasattr(action, "action"):
                action = action.action

            if isinstance(action, dict):
                for key in ("action", "actions", "act", "d_action"):
                    if key in action:
                        action = action[key]
                        break

            if hasattr(action, "tolist"):
                action = action.tolist()

            import numpy as np
            return np.asarray(action, dtype=np.int64).reshape(-1).tolist()
        except Exception:
            return []

    @classmethod
    def _extract_action_monitor_items(cls, action):
        """
        Extract button / target / skill-offset diagnosis from selected action.

        Expected flattened action:
            [button, move_x, move_z, skill_x, skill_z, target]

        如果你的项目 button 编号不同，只需要在 GameConfig 中设置：
            BUTTON_MOVE / BUTTON_ATTACK / BUTTON_SKILL1 / BUTTON_SKILL2 / BUTTON_SKILL3 / BUTTON_NONE
        """
        data = {}

        flat = cls._flatten_action(action)
        if len(flat) < 1:
            return data

        button = int(flat[0])
        skill_x = int(flat[3]) if len(flat) > 3 else 8
        skill_z = int(flat[4]) if len(flat) > 4 else 8
        target = int(flat[5]) if len(flat) > 5 else 0

        # 默认 button 编号按常见 1v1 动作头顺序。
        # 若实际不同，在 conf.py 的 GameConfig 里覆盖这些常量。
        BUTTON_MOVE = getattr(GameConfig, "BUTTON_MOVE", 0)
        BUTTON_ATTACK = getattr(GameConfig, "BUTTON_ATTACK", 1)
        BUTTON_SKILL1 = getattr(GameConfig, "BUTTON_SKILL1", 2)
        BUTTON_SKILL2 = getattr(GameConfig, "BUTTON_SKILL2", 3)
        BUTTON_SKILL3 = getattr(GameConfig, "BUTTON_SKILL3", 4)
        BUTTON_NONE = getattr(GameConfig, "BUTTON_NONE", 11)

        TARGET_ENEMY = 1
        TARGET_SOLDIER_START = 3
        TARGET_SOLDIER_END = 6
        TARGET_TOWER = 7
        TARGET_MONSTER = 8

        is_skill = button in {BUTTON_SKILL1, BUTTON_SKILL2, BUTTON_SKILL3}

        data["button_move_rate"] = 1.0 if button == BUTTON_MOVE else 0.0
        data["button_attack_rate"] = 1.0 if button == BUTTON_ATTACK else 0.0
        data["button_none_rate"] = 1.0 if button == BUTTON_NONE else 0.0
        data["button_skill1_rate"] = 1.0 if button == BUTTON_SKILL1 else 0.0
        data["button_skill2_rate"] = 1.0 if button == BUTTON_SKILL2 else 0.0
        data["button_skill3_rate"] = 1.0 if button == BUTTON_SKILL3 else 0.0

        data["target_soldier_rate"] = 1.0 if TARGET_SOLDIER_START <= target <= TARGET_SOLDIER_END else 0.0
        data["target_enemy_rate"] = 1.0 if target == TARGET_ENEMY else 0.0
        data["target_tower_rate"] = 1.0 if target == TARGET_TOWER else 0.0
        data["target_monster_rate"] = 1.0 if target == TARGET_MONSTER else 0.0

        if is_skill:
            data["skill_target_enemy_rate"] = 1.0 if target == TARGET_ENEMY else 0.0
            data["skill_target_soldier_rate"] = 1.0 if TARGET_SOLDIER_START <= target <= TARGET_SOLDIER_END else 0.0
            data["skill_target_tower_rate"] = 1.0 if target == TARGET_TOWER else 0.0
            # 16x16 skill offset 中心为 8，允许 7/8/9 作为接近中心。
            data["skill_center_rate"] = 1.0 if abs(skill_x - 8) <= 1 and abs(skill_z - 8) <= 1 else 0.0
        else:
            data["skill_target_enemy_rate"] = 0.0
            data["skill_target_soldier_rate"] = 0.0
            data["skill_target_tower_rate"] = 0.0
            data["skill_center_rate"] = 0.0

        return data

    def _get_main_hero_from_obs(self, obs):
        """Select controlled hero using player_id / camp instead of blindly using hero_states[0]."""
        if not isinstance(obs, dict):
            return None

        frame_state = obs.get("frame_state", {}) or {}
        hero_states = frame_state.get("hero_states", []) or []
        player_id = obs.get("player_id", None)
        player_camp = obs.get("camp", obs.get("player_camp", None))

        for hero in hero_states:
            if player_id is not None and hero.get("runtime_id") == player_id:
                return hero
            if player_id is not None and hero.get("player_id") == player_id:
                return hero

        for hero in hero_states:
            if player_camp is not None and hero.get("camp") == player_camp:
                return hero

        return hero_states[0] if hero_states else None

    def _extract_history_behavior_monitor_items(self, obs, state):
        """Extract stuck / grass diagnosis with per-episode history state."""
        data = {}

        try:
            hero = self._get_main_hero_from_obs(obs)
            if hero is None:
                return data

            loc = hero.get("location", {}) or {}
            x = self._safe_float(loc.get("x", 0.0))
            z = self._safe_float(loc.get("z", 0.0))

            is_in_grass = bool(hero.get("is_in_grass", False))
            hit_infos = hero.get("hit_target_info", []) or []
            real_cmd = hero.get("real_cmd", []) or []

            has_hit = len(hit_infos) > 0
            has_cmd = len(real_cmd) > 0

            if state.get("last_x", None) is None:
                moved = True
            else:
                dx = x - state.get("last_x", 0.0)
                dz = z - state.get("last_z", 0.0)
                moved = (dx * dx + dz * dz) > 10000.0  # distance > 100

            if is_in_grass:
                state["grass_steps"] = state.get("grass_steps", 0) + 1
                if not has_hit and not has_cmd:
                    state["grass_no_effective_steps"] = state.get("grass_no_effective_steps", 0) + 1
                else:
                    state["grass_no_effective_steps"] = 0
            else:
                state["grass_steps"] = 0
                state["grass_no_effective_steps"] = 0

            if not moved and not has_hit and not has_cmd:
                state["same_pos_steps"] = state.get("same_pos_steps", 0) + 1
            else:
                state["same_pos_steps"] = 0

            state["last_x"] = x
            state["last_z"] = z

            data["stuck_count"] = 1.0 if state.get("same_pos_steps", 0) >= 8 else 0.0
            data["grass_long_stay_count"] = 1.0 if state.get("grass_steps", 0) >= 12 else 0.0
            data["grass_no_effective_count"] = 1.0 if state.get("grass_no_effective_steps", 0) >= 8 else 0.0

        except Exception:
            return data

        return data

    def _extract_cake_behavior_monitor_items(self, obs, state):
        """Extract own-cake behavior monitor from current observation and history."""
        data = {}

        try:
            frame_state = obs.get("frame_state", {}) or {}
            hero = self._get_main_hero_from_obs(obs)
            if hero is None:
                return data

            cakes = frame_state.get("cakes", []) or []
            npc_states = frame_state.get("npc_states", []) or []

            hp_ratio = self._safe_float(hero.get("hp", 0)) / max(1.0, self._safe_float(hero.get("max_hp", 1)))
            main_camp = hero.get("camp")

            hero_loc = hero.get("location", {}) or {}
            hx = self._safe_float(hero_loc.get("x", 0))
            hz = self._safe_float(hero_loc.get("z", 0))

            own_tower = None
            for npc in npc_states:
                if int(self._safe_float(npc.get("sub_type", -1))) == 21 and npc.get("camp") == main_camp:
                    own_tower = npc
                    break

            if own_tower is None or not cakes:
                return data

            tower_loc = own_tower.get("location", {}) or {}
            tx = self._safe_float(tower_loc.get("x", 0))
            tz = self._safe_float(tower_loc.get("z", 0))

            valid = []
            for cake in cakes:
                if int(self._safe_float(cake.get("configId", cake.get("config_id", -1)))) != 5:
                    continue
                loc = (cake.get("collider", {}) or {}).get("location", {}) or {}
                cx = self._safe_float(loc.get("x", 0))
                cz = self._safe_float(loc.get("z", 0))
                dist_to_tower_sq = (cx - tx) ** 2 + (cz - tz) ** 2
                valid.append((dist_to_tower_sq, cx, cz))

            if not valid:
                return data

            _, cx, cz = min(valid, key=lambda item: item[0])
            dist = ((cx - hx) ** 2 + (cz - hz) ** 2) ** 0.5

            prev_dist = state.get("prev_own_cake_dist", None)
            state["prev_own_cake_dist"] = dist

            if hp_ratio < 0.4 and prev_dist is not None and dist < prev_dist:
                data["low_hp_own_cake_approach_count"] = 1.0
            else:
                data["low_hp_own_cake_approach_count"] = 0.0

            # 简化拾取代理：低血/中血且靠近己方血包。
            # 更严格拾取可在 reward_process.py 中结合 hp 上升和 cake 状态变化实现。
            if hp_ratio < 0.6 and dist < 1500:
                data["own_cake_pick_count"] = 1.0
            else:
                data["own_cake_pick_count"] = 0.0

        except Exception:
            return data

        return data

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        # camp_keys[i] is the camp key for agents[i] based on monitor_side
        # monitor_side 的 agent 对应 blue/red 取决于 monitor_side 配置
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            # Determine which camp this agent controls
            # 确定该 agent 控制哪个阵营
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
            }

            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                self.latest_training_metrics = self._flatten_training_metrics(training_metrics)
                now = time.time()
                if self.logger and now - self.last_training_metric_log_time >= 60:
                    compact_metrics = {
                        key: round(self._safe_float(self.latest_training_metrics.get(key, 0.0)), 6)
                        for key in sorted(LEARNER_ONLY_KEYS)
                        if key in self.latest_training_metrics
                    }
                    self.logger.info(f"training_metrics summary {compact_metrics}")
                    self.last_training_metric_log_time = now

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)

            # Call init_config on agents to get summoner skill selections
            # 调用 agent 的 init_config 获取召唤师技能选择，注入 usr_conf
            self._call_init_config(usr_conf)

            # Start a new environment
            # 启动新对局，返回初始环境状态
            env_obs = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]

            # Reset agents
            # 重置智能体
            self.reset_agents(observation)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0.0] * self.agent_num
            monitor_acc_list = [self._new_monitor_acc() for _ in range(self.agent_num)]
            behavior_state_list = [
                {
                    "grass_steps": 0,
                    "grass_no_effective_steps": 0,
                    "same_pos_steps": 0,
                    "last_x": None,
                    "last_z": None,
                    "prev_own_cake_dist": None,
                }
                for _ in range(self.agent_num)
            ]
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    frame_state = observation[str(i)]["frame_state"]
                    reward = agent.reward_manager.result(frame_state)
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += self._safe_float(reward.get("reward_sum", 0.0))
                    self._accumulate_monitor_items(monitor_acc_list[i], reward)
                    self._accumulate_monitor_items(monitor_acc_list[i], self._extract_env_monitor_items(observation[str(i)]))
                    self._accumulate_monitor_items(
                        monitor_acc_list[i],
                        self._extract_history_behavior_monitor_items(observation[str(i)], behavior_state_list[i]),
                    )
                    self._accumulate_monitor_items(
                        monitor_acc_list[i],
                        self._extract_cake_behavior_monitor_items(observation[str(i)], behavior_state_list[i]),
                    )

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])

                        # Action diagnostics should be accumulated for every predicted side,
                        # not only sampled side. Otherwise common_ai / monitor_side settings
                        # can make target/button/skill panels stay at 0.
                        action_monitor = self._extract_action_monitor_items(actions[index])
                        self._accumulate_monitor_items(monitor_acc_list[index], action_monitor)

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        frame_state = observation[str(i)]["frame_state"]
                        reward = agent.reward_manager.result(frame_state)
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += self._safe_float(reward.get("reward_sum", 0.0))
                        self._accumulate_monitor_items(monitor_acc_list[i], reward)
                        self._accumulate_monitor_items(monitor_acc_list[i], self._extract_env_monitor_items(observation[str(i)]))
                        self._accumulate_monitor_items(
                            monitor_acc_list[i],
                            self._extract_history_behavior_monitor_items(observation[str(i)], behavior_state_list[i]),
                        )
                        self._accumulate_monitor_items(
                            monitor_acc_list[i],
                            self._extract_cake_behavior_monitor_items(observation[str(i)], behavior_state_list[i]),
                        )

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"].get("reward_sum", 0.0),
                            )

                    # If monitor_side is common_ai / non-predict side, switch reporting to a predicted side.
                    report_side = monitor_side
                    if report_side >= len(self.do_predicts) or not self.do_predicts[report_side]:
                        for side_idx, flag in enumerate(self.do_predicts):
                            if flag:
                                report_side = side_idx
                                break
                    win_value = self._episode_win_value(
                        observation,
                        report_side,
                        terminated,
                        truncated,
                        (not terminated) and (not truncated) and is_train_test and frame_no >= 1000,
                    )
                    self._record_matchup_result(lineup, report_side, win_value)

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = self._finalize_monitor_items(monitor_acc_list[report_side])
                        monitor_data.update({
                            "episode_cnt": self.episode_cnt,
                            "frame_no": frame_no,
                            "reward": round(reward_sum_list[report_side], 2),
                            "win": win_value,
                        })
                        monitor_data.update(self._matchup_monitor_items(lineup, report_side, win_value))
                        monitor_data.update(self._matchup_rate_monitor_items())

                        # Merge learner-side PPO metrics.
                        # IMPORTANT: monitor panels with multiple lines expect every metric to be present
                        # on every report. If a learner metric is temporarily absent, keep its last cached
                        # value if available; otherwise report 0.0 only before the first learner update.
                        for key in LEARNER_ONLY_KEYS:
                            if key in self.latest_training_metrics:
                                monitor_data[key] = self.latest_training_metrics[key]
                            else:
                                monitor_data[key] = monitor_data.get(key, 0.0)

                        # Ensure every registered key has a value at every report.
                        # This prevents missing lines/gaps in multi-metric panels.
                        for key in MONITOR_KEYS:
                            monitor_data[key] = round(self._safe_float(monitor_data.get(key, 0.0)), 6)

                        if self.monitor:
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体要进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体要进行训练采样
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时, 无法获取到对手模型，因此将替换为最新模型
                            self.logger.info("Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[str(i)])
