#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

1v1 PPO training workflow.

修改原则：
1. 原有 D401 监控 key 全部保留。
2. 新增监控 key 只追加，不覆盖旧 key。
3. train_workflow.py 负责兜底上报所有 monitor_builder.py 中注册的新 key。
4. reward_process.py / agent.py 如果真实产出新诊断 key，会被自动累计并上报。
5. 环境基础指标从 frame_state 中低成本提取，避免面板空曲线。
"""

import os
import time

from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


# ============================================================
# Monitor keys
# 必须和 monitor_builder.py 中注册的 metrics_name 对齐。
# 缺失的 key 会兜底为 0.0，避免面板无曲线。
# ============================================================

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

NEW_MONITOR_KEYS = [
    # PPO / learner metrics
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

    # Environment metrics
    "episode_cnt",
    "frame_no",
    "reward",
    "win",
    "my_hp",
    "enemy_hp",
    "own_tower_hp_ratio",
    "enemy_tower_hp_ratio",
    "kill_count",
    "death_count",
    "cake_count",
    "neutral_count",
    "opponent_type",

    # Reward groups
    "reward_objective",
    "reward_growth_combat",
    "reward_behavior_safety",

    # Objective reward items
    "tower_hp_point",
    "kill",
    "death",

    # Growth / combat reward items
    "hp_point",
    "money",
    "exp",
    "last_hit",
    "lane_clear",
    "defense",
    "cake",
    "tower_risk",
    "stuck",
    "grass_behavior",

    # Lane / behavior diagnosis
    "enemy_soldier_count",
    "friendly_soldier_count",
    "enemy_soldier_near_own_tower_count",
    "own_tower_target_enemy_soldier_count",
    "defense_emergency_count",

    # Target diagnosis
    "target_soldier_rate",
    "target_enemy_rate",
    "target_tower_rate",
    "target_monster_rate",

    # Button diagnosis
    "button_move_rate",
    "button_attack_rate",
    "button_none_rate",
    "button_skill1_rate",
    "button_skill2_rate",
    "button_skill3_rate",

    # Skill diagnosis
    "skill_target_enemy_rate",
    "skill_target_soldier_rate",
    "skill_target_tower_rate",
    "skill_center_rate",

    # Abnormal behavior diagnosis
    "stuck_count",
    "grass_long_stay_count",
    "grass_no_effective_count",
    "unsafe_tower_entry_count",

    # Cake diagnosis
    "own_cake_pick_count",
    "low_hp_own_cake_approach_count",
]


def _dedup_keys(keys):
    out = []
    for key in keys:
        if key not in out:
            out.append(key)
    return out


MONITOR_KEYS = _dedup_keys(OLD_D401_KEYS + NEW_MONITOR_KEYS)


# 这些 key 如果每帧产生，按 episode 平均更有意义。
AVG_MONITOR_KEYS = {
    # Ratios / state values
    "my_hp",
    "enemy_hp",
    "own_tower_hp_ratio",
    "enemy_tower_hp_ratio",
    "enemy_soldier_count",
    "friendly_soldier_count",

    # PPO / health
    "hidden_norm",
    "feature_nan_count",
    "grad_norm",

    # Rates
    "target_soldier_rate",
    "target_enemy_rate",
    "target_tower_rate",
    "target_monster_rate",
    "button_move_rate",
    "button_attack_rate",
    "button_none_rate",
    "button_skill1_rate",
    "button_skill2_rate",
    "button_skill3_rate",
    "skill_target_enemy_rate",
    "skill_target_soldier_rate",
    "skill_target_tower_rate",
    "skill_center_rate",
}


# 三组 reward，如果 reward_process.py 没有直接给组 reward，
# train_workflow.py 会在 episode 末用分项自动合成。
REWARD_GROUPS_FOR_MONITOR = {
    "reward_objective": [
        "tower_hp_point",
        "kill",
        "death",
    ],
    "reward_growth_combat": [
        "hp_point",
        "money",
        "exp",
        "last_hit",
        "hero_hurt",
        "total_damage",
        "hero_damage",
        "skill_hit",
    ],
    "reward_behavior_safety": [
        "lane_clear",
        "defense",
        "cake",
        "tower_risk",
        "stuck",
        "no_ops",
        "grass_behavior",
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

    # Lineup iterator (112:Luban, 133:DiRenJie)
    # 阵容迭代器 (112:鲁班，133:狄仁杰)
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes([112, 133])

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

        # Latest learner-side PPO metrics from get_training_metrics()
        # learner 侧最新 PPO 指标
        self.latest_training_metrics = {}

    # ============================================================
    # Safe utils
    # ============================================================

    @staticmethod
    def _safe_float(value):
        """Convert a reward/monitor value to float safely."""
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
    def _safe_ratio(cls, cur, max_v):
        max_v = max(cls._safe_float(max_v), 1.0)
        return cls._safe_float(cur) / max_v

    @classmethod
    def _flatten_training_metrics(cls, metrics):
        """Flatten nested training metrics dict from learner."""
        flat = {}
        if not isinstance(metrics, dict):
            return flat

        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[sub_key] = cls._safe_float(sub_value)
            else:
                flat[key] = cls._safe_float(value)

        return flat

    # ============================================================
    # Monitor key helpers
    # ============================================================

    @staticmethod
    def _get_d401_monitor_keys():
        """Old D401 reward keys. Keep them for old monitor panels."""
        return OLD_D401_KEYS

    @staticmethod
    def _get_new_monitor_keys():
        """New monitor keys. Only append; do not remove old D401 keys."""
        return NEW_MONITOR_KEYS

    @staticmethod
    def _get_all_monitor_keys():
        """Old D401 keys + new monitor keys, deduplicated."""
        return MONITOR_KEYS

    @staticmethod
    def _new_monitor_acc():
        """Create per-side episode monitor accumulator."""
        return {
            "sum": {key: 0.0 for key in MONITOR_KEYS},
            "cnt": {key: 0 for key in MONITOR_KEYS},
        }

    @classmethod
    def _accumulate_monitor_items(cls, acc, data):
        """
        Accumulate old and new reward/monitor items over one episode.

        reward_process.py / agent.py 可以通过 reward dict 额外放入：
        - reward 分项
        - 行为诊断项
        - action rate
        - target rate
        """
        if not isinstance(data, dict):
            return

        for key in MONITOR_KEYS:
            if key in data:
                acc["sum"][key] += cls._safe_float(data.get(key, 0.0))
                acc["cnt"][key] += 1

    @classmethod
    def _finalize_monitor_items(cls, acc):
        """Finalize accumulated monitor items."""
        out = {}

        for key in MONITOR_KEYS:
            if key in AVG_MONITOR_KEYS:
                cnt = max(1, acc["cnt"].get(key, 0))
                out[key] = acc["sum"].get(key, 0.0) / cnt
            else:
                out[key] = acc["sum"].get(key, 0.0)

        # If reward_process.py did not directly emit grouped rewards,
        # build them from item sums.
        for group_key, item_keys in REWARD_GROUPS_FOR_MONITOR.items():
            if abs(out.get(group_key, 0.0)) < 1e-12:
                out[group_key] = sum(out.get(item_key, 0.0) for item_key in item_keys)

        return out

    # ============================================================
    # Environment monitor extraction
    # ============================================================

    def _extract_env_monitor_items(self, frame_state, monitor_side=0, opponent_type=0.0):
        """
        Extract low-cost environment metrics for panels.

        注意：
        这些指标来自当前 frame_state，只做兜底环境指标。
        reward_process.py 真实产出的同名 key 如果存在，会通过 reward dict 累计。
        """
        data = {}

        if not isinstance(frame_state, dict):
            return data

        try:
            hero_states = frame_state.get("hero_states", []) or []
            npc_states = frame_state.get("npc_states", []) or []
            cakes = frame_state.get("cakes", []) or []

            main_hero = None
            enemy_hero = None
            main_camp = None

            # 对于 observation[str(i)]，第一个 hero 通常就是该侧视角下的己方英雄。
            # 这里保持低成本兜底；更精确的指标应由 reward_process.py 输出。
            for hero in hero_states:
                if not isinstance(hero, dict):
                    continue

                if main_hero is None:
                    main_hero = hero
                    main_camp = hero.get("camp")
                elif hero.get("camp") != main_camp:
                    enemy_hero = hero
                    break

            if main_hero is not None:
                data["my_hp"] = self._safe_ratio(main_hero.get("hp", 0), main_hero.get("max_hp", 1))
                data["kill_count"] = self._safe_float(main_hero.get("kill_cnt", 0))
                data["death_count"] = self._safe_float(main_hero.get("dead_cnt", 0))
            else:
                data["my_hp"] = 0.0
                data["kill_count"] = 0.0
                data["death_count"] = 0.0

            if enemy_hero is not None:
                data["enemy_hp"] = self._safe_ratio(enemy_hero.get("hp", 0), enemy_hero.get("max_hp", 1))
            else:
                data["enemy_hp"] = 0.0

            own_tower = None
            enemy_tower = None
            friendly_soldier_count = 0
            enemy_soldier_count = 0
            neutral_count = 0
            enemy_soldier_near_own_tower_count = 0
            own_tower_target_enemy_soldier_count = 0

            enemy_soldier_runtime_ids = set()

            # First pass: identify soldiers and towers.
            for npc in npc_states:
                if not isinstance(npc, dict):
                    continue

                config_id = int(self._safe_float(npc.get("config_id", npc.get("configId", -1))))
                sub_type = int(self._safe_float(npc.get("sub_type", -1)))
                camp = npc.get("camp")
                hp = self._safe_float(npc.get("hp", 0.0))

                if hp <= 0:
                    continue

                # Soldiers: sub_type=11
                if sub_type == 11:
                    runtime_id = npc.get("runtime_id", None)
                    if main_camp is not None and camp == main_camp:
                        friendly_soldier_count += 1
                    else:
                        enemy_soldier_count += 1
                        if runtime_id is not None:
                            enemy_soldier_runtime_ids.add(runtime_id)

                # Normal towers: sub_type=21
                elif sub_type == 21:
                    if main_camp is not None and camp == main_camp:
                        own_tower = npc
                    else:
                        enemy_tower = npc

                # Neutral monster / river resource: config_id=6827
                if config_id == 6827:
                    neutral_count += 1

            if own_tower is not None:
                data["own_tower_hp_ratio"] = self._safe_ratio(
                    own_tower.get("hp", 0),
                    own_tower.get("max_hp", 1),
                )
                own_attack_target = own_tower.get("attack_target", 0)
                if own_attack_target in enemy_soldier_runtime_ids:
                    own_tower_target_enemy_soldier_count = 1
            else:
                data["own_tower_hp_ratio"] = 0.0

            if enemy_tower is not None:
                data["enemy_tower_hp_ratio"] = self._safe_ratio(
                    enemy_tower.get("hp", 0),
                    enemy_tower.get("max_hp", 1),
                )
            else:
                data["enemy_tower_hp_ratio"] = 0.0

            # Simple defense pressure proxy.
            if own_tower is not None:
                own_tower_loc = own_tower.get("location", {}) or {}
                own_tower_x = self._safe_float(own_tower_loc.get("x", 0.0))
                own_tower_z = self._safe_float(own_tower_loc.get("z", 0.0))
                own_tower_range = max(self._safe_float(own_tower.get("attack_range", 0.0)), 1.0)

                for npc in npc_states:
                    if not isinstance(npc, dict):
                        continue
                    if int(self._safe_float(npc.get("sub_type", -1))) != 11:
                        continue
                    if self._safe_float(npc.get("hp", 0.0)) <= 0:
                        continue
                    if main_camp is not None and npc.get("camp") == main_camp:
                        continue

                    loc = npc.get("location", {}) or {}
                    dx = self._safe_float(loc.get("x", 0.0)) - own_tower_x
                    dz = self._safe_float(loc.get("z", 0.0)) - own_tower_z
                    if dx * dx + dz * dz <= own_tower_range * own_tower_range:
                        enemy_soldier_near_own_tower_count += 1

            data["friendly_soldier_count"] = float(friendly_soldier_count)
            data["enemy_soldier_count"] = float(enemy_soldier_count)
            data["enemy_soldier_near_own_tower_count"] = float(enemy_soldier_near_own_tower_count)
            data["own_tower_target_enemy_soldier_count"] = float(own_tower_target_enemy_soldier_count)
            data["cake_count"] = float(len(cakes))
            data["neutral_count"] = float(neutral_count)
            data["opponent_type"] = self._safe_float(opponent_type)

        except Exception as e:
            if self.logger:
                self.logger.warning(f"extract env monitor failed: {e}")

        return data

    # ============================================================
    # Env init config
    # ============================================================

    def _call_init_config(self, usr_conf):
        """
        Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.

        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
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

    # ============================================================
    # Main episode loop
    # ============================================================

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                self.latest_training_metrics = self._flatten_training_metrics(training_metrics)

                for key, value in training_metrics.items():
                    if key == "env" and isinstance(value, dict):
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置，可以用长度为2的列表传入当前对局的阵容id
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
            latest_env_monitor_list = [dict() for _ in range(self.agent_num)]

            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            opponent_agent = self.env_conf_manager.get_opponent_agent()
            if opponent_agent == "common_ai":
                opponent_type = 1.0
            elif opponent_agent == "selfplay":
                opponent_type = 2.0
            else:
                opponent_type = 3.0

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                frame_state = observation[str(i)]["frame_state"]
                latest_env_monitor_list[i] = self._extract_env_monitor_items(
                    frame_state,
                    monitor_side=i,
                    opponent_type=opponent_type,
                )

                if do_sample:
                    reward = agent.reward_manager.result(frame_state)
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += self._safe_float(reward.get("reward_sum", 0.0))
                    self._accumulate_monitor_items(monitor_acc_list[i], reward)

            while True:
                # Initialize default actions.
                # If the agent does not make a decision, env.step uses the default action.
                # 初始化默认 actions，如果智能体不进行决策，则 env.step 使用默认 action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])

                        # Only sample when do_sample=True and is_eval=False.
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
                # 计算回报，作为当前环境状态 observation 的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    frame_state = observation[str(i)]["frame_state"]
                    latest_env_monitor_list[i] = self._extract_env_monitor_items(
                        frame_state,
                        monitor_side=i,
                        opponent_type=opponent_type,
                    )

                    if do_sample:
                        reward = agent.reward_manager.result(frame_state)
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += self._safe_float(reward.get("reward_sum", 0.0))
                        self._accumulate_monitor_items(monitor_acc_list[i], reward)

                # Normal end or timeout exit, run train_test will exit early.
                # 正常结束或超时退出，运行 train_test 时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, "
                        f"truncated:{truncated}, eval:{is_eval}, "
                        f"reward_sum:{reward_sum_list[monitor_side]}"
                    )

                    # Reward for saving the last state of the environment.
                    # 保存环境最后状态的 reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            final_reward = 0.0
                            if isinstance(observation.get(str(i), {}).get("reward", None), dict):
                                final_reward = observation[str(i)]["reward"].get("reward_sum", 0.0)

                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=final_reward,
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = self._finalize_monitor_items(
                            monitor_acc_list[monitor_side]
                        )

                        # Use latest env metrics as final-state environment snapshot.
                        # 环境指标使用最后一帧快照，避免 kill_count / tower_hp 等被逐帧累加。
                        monitor_data.update(latest_env_monitor_list[monitor_side])

                        monitor_data.update({
                            "episode_cnt": int(self.episode_cnt),
                            "frame_no": int(frame_no),
                            "reward": round(self._safe_float(reward_sum_list[monitor_side]), 4),
                            # If there is no explicit win flag, use reward as a fallback.
                            # 如果环境没有显式 win 字段，先用 reward 正负兜底。
                            "win": 1.0 if self._safe_float(reward_sum_list[monitor_side]) > 0 else 0.0,
                            "opponent_type": opponent_type,
                        })

                        # Merge learner-side PPO metrics if available.
                        # learner 指标覆盖同名 key。
                        for key in MONITOR_KEYS:
                            if key in self.latest_training_metrics:
                                monitor_data[key] = self.latest_training_metrics[key]

                        # Ensure every key registered in monitor_builder.py has a value.
                        # 确保所有 key 都有值，避免面板空曲线。
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

    # ============================================================
    # Agent reset / opponent loading
    # ============================================================

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

        # Load model according to the configuration.
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor.
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model.
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict.
                    # 如果对手是 common_ai 则不需要加载模型，也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False

                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool.
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型
                    agent.load_model(id="latest")

                else:
                    # Opponent model, model_id is checked from kaiwu.json.
                    # 选择 kaiwu.json 中设置的对手模型。
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(
                            f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}"
                        )

                    if is_train_test:
                        # Run train_test, cannot get opponent agent, so replace with latest model.
                        # 运行 train_test 时，无法获取到对手模型，因此将替换为最新模型
                        self.logger.info(
                            "Run train_test, cannot get opponent agent, so replace with latest model"
                        )
                        agent.load_model(id="latest")
                    else:
                        agent.load_opponent_agent(id=opponent_agent)

                    self.do_samples[i] = False

            # Reset agent.
            # 重置 agent
            agent.reset(observation[str(i)])