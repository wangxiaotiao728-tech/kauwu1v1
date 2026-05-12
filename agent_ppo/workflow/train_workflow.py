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
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


# Metrics used by monitor_builder.py. Missing keys are reported as 0.0 so panels do not disappear.
MONITOR_KEYS = [
    # environment
    "reward", "episode_cnt", "frame_no", "win",
    "my_hp", "enemy_hp", "own_tower_hp_ratio", "enemy_tower_hp_ratio",
    "kill_count", "death_count",
    # ppo
    "total_loss", "value_loss", "policy_loss", "entropy_loss",
    "approx_kl", "clip_fraction", "learning_rate", "hidden_norm", "feature_nan_count",
    # reward groups
    "reward_objective", "reward_growth_combat", "reward_behavior_safety",
    "tower_hp_point", "kill", "death",
    "money", "exp", "last_hit",
    "hp_point", "hero_hurt", "hero_damage",
    "lane_clear", "defense", "cake",
    "tower_risk", "stuck", "no_ops", "grass_behavior",
    "skill_hit", "total_damage",
    # behavior diagnostics
    "enemy_soldier_count", "friendly_soldier_count",
    "target_soldier_rate", "target_enemy_rate", "target_tower_rate", "target_monster_rate",
    "button_move_rate", "button_attack_rate", "button_none_rate",
    "button_skill1_rate", "button_skill2_rate", "button_skill3_rate",
    "skill_target_enemy_rate", "skill_target_soldier_rate", "skill_target_tower_rate", "skill_center_rate",
    "stuck_count", "grass_long_stay_count", "grass_no_effective_count", "unsafe_tower_entry_count",
    "defense_emergency_count", "enemy_soldier_near_own_tower_count",
    "own_cake_pick_count", "low_hp_own_cake_approach_count",
]

# These keys should be averaged over the episode if emitted per decision frame.
AVG_MONITOR_KEYS = {
    "my_hp", "enemy_hp", "own_tower_hp_ratio", "enemy_tower_hp_ratio",
    "enemy_soldier_count", "friendly_soldier_count",
    "target_soldier_rate", "target_enemy_rate", "target_tower_rate", "target_monster_rate",
    "button_move_rate", "button_attack_rate", "button_none_rate",
    "button_skill1_rate", "button_skill2_rate", "button_skill3_rate",
    "skill_target_enemy_rate", "skill_target_soldier_rate", "skill_target_tower_rate", "skill_center_rate",
    "hidden_norm", "feature_nan_count",
}

REWARD_GROUPS_FOR_MONITOR = {
    "reward_objective": ["tower_hp_point", "kill", "death"],
    "reward_growth_combat": [
        "hp_point", "money", "exp", "last_hit", "hero_hurt",
        "total_damage", "hero_damage", "skill_hit",
    ],
    "reward_behavior_safety": [
        "lane_clear", "defense", "cake", "tower_risk", "stuck",
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
        self.latest_training_metrics = {}

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

    def _extract_env_monitor_items(self, frame_state, monitor_side):
        """Extract low-cost environment metrics for panels."""
        data = {}
        try:
            hero_states = frame_state.get("hero_states", [])
            npc_states = frame_state.get("npc_states", [])
            main_camp = None
            my_hero = None
            enemy_hero = None
            for hero in hero_states:
                # reward manager main camp is based on agent; monitor_side observation is already local.
                # Here use the first matching runtime from observation perspective when possible.
                if my_hero is None:
                    my_hero = hero
                    main_camp = hero.get("camp", None)
                elif hero.get("camp") != main_camp:
                    enemy_hero = hero
            if my_hero:
                data["my_hp"] = self._safe_float(my_hero.get("hp", 0)) / max(1.0, self._safe_float(my_hero.get("max_hp", 1)))
                data["kill_count"] = self._safe_float(my_hero.get("kill_cnt", 0))
                data["death_count"] = self._safe_float(my_hero.get("dead_cnt", 0))
            if enemy_hero:
                data["enemy_hp"] = self._safe_float(enemy_hero.get("hp", 0)) / max(1.0, self._safe_float(enemy_hero.get("max_hp", 1)))
            own_tower = None
            enemy_tower = None
            friendly_soldiers = 0
            enemy_soldiers = 0
            for npc in npc_states:
                sub_type = int(npc.get("sub_type", -1))
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
                    else:
                        enemy_soldiers += 1
            if own_tower:
                data["own_tower_hp_ratio"] = self._safe_float(own_tower.get("hp", 0)) / max(1.0, self._safe_float(own_tower.get("max_hp", 1)))
            if enemy_tower:
                data["enemy_tower_hp_ratio"] = self._safe_float(enemy_tower.get("hp", 0)) / max(1.0, self._safe_float(enemy_tower.get("max_hp", 1)))
            data["friendly_soldier_count"] = friendly_soldiers
            data["enemy_soldier_count"] = enemy_soldiers
        except Exception:
            pass
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
                for key, value in training_metrics.items():
                    if key == "env" and isinstance(value, dict):
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

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
                    self._accumulate_monitor_items(monitor_acc_list[i], self._extract_env_monitor_items(frame_state, i))

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
                        self._accumulate_monitor_items(monitor_acc_list[i], self._extract_env_monitor_items(frame_state, i))

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

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = self._finalize_monitor_items(monitor_acc_list[monitor_side])
                        monitor_data.update({
                            "episode_cnt": self.episode_cnt,
                            "frame_no": frame_no,
                            "reward": round(reward_sum_list[monitor_side], 2),
                            "win": 1.0 if self._safe_float(reward_sum_list[monitor_side]) > 0 else 0.0,
                        })
                        # Merge learner-side PPO metrics if available.
                        for key in MONITOR_KEYS:
                            if key in self.latest_training_metrics:
                                monitor_data[key] = self.latest_training_metrics[key]

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
