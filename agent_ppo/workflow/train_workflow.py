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
import random
import math
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import CurriculumConfig, GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def cosine_schedule(start, end, progress):
    # progress 从 0 到 1，返回 start 到 end 的平滑调度值。
    progress = max(0.0, min(1.0, progress))
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))


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
        self.train_episode_cnt = 0
        self.last_report_monitor_time = 0
        self.current_opponent_agent = None
        self.common_ai_history = []

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        # camp_keys[i] is the camp key for agents[i] based on monitor_side
        # monitor_side 的 agent 对应 blue/red 取决于 monitor_side 配置
        monitor_side = self.env_conf_manager.get_monitor_side()
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
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)
            self.current_opponent_agent = self._select_opponent_agent(usr_conf, is_eval)
            self._set_usr_conf_opponent_agent(usr_conf, self.current_opponent_agent)
            self._apply_stage_hparams()

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
            extra_info = env_obs["extra_info"]

            # Reset agents
            # 重置智能体
            self.reset_agents(observation)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            if not is_eval:
                self.train_episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            reward_channel_sum_list = [dict() for _ in range(self.agent_num)]
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(
                f"Episode {self.episode_cnt} start, opponent_agent={self.current_opponent_agent}, "
                f"train_episode_cnt={self.train_episode_cnt}, usr_conf is {usr_conf}"
            )

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(
                        observation[str(i)]["frame_state"],
                        observation=observation[str(i)],
                        terminated=False,
                        truncated=False,
                    )
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]
                    self._accumulate_reward_channels(reward_channel_sum_list[i], reward)

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
                extra_info = env_obs["extra_info"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(
                            observation[str(i)]["frame_state"],
                            observation=observation[str(i)],
                            terminated=terminated,
                            truncated=truncated,
                        )
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        self._accumulate_reward_channels(reward_channel_sum_list[i], reward)

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
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        if self.monitor:
                            monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                            monitor_data.update(
                                self._env_monitor_metrics(
                                    observation=observation,
                                    monitor_side=monitor_side,
                                    frame_no=frame_no,
                                    truncated=truncated,
                                )
                            )
                            monitor_data.update(
                                {
                                    key: round(value, 4)
                                    for key, value in reward_channel_sum_list[monitor_side].items()
                                }
                            )
                            monitor_data.update(self._agent_monitor_metrics(self.agents[monitor_side]))
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    if not is_eval:
                        self._update_curriculum_history(observation, monitor_side)

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break

    def reset_agents(self, observation):
        opponent_agent = self.current_opponent_agent or self.env_conf_manager.get_opponent_agent()
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
                agent.load_model(id="latest", load_optimizer=self._should_load_optimizer())
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
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[str(i)])

    def _select_opponent_agent(self, usr_conf, is_eval):
        if is_eval and CurriculumConfig.EVAL_ALWAYS_COMMON_AI:
            return "common_ai"
        stage = self._stage()
        stage_opponent = stage.get("opponent")
        if stage_opponent == "common_ai":
            return "common_ai"
        if stage_opponent in ("common_ai_plus_short_selfplay", "selfplay_history_pool_common_ai_eval"):
            return self._select_stage_mixed_opponent(stage_opponent)
        base_opponent = self._get_usr_conf_opponent_agent(usr_conf) or self.env_conf_manager.get_opponent_agent()
        if is_eval or not CurriculumConfig.ENABLE_CURRICULUM:
            return base_opponent
        if self.train_episode_cnt < CurriculumConfig.COMMON_AI_WARMUP_EPISODES:
            return "common_ai"
        selfplay_prob = self._curriculum_selfplay_prob()
        return "selfplay" if random.random() < selfplay_prob else "common_ai"

    def _select_stage_mixed_opponent(self, stage_opponent):
        if stage_opponent == "common_ai_plus_short_selfplay":
            return "selfplay" if random.random() < 0.15 else "common_ai"
        roll = random.random()
        if roll < 0.20:
            return "common_ai"
        return "selfplay"

    def _stage(self):
        stage_name = os.environ.get("HOK_CURRICULUM_STAGE", CurriculumConfig.CURRENT_STAGE)
        return CurriculumConfig.CURRICULUM_STAGES.get(stage_name, CurriculumConfig.CURRICULUM_STAGES["S1_BASIC"])

    def _apply_stage_hparams(self):
        stage = self._stage()
        progress = min(1.0, self.train_episode_cnt / max(1, CurriculumConfig.SELFPLAY_RAMP_EPISODES))
        lr = cosine_schedule(stage.get("lr_start", 3e-4), stage.get("lr_end", 8e-5), progress)
        entropy_coef = cosine_schedule(stage.get("entropy_start", 0.02), stage.get("entropy_end", 0.005), progress)
        clip_eps = cosine_schedule(stage.get("clip_start", 0.20), stage.get("clip_end", 0.12), progress)
        for agent in self.agents:
            if hasattr(agent, "optimizer"):
                for group in agent.optimizer.param_groups:
                    group["lr"] = lr
            if hasattr(agent, "model"):
                agent.model.var_beta = entropy_coef
                agent.model.clip_param = clip_eps
                agent.model.ppo_epoch = int(stage.get("ppo_epoch", 1))

    def _should_load_optimizer(self):
        # 同阶段中断恢复可设置 HOK_LOAD_OPTIMIZER=1；跨阶段课程切换保持默认 False。
        return os.environ.get("HOK_LOAD_OPTIMIZER", "0").lower() in ("1", "true", "yes")

    def _accumulate_reward_channels(self, target, reward):
        for key in (
            "terminal",
            "tower",
            "tower_defense",
            "lane",
            "growth",
            "last_hit",
            "enhanced_tower",
            "resource",
            "cake",
            "skill",
            "death",
            "tower_risk",
        ):
            target[key] = target.get(key, 0.0) + float(reward.get(key, 0.0))

    def _agent_monitor_metrics(self, agent):
        hidden = getattr(agent, "lstm_hidden", None)
        hidden_norm = 0.0 if hidden is None else float(sum(float(x) * float(x) for x in hidden) ** 0.5)
        rule_controller = getattr(agent, "rule_controller", None)
        feature_process = getattr(agent, "feature_processes", None)
        optimizer = getattr(agent, "optimizer", None)
        learning_rate = 0.0
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            learning_rate = float(optimizer.param_groups[0].get("lr", 0.0) or 0.0)
        return {
            "hard_mask_rate": 0.0 if rule_controller is None else round(rule_controller.hard_mask_rate, 4),
            "rule_bias_count": 0 if rule_controller is None else rule_controller.rule_bias_count,
            "mask_fallback_count": 0 if rule_controller is None else rule_controller.mask_fallback_count,
            "feature_nan_count": 0 if feature_process is None else int(getattr(feature_process, "last_feature_nan_count", 0)),
            "hidden_norm": round(hidden_norm, 4),
            "learning_rate": round(learning_rate, 8),
        }

    def _env_monitor_metrics(self, observation, monitor_side, frame_no, truncated):
        obs = observation.get(str(monitor_side), {}) if isinstance(observation, dict) else {}
        frame_state = obs.get("frame_state", {}) or {}
        units = self._extract_monitor_units(frame_state, obs)
        my_hero = units["my_hero"]
        enemy_hero = units["enemy_hero"]
        own_tower = units["own_tower"]
        enemy_tower = units["enemy_tower"]
        return {
            "frame_no": int(frame_no or 0),
            "win": 1.0 if obs.get("win", 0) else 0.0,
            "timeout_rate": 1.0 if truncated else 0.0,
            "my_hp": round(self._hp_value(my_hero, default=0), 2),
            "enemy_hp": round(self._hp_value(enemy_hero, default=0), 2),
            "own_tower_hp": round(self._hp_value(own_tower, default=0), 2),
            "enemy_tower_hp": round(self._hp_value(enemy_tower, default=0), 2),
            "own_tower_hp_ratio": round(self._hp_ratio(own_tower), 4),
            "enemy_tower_hp_ratio": round(self._hp_ratio(enemy_tower), 4),
            "kill_count": self._unit_counter(my_hero, ("kill_cnt", "kill_count", "kill_num")),
            "death_count": self._unit_counter(my_hero, ("dead_cnt", "death_count", "dead_num")),
            "cake_count": len(frame_state.get("cake_states", []) or frame_state.get("cakes", []) or []),
            "neutral_count": units["neutral_count"],
            "opponent_type": self._opponent_type_value(),
        }

    def _curriculum_selfplay_prob(self):
        if self._common_ai_gate_passed():
            ramp_progress = max(0, self.train_episode_cnt - CurriculumConfig.COMMON_AI_WARMUP_EPISODES)
            ramp_ratio = min(1.0, ramp_progress / max(1, CurriculumConfig.SELFPLAY_RAMP_EPISODES))
            return CurriculumConfig.SELFPLAY_PROB_START + (
                CurriculumConfig.SELFPLAY_PROB_MAX - CurriculumConfig.SELFPLAY_PROB_START
            ) * ramp_ratio
        if self.train_episode_cnt >= CurriculumConfig.FORCE_SELFPLAY_AFTER_EPISODES:
            return CurriculumConfig.FALLBACK_SELFPLAY_PROB
        return 0.0

    def _common_ai_gate_passed(self):
        window = CurriculumConfig.COMMON_AI_GATE_WINDOW
        if len(self.common_ai_history) < window:
            return False
        recent = self.common_ai_history[-window:]
        avg_enemy_tower_hp = sum(item["enemy_tower_hp"] for item in recent) / window
        avg_death = sum(item["death"] for item in recent) / window
        return (
            avg_enemy_tower_hp <= CurriculumConfig.COMMON_AI_GATE_ENEMY_TOWER_HP
            and avg_death <= CurriculumConfig.COMMON_AI_GATE_DEATH
        )

    def _update_curriculum_history(self, observation, monitor_side):
        if self.current_opponent_agent != "common_ai":
            return
        obs = observation.get(str(monitor_side), {}) if isinstance(observation, dict) else {}
        frame_state = obs.get("frame_state", {}) or {}
        my_hero, enemy_tower = self._extract_monitor_hero_and_enemy_tower(frame_state, obs)
        if my_hero is None:
            return
        self.common_ai_history.append(
            {
                "enemy_tower_hp": self._hp_value(enemy_tower, default=12000),
                "death": my_hero.get("dead_cnt", 0),
                "win": obs.get("win", 0),
            }
        )
        max_history = max(CurriculumConfig.COMMON_AI_GATE_WINDOW * 5, 100)
        if len(self.common_ai_history) > max_history:
            self.common_ai_history = self.common_ai_history[-max_history:]

    def _extract_monitor_hero_and_enemy_tower(self, frame_state, obs):
        units = self._extract_monitor_units(frame_state, obs)
        return units["my_hero"], units["enemy_tower"]

    def _extract_monitor_units(self, frame_state, obs):
        player_id = obs.get("player_id")
        player_camp = obs.get("player_camp", obs.get("camp"))
        my_hero = None
        enemy_hero = None
        for hero in frame_state.get("hero_states", []) or []:
            if hero.get("runtime_id") == player_id:
                my_hero = hero
                break
        if my_hero is None:
            for hero in frame_state.get("hero_states", []) or []:
                if self._same_camp(hero.get("camp"), player_camp):
                    my_hero = hero
                    break
        my_camp = my_hero.get("camp") if my_hero else player_camp
        for hero in frame_state.get("hero_states", []) or []:
            if not self._same_camp(hero.get("camp"), my_camp):
                enemy_hero = hero
                break
        own_tower = None
        enemy_tower = None
        neutral_count = 0
        for npc in frame_state.get("npc_states", []) or []:
            if self._is_tower(npc):
                if self._same_camp(npc.get("camp"), my_camp):
                    own_tower = own_tower or npc
                else:
                    enemy_tower = enemy_tower or npc
            elif self._is_monster(npc):
                neutral_count += 1
        return {
            "my_hero": my_hero,
            "enemy_hero": enemy_hero,
            "own_tower": own_tower,
            "enemy_tower": enemy_tower,
            "neutral_count": neutral_count,
        }

    def _get_usr_conf_opponent_agent(self, usr_conf):
        try:
            return usr_conf.get("episode", {}).get("opponent_agent")
        except AttributeError:
            return None

    def _set_usr_conf_opponent_agent(self, usr_conf, opponent_agent):
        if isinstance(usr_conf, dict):
            usr_conf.setdefault("episode", {})["opponent_agent"] = opponent_agent

    def _is_tower(self, npc):
        sub_type = npc.get("sub_type")
        return sub_type == 21 or "TOWER" in str(sub_type).upper()

    def _is_monster(self, npc):
        sub_type = str(npc.get("sub_type", "")).upper()
        npc_type = str(npc.get("type", npc.get("npc_type", ""))).upper()
        camp = self._camp_value(npc.get("camp"))
        return camp not in (1, 2) and ("MONSTER" in sub_type or "MONSTER" in npc_type or npc.get("camp") in (0, "0"))

    def _hp_value(self, obj, default=0):
        if not obj:
            return default
        return obj.get("hp", default)

    def _hp_ratio(self, obj):
        if not obj:
            return 0.0
        hp = float(obj.get("hp", 0.0) or 0.0)
        max_hp = float(obj.get("max_hp", obj.get("hp_max", hp)) or hp or 1.0)
        return hp / max(max_hp, 1.0)

    def _unit_counter(self, unit, keys):
        if not unit:
            return 0
        for key in keys:
            if key in unit:
                return int(unit.get(key) or 0)
        return 0

    def _opponent_type_value(self):
        if self.current_opponent_agent == "common_ai":
            return 0
        if self.current_opponent_agent == "selfplay":
            return 1
        return 2

    def _same_camp(self, left, right):
        return self._camp_value(left) == self._camp_value(right)

    def _camp_value(self, camp):
        if isinstance(camp, str):
            if camp.endswith("_1") or camp == "1":
                return 1
            if camp.endswith("_2") or camp == "2":
                return 2
        return camp
