#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor builder for 1v1 PPO.
- Legal Chinese names: only Chinese/English/digits/_-/space, length <= 20.
- Related metrics are placed in the same line panel for easier comparison.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _precision(metric):
    if metric == "learning_rate":
        return "0.00000001"
    if metric.endswith("_rate") or metric.endswith("_ratio") or metric in {"win", "approx_kl", "clip_fraction"}:
        return "0.0001"
    return "0.01"


def _add_multi_metric_panels(builder, panels):
    """Add panels. Each panel can contain one or more line metrics."""
    for panel_name, panel_name_en, metrics in panels:
        builder = builder.add_panel(
            name=panel_name,
            name_en=panel_name_en,
            type="line",
        )
        for metric in metrics:
            builder = builder.add_metric(
                metrics_name=metric,
                expr=f"round(avg({metric}{{}}), {_precision(metric)})",
            )
        builder = builder.end_panel()
    return builder


def build_monitor():
    monitor = MonitorConfigBuilder()

    environment_panels = [
        ("塔血对比", "tower_compare", ["own_tower_hp_ratio", "enemy_tower_hp_ratio"]),
        ("英雄血量", "hero_hp", ["my_hp", "enemy_hp"]),
        ("击杀死亡", "kill_death", ["kill_count", "death_count"]),
        ("胜率", "win", ["win"]),
        ("总奖励", "reward", ["reward"]),
        ("对局信息", "episode_info", ["episode_cnt", "frame_no"]),
    ]

    ppo_panels = [
        ("损失对比", "loss_compare", ["total_loss", "value_loss", "policy_loss"]),
        ("PPO稳定", "ppo_stable", ["approx_kl", "clip_fraction"]),
        ("熵损失", "entropy_loss", ["entropy_loss"]),
        ("学习率", "learning_rate", ["learning_rate"]),
        ("训练健康", "train_health", ["hidden_norm", "feature_nan_count"]),
    ]

    reward_panels = [
        ("奖励三组", "reward_groups", ["reward_objective", "reward_growth_combat", "reward_behavior_safety"]),
        ("目标奖励", "objective_reward", ["tower_hp_point", "kill", "death"]),
        ("发育奖励", "growth_reward", ["money", "exp", "last_hit"]),
        ("血量伤害", "combat_reward", ["hp_point", "hero_hurt", "hero_damage"]),
        ("清线防守", "lane_defense", ["lane_clear", "defense", "cake"]),
        ("风险行为", "risk_behavior", ["tower_risk", "stuck", "no_ops", "grass_behavior"]),
        ("技能奖励", "skill_reward", ["skill_hit", "total_damage"]),
    ]

    behavior_panels = [
        ("兵线数量", "soldier_count", ["enemy_soldier_count", "friendly_soldier_count"]),
        ("目标选择", "target_select", ["target_soldier_rate", "target_enemy_rate", "target_tower_rate", "target_monster_rate"]),
        ("基础动作", "basic_action", ["button_move_rate", "button_attack_rate", "button_none_rate"]),
        ("技能动作", "skill_action", ["button_skill1_rate", "button_skill2_rate", "button_skill3_rate"]),
        ("技能目标", "skill_target", ["skill_target_enemy_rate", "skill_target_soldier_rate", "skill_target_tower_rate", "skill_center_rate"]),
        ("异常行为", "bad_behavior", ["stuck_count", "grass_long_stay_count", "grass_no_effective_count", "unsafe_tower_entry_count"]),
        ("防守行为", "defense_behavior", ["defense_emergency_count", "enemy_soldier_near_own_tower_count"]),
        ("血包行为", "cake_behavior", ["own_cake_pick_count", "low_hp_own_cake_approach_count"]),
    ]

    builder = monitor.title("智能决策1v1")
    builder = builder.add_group(group_name="环境指标", group_name_en="environment")
    builder = _add_multi_metric_panels(builder, environment_panels).end_group()

    builder = builder.add_group(group_name="PPO指标", group_name_en="ppo")
    builder = _add_multi_metric_panels(builder, ppo_panels).end_group()

    builder = builder.add_group(group_name="奖励分组", group_name_en="reward")
    builder = _add_multi_metric_panels(builder, reward_panels).end_group()

    builder = builder.add_group(group_name="行为诊断", group_name_en="behavior")
    builder = _add_multi_metric_panels(builder, behavior_panels).end_group()

    return builder.build()
