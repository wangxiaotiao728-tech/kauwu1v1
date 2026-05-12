#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
D401 replica monitor builder.

原则：
1. 原有 baseline 面板全部保留。
2. 原有 D401 奖励面板全部保留。
3. 新方案指标只追加。
4. 有对比价值的新指标合并在同一张 line panel。
5. 面板中文名合法：1~20 字符，仅中英文、数字、_、-、空格。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _precision(metric):
    if metric == "learning_rate":
        return "0.00000001"

    if metric in {
        "win",
        "timeout_rate",
        "approx_kl",
        "clip_fraction",
        "entropy_beta",
        "ppo_clip",
    }:
        return "0.0001"

    if (
        metric.endswith("_rate")
        or metric.endswith("_ratio")
        or metric.endswith("_norm")
        or metric.endswith("_fraction")
    ):
        return "0.0001"

    return "0.01"


def _add_metric(builder, metric):
    return builder.add_metric(
        metrics_name=metric,
        expr=f"round(avg({metric}{{}}), {_precision(metric)})",
    )


def _add_multi_metric_panels(builder, panels):
    """
    panels:
        [
            ("中文面板名", "english_panel_name", ["metric_a", "metric_b"]),
            ...
        ]
    """
    for panel_name, panel_name_en, metrics in panels:
        builder = builder.add_panel(
            name=panel_name,
            name_en=panel_name_en,
            type="line",
        )

        for metric in metrics:
            builder = _add_metric(builder, metric)

        builder = builder.end_panel()

    return builder


def build_monitor():
    """
    创建监控面板。

    原有面板全部保留；
    新增面板只追加；
    有对比意义的新指标合并到同一张曲线图。
    """
    monitor = MonitorConfigBuilder()

    # ============================================================
    # 原有算法指标：保留，不改名
    # ============================================================
    builder = (
        monitor.title("智能决策1v1")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="round(avg(reward{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="round(avg(total_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="round(avg(value_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="round(avg(policy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="round(avg(entropy_loss{}), 0.01)",
        )
        .end_panel()
        .end_group()
    )

    # ============================================================
    # 原有 D401 奖励：保留，不改名
    # ============================================================
    builder = (
        builder
        .add_group(
            group_name="D401奖励",
            group_name_en="d401_reward",
        )
        .add_panel(
            name="承伤",
            name_en="hero_hurt",
            type="line",
        )
        .add_metric(
            metrics_name="hero_hurt",
            expr="round(avg(hero_hurt{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="总伤",
            name_en="total_damage",
            type="line",
        )
        .add_metric(
            metrics_name="total_damage",
            expr="round(avg(total_damage{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="英伤",
            name_en="hero_damage",
            type="line",
        )
        .add_metric(
            metrics_name="hero_damage",
            expr="round(avg(hero_damage{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="暴击",
            name_en="crit",
            type="line",
        )
        .add_metric(
            metrics_name="crit",
            expr="round(avg(crit{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="技命中",
            name_en="skill_hit",
            type="line",
        )
        .add_metric(
            metrics_name="skill_hit",
            expr="round(avg(skill_hit{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="无操作",
            name_en="no_ops",
            type="line",
        )
        .add_metric(
            metrics_name="no_ops",
            expr="round(avg(no_ops{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="草丛",
            name_en="in_grass",
            type="line",
        )
        .add_metric(
            metrics_name="in_grass",
            expr="round(avg(in_grass{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="塔下",
            name_en="under_tower_behavior",
            type="line",
        )
        .add_metric(
            metrics_name="under_tower_behavior",
            expr="round(avg(under_tower_behavior{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="被动",
            name_en="passive_skills",
            type="line",
        )
        .add_metric(
            metrics_name="passive_skills",
            expr="round(avg(passive_skills{}), 0.01)",
        )
        .end_panel()
        .end_group()
    )

    # ============================================================
    # 新增训练指标：只追加
    # ============================================================
    new_train_panels = [
        ("PPO稳定", "ppo_stable", [
            "approx_kl",
            "clip_fraction",
        ]),
        ("学习动态", "learn_dynamic", [
            "learning_rate",
            "entropy_beta",
            "ppo_clip",
        ]),
        ("训练健康", "train_health", [
            "grad_norm",
            "hidden_norm",
            "feature_nan_count",
        ]),
    ]

    builder = builder.add_group(
        group_name="新增训练",
        group_name_en="new_train",
    )
    builder = _add_multi_metric_panels(builder, new_train_panels).end_group()

    # ============================================================
    # 新增环境指标：只追加
    # ============================================================
    new_env_panels = [
        ("塔血对比", "tower_compare", [
            "own_tower_hp_ratio",
            "enemy_tower_hp_ratio",
        ]),
        ("英雄血量", "hero_hp", [
            "my_hp",
            "enemy_hp",
        ]),
        ("击杀死亡", "kill_death", [
            "kill_count",
            "death_count",
        ]),
        ("兵线数量", "soldier_count", [
            "enemy_soldier_count",
            "friendly_soldier_count",
        ]),
        ("资源数量", "resource_count", [
            "cake_count",
            "neutral_count",
        ]),
        ("对局信息", "episode_info", [
            "episode_cnt",
            "frame_no",
            "opponent_type",
        ]),
    ]

    builder = builder.add_group(
        group_name="新增环境",
        group_name_en="new_env",
    )
    builder = _add_multi_metric_panels(builder, new_env_panels).end_group()

    # ============================================================
    # 新增三组奖励：只追加
    # ============================================================
    new_reward_panels = [
        ("奖励三组", "reward_groups", [
            "reward_objective",
            "reward_growth_combat",
            "reward_behavior_safety",
        ]),
        ("目标奖励", "objective_reward", [
            "tower_hp_point",
            "kill",
            "death",
        ]),
        ("发育奖励", "growth_reward", [
            "money",
            "exp",
            "last_hit",
        ]),
        ("血量伤害", "combat_reward", [
            "hp_point",
            "hero_hurt",
            "hero_damage",
            "total_damage",
        ]),
        ("清线防守", "lane_defense", [
            "lane_clear",
            "defense",
            "cake",
        ]),
        ("风险行为", "risk_behavior", [
            "tower_risk",
            "stuck",
            "no_ops",
            "grass_behavior",
        ]),
    ]

    builder = builder.add_group(
        group_name="新增奖励",
        group_name_en="new_reward",
    )
    builder = _add_multi_metric_panels(builder, new_reward_panels).end_group()

    # ============================================================
    # 新增行为诊断：只追加
    # ============================================================
    new_behavior_panels = [
        ("目标选择", "target_select", [
            "target_soldier_rate",
            "target_enemy_rate",
            "target_tower_rate",
            "target_monster_rate",
        ]),
        ("基础动作", "basic_action", [
            "button_move_rate",
            "button_attack_rate",
            "button_none_rate",
        ]),
        ("技能动作", "skill_action", [
            "button_skill1_rate",
            "button_skill2_rate",
            "button_skill3_rate",
        ]),
        ("技能目标", "skill_target", [
            "skill_target_enemy_rate",
            "skill_target_soldier_rate",
            "skill_target_tower_rate",
            "skill_center_rate",
        ]),
        ("异常行为", "bad_behavior", [
            "stuck_count",
            "grass_long_stay_count",
            "grass_no_effective_count",
            "unsafe_tower_entry_count",
        ]),
        ("防守行为", "defense_behavior", [
            "defense_emergency_count",
            "enemy_soldier_near_own_tower_count",
            "own_tower_target_enemy_soldier_count",
        ]),
        ("血包行为", "cake_behavior", [
            "own_cake_pick_count",
            "low_hp_own_cake_approach_count",
        ]),
    ]

    builder = builder.add_group(
        group_name="新增行为",
        group_name_en="new_behavior",
    )
    builder = _add_multi_metric_panels(builder, new_behavior_panels).end_group()

    return builder.build()