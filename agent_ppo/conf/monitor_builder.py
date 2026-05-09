#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
D401 replica monitor builder.
Keep the original baseline chain style and only append D401 reward item panels.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    Create custom monitor panel configurations.

    Baseline metrics:
        reward, total_loss, value_loss, policy_loss, entropy_loss
    D401 reward item metrics:
        hero_hurt, total_damage, hero_damage, crit, skill_hit,
        no_ops, in_grass, under_tower_behavior, passive_skills
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        # ===== Baseline original panels =====
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
        # ===== D401 reward item panels =====
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
        .build()
    )
    return config_dict
