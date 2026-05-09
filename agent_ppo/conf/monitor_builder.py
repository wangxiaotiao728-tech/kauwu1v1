#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_line_panels(builder, panels):
    for panel in panels:
        if len(panel) == 3:
            name, metric, precision = panel
        else:
            name, metric = panel
            precision = "0.01"
        builder = (
            builder.add_panel(name=name, name_en=metric, type="line")
            .add_metric(metrics_name=metric, expr=f"round(avg({metric}{{}}), {precision})")
            .end_panel()
        )
    return builder


def build_monitor():
    """
    创建自定义监控面板。

    环境指标：还原原面板中用于观察对局进程和环境状态的字段。
    算法指标：新增 PPO、规则控制、奖励分项等训练诊断字段。
    """
    monitor = MonitorConfigBuilder()

    # 环境指标：来自环境观测和 episode 结果，方便确认训练是否真的在推进。
    environment_panels = [
        ("对局数", "episode_cnt"),
        ("总奖励", "reward"),
        ("胜利率", "win"),
        ("超时率", "timeout_rate"),
        ("对局帧数", "frame_no"),
        ("己方英雄血量", "my_hp"),
        ("敌方英雄血量", "enemy_hp"),
        ("己方塔血量", "own_tower_hp"),
        ("敌方塔血量", "enemy_tower_hp"),
        ("己方塔血比", "own_tower_hp_ratio"),
        ("敌方塔血比", "enemy_tower_hp_ratio"),
        ("击杀次数", "kill_count"),
        ("死亡次数", "death_count"),
        ("血包数量", "cake_count"),
        ("中立资源数量", "neutral_count"),
        ("对手类型", "opponent_type"),
    ]
    builder = monitor.title("王者荣耀1v1").add_group(group_name="环境指标", group_name_en="environment")
    builder = _add_line_panels(builder, environment_panels).end_group()

    # 算法指标：来自 PPO 更新、规则控制和奖励通道，定位学习稳定性问题。
    algorithm_panels = [
        ("总损失", "total_loss"),
        ("价值损失", "value_loss"),
        ("策略损失", "policy_loss"),
        ("熵损失", "entropy_loss"),
        ("KL", "approx_kl"),
        ("裁剪比例", "clip_fraction"),
        ("学习率", "learning_rate", "0.00000001"),
        ("硬掩码率", "hard_mask_rate"),
        ("规则偏置数", "rule_bias_count"),
        ("掩码兜底数", "mask_fallback_count"),
        ("特征NaN数", "feature_nan_count"),
        ("隐状态范数", "hidden_norm"),
        ("推塔奖励", "tower"),
        ("守塔奖励", "tower_defense"),
        ("兵线奖励", "lane"),
        ("成长奖励", "growth"),
        ("补刀奖励", "last_hit"),
        ("死亡惩罚", "death"),
        ("塔下风险", "tower_risk"),
        ("资源奖励", "resource"),
        ("血包奖励", "cake"),
        ("技能奖励", "skill"),
        ("终局奖励", "terminal"),
    ]
    builder = builder.add_group(group_name="算法指标", group_name_en="algorithm")
    builder = _add_line_panels(builder, algorithm_panels).end_group()

    return builder.build()
