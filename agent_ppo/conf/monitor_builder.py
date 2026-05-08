#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()
    builder = monitor.title("智能决策1v1").add_group(group_name="算法指标", group_name_en="algorithm")
    panels = [
        ("总奖励", "reward"),
        ("总损失", "total_loss"),
        ("价值损失", "value_loss"),
        ("策略损失", "policy_loss"),
        ("熵", "entropy_loss"),
        ("KL", "approx_kl"),
        ("裁剪率", "clip_fraction"),
        ("学习率", "learning_rate"),
        ("规则屏蔽率", "hard_mask_rate"),
        ("规则偏置数", "rule_bias_count"),
        ("特征NaN", "feature_nan_count"),
        ("隐状态范数", "hidden_norm"),
        ("推塔奖励", "tower"),
        ("保塔惩罚", "tower_defense"),
        ("兵线奖励", "lane"),
        ("成长奖励", "growth"),
        ("补刀奖励", "last_hit"),
        ("死亡惩罚", "death"),
        ("塔风险", "tower_risk"),
        ("资源奖励", "resource"),
        ("技能奖励", "skill"),
        ("终局奖励", "terminal"),
    ]
    for name, metric in panels:
        builder = (
            builder.add_panel(name=name, name_en=metric, type="line")
            .add_metric(metrics_name=metric, expr=f"round(avg({metric}{{}}), 0.01)")
            .end_panel()
        )
    config_dict = builder.end_group().build()
    return config_dict
