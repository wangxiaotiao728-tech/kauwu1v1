#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
1v1 monitor builder.

面板原则：
1. 算法损失只保留 learner 侧真实上报的核心指标。
2. 旧 D401 指标不再单独建一组，合并到奖励/行为面板，避免重复。
3. 面板标题和图例尽量使用中文，表达式仍引用真实 metric key。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


DYNAMIC_GAUGE_METRICS = {
    "learning_rate",
    "entropy_beta",
    "ppo_clip",
}


METRIC_LABELS = {
    # algorithm
    "reward": "累计回报",
    "total_loss": "总损失",
    "value_loss": "价值损失",
    "policy_loss": "策略损失",
    "entropy_loss": "熵损失",
    "approx_kl": "近似KL",
    "clip_fraction": "裁剪比例",
    "learning_rate": "学习率",
    "entropy_beta": "熵系数",
    "ppo_clip": "PPO裁剪",
    "grad_norm": "梯度范数",
    "hidden_norm": "隐状态范数",
    "feature_nan_count": "特征NaN数",
    "global_value_loss": "全局价值损失",
    "group_value_loss": "分组价值损失",
    "policy_head_count": "策略头数",
    "policy_loss_per_head": "单头策略损失",
    "entropy_loss_per_head": "单头熵损失",
    "diagnostic_total_loss": "诊断总损失",
    "policy_value_abs_ratio": "策略价值比",
    "adv_abs_mean": "优势均值",
    "adv_std": "优势标准差",
    "value_target_abs_mean": "价值目标均值",
    "value_pred_abs_mean": "价值预测均值",
    "group_value_target_abs_mean": "分组价值目标均值",
    "train_frame_ratio": "训练帧比例",

    # environment
    "win": "胜率",
    "own_tower_hp_ratio": "己方塔血",
    "enemy_tower_hp_ratio": "敌方塔血",
    "my_hp": "己方血量",
    "enemy_hp": "敌方血量",
    "kill_count": "击杀",
    "death_count": "死亡",
    "my_money": "己方金币",
    "enemy_money": "敌方金币",
    "my_exp": "己方经验",
    "enemy_exp": "敌方经验",
    "my_level": "己方等级",
    "enemy_level": "敌方等级",
    "enemy_soldier_count": "敌方小兵数",
    "friendly_soldier_count": "己方小兵数",
    "cake_count": "血包数",
    "neutral_count": "中立资源数",
    "episode_cnt": "对局数",
    "frame_no": "帧号",

    # reward
    "reward_objective": "目标奖励",
    "reward_growth_combat": "发育战斗奖励",
    "reward_behavior_safety": "行为安全奖励",
    "tower_hp_point": "塔血奖励",
    "kill": "击杀奖励",
    "death": "死亡惩罚",
    "money": "金币奖励",
    "exp": "经验奖励",
    "last_hit": "补刀奖励",
    "hp_point": "血量奖励",
    "hero_hurt": "承伤",
    "hero_damage": "英雄伤害",
    "total_damage": "总伤害",
    "enemy_pressure": "接敌压力",
    "combat_intent": "战斗意图",
    "trade_advantage": "换血优势",
    "skill_hit": "技能命中",
    "crit": "暴击",
    "lane_clear": "清线",
    "defense": "防守",
    "cake": "血包",
    "forward": "前压",
    "lane_presence": "兵线存在",
    "tower_risk": "塔下风险",
    "stuck": "卡住",
    "no_ops": "无操作",
    "home_idle": "泉水停留",
    "respawn_leave_base": "复活离家",
    "grass_behavior": "草丛行为",
    "in_grass": "草丛率",
    "under_tower_behavior": "塔下行为",
    "passive_skills": "被动技能",

    # behavior
    "target_soldier_rate": "目标小兵",
    "target_enemy_rate": "目标英雄",
    "target_tower_rate": "目标防御塔",
    "target_monster_rate": "目标野怪",
    "button_move_rate": "移动",
    "button_attack_rate": "普攻",
    "button_none_rate": "空动作",
    "button_skill1_rate": "技能1",
    "button_skill2_rate": "技能2",
    "button_skill3_rate": "技能3",
    "skill_target_enemy_rate": "技能打英雄",
    "skill_target_soldier_rate": "技能打小兵",
    "skill_target_tower_rate": "技能打塔",
    "skill_center_rate": "技能空放中心",
    "stuck_count": "卡住次数",
    "grass_long_stay_count": "草丛久留",
    "grass_no_effective_count": "草丛无收益",
    "unsafe_tower_entry_count": "危险进塔",
    "defense_emergency_count": "防守紧急",
    "enemy_soldier_near_own_tower_count": "塔前敌兵",
    "own_tower_target_enemy_soldier_count": "塔打敌兵",
    "own_cake_pick_count": "己方血包拾取",
    "low_hp_own_cake_approach_count": "低血靠近血包",
}


def _precision(metric):
    if metric == "learning_rate":
        return "0.00000001"

    if metric in {
        "win",
        "approx_kl",
        "clip_fraction",
        "entropy_beta",
        "ppo_clip",
        "total_loss",
        "value_loss",
        "policy_loss",
        "entropy_loss",
        "global_value_loss",
        "group_value_loss",
        "policy_head_count",
        "policy_loss_per_head",
        "entropy_loss_per_head",
        "diagnostic_total_loss",
        "policy_value_abs_ratio",
        "adv_abs_mean",
        "adv_std",
        "value_target_abs_mean",
        "value_pred_abs_mean",
        "group_value_target_abs_mean",
        "train_frame_ratio",
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


def _metric_label(metric):
    return METRIC_LABELS.get(metric, metric)


def _add_metric(builder, metric):
    agg = "max" if metric in DYNAMIC_GAUGE_METRICS else "avg"
    return builder.add_metric(
        metrics_name=_metric_label(metric),
        expr=f"round({agg}({metric}{{}}), {_precision(metric)})",
    )


def _add_panel(builder, name, name_en, metrics):
    builder = builder.add_panel(
        name=name,
        name_en=name_en,
        type="line",
    )
    for metric in metrics:
        builder = _add_metric(builder, metric)
    return builder.end_panel()


def _add_multi_metric_panels(builder, panels):
    for panel_name, panel_name_en, metrics in panels:
        builder = _add_panel(builder, panel_name, panel_name_en, metrics)
    return builder


def build_monitor():
    monitor = MonitorConfigBuilder()

    builder = monitor.title("智能决策1v1")

    algorithm_panels = [
        ("累计回报", "reward", ["reward"]),
        ("总损失", "total_loss", ["total_loss"]),
        ("价值损失", "value_loss", ["value_loss"]),
        ("策略损失", "policy_loss", ["policy_loss"]),
        ("熵损失", "entropy_loss", ["entropy_loss"]),
        (
            "归一损失诊断",
            "normalized_loss_debug",
            ["diagnostic_total_loss", "value_loss", "policy_loss_per_head", "entropy_loss_per_head"],
        ),
        ("损失量级诊断", "loss_scale_debug", ["policy_value_abs_ratio", "policy_head_count"]),
        ("PPO稳定性", "ppo_stability", ["approx_kl", "clip_fraction"]),
        ("学习动态", "learn_dynamic", ["learning_rate", "entropy_beta", "ppo_clip"]),
        ("训练健康", "train_health", ["grad_norm", "hidden_norm", "feature_nan_count"]),
        (
            "价值网络诊断",
            "value_debug",
            [
                "global_value_loss",
                "group_value_loss",
                "policy_head_count",
                "policy_loss_per_head",
                "entropy_loss_per_head",
                "diagnostic_total_loss",
                "policy_value_abs_ratio",
                "adv_abs_mean",
                "adv_std",
                "value_target_abs_mean",
                "value_pred_abs_mean",
                "group_value_target_abs_mean",
                "train_frame_ratio",
            ],
        ),
    ]
    builder = builder.add_group(group_name="算法指标", group_name_en="algorithm")
    builder = _add_multi_metric_panels(builder, algorithm_panels).end_group()

    env_panels = [
        ("胜负", "win", ["win"]),
        ("塔血对比", "tower_compare", ["own_tower_hp_ratio", "enemy_tower_hp_ratio"]),
        ("英雄血量", "hero_hp", ["my_hp", "enemy_hp"]),
        ("击杀死亡", "kill_death", ["kill_count", "death_count"]),
        ("经济等级", "economy", ["my_money", "enemy_money", "my_exp", "enemy_exp", "my_level", "enemy_level"]),
        ("兵线资源", "soldier_resource", ["enemy_soldier_count", "friendly_soldier_count", "cake_count", "neutral_count"]),
        ("对局信息", "episode_info", ["episode_cnt", "frame_no"]),
    ]
    builder = builder.add_group(group_name="环境指标", group_name_en="environment")
    builder = _add_multi_metric_panels(builder, env_panels)
    builder = (
        builder
        .add_panel(name="阵容胜率", name_en="matchup_win_rate", type="line")
        .add_metric(metrics_name="鲁班vs鲁班", expr="round(avg(matchup_112_vs_112_win_rate{}), 0.0001)")
        .add_metric(metrics_name="鲁班vs狄仁杰", expr="round(avg(matchup_112_vs_133_win_rate{}), 0.0001)")
        .add_metric(metrics_name="狄仁杰vs鲁班", expr="round(avg(matchup_133_vs_112_win_rate{}), 0.0001)")
        .add_metric(metrics_name="狄仁杰vs狄仁杰", expr="round(avg(matchup_133_vs_133_win_rate{}), 0.0001)")
        .end_panel()
        .end_group()
    )

    reward_panels = [
        ("奖励三组", "reward_groups", ["reward_objective", "reward_growth_combat", "reward_behavior_safety"]),
        ("目标奖励", "objective_reward", ["tower_hp_point", "kill", "death"]),
        ("发育奖励", "growth_reward", ["money", "exp", "last_hit"]),
        (
            "战斗奖励",
            "combat_reward",
            [
                "hp_point",
                "hero_hurt",
                "hero_damage",
                "total_damage",
                "skill_hit",
                "enemy_pressure",
                "combat_intent",
                "trade_advantage",
                "crit",
            ],
        ),
        ("清线防守", "lane_defense", ["lane_clear", "defense", "cake", "forward", "lane_presence", "respawn_leave_base"]),
        (
            "风险行为奖励",
            "risk_behavior_reward",
            ["tower_risk", "stuck", "no_ops", "home_idle", "respawn_leave_base", "grass_behavior", "in_grass", "under_tower_behavior", "passive_skills"],
        ),
    ]
    builder = builder.add_group(group_name="奖励指标", group_name_en="reward")
    builder = _add_multi_metric_panels(builder, reward_panels).end_group()

    behavior_panels = [
        ("目标选择", "target_select", ["target_soldier_rate", "target_enemy_rate", "target_tower_rate", "target_monster_rate"]),
        ("基础动作", "basic_action", ["button_move_rate", "button_attack_rate", "button_none_rate"]),
        ("技能动作", "skill_action", ["button_skill1_rate", "button_skill2_rate", "button_skill3_rate"]),
        (
            "技能目标",
            "skill_target",
            ["skill_target_enemy_rate", "skill_target_soldier_rate", "skill_target_tower_rate", "skill_center_rate"],
        ),
        ("异常行为", "bad_behavior", ["stuck_count", "grass_long_stay_count", "grass_no_effective_count", "unsafe_tower_entry_count"]),
        (
            "防守行为",
            "defense_behavior",
            ["defense_emergency_count", "enemy_soldier_near_own_tower_count", "own_tower_target_enemy_soldier_count"],
        ),
        ("血包行为", "cake_behavior", ["own_cake_pick_count", "low_hp_own_cake_approach_count"]),
    ]
    builder = builder.add_group(group_name="行为诊断", group_name_en="behavior")
    builder = _add_multi_metric_panels(builder, behavior_panels).end_group()

    return builder.build()
