#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""规则控制器：只产出 hard mask 和 logit bias，不直接改写最终动作。"""

from dataclasses import dataclass
import os

import numpy as np

from agent_ppo.conf.conf import Config, CurriculumConfig, RuleConfig


RULE_STATES = {
    "MODEL": 0,
    "FINISH_TOWER": 1,
    "DEATH_WINDOW_PUSH": 2,
    "RETREAT_TOWER_AGGRO": 3,
    "DEFEND_TOWER": 4,
    "PUSH_WITH_MINION": 5,
    "ENHANCED_POKE_TOWER": 6,
    "PICK_CAKE_SAFE": 7,
    "CONTEST_RESOURCE": 8,
}


@dataclass
class RuleOutput:
    hard_mask: list
    logit_bias: list
    state_id: int
    debug: dict


class RuleController:
    def __init__(self, config=Config):
        self.config = config
        self.hard_mask_count = 0
        self.total_mask_count = 0
        self.mask_fallback_count = 0
        self.rule_bias_count = 0

    def compute(self, feature, aux, official_masks):
        hard_mask = [np.zeros(size, dtype=np.float32) for size in Config.LABEL_SIZE_LIST]
        logit_bias = [np.zeros(size, dtype=np.float32) for size in Config.LABEL_SIZE_LIST]
        state_id = RULE_STATES["MODEL"]
        stage = self._stage()
        enable_rule_bias = bool(stage.get("enable_rule_bias", False))
        enable_cake_bias = bool(stage.get("enable_cake_bias", False))
        enable_resource_bias = bool(stage.get("enable_resource_bias", False))

        my_dead = bool(aux.get("my_hp_ratio", 1.0) <= 0.0)
        direct_push = bool(aux.get("direct_push_window", False))
        defense = bool(aux.get("defense_emergency", False))
        cake_safe = bool(aux.get("cake_observed", False) and aux.get("my_hp_ratio", 1.0) < RuleConfig.BLOOD_PACK_HP_RATIO)
        tower_low = aux.get("enemy_tower_hp_ratio", 1.0) < RuleConfig.FINISH_TOWER_HP_RATIO
        neutral_safe = bool(
            aux.get("neutral_observed", False)
            and aux.get("resource_allowed", False)
            and len(feature) > RuleConfig.NEUTRAL_SAFE_FEATURE_INDEX
            and feature[RuleConfig.NEUTRAL_SAFE_FEATURE_INDEX] > 0.5
        )

        if my_dead:
            self._allow_only(hard_mask, RuleConfig.NOOP_ACTION)
            state_id = RULE_STATES["MODEL"]
        elif enable_rule_bias and tower_low and direct_push:
            self._bias_action(logit_bias, RuleConfig.FINISH_TOWER_ACTION, RuleConfig.FINISH_TOWER_BIAS)
            state_id = RULE_STATES["FINISH_TOWER"]
        elif enable_rule_bias and direct_push:
            self._bias_action(logit_bias, RuleConfig.FINISH_TOWER_ACTION, RuleConfig.DIRECT_PUSH_BIAS)
            state_id = RULE_STATES["DEATH_WINDOW_PUSH"]
        elif enable_rule_bias and defense:
            self._bias_head(logit_bias, 0, RuleConfig.DEFEND_BUTTON, RuleConfig.DEFEND_TOWER_BIAS)
            state_id = RULE_STATES["DEFEND_TOWER"]
        elif enable_rule_bias and enable_cake_bias and cake_safe and not direct_push and not defense:
            self._bias_head(logit_bias, 0, RuleConfig.CAKE_BUTTON, RuleConfig.PICK_CAKE_BIAS)
            state_id = RULE_STATES["PICK_CAKE_SAFE"]
        elif enable_rule_bias and enable_resource_bias and neutral_safe:
            self._bias_head(logit_bias, 0, RuleConfig.RESOURCE_BUTTON, RuleConfig.CONTEST_RESOURCE_BIAS)
            state_id = RULE_STATES["CONTEST_RESOURCE"]

        # 低血量且塔下风险高时，禁止继续进攻并偏置到撤退/防守动作。
        tower_risk = (
            float(feature[RuleConfig.TOWER_RISK_FEATURE_INDEX])
            if len(feature) > RuleConfig.TOWER_RISK_FEATURE_INDEX
            else 0.0
        )
        if (
            enable_rule_bias
            and aux.get("my_hp_ratio", 1.0) < RuleConfig.LOW_HP_RISK_RATIO
            and tower_risk > RuleConfig.TOWER_RISK_HARD_MASK_THRESHOLD
        ):
            for button in RuleConfig.RETREAT_FORBID_BUTTONS:
                if button < hard_mask[0].shape[0]:
                    hard_mask[0][button] = 1.0
            self._bias_head(logit_bias, 0, RuleConfig.RETREAT_BUTTON, RuleConfig.RETREAT_TOWER_BIAS)
            state_id = RULE_STATES["RETREAT_TOWER_AGGRO"]

        if self.hard_mask_rate > RuleConfig.HARD_MASK_RATE_LIMIT and not my_dead:
            # hard mask 过高时降级成 bias-only，避免规则长期压死策略学习。
            hard_mask = [np.zeros_like(mask) for mask in hard_mask]

        logit_bias = [np.clip(bias, -RuleConfig.LOGIT_BIAS_ABS_MAX, RuleConfig.LOGIT_BIAS_ABS_MAX) for bias in logit_bias]
        self._update_stats(hard_mask, logit_bias)
        debug = {
            "hard_mask_rate": self.hard_mask_rate,
            "rule_bias_count": self.rule_bias_count,
            "mask_fallback_count": self.mask_fallback_count,
            "enable_rule_bias": enable_rule_bias,
        }
        return RuleOutput(hard_mask=hard_mask, logit_bias=logit_bias, state_id=state_id, debug=debug)

    @property
    def hard_mask_rate(self):
        if self.total_mask_count <= 0:
            return 0.0
        return self.hard_mask_count / self.total_mask_count

    def compose_final_masks(self, official_masks, rule_output):
        final_masks = []
        for official, forbid in zip(official_masks, rule_output.hard_mask):
            official = np.asarray(official, dtype=np.float32)
            forbid = np.asarray(forbid, dtype=np.float32)
            final = official * (1.0 - forbid)
            if final.sum() <= 0:
                final = np.zeros_like(official, dtype=np.float32)
                final[0] = 1.0
                self.mask_fallback_count += 1
            final_masks.append(final)
        return final_masks

    def _stage(self):
        stage_name = os.environ.get("HOK_CURRICULUM_STAGE", CurriculumConfig.CURRENT_STAGE)
        return CurriculumConfig.CURRICULUM_STAGES.get(stage_name, CurriculumConfig.CURRICULUM_STAGES["S1_BASIC"])

    def _allow_only(self, hard_mask, action):
        for head, value in enumerate(action):
            hard_mask[head][:] = 1.0
            if 0 <= value < hard_mask[head].shape[0]:
                hard_mask[head][value] = 0.0

    def _bias_action(self, logit_bias, action, amount):
        for head, value in enumerate(action):
            if 0 <= value < logit_bias[head].shape[0]:
                logit_bias[head][value] += amount

    def _bias_head(self, logit_bias, head, value, amount):
        if 0 <= head < len(logit_bias) and 0 <= value < logit_bias[head].shape[0]:
            logit_bias[head][value] += amount

    def _update_stats(self, hard_mask, logit_bias):
        for mask in hard_mask:
            self.hard_mask_count += int(np.sum(mask > 0.0))
            self.total_mask_count += int(mask.size)
        self.rule_bias_count += sum(int(np.sum(np.abs(bias) > 1e-6)) for bias in logit_bias)
