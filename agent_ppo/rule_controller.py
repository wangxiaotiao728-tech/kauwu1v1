#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""规则控制器：只产出 hard mask 和 logit bias，不事后篡改动作。"""

from dataclasses import dataclass

import numpy as np

from agent_ppo.conf.conf import Config, RuleConfig


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
        debug = {}

        my_dead = bool(aux.get("my_hp_ratio", 1.0) <= 0.0)
        direct_push = bool(aux.get("direct_push_window", False))
        defense = bool(aux.get("defense_emergency", False))
        cake_safe = bool(aux.get("cake_observed", False) and aux.get("my_hp_ratio", 1.0) < RuleConfig.BLOOD_PACK_HP_RATIO)
        tower_low = aux.get("enemy_tower_hp_ratio", 1.0) < RuleConfig.FINISH_TOWER_HP_RATIO

        if my_dead:
            self._allow_only(hard_mask, RuleConfig.NOOP_ACTION)
            state_id = RULE_STATES["MODEL"]
        elif tower_low and direct_push:
            self._bias_action(logit_bias, [3, 15, 15, 15, 15, 7], 2.0)
            state_id = RULE_STATES["FINISH_TOWER"]
        elif direct_push:
            self._bias_action(logit_bias, [3, 15, 15, 15, 15, 7], 1.2)
            state_id = RULE_STATES["DEATH_WINDOW_PUSH"]
        elif defense:
            self._bias_head(logit_bias, 0, 2, 0.8)
            state_id = RULE_STATES["DEFEND_TOWER"]
        elif cake_safe and not direct_push and not defense:
            self._bias_head(logit_bias, 0, 2, 0.5)
            state_id = RULE_STATES["PICK_CAKE_SAFE"]

        # 低血且处于塔风险时，禁止明显进攻按钮，其余场景使用 bias。
        tower_risk = float(feature[488]) if len(feature) > 488 else 0.0
        if aux.get("my_hp_ratio", 1.0) < RuleConfig.LOW_HP_RISK_RATIO and tower_risk > 0.65:
            for button in (3, 4, 5, 6, 8):
                if button < hard_mask[0].shape[0]:
                    hard_mask[0][button] = 1.0
            self._bias_head(logit_bias, 0, 2, 1.0)
            state_id = RULE_STATES["RETREAT_TOWER_AGGRO"]

        logit_bias = [np.clip(bias, -RuleConfig.LOGIT_BIAS_ABS_MAX, RuleConfig.LOGIT_BIAS_ABS_MAX) for bias in logit_bias]
        self._update_stats(hard_mask, logit_bias)
        debug["hard_mask_rate"] = self.hard_mask_rate
        debug["rule_bias_count"] = self.rule_bias_count
        debug["mask_fallback_count"] = self.mask_fallback_count
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
