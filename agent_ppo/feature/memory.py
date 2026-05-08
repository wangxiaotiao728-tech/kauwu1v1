#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""历史记忆模块，只输出事实和带置信度的估计，不直接决策。"""

import math
from collections import deque


class MemoryProcess:
    def __init__(self):
        self.reset()

    def reset(self):
        self.enemy_last_seen = {
            "x": 0.0,
            "z": 0.0,
            "hp_ratio": 0.0,
            "frame": -1,
            "near_grass": 0.0,
        }
        self.events_30 = deque(maxlen=30)
        self.events_60 = deque(maxlen=60)
        self.tower_memory = {
            "enemy_tower_hp_last": 1.0,
            "own_tower_hp_last": 1.0,
            "enemy_tower_hp_delta_1": 0.0,
            "own_tower_hp_delta_1": 0.0,
            "tower_aggro_count": 0.0,
            "estimated_tower_range": 8500.0,
            "tower_range_confidence": 0.0,
        }
        self.combo_memory = {
            "last_skill_used": 0.0,
            "steps_since_last_skill": 999.0,
            "steps_since_last_attack": 999.0,
            "enhanced_attack_ready": 0.0,
            "enhanced_attack_confidence": 0.0,
            "combo_window_active": 0.0,
        }
        self.cake_memory = {
            "last_cake_seen": 0.0,
            "last_cake_pick_frame": -1,
            "hp_delta_after_cake": 0.0,
            "speed_delta_after_cake": 0.0,
            "buff_added_after_cake": 0.0,
            "cake_effect_confidence": 0.0,
        }
        self.neutral_memory = {
            "neutral_last_seen": 0.0,
            "neutral_recently_killed": 0.0,
            "neutral_spawn_estimate": 0.0,
            "neutral_spawn_confidence": 0.0,
        }

    def update_before_action(self, observation, frame_no):
        frame_state = observation.get("frame_state", {}) or {}
        my_hero, enemy_hero = self._split_heroes(frame_state, observation)
        my_camp = my_hero.get("camp") if my_hero else observation.get("player_camp", observation.get("camp"))
        own_tower, enemy_tower, friendly_minions, enemy_minions, monsters = self._split_npcs(frame_state, my_camp)

        if enemy_hero:
            ex, ez = self._pos(enemy_hero)
            self.enemy_last_seen.update(
                {
                    "x": ex,
                    "z": ez,
                    "hp_ratio": self._hp_ratio(enemy_hero),
                    "frame": frame_no,
                    "near_grass": 1.0 if enemy_hero.get("is_in_grass", False) else 0.0,
                }
            )

        enemy_tower_hp = self._hp_ratio(enemy_tower)
        own_tower_hp = self._hp_ratio(own_tower)
        self.tower_memory["enemy_tower_hp_delta_1"] = max(
            0.0, self.tower_memory["enemy_tower_hp_last"] - enemy_tower_hp
        )
        self.tower_memory["own_tower_hp_delta_1"] = max(0.0, self.tower_memory["own_tower_hp_last"] - own_tower_hp)
        self.tower_memory["enemy_tower_hp_last"] = enemy_tower_hp
        self.tower_memory["own_tower_hp_last"] = own_tower_hp
        if my_hero and enemy_tower and enemy_tower.get("attack_target") == my_hero.get("runtime_id"):
            self.tower_memory["tower_aggro_count"] += 1.0
            self._sample_tower_range(my_hero, enemy_tower)

        event = {
            "enemy_observed": 1.0 if enemy_hero else 0.0,
            "attack_hero": 1.0 if self._target_is(enemy_hero, my_hero) else 0.0,
            "attack_tower": 1.0 if self._target_is(enemy_hero, own_tower) else 0.0,
            "clear_wave": 1.0 if self._target_in(enemy_hero, friendly_minions) else 0.0,
            "near_neutral": 1.0 if enemy_hero and monsters and min(self._dist(self._pos(enemy_hero), self._pos(m)) for m in monsters) < 8000 else 0.0,
            "damage_to_my_tower": self.tower_memory["own_tower_hp_delta_1"],
        }
        self.events_30.append(event)
        self.events_60.append(event)

        self.combo_memory["steps_since_last_skill"] += 1.0
        self.combo_memory["steps_since_last_attack"] += 1.0
        if self._real_cmd_has_skill((my_hero or {}).get("real_cmd", []) or []):
            self.combo_memory["last_skill_used"] = 1.0
            self.combo_memory["steps_since_last_skill"] = 0.0
            self.combo_memory["combo_window_active"] = 1.0
        else:
            self.combo_memory["combo_window_active"] = 1.0 if self.combo_memory["steps_since_last_skill"] <= 8 else 0.0
        if self._real_cmd_has_common_attack((my_hero or {}).get("real_cmd", []) or []):
            self.combo_memory["steps_since_last_attack"] = 0.0
        self.combo_memory["enhanced_attack_ready"] = self._enhanced_attack_ready(my_hero)
        self.combo_memory["enhanced_attack_confidence"] = 1.0 if my_hero else 0.0

        cakes = frame_state.get("cakes", []) or []
        self.cake_memory["last_cake_seen"] = 1.0 if cakes else 0.0
        self.neutral_memory["neutral_last_seen"] = 1.0 if monsters else 0.0

    def update_after_action(self, observation, action, reward_aux, frame_no):
        # 预留给 cake 效果、资源击杀等跨帧结果估计。
        if reward_aux and reward_aux.get("cake_safe_pick", 0.0) > 0:
            self.cake_memory["last_cake_pick_frame"] = frame_no
            self.cake_memory["cake_effect_confidence"] = max(self.cake_memory["cake_effect_confidence"], 0.2)

    def export_features(self):
        return {
            "enemy_last_seen": dict(self.enemy_last_seen),
            "opponent_behavior": {
                "attack_hero_rate_30": self._rate(self.events_30, "attack_hero"),
                "attack_tower_rate_30": self._rate(self.events_30, "attack_tower"),
                "clear_wave_rate_30": self._rate(self.events_30, "clear_wave"),
                "near_neutral_rate_60": self._rate(self.events_60, "near_neutral"),
                "missing_rate_60": 1.0 - self._rate(self.events_60, "enemy_observed"),
                "damage_to_my_tower_rate_60": self._rate(self.events_60, "damage_to_my_tower"),
                "behavior_change_score": abs(self._rate(self.events_30, "attack_tower") - self._rate(self.events_60, "attack_tower")),
            },
            "tower_memory": dict(self.tower_memory),
            "combo_memory": dict(self.combo_memory),
            "cake_memory": dict(self.cake_memory),
            "neutral_memory": dict(self.neutral_memory),
        }

    def _sample_tower_range(self, my_hero, enemy_tower):
        dist = self._dist(self._pos(my_hero), self._pos(enemy_tower))
        if dist >= 100000:
            return
        conf = self.tower_memory["tower_range_confidence"]
        old = self.tower_memory["estimated_tower_range"]
        self.tower_memory["estimated_tower_range"] = old * conf + dist * (1.0 - conf)
        self.tower_memory["tower_range_confidence"] = min(1.0, conf + 1.0 / 200.0)

    def _rate(self, events, key):
        if not events:
            return 0.0
        return sum(float(e.get(key, 0.0)) for e in events) / len(events)

    def _split_heroes(self, frame_state, observation):
        player_id = observation.get("player_id")
        player_camp = observation.get("player_camp", observation.get("camp"))
        heroes = frame_state.get("hero_states", []) or []
        my_hero, enemy_hero = None, None
        for hero in heroes:
            if hero.get("runtime_id") == player_id:
                my_hero = hero
                break
        if my_hero is None:
            for hero in heroes:
                if self._same_camp(hero.get("camp"), player_camp):
                    my_hero = hero
                    break
        my_camp = my_hero.get("camp") if my_hero else player_camp
        for hero in heroes:
            if hero is not my_hero and not self._same_camp(hero.get("camp"), my_camp):
                enemy_hero = hero
                break
        return my_hero, enemy_hero

    def _split_npcs(self, frame_state, my_camp):
        own_tower, enemy_tower, friendly_minions, enemy_minions, monsters = None, None, [], [], []
        for npc in frame_state.get("npc_states", []) or []:
            camp = npc.get("camp")
            if self._is_tower(npc):
                if self._same_camp(camp, my_camp):
                    own_tower = npc
                else:
                    enemy_tower = npc
            elif self._is_monster(npc):
                monsters.append(npc)
            elif self._same_camp(camp, my_camp):
                friendly_minions.append(npc)
            elif camp not in (0, None, "0"):
                enemy_minions.append(npc)
        return own_tower, enemy_tower, friendly_minions, enemy_minions, monsters

    def _is_tower(self, npc):
        sub_type = npc.get("sub_type")
        return sub_type == 21 or "TOWER" in str(sub_type).upper()

    def _is_monster(self, npc):
        actor_type = str(npc.get("actor_type", "")).upper()
        sub_type = str(npc.get("sub_type", "")).upper()
        return "MONSTER" in actor_type or "MONSTER" in sub_type or self._camp_value(npc.get("camp")) in (0, None, "0")

    def _target_is(self, source, target):
        return bool(source and target and source.get("attack_target") == target.get("runtime_id"))

    def _target_in(self, source, targets):
        if not source:
            return False
        target_id = source.get("attack_target")
        return target_id is not None and target_id in {t.get("runtime_id") for t in targets}

    def _enhanced_attack_ready(self, hero):
        if not hero:
            return 0.0
        buffs = ((hero.get("buff_state") or {}).get("buff_skills", []) or [])
        marks = ((hero.get("buff_state") or {}).get("buff_marks", []) or [])
        return 1.0 if buffs or marks else 0.0

    def _real_cmd_has_skill(self, real_cmd):
        return any(cmd.get("dir_skill") or cmd.get("pos_skill") or cmd.get("obj_skill") for cmd in real_cmd)

    def _real_cmd_has_common_attack(self, real_cmd):
        return any(cmd.get("attack_common") or cmd.get("attack_actor") for cmd in real_cmd)

    def _hp_ratio(self, obj):
        return self._clamp(self._safe_div((obj or {}).get("hp", 0), (obj or {}).get("max_hp", 0)))

    def _pos(self, obj):
        loc = (obj or {}).get("location", {}) or {}
        return loc.get("x", 100000), loc.get("z", 100000)

    def _dist(self, a, b):
        ax, az = a
        bx, bz = b
        if 100000 in (ax, az, bx, bz):
            return 100000.0
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)

    def _same_camp(self, left, right):
        return self._camp_value(left) == self._camp_value(right)

    def _camp_value(self, camp):
        if isinstance(camp, str):
            if camp.endswith("_1") or camp == "1":
                return 1
            if camp.endswith("_2") or camp == "2":
                return 2
        return camp

    def _safe_div(self, num, den):
        try:
            return num / den if den else 0.0
        except (TypeError, ZeroDivisionError):
            return 0.0

    def _clamp(self, value, lo=0.0, hi=1.0):
        try:
            if math.isnan(value) or math.isinf(value):
                return 0.0
        except TypeError:
            return 0.0
        return max(lo, min(hi, float(value)))
