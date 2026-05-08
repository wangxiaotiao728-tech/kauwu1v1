#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""固定 512 维特征构造。"""

import math

import numpy as np

from agent_ppo.conf.conf import Config, RuleConfig
from agent_ppo.feature.feature_schema import FEATURE_DIM, IDX
from agent_ppo.feature.memory import MemoryProcess


def safe_float(x, default=0.0):
    try:
        x = float(x)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


def n01(x, lo, hi):
    x = safe_float(x)
    return float(np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0))


def nsym(x, bound):
    x = safe_float(x)
    return float(np.clip(x / (bound + 1e-6), -1.0, 1.0))


def flag(x):
    return 1.0 if bool(x) else 0.0


class FeatureBuilder:
    def __init__(self, memory, config=Config):
        self.memory = memory
        self.config = config

    def build(self, observation, frame_no, legal_action=None, sub_action_mask=None):
        feature = np.zeros(FEATURE_DIM, dtype=np.float32)
        frame_state = observation.get("frame_state", {}) or {}
        my_hero, enemy_hero = self._split_heroes(frame_state, observation)
        my_camp = (my_hero or {}).get("camp", observation.get("player_camp", observation.get("camp")))
        own_tower, enemy_tower, friendly_minions, enemy_minions, neutral_units = self._split_npcs(frame_state, my_camp)
        cakes = frame_state.get("cakes", []) or []
        memory = self.memory.export_features()

        direct_push_window = self._direct_push_window(my_hero, enemy_hero, enemy_tower, friendly_minions)
        defense_emergency = self._defense_emergency(own_tower, enemy_minions, memory)
        resource_allowed = (not direct_push_window) and (not defense_emergency) and self._hp_ratio(my_hero) > 0.45

        self._fill_global(feature, frame_no, direct_push_window, defense_emergency, resource_allowed)
        self._fill_my_hero(feature, my_hero)
        self._fill_enemy(feature, frame_no, enemy_hero, memory)
        self._fill_skill_combo(feature, my_hero, memory)
        self._fill_tower(feature, my_hero, own_tower, enemy_tower, friendly_minions, memory)
        self._fill_lane(feature, my_hero, own_tower, enemy_tower, friendly_minions, enemy_minions)
        self._fill_resource(feature, my_hero, enemy_hero, cakes, neutral_units, memory)
        self._fill_vision(feature, my_hero, enemy_hero)
        self._fill_opponent_behavior(feature, memory)
        self._fill_rule_debug(feature, direct_push_window, defense_emergency, resource_allowed, legal_action)

        feature[496:512] = 0.0
        feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        aux = {
            "direct_push_window": direct_push_window,
            "defense_emergency": defense_emergency,
            "resource_allowed": resource_allowed,
            "my_hp_ratio": self._hp_ratio(my_hero),
            "enemy_tower_hp_ratio": self._hp_ratio(enemy_tower),
            "own_tower_hp_ratio": self._hp_ratio(own_tower),
            "enemy_observed": enemy_hero is not None,
            "cake_observed": len(cakes) > 0,
            "neutral_observed": len(neutral_units) > 0,
        }
        return feature, aux

    def _fill_global(self, feature, frame_no, direct_push_window, defense_emergency, resource_allowed):
        setv(feature, IDX["frame_progress"], n01(frame_no, 0, 20000))
        setv(feature, IDX["timeout_progress"], n01(frame_no, 16000, 22000))
        setv(feature, IDX["early_phase"], flag(frame_no < 5000))
        setv(feature, IDX["mid_phase"], flag(5000 <= frame_no < 13000))
        setv(feature, IDX["late_phase"], flag(frame_no >= 13000))
        setv(feature, IDX["direct_push_window"], flag(direct_push_window))
        setv(feature, IDX["defense_emergency"], flag(defense_emergency))
        setv(feature, IDX["resource_allowed"], flag(resource_allowed))

    def _fill_my_hero(self, feature, my_hero):
        setv(feature, IDX["my_hp_ratio"], self._hp_ratio(my_hero))
        setv(feature, IDX["my_level_norm"], n01((my_hero or {}).get("level", 1), 1, 15))
        setv(feature, IDX["my_gold_norm"], n01(self._money_value(my_hero), 0, 20000))
        setv(feature, IDX["my_alive"], flag(self._alive(my_hero)))
        x, z = self._pos(my_hero)
        setv(feature, 32, nsym(x, 60000))
        setv(feature, 33, nsym(z, 60000))
        setv(feature, 34, n01((my_hero or {}).get("attack_range", 0), 0, 12000))
        setv(feature, 35, n01((my_hero or {}).get("mov_spd", 0), 0, 1200))
        setv(feature, 36, n01((my_hero or {}).get("phy_atk", 0), 0, 1500))
        setv(feature, 37, n01((my_hero or {}).get("phy_def", 0), 0, 1500))
        setv(feature, 38, n01((my_hero or {}).get("mgc_def", 0), 0, 1500))
        setv(feature, 39, n01((my_hero or {}).get("kill_cnt", 0), 0, 10))
        setv(feature, 40, n01((my_hero or {}).get("dead_cnt", 0), 0, 10))
        setv(feature, 41, flag((my_hero or {}).get("is_in_grass", False)))

    def _fill_enemy(self, feature, frame_no, enemy_hero, memory):
        enemy_seen = memory["enemy_last_seen"]
        observed = enemy_hero is not None
        setv(feature, IDX["enemy_observed"], flag(observed))
        missing = 0 if observed else max(0, frame_no - int(enemy_seen.get("frame", -1)))
        setv(feature, IDX["enemy_not_observed_steps"], n01(missing, 0, 1200))
        setv(feature, IDX["enemy_alive"], flag(self._alive(enemy_hero)) if observed else 0.0)
        setv(feature, IDX["enemy_hp_ratio"], self._hp_ratio(enemy_hero) if observed else 0.0)
        if observed:
            ex, ez = self._pos(enemy_hero)
            setv(feature, 88, nsym(ex, 60000))
            setv(feature, 89, nsym(ez, 60000))
            setv(feature, 90, n01((enemy_hero or {}).get("level", 1), 1, 15))
            setv(feature, 91, n01(self._money_value(enemy_hero), 0, 20000))
        setv(feature, IDX["enemy_last_seen_x"], nsym(enemy_seen.get("x", 0), 60000))
        setv(feature, IDX["enemy_last_seen_z"], nsym(enemy_seen.get("z", 0), 60000))
        setv(feature, IDX["enemy_last_seen_time_norm"], n01(missing, 0, 1200))
        setv(feature, 99, safe_float(enemy_seen.get("hp_ratio", 0.0)))
        setv(feature, 100, safe_float(enemy_seen.get("near_grass", 0.0)))

    def _fill_skill_combo(self, feature, my_hero, memory):
        slots = (((my_hero or {}).get("skill_state") or {}).get("slot_states", []) or [])
        normal_slots = [slot for slot in slots if slot.get("configId") not in (80102, 80103, 80104, 80105, 80107, 80108, 80109, 80110, 80115, 80121)]
        normal_slots = sorted(normal_slots, key=lambda item: item.get("slot_type", 0))
        for i in range(4):
            slot = normal_slots[i] if i < len(normal_slots) else {}
            base = 144 + i * 8
            cooldown_max = max(1.0, safe_float(slot.get("cooldown_max", 1.0), 1.0))
            setv(feature, base, n01(slot.get("level", 0), 0, 6))
            setv(feature, base + 1, flag(slot.get("usable", False)))
            setv(feature, base + 2, n01(slot.get("cooldown", 0), 0, cooldown_max))
            setv(feature, base + 3, flag(slot.get("succUsedInFrame", 0) > 0))
        combo = memory["combo_memory"]
        setv(feature, 184, n01(combo.get("steps_since_last_skill", 999), 0, 120))
        setv(feature, 185, n01(combo.get("steps_since_last_attack", 999), 0, 120))
        setv(feature, 186, combo.get("enhanced_attack_ready", 0.0))
        setv(feature, 187, combo.get("enhanced_attack_confidence", 0.0))
        setv(feature, 188, combo.get("combo_window_active", 0.0))

    def _fill_tower(self, feature, my_hero, own_tower, enemy_tower, friendly_minions, memory):
        tower_memory = memory["tower_memory"]
        setv(feature, IDX["enemy_tower_hp_ratio"], self._hp_ratio(enemy_tower))
        setv(feature, IDX["own_tower_hp_ratio"], self._hp_ratio(own_tower))
        my_pos = self._pos(my_hero)
        setv(feature, 209, n01(self._dist(my_pos, self._pos(enemy_tower)), 0, 30000))
        setv(feature, 225, n01(self._dist(my_pos, self._pos(own_tower)), 0, 30000))
        setv(feature, 210, flag(self._tower_targets_self(my_hero, enemy_tower)))
        setv(feature, 211, flag(self._friendly_minion_tanking(enemy_tower, friendly_minions)))
        setv(feature, 212, flag(self._can_attack_tower(my_hero, enemy_tower)))
        setv(feature, 213, n01(tower_memory.get("enemy_tower_hp_delta_1", 0.0), 0, 0.2))
        setv(feature, 226, n01(tower_memory.get("own_tower_hp_delta_1", 0.0), 0, 0.2))
        setv(feature, 240, n01(tower_memory.get("estimated_tower_range", RuleConfig.DEFAULT_TOWER_RANGE), 0, 12000))
        setv(feature, 241, tower_memory.get("tower_range_confidence", 0.0))
        setv(feature, 488, self._tower_risk(my_hero, enemy_tower))

    def _fill_lane(self, feature, my_hero, own_tower, enemy_tower, friendly_minions, enemy_minions):
        setv(feature, IDX["friendly_minion_count"], n01(len(friendly_minions), 0, 8))
        setv(feature, IDX["enemy_minion_count"], n01(len(enemy_minions), 0, 8))
        setv(feature, 274, n01(sum(self._hp_ratio(u) for u in friendly_minions), 0, 8))
        setv(feature, 275, n01(sum(self._hp_ratio(u) for u in enemy_minions), 0, 8))
        my_pos = self._pos(my_hero)
        enemy_sorted = sorted(enemy_minions, key=lambda unit: self._dist(self._pos(unit), my_pos))
        friendly_sorted = sorted(friendly_minions, key=lambda unit: self._dist(self._pos(unit), my_pos))
        for i, unit in enumerate(enemy_sorted[:6]):
            base = 288 + i * 4
            setv(feature, base, self._hp_ratio(unit))
            setv(feature, base + 1, n01(self._dist(self._pos(unit), my_pos), 0, 20000))
            setv(feature, base + 2, n01(self._dist(self._pos(unit), self._pos(own_tower)), 0, 30000))
            setv(feature, base + 3, flag(unit.get("attack_target") == (my_hero or {}).get("runtime_id")))
        for i, unit in enumerate(friendly_sorted[:6]):
            base = 320 + i * 4
            setv(feature, base, self._hp_ratio(unit))
            setv(feature, base + 1, n01(self._dist(self._pos(unit), self._pos(enemy_tower)), 0, 30000))

    def _fill_resource(self, feature, my_hero, enemy_hero, cakes, neutral_units, memory):
        setv(feature, IDX["cake_observed"], flag(cakes))
        if cakes and my_hero:
            cake = min(cakes, key=lambda c: self._dist(self._pos(my_hero), self._cake_pos(c)))
            cx, cz = self._cake_pos(cake)
            setv(feature, 361, nsym(cx, 60000))
            setv(feature, 362, nsym(cz, 60000))
            setv(feature, 363, n01(self._dist(self._pos(my_hero), (cx, cz)), 0, 30000))
            safe = self._hp_ratio(my_hero) < RuleConfig.BLOOD_PACK_HP_RATIO
            if enemy_hero:
                safe = safe and self._dist(self._pos(enemy_hero), (cx, cz)) > RuleConfig.ENEMY_NEAR_DIST
            setv(feature, 364, flag(safe))
        cake_memory = memory["cake_memory"]
        setv(feature, 365, cake_memory.get("cake_effect_confidence", 0.0))
        setv(feature, IDX["neutral_observed"], flag(neutral_units))
        if neutral_units and my_hero:
            neutral = min(neutral_units, key=lambda n: self._dist(self._pos(my_hero), self._pos(n)))
            setv(feature, 377, self._hp_ratio(neutral))
            setv(feature, 378, n01(self._dist(self._pos(my_hero), self._pos(neutral)), 0, 30000))
            setv(feature, 379, flag(self._hp_ratio(neutral) < 0.25))
        neutral_memory = memory["neutral_memory"]
        setv(feature, 392, neutral_memory.get("neutral_spawn_estimate", 0.0))
        setv(feature, 393, neutral_memory.get("neutral_spawn_confidence", 0.0))

    def _fill_vision(self, feature, my_hero, enemy_hero):
        setv(feature, 408, flag((my_hero or {}).get("is_in_grass", False)))
        setv(feature, 409, flag((enemy_hero or {}).get("is_in_grass", False)) if enemy_hero else 0.0)
        setv(feature, 410, n01(self._dist(self._pos(my_hero), self._pos(enemy_hero)), 0, 30000) if enemy_hero else 1.0)

    def _fill_opponent_behavior(self, feature, memory):
        behavior = memory["opponent_behavior"]
        setv(feature, IDX["enemy_attack_hero_rate_30"], behavior.get("attack_hero_rate_30", 0.0))
        setv(feature, IDX["enemy_attack_tower_rate_30"], behavior.get("attack_tower_rate_30", 0.0))
        setv(feature, 442, behavior.get("clear_wave_rate_30", 0.0))
        setv(feature, 443, behavior.get("near_neutral_rate_60", 0.0))
        setv(feature, 444, behavior.get("missing_rate_60", 0.0))
        setv(feature, 445, behavior.get("damage_to_my_tower_rate_60", 0.0))
        setv(feature, 446, behavior.get("behavior_change_score", 0.0))

    def _fill_rule_debug(self, feature, direct_push_window, defense_emergency, resource_allowed, legal_action):
        setv(feature, 472, flag(direct_push_window))
        setv(feature, 473, flag(defense_emergency))
        setv(feature, 474, flag(resource_allowed))
        if legal_action is not None:
            arr = np.asarray(legal_action)
            setv(feature, 475, n01(arr.sum(), 0, max(1, arr.size)))

    def _direct_push_window(self, my_hero, enemy_hero, enemy_tower, friendly_minions):
        enemy_dead = enemy_hero is not None and not self._alive(enemy_hero)
        tower_finish = enemy_tower is not None and self._hp_ratio(enemy_tower) < RuleConfig.FINISH_TOWER_HP_RATIO
        minion_tanking = self._friendly_minion_tanking(enemy_tower, friendly_minions)
        enhanced_can_hit = self._can_attack_tower(my_hero, enemy_tower)
        return bool(enemy_dead or tower_finish or minion_tanking or enhanced_can_hit)

    def _defense_emergency(self, own_tower, enemy_minions, memory):
        own_tower_low = own_tower is not None and self._hp_ratio(own_tower) < RuleConfig.LOW_HP_RISK_RATIO
        own_tower_damaged = memory["tower_memory"].get("own_tower_hp_delta_1", 0.0) > 0.01
        return bool(own_tower_low or (enemy_minions and own_tower_damaged))

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
        my_camp = (my_hero or {}).get("camp", player_camp)
        for hero in heroes:
            if hero is not my_hero and not self._same_camp(hero.get("camp"), my_camp):
                enemy_hero = hero
                break
        return my_hero, enemy_hero

    def _split_npcs(self, frame_state, my_camp):
        own_tower, enemy_tower, friendly_minions, enemy_minions, neutral_units = None, None, [], [], []
        for npc in frame_state.get("npc_states", []) or []:
            camp = npc.get("camp")
            if self._is_tower(npc):
                if self._same_camp(camp, my_camp):
                    own_tower = npc
                else:
                    enemy_tower = npc
            elif self._is_monster(npc):
                neutral_units.append(npc)
            elif self._same_camp(camp, my_camp):
                friendly_minions.append(npc)
            elif camp not in (0, None, "0"):
                enemy_minions.append(npc)
        return own_tower, enemy_tower, friendly_minions, enemy_minions, neutral_units

    def _is_tower(self, npc):
        sub_type = npc.get("sub_type")
        return sub_type == 21 or "TOWER" in str(sub_type).upper()

    def _is_monster(self, npc):
        actor_type = str(npc.get("actor_type", "")).upper()
        sub_type = str(npc.get("sub_type", "")).upper()
        return "MONSTER" in actor_type or "MONSTER" in sub_type or self._camp_value(npc.get("camp")) in (0, None, "0")

    def _tower_targets_self(self, my_hero, enemy_tower):
        return bool(my_hero and enemy_tower and enemy_tower.get("attack_target") == my_hero.get("runtime_id"))

    def _tower_risk(self, my_hero, enemy_tower):
        if not my_hero or not enemy_tower:
            return 0.0
        dist = self._dist(self._pos(my_hero), self._pos(enemy_tower))
        tower_range = safe_float(enemy_tower.get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE))
        smooth = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, (tower_range - dist) / RuleConfig.TOWER_RISK_SIGMOID_SCALE))))
        target_self = 1.0 if enemy_tower.get("attack_target") == my_hero.get("runtime_id") else 0.0
        low_hp = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, (RuleConfig.LOW_HP_RISK_RATIO - self._hp_ratio(my_hero)) / 0.08))))
        return float(np.clip(smooth * (0.35 + 0.45 * target_self + 0.20 * low_hp), 0.0, 1.0))

    def _friendly_minion_tanking(self, enemy_tower, friendly_minions):
        target = (enemy_tower or {}).get("attack_target")
        return target is not None and target in {unit.get("runtime_id") for unit in friendly_minions}

    def _can_attack_tower(self, my_hero, enemy_tower):
        if not my_hero or not enemy_tower:
            return False
        return self._dist(self._pos(my_hero), self._pos(enemy_tower)) <= safe_float(my_hero.get("attack_range", 0)) + RuleConfig.ATTACK_TOWER_EXTRA_RANGE

    def _hp_ratio(self, obj):
        return float(np.clip(safe_float((obj or {}).get("hp", 0)) / (safe_float((obj or {}).get("max_hp", 0)) + 1e-6), 0.0, 1.0))

    def _alive(self, obj):
        return bool(obj and safe_float(obj.get("hp", 0)) > 0)

    def _money_value(self, hero):
        return safe_float((hero or {}).get("money_cnt", (hero or {}).get("money", 0)))

    def _pos(self, obj):
        loc = (obj or {}).get("location", {}) or {}
        return safe_float(loc.get("x", 100000), 100000), safe_float(loc.get("z", 100000), 100000)

    def _cake_pos(self, cake):
        loc = ((cake.get("collider", {}) or {}).get("location", {}) or {})
        return safe_float(loc.get("x", 100000), 100000), safe_float(loc.get("z", 100000), 100000)

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


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.memory = MemoryProcess()
        self.builder = FeatureBuilder(self.memory, Config)
        self.last_aux = {}

    def reset(self, camp):
        self.camp = camp
        self.memory.reset()
        self.last_aux = {}

    def process_feature(self, observation):
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = frame_state.get("frame_no", observation.get("frame_no", 0))
        self.memory.update_before_action(observation, frame_no)
        feature, aux = self.builder.build(
            observation,
            frame_no,
            legal_action=observation.get("legal_action"),
            sub_action_mask=observation.get("sub_action_mask"),
        )
        self.last_aux = aux
        return feature

    def update_after_action(self, observation, action, reward_aux=None):
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = frame_state.get("frame_no", observation.get("frame_no", 0))
        self.memory.update_after_action(observation, action, reward_aux or {}, frame_no)


def setv(feature, index, value, lo=0.0, hi=1.0):
    if 0 <= index < len(feature):
        feature[index] = np.clip(safe_float(value), lo, hi)
