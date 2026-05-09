#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""历史记忆模块，只输出事实和带置信度的估计，不直接决策。"""

import math
from collections import deque

from agent_ppo.feature.cake_rune_explorer import CakeRuneExplorer
from agent_ppo.feature.neutral_objective import NeutralObjectiveMemory
from agent_ppo.feature.opponent_behavior_memory import OpponentBehaviorMemory
from agent_ppo.feature.tower_range_explorer import TowerRangeExplorer


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
        self.lane_history_30 = deque(maxlen=30)
        self.lane_history_60 = deque(maxlen=60)
        self.enemy_tower_hp_history = deque(maxlen=60)
        self.own_tower_hp_history = deque(maxlen=60)
        self.last_hit_count = 0.0
        self.tower_memory = {
            "enemy_tower_hp_last": 1.0,
            "own_tower_hp_last": 1.0,
            "enemy_tower_hp_delta_1": 0.0,
            "own_tower_hp_delta_1": 0.0,
            "tower_aggro_count": 0.0,
            "estimated_tower_range": 8500.0,
            "tower_range_confidence": 0.0,
            "enemy_tower_hp_delta_30": 0.0,
            "own_tower_hp_delta_30": 0.0,
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
        self.tower_range_explorer = TowerRangeExplorer()
        self.cake_rune_explorer = CakeRuneExplorer()
        self.neutral_objective = NeutralObjectiveMemory()
        self.opponent_behavior_memory = OpponentBehaviorMemory()

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
        self.enemy_tower_hp_history.append(enemy_tower_hp)
        self.own_tower_hp_history.append(own_tower_hp)
        self.tower_memory["enemy_tower_hp_delta_1"] = max(
            0.0, self.tower_memory["enemy_tower_hp_last"] - enemy_tower_hp
        )
        self.tower_memory["own_tower_hp_delta_1"] = max(0.0, self.tower_memory["own_tower_hp_last"] - own_tower_hp)
        self.tower_memory["enemy_tower_hp_delta_30"] = max(
            0.0, (self.enemy_tower_hp_history[0] if self.enemy_tower_hp_history else enemy_tower_hp) - enemy_tower_hp
        )
        self.tower_memory["own_tower_hp_delta_30"] = max(
            0.0, (self.own_tower_hp_history[0] if self.own_tower_hp_history else own_tower_hp) - own_tower_hp
        )
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
        self.opponent_behavior_memory.update(frame_no, observation, enemy_hero is not None, event)

        lane_score = self._lane_score(own_tower, enemy_tower, friendly_minions, enemy_minions)
        self.lane_history_30.append(lane_score)
        self.lane_history_60.append(lane_score)
        self.last_hit_count += self._last_hit_count(frame_state, observation.get("player_id"))

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
        self.tower_range_explorer.sample(
            frame_no,
            my_hero or {},
            self._pos(enemy_tower),
            self._pos(own_tower),
            (enemy_tower or {}).get("attack_target"),
            (own_tower or {}).get("attack_target"),
            friendly_minions,
            enemy_minions,
        )
        self.cake_rune_explorer.sample(
            frame_no,
            cakes,
            self._pos(my_hero),
            self._pos(enemy_hero) if enemy_hero else None,
            self._hp_ratio(my_hero),
            (my_hero or {}).get("mov_spd", 0),
            ((my_hero or {}).get("buff_state") or {}),
            [
                slot.get("cooldown", 0)
                for slot in (((my_hero or {}).get("skill_state") or {}).get("slot_states", []) or [])
            ],
        )
        self.neutral_objective.sample(
            frame_no,
            monsters,
            self._pos(my_hero),
            self._pos(enemy_hero) if enemy_hero else None,
            event,
        )

    def update_after_action(self, observation, action, reward_aux, frame_no):
        # 预留给 cake 效果、资源击杀等跨帧结果估计。
        if reward_aux and reward_aux.get("cake_safe_pick", 0.0) > 0:
            self.cake_memory["last_cake_pick_frame"] = frame_no
            self.cake_memory["cake_effect_confidence"] = max(self.cake_memory["cake_effect_confidence"], 0.2)
        if reward_aux and reward_aux.get("neutral_result", 0.0) > 0:
            self.neutral_memory["neutral_recently_killed"] = 1.0

    def export_features(self):
        tower_export = self.tower_range_explorer.export()
        cake_export = self.cake_rune_explorer.export()
        neutral_export = self.neutral_objective.export()
        opponent_export = self.opponent_behavior_memory.export()
        tower_memory = dict(self.tower_memory)
        tower_memory["estimated_tower_range"] = tower_export["estimated_enemy_tower_range"]
        tower_memory["estimated_own_tower_range"] = tower_export["estimated_own_tower_range"]
        tower_memory["tower_range_confidence"] = tower_export["tower_range_confidence"]
        tower_memory["self_in_enemy_tower_range_estimated"] = tower_export["self_in_enemy_tower_range_estimated"]
        tower_memory["safe_distance_to_enemy_tower"] = tower_export["safe_distance_to_enemy_tower"]
        cake_memory = dict(self.cake_memory)
        cake_memory.update(
            {
                "cake_effect_confidence": cake_export["cake_effect_confidence"],
                "hp_delta_after_cake": cake_export["hp_delta_after_cake"],
                "speed_delta_after_cake": cake_export["speed_delta_after_cake"],
                "buff_added_after_cake": cake_export["buff_added_after_cake"],
            }
        )
        neutral_memory = dict(self.neutral_memory)
        neutral_memory.update(
            {
                "neutral_last_seen": neutral_export["neutral_observed"],
                "neutral_hp_ratio": neutral_export["neutral_hp_ratio"],
                "neutral_dist_to_self": neutral_export["neutral_dist_to_self"],
                "neutral_dist_to_enemy": neutral_export["neutral_dist_to_enemy"],
                "neutral_low_hp": neutral_export["neutral_low_hp"],
                "neutral_contested_score": neutral_export["neutral_contested_score"],
                "neutral_safe_to_take": neutral_export["neutral_safe_to_take"],
                "neutral_recently_killed": neutral_export["neutral_recently_killed"],
                "neutral_spawn_estimate": neutral_export["neutral_spawn_estimate"],
                "neutral_time_to_spawn": neutral_export["neutral_time_to_spawn"],
                "neutral_spawn_confidence": neutral_export["neutral_spawn_confidence"],
            }
        )
        lane_30_start = self.lane_history_30[0] if self.lane_history_30 else 0.0
        lane_60_start = self.lane_history_60[0] if self.lane_history_60 else 0.0
        lane_now = self.lane_history_30[-1] if self.lane_history_30 else 0.0
        return {
            "enemy_last_seen": dict(self.enemy_last_seen),
            "opponent_behavior": {
                "attack_hero_rate_30": opponent_export["enemy_attack_hero_rate_30"],
                "attack_tower_rate_30": opponent_export["enemy_attack_tower_rate_30"],
                "clear_wave_rate_30": opponent_export["enemy_clear_wave_rate_30"],
                "near_neutral_rate_60": opponent_export["enemy_near_neutral_rate_60"],
                "missing_rate_60": opponent_export["enemy_missing_rate_60"],
                "damage_to_my_tower_rate_60": opponent_export["enemy_damage_to_my_tower_rate_60"],
                "behavior_change_score": opponent_export["enemy_behavior_change_score"],
            },
            "tower_memory": tower_memory,
            "combo_memory": dict(self.combo_memory),
            "cake_memory": cake_memory,
            "neutral_memory": neutral_memory,
            "lane_memory": {
                "lane_push_delta_30": lane_now - lane_30_start,
                "lane_push_delta_60": lane_now - lane_60_start,
                "last_hit_count": self.last_hit_count,
                "recent_clear_wave_efficiency": max(0.0, lane_now - lane_30_start),
            },
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
            if (
                hero is not my_hero
                and not self._same_camp(hero.get("camp"), my_camp)
                and self._visible_to_camp(hero, my_camp)
            ):
                enemy_hero = hero
                break
        return my_hero, enemy_hero

    def _visible_to_camp(self, obj, camp):
        visible = (obj or {}).get("camp_visible", None)
        camp_idx = self._camp_value(camp)
        if isinstance(visible, list) and camp_idx in (1, 2) and len(visible) >= camp_idx:
            return bool(visible[camp_idx - 1])
        return obj is not None

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

    def _lane_score(self, own_tower, enemy_tower, friendly_minions, enemy_minions):
        enemy_tower_pos = self._pos(enemy_tower)
        own_tower_pos = self._pos(own_tower)
        friendly_front = min((self._dist(self._pos(unit), enemy_tower_pos) for unit in friendly_minions), default=30000.0)
        enemy_front = min((self._dist(self._pos(unit), own_tower_pos) for unit in enemy_minions), default=30000.0)
        return (
            0.35 * self._safe_div(len(friendly_minions) - len(enemy_minions), 8.0)
            + 0.35 * (1.0 - self._safe_div(friendly_front, 30000.0))
            - 0.30 * (1.0 - self._safe_div(enemy_front, 30000.0))
        )

    def _last_hit_count(self, frame_state, player_id):
        cnt = 0.0
        frame_action = frame_state.get("frame_action", {}) or {}
        for dead in frame_action.get("dead_action", []) or []:
            killer = dead.get("killer", {}) or {}
            death = dead.get("death", {}) or {}
            if killer.get("runtime_id") != player_id:
                continue
            if death.get("actor_type") == "ACTOR_TYPE_HERO":
                continue
            if "TOWER" in str(death.get("sub_type", "")).upper():
                continue
            cnt += 1.0
        return cnt

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
