#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Final reward process for D401-256.

Returns per-item raw rewards, weighted 3-group rewards, and reward_sum.
The policy/value pipeline consumes reward_groups in feature.definition.
"""

import math
from typing import Any, Dict, List, Optional

from agent_ppo.conf.conf import GameConfig

DIST_NORM = 60000.0


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_any(obj, *keys, default=None):
    for key in keys:
        val = _get(obj, key, None)
        if val is not None:
            return val
    return default


def _to_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return default


def _loc(actor):
    loc = _get_any(actor, "location", default={}) or {}
    if isinstance(loc, dict):
        return {"x": float(loc.get("x", 0) or 0), "z": float(loc.get("z", 0) or 0)}
    return {"x": 0.0, "z": 0.0}


def _dist(a, b):
    if a is None or b is None:
        return DIST_NORM
    la, lb = _loc(a), _loc(b)
    dx, dz = la["x"] - lb["x"], la["z"] - lb["z"]
    return math.sqrt(dx * dx + dz * dz)


def _hp_rate(actor):
    hp = float(_get_any(actor, "hp", default=0) or 0)
    max_hp = max(1.0, float(_get_any(actor, "max_hp", "maxHp", default=1) or 1))
    return max(0.0, min(1.0, hp / max_hp))


def _alive(actor):
    return float(_get_any(actor, "hp", default=0) or 0) > 0


def _is_visible_to(actor, camp):
    if actor is None:
        return False
    visible = _get_any(actor, "camp_visible", "campVisible", default=None)
    if isinstance(visible, list) and len(visible) >= 2:
        try:
            idx = 0 if int(camp) == 1 else 1
            return bool(visible[idx])
        except Exception:
            return True
    loc = _loc(actor)
    if abs(loc["x"]) > 90000 or abs(loc["z"]) > 90000:
        return False
    return True


def _runtime_id(actor):
    return _get_any(actor, "runtime_id", "runtimeId", default=None)


def _config_id(actor):
    return _get_any(actor, "config_id", "configId", default=None)


def _sub_type(actor):
    return _get_any(actor, "sub_type", "subType", default=None)


def _camp(actor):
    return _get_any(actor, "camp", default=None)


def _is_soldier(npc):
    return npc is not None and _alive(npc) and (
        _to_int(_sub_type(npc)) in GameConfig.SOLDIER_SUB_TYPES
        or _to_int(_config_id(npc)) in GameConfig.SOLDIER_CONFIG_IDS
    )


def _is_tower(npc):
    return npc is not None and _alive(npc) and (
        _to_int(_sub_type(npc)) in GameConfig.TOWER_SUB_TYPES
        or _to_int(_config_id(npc)) in GameConfig.TOWER_CONFIG_IDS
    )


def _is_monster(npc):
    return npc is not None and _alive(npc) and _to_int(_config_id(npc)) in GameConfig.MONSTER_CONFIG_IDS


def _slot_states(hero):
    state = _get_any(hero, "skill_state", default={}) or {}
    return _get(state, "slot_states", []) or []


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.prev = {}
        self.has_seen_minions = False
        self.prev_enemy_soldier_count = 0
        self.prev_friendly_soldier_count = 0
        self.prev_near_own_enemy_soldier_count = 0
        self.prev_enemy_soldier_ids = set()
        self.prev_friendly_soldier_ids = set()
        self.prev_lane_visible = False
        self.last_self_pos = None
        self.no_effective_action_steps = 0
        self.grass_steps = 0
        self.grass_no_effective_steps = 0
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG

    def _find_heroes(self, frame_data):
        main, enemy = None, None
        for hero in frame_data.get("hero_states", []):
            rid = _runtime_id(hero)
            pid = _get_any(hero, "player_id", default=None)
            if rid == self.main_hero_player_id or pid == self.main_hero_player_id:
                main = hero
                self.main_hero_camp = _camp(hero)
            else:
                enemy = hero
        # Fallback by cached camp.
        if main is None and self.main_hero_camp != -1:
            for hero in frame_data.get("hero_states", []):
                if _camp(hero) == self.main_hero_camp:
                    main = hero
                else:
                    enemy = hero
        return main, enemy

    def _split_npcs(self, frame_data, camp):
        soldiers, friendly_soldiers, enemy_soldiers, towers, monsters = [], [], [], [], []
        for npc in frame_data.get("npc_states", []):
            if _is_soldier(npc):
                soldiers.append(npc)
                if _camp(npc) == camp:
                    friendly_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
            elif _is_tower(npc):
                towers.append(npc)
            elif _is_monster(npc):
                monsters.append(npc)
        own_tower, enemy_tower = None, None
        for t in towers:
            if _camp(t) == camp:
                own_tower = t
            else:
                enemy_tower = t
        return friendly_soldiers, enemy_soldiers, own_tower, enemy_tower, monsters

    def _split_cakes(self, frame_data, own_tower, enemy_tower):
        cakes = []
        for cake in frame_data.get("cakes", []) or []:
            if _to_int(_get_any(cake, "configId", "config_id", default=-1)) not in GameConfig.CAKE_CONFIG_IDS:
                continue
            collider = _get(cake, "collider", {}) or {}
            loc = _get(collider, "location", {}) or {}
            if not isinstance(loc, dict):
                continue
            cakes.append((cake, {"x": float(loc.get("x", 0) or 0), "z": float(loc.get("z", 0) or 0)}))
        if not cakes:
            return None, None
        own_pos = _loc(own_tower) if own_tower is not None else {"x": -15000.0, "z": -15000.0}
        enemy_pos = _loc(enemy_tower) if enemy_tower is not None else {"x": 15000.0, "z": 15000.0}
        own = min(cakes, key=lambda item: math.hypot(item[1]["x"] - own_pos["x"], item[1]["z"] - own_pos["z"]))
        remain = [c for c in cakes if c is not own]
        enemy = min(remain, key=lambda item: math.hypot(item[1]["x"] - enemy_pos["x"], item[1]["z"] - enemy_pos["z"])) if remain else None
        return own, enemy

    def _metric(self, key, cur, scale=1.0):
        last = self.prev.get(key, cur)
        self.prev[key] = cur
        return (cur - last) / max(1e-6, scale)

    def _tracked_hp_rate(self, key, actor, default=1.0):
        """Use last visible hp for fogged actors instead of treating them as dead."""
        if actor is None:
            return float(self.prev.get(key, default))
        cur = _hp_rate(actor)
        self.prev[key] = cur
        return cur

    def _tracked_dist(self, key, a, b, default=DIST_NORM):
        """Use last visible distance when a known objective is temporarily hidden."""
        if a is None or b is None:
            return float(self.prev.get(key, default))
        cur = _dist(a, b)
        self.prev[key] = cur
        return cur

    def _has_real_cmd(self, hero):
        return len(_get_any(hero, "real_cmd", default=[]) or []) > 0

    def _hit_any(self, hero):
        return len(_get_any(hero, "hit_target_info", default=[]) or []) > 0

    def _hit_target_ids(self, hero):
        ids = set()
        for info in _get_any(hero, "hit_target_info", "hitTargetInfo", default=[]) or []:
            target_id = _get_any(info, "hit_target", "hitTarget", "runtime_id", "runtimeId", default=None)
            if target_id is not None:
                ids.add(target_id)
        return ids

    def _update_stuck_grass(self, hero, defense_emergency, enemy_soldier_reduced):
        pos = _loc(hero)
        if self.last_self_pos is None:
            moved = DIST_NORM
        else:
            moved = math.hypot(pos["x"] - self.last_self_pos[0], pos["z"] - self.last_self_pos[1])
        self.last_self_pos = (pos["x"], pos["z"])
        has_cmd = self._has_real_cmd(hero)
        hit_any = self._hit_any(hero)
        if _alive(hero) and moved < 100 and (not has_cmd) and (not hit_any):
            self.no_effective_action_steps += 1
        else:
            self.no_effective_action_steps = 0
        stuck = 0.0
        if self.no_effective_action_steps >= 4:
            stuck = min(1.0, (self.no_effective_action_steps - 3) / 9.0)

        in_grass = bool(_get_any(hero, "is_in_grass", default=False))
        grass_behavior = 0.0
        if in_grass:
            self.grass_steps += 1
            if not has_cmd and not hit_any and enemy_soldier_reduced <= 0:
                self.grass_no_effective_steps += 1
            else:
                self.grass_no_effective_steps = 0
            if hit_any:
                grass_behavior += 0.3
            if enemy_soldier_reduced > 0:
                grass_behavior += 0.2
            if defense_emergency and self.grass_steps >= 6 and enemy_soldier_reduced <= 0:
                grass_behavior -= 1.0
            if self.grass_no_effective_steps >= 8:
                grass_behavior -= 1.0
        else:
            self.grass_steps = 0
            self.grass_no_effective_steps = 0
        return stuck, grass_behavior

    def _calc_skill_hit(self, hero):
        # Reward effective hits on any target; do not punish non-hero skill usage.
        return 1.0 if self._hit_any(hero) else 0.0

    def _calc_cake(self, hero, own_cake):
        cur_hp = _hp_rate(hero)
        own_cake_exists = own_cake is not None
        if own_cake is not None:
            loc = own_cake[1]
            pos = _loc(hero)
            dist = math.hypot(loc["x"] - pos["x"], loc["z"] - pos["z"])
        else:
            dist = DIST_NORM
        prev_hp = self.prev.get("my_hp_ratio_for_cake", cur_hp)
        prev_dist = self.prev.get("own_cake_dist", dist)
        prev_exists = self.prev.get("own_cake_exists", 0.0) > 0.5
        self.prev["my_hp_ratio_for_cake"] = cur_hp
        self.prev["own_cake_dist"] = dist
        self.prev["own_cake_exists"] = 1.0 if own_cake_exists else 0.0
        picked = prev_exists and prev_dist < 2000 and cur_hp > prev_hp + 0.03
        if picked and prev_hp < GameConfig.CAKE_LOW_HP_THRESHOLD:
            return 1.0
        if picked and prev_hp < GameConfig.CAKE_MEDIUM_HP_THRESHOLD:
            return 0.3
        if cur_hp < GameConfig.CAKE_LOW_HP_THRESHOLD and own_cake_exists:
            return max(0.0, (prev_dist - dist) / 10000.0) * 0.2
        return 0.0

    def result(self, frame_data):
        frame_no = int(frame_data.get("frame_no", 0) or 0)
        hero, enemy = self._find_heroes(frame_data)
        if hero is None:
            return {"reward_sum": 0.0, "reward_groups": {name: 0.0 for name in GameConfig.REWARD_GROUP_NAMES}}
        camp = _camp(hero)
        friendly_soldiers, enemy_soldiers, own_tower, enemy_tower, monsters = self._split_npcs(frame_data, camp)
        own_cake, enemy_cake = self._split_cakes(frame_data, own_tower, enemy_tower)

        # Main deltas/potentials.
        my_hp = _hp_rate(hero)
        own_tower_hp = self._tracked_hp_rate("tracked_own_tower_hp", own_tower, 1.0)
        enemy_tower_hp = self._tracked_hp_rate("tracked_enemy_tower_hp", enemy_tower, 1.0)
        enemy_visible = enemy is not None and _is_visible_to(enemy, camp)
        enemy_hp = self._tracked_hp_rate("tracked_enemy_hp", enemy if enemy_visible else None, 1.0)
        tower_score = own_tower_hp - enemy_tower_hp
        hp_score = my_hp - enemy_hp
        money_score = float(_get_any(hero, "money_cnt", "money", default=0) or 0)
        exp_score = float(_get_any(hero, "level", default=0) or 0) * 2000.0 + float(_get_any(hero, "exp", default=0) or 0)
        kill_cnt = float(_get_any(hero, "kill_cnt", default=0) or 0)
        death_cnt = float(_get_any(hero, "dead_cnt", default=0) or 0)
        total_damage = float(_get_any(hero, "total_hurt", default=0) or 0)
        hero_damage = float(_get_any(hero, "total_hurt_to_hero", default=0) or 0)
        hero_hurt = float(_get_any(hero, "total_be_hurt_by_hero", default=0) or 0)

        enemy_count, friendly_count = len(enemy_soldiers), len(friendly_soldiers)
        total_soldier_count = enemy_count + friendly_count
        lane_visible = total_soldier_count > 0
        current_enemy_soldier_ids = set(_runtime_id(s) for s in enemy_soldiers)
        current_friendly_soldier_ids = set(_runtime_id(s) for s in friendly_soldiers)
        hit_prev_enemy_soldier = len(self.prev_enemy_soldier_ids & self._hit_target_ids(hero)) > 0
        if not lane_visible:
            enemy_soldier_reduced = 1.0 if self.has_seen_minions and self.prev_enemy_soldier_count > 0 and hit_prev_enemy_soldier else 0.0
            lane_adv_delta = 0.0
            self.has_seen_minions = False
            self.prev_enemy_soldier_count = 0
            self.prev_friendly_soldier_count = 0
            self.prev_enemy_soldier_ids = set()
            self.prev_friendly_soldier_ids = set()
        elif not self.has_seen_minions:
            self.has_seen_minions = True
            self.prev_enemy_soldier_count = enemy_count
            self.prev_friendly_soldier_count = friendly_count
            self.prev_enemy_soldier_ids = current_enemy_soldier_ids
            self.prev_friendly_soldier_ids = current_friendly_soldier_ids
            enemy_soldier_reduced = 0.0
            lane_adv_delta = 0.0
        else:
            enemy_soldier_reduced = max(0.0, float(self.prev_enemy_soldier_count - enemy_count))
            prev_adv = self.prev_friendly_soldier_count - self.prev_enemy_soldier_count
            cur_adv = friendly_count - enemy_count
            lane_adv_delta = float(cur_adv - prev_adv)
            self.prev_enemy_soldier_count = enemy_count
            self.prev_friendly_soldier_count = friendly_count
            self.prev_enemy_soldier_ids = current_enemy_soldier_ids
            self.prev_friendly_soldier_ids = current_friendly_soldier_ids
        self.prev_lane_visible = lane_visible

        own_range = float(_get_any(own_tower, "attack_range", default=8800) or 8800)
        enemy_near_own = [s for s in enemy_soldiers if _dist(s, own_tower) <= own_range * 1.15]
        near_own_reduced = 0.0 if not lane_visible else max(0.0, float(self.prev_near_own_enemy_soldier_count - len(enemy_near_own)))
        self.prev_near_own_enemy_soldier_count = len(enemy_near_own)
        enemy_ids = current_enemy_soldier_ids
        friendly_ids = current_friendly_soldier_ids
        own_tower_target_enemy_soldier = _get_any(own_tower, "attack_target", default=0) in enemy_ids
        defense_emergency = own_tower_target_enemy_soldier or len(enemy_near_own) > 0

        stuck, grass_behavior = self._update_stuck_grass(hero, defense_emergency, enemy_soldier_reduced)

        enemy_range = float(_get_any(enemy_tower, "attack_range", default=8800) or 8800)
        in_enemy_tower = _dist(hero, enemy_tower) <= enemy_range
        enemy_tower_target = _get_any(enemy_tower, "attack_target", default=0)
        friendly_tanking = enemy_tower_target in friendly_ids
        attacking_enemy_tower = _get_any(hero, "attack_target", default=0) == _runtime_id(enemy_tower)
        tower_risk = 0.0
        if in_enemy_tower and not friendly_tanking:
            tower_risk -= 1.0
        if enemy_tower_target == _runtime_id(hero):
            tower_risk -= 2.0
        if friendly_tanking and attacking_enemy_tower:
            tower_risk += 0.5

        defense = 0.0
        if defense_emergency:
            if near_own_reduced > 0:
                defense += near_own_reduced
            own_tower_hp_delta = self._metric("own_tower_hp_for_defense", own_tower_hp, 1.0)
            if own_tower_hp_delta < 0:
                defense -= 1.0
            if _dist(hero, own_tower) > 15000 and enemy_soldier_reduced <= 0 and not self._hit_any(hero):
                defense -= 0.2
        else:
            self.prev["own_tower_hp_for_defense"] = own_tower_hp

        lane_clear = enemy_soldier_reduced + lane_adv_delta * 0.3
        if defense_emergency and near_own_reduced > 0:
            lane_clear += 0.5 * near_own_reduced
        lane_clear = max(-1.0, min(3.0, lane_clear))

        no_ops = 0.0
        if _alive(hero) and (not self._has_real_cmd(hero)) and (not self._hit_any(hero)):
            no_ops = min(1.0, self.no_effective_action_steps / 10.0)
        dist_enemy_tower = self._tracked_dist("tracked_enemy_tower_dist", hero, enemy_tower, DIST_NORM)
        dist_own_tower = self._tracked_dist("tracked_own_tower_dist", hero, own_tower, DIST_NORM)
        prev_enemy_tower_dist = self.prev.get("enemy_tower_dist", dist_enemy_tower)
        prev_own_tower_dist = self.prev.get("own_tower_dist", dist_own_tower)
        self.prev["enemy_tower_dist"] = dist_enemy_tower
        self.prev["own_tower_dist"] = dist_own_tower
        forward = (prev_enemy_tower_dist - dist_enemy_tower) / 10000.0
        leave_home = (dist_own_tower - prev_own_tower_dist) / 10000.0
        forward += 0.3 * leave_home
        if frame_no > 300 and (not defense_emergency) and my_hp > 0.55 and dist_own_tower < 14000:
            forward -= 0.20
        forward = max(-1.0, min(1.0, forward))

        reward = {
            "tower_hp_point": self._metric("tower_score", tower_score, 1.0),
            "hp_point": self._metric("hp_score", hp_score, 1.0),
            "money": self._metric("money", money_score, 500.0),
            "exp": self._metric("exp", exp_score, 500.0),
            "last_hit": enemy_soldier_reduced,
            "lane_clear": lane_clear,
            "defense": defense,
            "cake": self._calc_cake(hero, own_cake),
            "tower_risk": tower_risk,
            "stuck": stuck,
            "no_ops": no_ops,
            "forward": forward,
            "grass_behavior": grass_behavior,
            "skill_hit": self._calc_skill_hit(hero),
            "hero_hurt": max(0.0, self._metric("hero_hurt", hero_hurt, 20000.0)),
            "total_damage": max(0.0, self._metric("total_damage", total_damage, 60000.0)),
            "hero_damage": max(0.0, self._metric("hero_damage", hero_damage, 20000.0)),
            "kill": max(0.0, self._metric("kill", kill_cnt, 1.0)),
            "death": max(0.0, self._metric("death", death_cnt, 1.0)),
            "bad_skill": 0.0,
            "passive_skills": 0.0,
            "crit": 0.0,
            "ep_rate": 0.0,
            "monster_resource": 0.0,
        }

        # Time decay for shaping items only.
        if self.time_scale_arg > 0:
            decay = math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
            for key in list(reward.keys()):
                if key not in GameConfig.NO_DECAY_REWARD_KEYS:
                    reward[key] *= decay

        reward_groups = {name: 0.0 for name in GameConfig.REWARD_GROUP_NAMES}
        assigned = set()
        for group, keys in GameConfig.REWARD_GROUPS.items():
            for key in keys:
                assigned.add(key)
                reward_groups[group] += reward.get(key, 0.0) * GameConfig.REWARD_WEIGHT_DICT.get(key, 0.0)
        # Keep any later ungrouped non-zero reward in behavior_safety.
        for key, value in reward.items():
            if key in assigned:
                continue
            reward_groups["behavior_safety"] = reward_groups.get("behavior_safety", 0.0) + value * GameConfig.REWARD_WEIGHT_DICT.get(key, 0.0)
        reward_sum = sum(reward_groups.values())
        reward["reward_groups"] = reward_groups
        reward["reward_objective"] = reward_groups.get("objective", 0.0)
        reward["reward_growth_combat"] = reward_groups.get("growth_combat", 0.0)
        reward["reward_behavior_safety"] = reward_groups.get("behavior_safety", 0.0)
        reward["reward_sum"] = reward_sum
        return reward
