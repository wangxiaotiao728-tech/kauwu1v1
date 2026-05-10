#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""D401 lane/skill feature processor.

This version keeps the D401 replica model/algorithm/reward design, but expands
baseline observation features from 10 dims to 128 dims so the current rewards
(lane_clear, last_hit, skill_hit, bad_skill, under_tower_behavior, passive_skills)
have matching state information.

Feature layout, fixed at 128 dims:
0   - 15   self hero
16  - 39   enemy hero / visibility / last seen
40  - 59   skills / summoner / passive
60  - 87   lane/minions
88  - 103  towers / tower safety
104 - 119  target and objective summaries
120 - 127  behavior history / misc
"""

import math
from typing import Any, Dict, List, Optional, Tuple

FEATURE_DIM = 128
MAP_ABS_MAX = 60000.0
DIST_NORM = 60000.0
MAX_MONEY = 30000.0
MAX_LEVEL = 15.0
MAX_FRAME = 20000.0
MAX_MINION_COUNT = 10.0
MAX_SKILL_CD = 60.0


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(v)
    except Exception:
        return lo
    if math.isnan(v) or math.isinf(v):
        return lo
    return max(lo, min(hi, v))


def _norm_pos(v: float) -> float:
    return _clip((float(v) + MAP_ABS_MAX) / (2.0 * MAP_ABS_MAX), 0.0, 1.0)


def _norm_signed(v: float, scale: float = DIST_NORM) -> float:
    # map [-scale, scale] to [0, 1]
    return _clip((float(v) / max(scale, 1.0) + 1.0) * 0.5, 0.0, 1.0)


def _norm_dist(v: float, scale: float = DIST_NORM) -> float:
    return _clip(float(v) / max(scale, 1.0), 0.0, 1.0)


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_any(obj: Any, *keys: str, default=None):
    """Read keys from object, actor_state and actor_state.values defensively."""
    for key in keys:
        val = _get(obj, key, None)
        if val is not None:
            return val
    actor_state = _get(obj, "actor_state", None)
    if actor_state is not None:
        for key in keys:
            val = _get(actor_state, key, None)
            if val is not None:
                return val
        values = _get(actor_state, "values", None)
        if values is not None:
            for key in keys:
                val = _get(values, key, None)
                if val is not None:
                    return val
    return default


def _loc(actor: Any) -> Dict[str, float]:
    loc = _get_any(actor, "location", "loc", "position", default=None)
    if isinstance(loc, dict):
        return {"x": float(loc.get("x", 0) or 0), "z": float(loc.get("z", 0) or 0)}
    return {"x": 0.0, "z": 0.0}


def _dist(a: Any, b: Any) -> float:
    if a is None or b is None:
        return DIST_NORM
    la, lb = _loc(a), _loc(b)
    dx = la["x"] - lb["x"]
    dz = la["z"] - lb["z"]
    return math.sqrt(dx * dx + dz * dz)


def _hp_rate(actor: Any) -> float:
    hp = float(_get_any(actor, "hp", default=0) or 0)
    max_hp = max(1.0, float(_get_any(actor, "max_hp", "maxHp", default=1) or 1))
    return _clip(hp / max_hp, 0.0, 1.0)


def _ep_rate(actor: Any) -> float:
    ep = float(_get_any(actor, "ep", default=0) or 0)
    max_ep = max(1.0, float(_get_any(actor, "max_ep", "maxEp", default=1) or 1))
    return _clip(ep / max_ep, 0.0, 1.0)


def _is_alive(actor: Any) -> bool:
    return _hp_rate(actor) > 0.0


def _camp(actor: Any):
    return _get_any(actor, "camp", default=None)


def _is_tower(npc: Any) -> bool:
    sub = _get_any(npc, "sub_type", "subType", default=-1)
    return sub == 21 or str(sub).upper().find("TOWER") >= 0


def _is_soldier(npc: Any) -> bool:
    if npc is None or _is_tower(npc) or not _is_alive(npc):
        return False
    sub = _get_any(npc, "sub_type", "subType", default=-1)
    # Current baseline reward uses a coarse non-tower NPC rule. Keep it aligned,
    # but still reject explicit monster/resource names when they are available.
    name = str(_get_any(npc, "name", "config_name", "actor_name", default="")).lower()
    if "monster" in name or "neutral" in name or "buff" in name:
        return False
    return sub not in (21, -1)


def _safe_camp_equal(a, b) -> bool:
    return a == b or str(a) == str(b)


def _skill_slots(hero: Any) -> List[Any]:
    skill_state = _get_any(hero, "skill_state", "skillState", default={}) or {}
    slots = _get(skill_state, "slot_states", None)
    if slots is None:
        slots = _get(skill_state, "slotStates", None)
    return slots or []


def _slot_cd_ratio(slot: Any) -> float:
    cd = _get_any(slot, "cd", "cooldown", "cool_down", "left_cd", "leftCoolDown", "cooldown_ms", default=0) or 0
    # Some versions use ms.
    cd = float(cd)
    if cd > 1000:
        cd = cd / 1000.0
    return _clip(cd / MAX_SKILL_CD, 0.0, 1.0)


def _slot_available(slot: Any) -> float:
    usable = _get_any(slot, "usable", "canUse", "can_use", "available", default=None)
    if usable is not None:
        return 1.0 if bool(usable) else 0.0
    return 1.0 if _slot_cd_ratio(slot) <= 1e-6 else 0.0


class FeatureProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.last_enemy_seen = {"x": 0.0, "z": 0.0, "hp": 0.0, "frame": 0}
        self.last_self_pos: Optional[Tuple[float, float]] = None
        self.same_position_steps = 0
        self.last_frame_no = 0

    def _find_heroes(self, frame_state):
        main_hero, enemy_hero = None, None
        heroes = frame_state.get("hero_states", []) or []
        for hero in heroes:
            if _safe_camp_equal(_camp(hero), self.main_camp):
                main_hero = hero
            else:
                enemy_hero = hero
        if main_hero is None and heroes:
            main_hero = heroes[0]
            for hero in heroes[1:]:
                if not _safe_camp_equal(_camp(hero), _camp(main_hero)):
                    enemy_hero = hero
                    break
        return main_hero, enemy_hero

    def _find_towers_and_minions(self, frame_state, main_camp):
        own_tower, enemy_tower = None, None
        friendly_minions, enemy_minions = [], []
        for npc in frame_state.get("npc_states", []) or []:
            npc_camp = _camp(npc)
            if _is_tower(npc):
                if _safe_camp_equal(npc_camp, main_camp):
                    own_tower = npc
                else:
                    enemy_tower = npc
                continue
            if _is_soldier(npc):
                if _safe_camp_equal(npc_camp, main_camp):
                    friendly_minions.append(npc)
                else:
                    enemy_minions.append(npc)
        return own_tower, enemy_tower, friendly_minions, enemy_minions

    def _nearest(self, source, units):
        if source is None or not units:
            return None, DIST_NORM
        best, best_d = None, DIST_NORM
        for u in units:
            d = _dist(source, u)
            if d < best_d:
                best, best_d = u, d
        return best, best_d

    def _minion_hp_sum(self, units):
        return sum(_hp_rate(u) for u in units)

    def _minion_front_pos(self, units, enemy_tower):
        if not units or enemy_tower is None:
            return 0.0
        # Smaller distance to enemy tower means more advanced wave.
        dists = [_dist(u, enemy_tower) for u in units]
        return 1.0 - _norm_dist(min(dists), DIST_NORM)

    def _count_near(self, units, target, radius=9000.0):
        if target is None:
            return 0
        return sum(1 for u in units if _dist(u, target) <= radius)

    def _tower_target_flags(self, tower, self_hero, minions):
        target = _get_any(tower, "attack_target", "attackTarget", "target", default=None)
        self_id = _get_any(self_hero, "runtime_id", "runtimeId", default=None)
        if target is None:
            return 0.0, 0.0
        target_self = 1.0 if self_id is not None and target == self_id else 0.0
        target_minion = 0.0
        for m in minions:
            mid = _get_any(m, "runtime_id", "runtimeId", default=None)
            if mid is not None and target == mid:
                target_minion = 1.0
                break
        return target_self, target_minion

    def _tower_range(self, tower):
        return float(_get_any(tower, "attack_range", "attackRange", default=8000) or 8000)

    def _process_history(self, self_hero, frame_no):
        pos = _loc(self_hero)
        cur = (pos["x"], pos["z"])
        if self.last_self_pos is None:
            self.same_position_steps = 0
        else:
            dx = cur[0] - self.last_self_pos[0]
            dz = cur[1] - self.last_self_pos[1]
            if math.sqrt(dx * dx + dz * dz) < 100.0:
                self.same_position_steps += 1
            else:
                self.same_position_steps = 0
        self.last_self_pos = cur
        self.last_frame_no = frame_no

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        frame_no = int(frame_state.get("frame_no", 0) or 0)
        self_hero, enemy_hero = self._find_heroes(frame_state)
        main_camp = _camp(self_hero) if self_hero is not None else self.main_camp
        own_tower, enemy_tower, friendly_minions, enemy_minions = self._find_towers_and_minions(frame_state, main_camp)

        if self_hero is not None:
            self._process_history(self_hero, frame_no)

        # Enemy visibility/last seen.
        enemy_observed = 1.0 if enemy_hero is not None and _is_alive(enemy_hero) else 0.0
        if enemy_observed:
            eloc = _loc(enemy_hero)
            self.last_enemy_seen = {"x": eloc["x"], "z": eloc["z"], "hp": _hp_rate(enemy_hero), "frame": frame_no}
        not_seen_steps = max(0, frame_no - int(self.last_enemy_seen.get("frame", 0) or 0))

        f: List[float] = []

        # 0-15 self hero
        sloc = _loc(self_hero)
        f.extend([
            1.0 if _is_alive(self_hero) else 0.0,
            _hp_rate(self_hero),
            _ep_rate(self_hero),
            _clip(float(_get_any(self_hero, "level", default=1) or 1) / MAX_LEVEL),
            _clip(float(_get_any(self_hero, "exp", default=0) or 0) / 2500.0),
            _clip(float(_get_any(self_hero, "money_cnt", "money", default=0) or 0) / MAX_MONEY),
            _norm_pos(sloc["x"]),
            _norm_pos(sloc["z"]),
            _norm_dist(_get_any(self_hero, "attack_range", "attackRange", default=0) or 0, 15000.0),
            _norm_dist(_get_any(self_hero, "move_speed", "moveSpeed", default=0) or 0, 20000.0),
            _norm_dist(_dist(self_hero, enemy_tower)),
            _norm_dist(_dist(self_hero, own_tower)),
            _clip(float(_get_any(self_hero, "kill_cnt", default=0) or 0) / 10.0),
            _clip(float(_get_any(self_hero, "dead_cnt", default=0) or 0) / 10.0),
            _norm_dist(float(_get_any(self_hero, "total_hurt", "totalHurt", default=0) or 0), 60000.0),
            _norm_dist(float(_get_any(self_hero, "total_be_hurt_by_hero", "totalBeHurtByHero", default=0) or 0), 30000.0),
        ])

        # 16-39 enemy hero / visibility / last seen
        if enemy_hero is not None:
            epos = _loc(enemy_hero)
            enemy_vals = [
                enemy_observed,
                _hp_rate(enemy_hero),
                _clip(float(_get_any(enemy_hero, "level", default=1) or 1) / MAX_LEVEL),
                _clip(float(_get_any(enemy_hero, "exp", default=0) or 0) / 2500.0),
                _clip(float(_get_any(enemy_hero, "money_cnt", "money", default=0) or 0) / MAX_MONEY),
                _norm_pos(epos["x"]),
                _norm_pos(epos["z"]),
                _norm_dist(_dist(self_hero, enemy_hero)),
                _norm_dist(_dist(enemy_hero, own_tower)),
                _norm_dist(_dist(enemy_hero, enemy_tower)),
                _norm_signed(epos["x"] - sloc["x"]),
                _norm_signed(epos["z"] - sloc["z"]),
                1.0 if _is_alive(enemy_hero) else 0.0,
            ]
        else:
            enemy_vals = [0.0] * 13
        enemy_vals.extend([
            _norm_pos(float(self.last_enemy_seen.get("x", 0.0))),
            _norm_pos(float(self.last_enemy_seen.get("z", 0.0))),
            _clip(float(self.last_enemy_seen.get("hp", 0.0))),
            _clip(not_seen_steps / 200.0),
            _norm_dist(_dist(enemy_hero, self_hero)) if enemy_hero is not None else 1.0,
            _norm_dist(_get_any(enemy_hero, "attack_range", "attackRange", default=0) or 0, 15000.0) if enemy_hero is not None else 0.0,
            _norm_dist(float(_get_any(enemy_hero, "total_hurt_to_hero", "totalHurtToHero", default=0) or 0), 30000.0) if enemy_hero is not None else 0.0,
            _norm_dist(float(_get_any(enemy_hero, "total_hurt", "totalHurt", default=0) or 0), 60000.0) if enemy_hero is not None else 0.0,
            _clip(float(frame_no) / MAX_FRAME),
            1.0 if enemy_observed and _dist(self_hero, enemy_hero) <= float(_get_any(self_hero, "attack_range", "attackRange", default=0) or 0) else 0.0,
            1.0 if enemy_observed and _dist(self_hero, enemy_hero) <= 12000.0 else 0.0,
        ])
        f.extend(enemy_vals[:24])

        # 40-59 skills / passive / summoner
        slots = _skill_slots(self_hero)
        for i in range(3):
            slot = slots[i] if i < len(slots) else None
            f.append(_slot_available(slot))
            f.append(_slot_cd_ratio(slot))
        # Summoner / extra skill: use 4th slot when present.
        slot = slots[3] if len(slots) > 3 else None
        f.extend([_slot_available(slot), _slot_cd_ratio(slot)])
        used_total, hit_total = 0.0, 0.0
        for s in slots:
            used_total += float(_get_any(s, "usedTimes", "used_times", default=0) or 0)
            hit_total += float(_get_any(s, "hitHeroTimes", "hit_hero_times", default=0) or 0)
        f.extend([
            _clip(hit_total / max(1.0, used_total)),
            _clip(max(0.0, used_total - hit_total) / 50.0),
        ])
        passive_skills = _get_any(self_hero, "passive_skill", default=[]) or []
        passive_level = sum(float(_get(p, "level", 0) or 0) for p in passive_skills)
        passive_triggered = 1.0 if any(bool(_get(p, "triggered", False)) for p in passive_skills) else 0.0
        # A lightweight enhanced attack proxy: passive triggered or high passive level.
        enhanced_ready = 1.0 if passive_triggered or passive_level >= 3 else 0.0
        f.extend([
            _clip(passive_level / 5.0),
            passive_triggered,
            enhanced_ready,
            1.0 if slots and any(_slot_available(s) for s in slots[:3]) else 0.0,
            1.0 if enemy_observed and _dist(self_hero, enemy_hero) <= 14000.0 else 0.0,
            1.0 if len(enemy_minions) > 0 and self._nearest(self_hero, enemy_minions)[1] <= 14000.0 else 0.0,
            _clip(float(_get_any(self_hero, "attack_speed", "attackSpeed", default=0) or 0) / 20000.0),
            _clip(float(_get_any(self_hero, "crit_rate", default=0) or 0) / 10000.0),
            _clip(float(_get_any(self_hero, "crit_effe", default=0) or 0) / 10000.0),
            0.0,
        ])

        # 60-87 lane/minions
        nearest_enemy_minion, nemd = self._nearest(self_hero, enemy_minions)
        nearest_friendly_minion, nfmd = self._nearest(self_hero, friendly_minions)
        lowest_enemy = min(enemy_minions, key=lambda u: _hp_rate(u), default=None)
        lowest_enemy_hp = _hp_rate(lowest_enemy) if lowest_enemy is not None else 0.0
        f.extend([
            _clip(len(friendly_minions) / MAX_MINION_COUNT),
            _clip(len(enemy_minions) / MAX_MINION_COUNT),
            _clip(self._minion_hp_sum(friendly_minions) / MAX_MINION_COUNT),
            _clip(self._minion_hp_sum(enemy_minions) / MAX_MINION_COUNT),
            _clip((self._minion_hp_sum(friendly_minions) - self._minion_hp_sum(enemy_minions)) / MAX_MINION_COUNT * 0.5 + 0.5),
            _clip((len(friendly_minions) - len(enemy_minions)) / MAX_MINION_COUNT * 0.5 + 0.5),
            _norm_dist(nemd),
            _norm_dist(nfmd),
            lowest_enemy_hp,
            _norm_dist(_dist(self_hero, lowest_enemy)),
            self._minion_front_pos(friendly_minions, enemy_tower),
            self._minion_front_pos(enemy_minions, own_tower),
            _clip((self._minion_front_pos(friendly_minions, enemy_tower) - self._minion_front_pos(enemy_minions, own_tower)) * 0.5 + 0.5),
            _clip(self._count_near(friendly_minions, enemy_tower) / MAX_MINION_COUNT),
            _clip(self._count_near(enemy_minions, own_tower) / MAX_MINION_COUNT),
            1.0 if enemy_tower is not None and any(_dist(m, enemy_tower) <= self._tower_range(enemy_tower) * 1.1 for m in friendly_minions) else 0.0,
            1.0 if own_tower is not None and any(_dist(m, own_tower) <= self._tower_range(own_tower) * 1.1 for m in enemy_minions) else 0.0,
            1.0 if lowest_enemy is not None and lowest_enemy_hp < 0.25 and _dist(self_hero, lowest_enemy) <= float(_get_any(self_hero, "attack_range", "attackRange", default=0) or 0) * 1.2 else 0.0,
            _norm_pos(_loc(nearest_enemy_minion)["x"]) if nearest_enemy_minion is not None else 0.0,
            _norm_pos(_loc(nearest_enemy_minion)["z"]) if nearest_enemy_minion is not None else 0.0,
            _hp_rate(nearest_enemy_minion) if nearest_enemy_minion is not None else 0.0,
            _norm_pos(_loc(lowest_enemy)["x"]) if lowest_enemy is not None else 0.0,
            _norm_pos(_loc(lowest_enemy)["z"]) if lowest_enemy is not None else 0.0,
            _clip(sum(1 for m in enemy_minions if _hp_rate(m) < 0.35) / MAX_MINION_COUNT),
            _clip(sum(1 for m in friendly_minions if _hp_rate(m) < 0.35) / MAX_MINION_COUNT),
            1.0 if enemy_minions else 0.0,
            1.0 if friendly_minions else 0.0,
            _clip(len(enemy_minions) / max(1.0, len(enemy_minions) + len(friendly_minions))),
        ])

        # 88-103 towers / tower safety
        enemy_tower_range = self._tower_range(enemy_tower) if enemy_tower is not None else 8000.0
        own_tower_range = self._tower_range(own_tower) if own_tower is not None else 8000.0
        enemy_tower_target_self, enemy_tower_target_minion = self._tower_target_flags(enemy_tower, self_hero, friendly_minions)
        own_tower_target_enemy, own_tower_target_minion = self._tower_target_flags(own_tower, enemy_hero, enemy_minions)
        self_enemy_tower_dist = _dist(self_hero, enemy_tower)
        f.extend([
            _hp_rate(enemy_tower),
            _hp_rate(own_tower),
            _norm_dist(self_enemy_tower_dist),
            _norm_dist(_dist(self_hero, own_tower)),
            _norm_dist(_dist(enemy_hero, own_tower)) if enemy_hero is not None else 1.0,
            _norm_dist(_dist(enemy_hero, enemy_tower)) if enemy_hero is not None else 1.0,
            1.0 if self_enemy_tower_dist <= enemy_tower_range else 0.0,
            enemy_tower_target_self,
            enemy_tower_target_minion,
            own_tower_target_enemy,
            own_tower_target_minion,
            1.0 if enemy_tower_target_minion > 0.5 and self_enemy_tower_dist <= enemy_tower_range * 1.25 else 0.0,
            1.0 if self_enemy_tower_dist > enemy_tower_range and self_enemy_tower_dist <= enemy_tower_range * 1.35 else 0.0,
            1.0 if _hp_rate(enemy_tower) < 0.2 else 0.0,
            1.0 if _hp_rate(own_tower) < 0.2 else 0.0,
            _clip(enemy_tower_range / 20000.0),
        ])

        # 104-119 target/objective summaries
        target_units = [enemy_hero, enemy_tower, lowest_enemy, nearest_enemy_minion, nearest_friendly_minion]
        for unit in target_units[:4]:
            f.append(1.0 if unit is not None else 0.0)
            f.append(_hp_rate(unit) if unit is not None else 0.0)
            f.append(_norm_dist(_dist(self_hero, unit)) if unit is not None else 1.0)
        f.extend([
            1.0 if enemy_observed and enemy_hero is not None and _dist(self_hero, enemy_hero) <= float(_get_any(self_hero, "attack_range", "attackRange", default=0) or 0) * 1.2 else 0.0,
            1.0 if enemy_tower is not None and self_enemy_tower_dist <= enemy_tower_range * 1.25 else 0.0,
            1.0 if lowest_enemy is not None and lowest_enemy_hp < 0.25 else 0.0,
            _clip(float(frame_no) / MAX_FRAME),
        ])

        # 120-127 behavior history / misc
        behav = _get_any(self_hero, "behav_mode", default="")
        is_idle = 1.0 if str(behav) == "State_Idle" or behav == 0 else 0.0
        f.extend([
            is_idle,
            _clip(self.same_position_steps / 30.0),
            1.0 if _dist(self_hero, enemy_tower) < _dist(self_hero, own_tower) else 0.0,
            1.0 if _hp_rate(self_hero) < 0.35 else 0.0,
            1.0 if enemy_observed and _hp_rate(enemy_hero) < 0.35 else 0.0,
            _clip(float(_get_any(self_hero, "money_cnt", "money", default=0) or 0) - float(_get_any(enemy_hero, "money_cnt", "money", default=0) or 0), -MAX_MONEY, MAX_MONEY) / (2 * MAX_MONEY) + 0.5 if enemy_hero is not None else 0.5,
            _clip((float(_get_any(self_hero, "level", default=1) or 1) - float(_get_any(enemy_hero, "level", default=1) or 1)) / MAX_LEVEL * 0.5 + 0.5) if enemy_hero is not None else 0.5,
            1.0,
        ])

        if len(f) < FEATURE_DIM:
            f.extend([0.0] * (FEATURE_DIM - len(f)))
        elif len(f) > FEATURE_DIM:
            f = f[:FEATURE_DIM]

        # Final safety: replace nan/inf and clamp to a modest range.
        out = []
        for v in f:
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    fv = 0.0
            except Exception:
                fv = 0.0
            out.append(_clip(fv, 0.0, 1.0))
        return out
