#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Official-protocol strict 128-dim feature processor for D401 replica.

This version keeps the D401 model/algorithm design, but strictly extracts
features from the official 1v1 observation protocol:
- hero_states: heroes, skill_state, passive_skill, buff_state, real_cmd
- npc_states: NPCs, separated by actor_type/sub_type/config_id
- cakes: function objects, treated as blood-pack/resource via collider.location
- frame_action/map_state are not used for ordinary damage or map CNN features

Important: NPC type matching is strict. String enum names are recognized directly.
For integer enum values, only ACTOR_SUB_TOWER=21 is enabled by default because it
is used by the baseline tower processor. Fill GameConfig.SOLDIER_SUB_TYPES /
MONSTER_ACTOR_TYPES / *_CONFIG_IDS after checking NPC_SCAN logs if your env emits
integer enum values.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from agent_ppo.conf.conf import GameConfig

FEATURE_DIM = 128
MAP_ABS_MAX = 60000.0
DIST_NORM = 60000.0
MAX_MONEY = 30000.0
MAX_LEVEL = 15.0
MAX_FRAME = 20000.0
MAX_MINION_COUNT = 10.0
MAX_SKILL_CD_MS = 60000.0
MAX_SKILL_CD_SEC = 60.0


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
    """Read protocol keys defensively, including legacy actor_state wrappers."""
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


def _enum_str(value: Any) -> str:
    return str(value).upper() if value is not None else ""


def _enum_matches(value: Any, names: set, nums: set) -> bool:
    s = _enum_str(value)
    if s in names:
        return True
    try:
        return int(value) in nums
    except Exception:
        return False


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


def _safe_camp_equal(a, b) -> bool:
    return a == b or str(a) == str(b)


def _config_id(actor: Any):
    return _get_any(actor, "config_id", "configId", default=None)


def _actor_type(actor: Any):
    return _get_any(actor, "actor_type", "actorType", default=None)


def _sub_type(actor: Any):
    return _get_any(actor, "sub_type", "subType", default=None)


def _is_tower(npc: Any) -> bool:
    """Official strict ordinary tower check: ACTOR_SUB_TOWER only."""
    sub = _sub_type(npc)
    return _enum_matches(sub, {"ACTOR_SUB_TOWER"}, set(getattr(GameConfig, "TOWER_SUB_TYPES", {21})))


def _is_soldier(npc: Any) -> bool:
    """Official strict soldier check: ACTOR_SUB_SOLDIER only."""
    if npc is None or not _is_alive(npc):
        return False
    sub = _sub_type(npc)
    cfg = _config_id(npc)
    if _enum_matches(sub, {"ACTOR_SUB_SOLDIER"}, set(getattr(GameConfig, "SOLDIER_SUB_TYPES", set()))):
        return True
    try:
        return int(cfg) in set(getattr(GameConfig, "SOLDIER_CONFIG_IDS", set()))
    except Exception:
        return False


def _is_monster(npc: Any) -> bool:
    """Official strict monster/neutral check: ACTOR_TYPE_MONSTER or configured ids."""
    if npc is None or not _is_alive(npc):
        return False
    actor_type = _actor_type(npc)
    cfg = _config_id(npc)
    if _enum_matches(actor_type, {"ACTOR_TYPE_MONSTER"}, set(getattr(GameConfig, "MONSTER_ACTOR_TYPES", set()))):
        return True
    try:
        return int(cfg) in set(getattr(GameConfig, "MONSTER_CONFIG_IDS", set()))
    except Exception:
        return False


def _skill_slots(hero: Any) -> List[Any]:
    skill_state = _get_any(hero, "skill_state", "skillState", default={}) or {}
    slots = _get(skill_state, "slot_states", None)
    if slots is None:
        slots = _get(skill_state, "slotStates", None)
    return slots or []


def _slot_cd_ratio(slot: Any) -> float:
    cd = float(_get_any(slot, "cooldown", "cd", "cool_down", "left_cd", "leftCoolDown", default=0) or 0)
    cd_max = float(_get_any(slot, "cooldown_max", "cooldownMax", default=0) or 0)
    if cd_max > 0:
        return _clip(cd / max(cd_max, 1.0), 0.0, 1.0)
    # Fallback only when cooldown_max is absent.
    if cd > 1000:
        return _clip(cd / MAX_SKILL_CD_MS, 0.0, 1.0)
    return _clip(cd / MAX_SKILL_CD_SEC, 0.0, 1.0)


def _slot_available(slot: Any) -> float:
    usable = _get_any(slot, "usable", "canUse", "can_use", "available", default=None)
    if usable is not None:
        return 1.0 if bool(usable) else 0.0
    return 1.0 if _slot_cd_ratio(slot) <= 1e-6 else 0.0


def _runtime_id(actor: Any):
    return _get_any(actor, "runtime_id", "runtimeId", default=None)


class FeatureProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.last_enemy_seen = {"x": 0.0, "z": 0.0, "hp": 0.0, "frame": 0}
        self.last_self_pos: Optional[Tuple[float, float]] = None
        self.same_position_steps = 0
        self.last_frame_no = 0
        self._npc_scan_seen = set()

    def _debug_scan_npc_types(self, frame_state, frame_no):
        if not getattr(GameConfig, "DEBUG_NPC_SCAN", False):
            return
        if frame_no > int(getattr(GameConfig, "DEBUG_NPC_SCAN_MAX_FRAME", 200)):
            return
        for npc in frame_state.get("npc_states", []) or []:
            key = (_config_id(npc), _actor_type(npc), _sub_type(npc), _camp(npc))
            if key in self._npc_scan_seen:
                continue
            self._npc_scan_seen.add(key)
            print(
                "[NPC_SCAN]",
                "frame=", frame_no,
                "config_id=", _config_id(npc),
                "actor_type=", _actor_type(npc),
                "sub_type=", _sub_type(npc),
                "camp=", _camp(npc),
                "hp=", _get_any(npc, "hp", default=None),
                "max_hp=", _get_any(npc, "max_hp", default=None),
                "loc=", _get_any(npc, "location", default=None),
            )

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

    def _find_towers_minions_monsters(self, frame_state, main_camp):
        own_tower, enemy_tower = None, None
        friendly_minions, enemy_minions, monsters = [], [], []
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
                continue
            if _is_monster(npc):
                monsters.append(npc)
        return own_tower, enemy_tower, friendly_minions, enemy_minions, monsters

    def _nearest(self, source, units):
        if source is None or not units:
            return None, DIST_NORM
        best, best_d = None, DIST_NORM
        for u in units:
            d = _dist(source, u)
            if d < best_d:
                best, best_d = u, d
        return best, best_d

    def _nearest_k(self, source, units, k=4):
        if source is None or not units:
            return []
        return sorted(units, key=lambda u: _dist(source, u))[:k]

    def _minion_hp_sum(self, units):
        return sum(_hp_rate(u) for u in units)

    def _minion_front_pos(self, units, enemy_tower):
        if not units or enemy_tower is None:
            return 0.0
        dists = [_dist(u, enemy_tower) for u in units]
        return 1.0 - _norm_dist(min(dists), DIST_NORM)

    def _count_near(self, units, target, radius=9000.0):
        if target is None:
            return 0
        return sum(1 for u in units if _dist(u, target) <= radius)

    def _tower_target_flags(self, tower, self_hero, minions):
        target = _get_any(tower, "attack_target", "attackTarget", default=None)
        self_id = _runtime_id(self_hero)
        if target is None:
            return 0.0, 0.0
        target_self = 1.0 if self_id is not None and target == self_id else 0.0
        target_minion = 0.0
        for m in minions:
            mid = _runtime_id(m)
            if mid is not None and target == mid:
                target_minion = 1.0
                break
        return target_self, target_minion

    def _tower_range(self, tower):
        # Official protocol contains attack_range. No hard tower-range explorer here.
        return float(_get_any(tower, "attack_range", "attackRange", default=0) or 0)

    def _cake_summary(self, frame_state, self_hero):
        cakes = frame_state.get("cakes", []) or []
        if not cakes or self_hero is None:
            return 0.0, 1.0, 0.0
        best_dist = DIST_NORM
        best_radius = 0.0
        for cake in cakes:
            collider = _get(cake, "collider", {}) or {}
            loc = _get(collider, "location", None)
            if not isinstance(loc, dict):
                continue
            fake = {"location": loc}
            d = _dist(self_hero, fake)
            if d < best_dist:
                best_dist = d
                best_radius = float(_get(collider, "radius", 0) or 0)
        return 1.0, _norm_dist(best_dist), _norm_dist(best_radius, 10000.0)

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

    def _enemy_visible_to_main(self, enemy_hero, main_camp):
        if enemy_hero is None or not _is_alive(enemy_hero):
            return 0.0
        visible = _get_any(enemy_hero, "camp_visible", default=None)
        if isinstance(visible, (list, tuple)):
            try:
                camp_int = int(main_camp)
                # Protocol: camp 1=blue, 2=red; camp_visible[0]=blue, [1]=red.
                idx = camp_int - 1
                if 0 <= idx < len(visible):
                    return 1.0 if bool(visible[idx]) else 0.0
            except Exception:
                pass
        # If enemy exists in observation but visibility array is absent, treat as observed.
        return 1.0

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        frame_no = int(frame_state.get("frame_no", 0) or 0)
        self._debug_scan_npc_types(frame_state, frame_no)

        self_hero, enemy_hero = self._find_heroes(frame_state)
        main_camp = _camp(self_hero) if self_hero is not None else self.main_camp
        own_tower, enemy_tower, friendly_minions, enemy_minions, monsters = self._find_towers_minions_monsters(
            frame_state, main_camp
        )

        if self_hero is not None:
            self._process_history(self_hero, frame_no)

        enemy_observed = self._enemy_visible_to_main(enemy_hero, main_camp)
        if enemy_observed:
            eloc = _loc(enemy_hero)
            self.last_enemy_seen = {"x": eloc["x"], "z": eloc["z"], "hp": _hp_rate(enemy_hero), "frame": frame_no}
        not_seen_steps = max(0, frame_no - int(self.last_enemy_seen.get("frame", 0) or 0))

        f: List[float] = []
        sloc = _loc(self_hero)

        # 0-15 self hero
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
            _norm_dist(_get_any(self_hero, "mov_spd", "move_speed", "moveSpeed", default=0) or 0, 20000.0),
            _norm_dist(_dist(self_hero, enemy_tower)),
            _norm_dist(_dist(self_hero, own_tower)),
            _clip(float(_get_any(self_hero, "kill_cnt", default=0) or 0) / 10.0),
            _clip(float(_get_any(self_hero, "dead_cnt", default=0) or 0) / 10.0),
            _norm_dist(float(_get_any(self_hero, "total_hurt", default=0) or 0), 60000.0),
            _norm_dist(float(_get_any(self_hero, "total_be_hurt_by_hero", default=0) or 0), 30000.0),
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
            _norm_dist(_get_any(enemy_hero, "attack_range", "attackRange", default=0) or 0, 15000.0)
            if enemy_hero is not None else 0.0,
            _norm_dist(float(_get_any(enemy_hero, "total_hurt_to_hero", default=0) or 0), 30000.0)
            if enemy_hero is not None else 0.0,
            _norm_dist(float(_get_any(enemy_hero, "total_hurt", default=0) or 0), 60000.0)
            if enemy_hero is not None else 0.0,
            _clip(float(frame_no) / MAX_FRAME),
            1.0 if enemy_observed and _dist(self_hero, enemy_hero) <= float(_get_any(self_hero, "attack_range", default=0) or 0) else 0.0,
            1.0 if enemy_observed and _dist(self_hero, enemy_hero) <= 12000.0 else 0.0,
        ])
        f.extend(enemy_vals[:24])

        # 40-59 skills / passive / summoner
        slots = _skill_slots(self_hero)
        for i in range(3):
            slot = slots[i] if i < len(slots) else None
            f.append(_slot_available(slot))
            f.append(_slot_cd_ratio(slot))
        slot = slots[3] if len(slots) > 3 else None
        f.extend([_slot_available(slot), _slot_cd_ratio(slot)])
        used_total, hit_total, succ_frame, combo_left = 0.0, 0.0, 0.0, 0.0
        for s in slots:
            used_total += float(_get_any(s, "usedTimes", default=0) or 0)
            hit_total += float(_get_any(s, "hitHeroTimes", default=0) or 0)
            succ_frame += float(_get_any(s, "succUsedInFrame", default=0) or 0)
            combo_left = max(combo_left, float(_get_any(s, "comboEffectTime", default=0) or 0))
        buff_state = _get_any(self_hero, "buff_state", default={}) or {}
        buff_marks = _get(buff_state, "buff_marks", []) or []
        buff_layers = sum(float(_get(b, "layer", 0) or 0) for b in buff_marks)
        passive_skills = _get_any(self_hero, "passive_skill", default=[]) or []
        passive_cd_min = min([float(_get(p, "cooldown", 0) or 0) for p in passive_skills], default=0.0)
        f.extend([
            _clip(hit_total / max(1.0, used_total)),
            _clip(max(0.0, used_total - hit_total) / 50.0),
            _clip(succ_frame / 5.0),
            _clip(combo_left / 10000.0),
            _clip(len(passive_skills) / 5.0),
            _clip(passive_cd_min / MAX_SKILL_CD_MS),
            _clip(buff_layers / 10.0),
            1.0 if slots and any(_slot_available(s) for s in slots[:3]) else 0.0,
            1.0 if enemy_observed and _dist(self_hero, enemy_hero) <= 14000.0 else 0.0,
            1.0 if len(enemy_minions) > 0 and self._nearest(self_hero, enemy_minions)[1] <= 14000.0 else 0.0,
            _clip(float(_get_any(self_hero, "atk_spd", "attack_speed", default=0) or 0) / 20000.0),
            _clip(float(_get_any(self_hero, "crit_rate", default=0) or 0) / 10000.0),
        ])

        # 60-87 lane/minions. Only ACTOR_SUB_SOLDIER is counted.
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
            1.0 if enemy_tower is not None and self._tower_range(enemy_tower) > 0 and any(_dist(m, enemy_tower) <= self._tower_range(enemy_tower) * 1.1 for m in friendly_minions) else 0.0,
            1.0 if own_tower is not None and self._tower_range(own_tower) > 0 and any(_dist(m, own_tower) <= self._tower_range(own_tower) * 1.1 for m in enemy_minions) else 0.0,
            1.0 if lowest_enemy is not None and lowest_enemy_hp < 0.25 and _dist(self_hero, lowest_enemy) <= float(_get_any(self_hero, "attack_range", default=0) or 0) * 1.2 else 0.0,
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

        # 88-103 towers / tower safety. Only ACTOR_SUB_TOWER is used.
        enemy_tower_range = self._tower_range(enemy_tower) if enemy_tower is not None else 0.0
        own_tower_range = self._tower_range(own_tower) if own_tower is not None else 0.0
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
            1.0 if enemy_tower_range > 0 and self_enemy_tower_dist <= enemy_tower_range else 0.0,
            enemy_tower_target_self,
            enemy_tower_target_minion,
            own_tower_target_enemy,
            own_tower_target_minion,
            1.0 if enemy_tower_target_minion > 0.5 and enemy_tower_range > 0 and self_enemy_tower_dist <= enemy_tower_range * 1.25 else 0.0,
            1.0 if enemy_tower_range > 0 and self_enemy_tower_dist > enemy_tower_range and self_enemy_tower_dist <= enemy_tower_range * 1.35 else 0.0,
            1.0 if _hp_rate(enemy_tower) < 0.2 else 0.0,
            1.0 if _hp_rate(own_tower) < 0.2 else 0.0,
            _clip(enemy_tower_range / 20000.0),
        ])

        # 104-119 official target summaries: Enemy, Self, Tower, Monster, Soldier slots.
        nearest_monster, nmod = self._nearest(self_hero, monsters)
        nearest_soldiers = self._nearest_k(self_hero, friendly_minions + enemy_minions, k=4)
        f.extend([
            enemy_observed,
            _hp_rate(enemy_hero) if enemy_hero is not None else 0.0,
            _norm_dist(_dist(self_hero, enemy_hero)) if enemy_hero is not None else 1.0,
            _hp_rate(self_hero),
            1.0 if enemy_tower is not None else 0.0,
            _hp_rate(enemy_tower),
            _norm_dist(self_enemy_tower_dist),
            1.0 if nearest_monster is not None else 0.0,
            _hp_rate(nearest_monster) if nearest_monster is not None else 0.0,
            _norm_dist(nmod),
        ])
        for i in range(4):
            unit = nearest_soldiers[i] if i < len(nearest_soldiers) else None
            f.append(1.0 if unit is not None else 0.0)
        # Last two target slots summarize nearest official Soldier target.
        nearest_official_soldier = nearest_soldiers[0] if nearest_soldiers else None
        f.extend([
            _hp_rate(nearest_official_soldier) if nearest_official_soldier is not None else 0.0,
            _clip(1.0 if nearest_official_soldier is not None and _safe_camp_equal(_camp(nearest_official_soldier), main_camp) else 0.0),
        ])

        # 120-127 behavior history / cake / misc
        behav = _get_any(self_hero, "behav_mode", default="")
        is_idle = 1.0 if str(behav) == "State_Idle" or behav == 0 else 0.0
        cake_exists, cake_dist, cake_radius = self._cake_summary(frame_state, self_hero)
        f.extend([
            is_idle,
            _clip(self.same_position_steps / 30.0),
            1.0 if _dist(self_hero, enemy_tower) < _dist(self_hero, own_tower) else 0.0,
            1.0 if _hp_rate(self_hero) < 0.35 else 0.0,
            cake_exists,
            cake_dist,
            cake_radius,
            _clip(float(frame_no) / MAX_FRAME),
        ])

        if len(f) < FEATURE_DIM:
            f.extend([0.0] * (FEATURE_DIM - len(f)))
        elif len(f) > FEATURE_DIM:
            f = f[:FEATURE_DIM]

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
