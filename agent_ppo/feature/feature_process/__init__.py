#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Final 256-dim feature processor for HoK 1v1 PPO.

Feature layout must match Config.FEATURE_GROUP_SIZES:
[32 self hero, 32 enemy hero, 56 skill slots, 40 lane/NPC,
 32 tower/monster/cake/time, 40 target summary, 24 history/risk].
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from agent_ppo.conf.conf import GameConfig, DimConfig

FEATURE_DIM = int(DimConfig.DIM_OF_FEATURE[0])
MAP_ABS_MAX = 60000.0
DIST_NORM = 60000.0
MAX_FRAME = 20000.0
MAX_MONEY = 30000.0
MAX_LEVEL = 15.0
MAX_EXP = 2000.0
MAX_ATTR = 10000.0
MAX_SKILL_CD = 120000.0
MAX_COUNT = 10.0


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(v)
    except Exception:
        return lo
    if math.isnan(v) or math.isinf(v):
        return lo
    return max(lo, min(hi, v))


def _to_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return default


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_any(obj: Any, *keys: str, default=None):
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


def _forward(actor: Any) -> Dict[str, float]:
    f = _get_any(actor, "forward", default=None)
    if isinstance(f, dict):
        return {"x": float(f.get("x", 0) or 0), "z": float(f.get("z", 0) or 0)}
    return {"x": 0.0, "z": 0.0}


def _dist_actor(a: Any, b: Any) -> float:
    if a is None or b is None:
        return DIST_NORM
    return _dist_loc(_loc(a), _loc(b))


def _dist_loc(a: Dict[str, float], b: Dict[str, float]) -> float:
    dx = float(a.get("x", 0)) - float(b.get("x", 0))
    dz = float(a.get("z", 0)) - float(b.get("z", 0))
    return math.sqrt(dx * dx + dz * dz)


def _norm_pos(v: float) -> float:
    return _clip((float(v) + MAP_ABS_MAX) / (2.0 * MAP_ABS_MAX))


def _norm_signed(v: float, scale: float = DIST_NORM) -> float:
    return _clip((float(v) / max(scale, 1.0) + 1.0) * 0.5)


def _norm_dist(v: float, scale: float = DIST_NORM) -> float:
    return _clip(float(v) / max(scale, 1.0))


def _hp_rate(actor: Any) -> float:
    hp = float(_get_any(actor, "hp", default=0) or 0)
    max_hp = max(1.0, float(_get_any(actor, "max_hp", "maxHp", default=1) or 1))
    return _clip(hp / max_hp)


def _ep_rate(actor: Any) -> float:
    ep = float(_get_any(actor, "ep", default=0) or 0)
    max_ep = max(1.0, float(_get_any(actor, "max_ep", "maxEp", default=1) or 1))
    return _clip(ep / max_ep)


def _alive(actor: Any) -> bool:
    return float(_get_any(actor, "hp", default=0) or 0) > 0


def _camp(actor: Any):
    return _get_any(actor, "camp", default=None)


def _runtime_id(actor: Any):
    return _get_any(actor, "runtime_id", "runtimeId", default=None)


def _config_id(actor: Any):
    return _get_any(actor, "config_id", "configId", default=None)


def _sub_type(actor: Any):
    return _get_any(actor, "sub_type", "subType", default=None)


def _actor_type(actor: Any):
    return _get_any(actor, "actor_type", "actorType", default=None)


def _is_soldier(npc: Any) -> bool:
    if npc is None or not _alive(npc):
        return False
    sub_type = _to_int(_sub_type(npc))
    cfg = _to_int(_config_id(npc))
    return sub_type in GameConfig.SOLDIER_SUB_TYPES or cfg in GameConfig.SOLDIER_CONFIG_IDS


def _is_tower(npc: Any) -> bool:
    if npc is None or not _alive(npc):
        return False
    sub_type = _to_int(_sub_type(npc))
    cfg = _to_int(_config_id(npc))
    return sub_type in GameConfig.TOWER_SUB_TYPES or cfg in GameConfig.TOWER_CONFIG_IDS


def _is_excluded_npc(npc: Any) -> bool:
    sub_type = _to_int(_sub_type(npc))
    cfg = _to_int(_config_id(npc))
    return sub_type in GameConfig.EXCLUDED_NPC_SUB_TYPES or cfg in GameConfig.BASE_OR_SPRING_CONFIG_IDS


def _is_monster(npc: Any) -> bool:
    if npc is None or not _alive(npc):
        return False
    return _to_int(_config_id(npc)) in GameConfig.MONSTER_CONFIG_IDS


def _slot_by_type(hero: Any) -> Dict[int, Any]:
    skill_state = _get_any(hero, "skill_state", "skillState", default={}) or {}
    slots = _get(skill_state, "slot_states", None)
    if slots is None:
        slots = _get(skill_state, "slotStates", None)
    slots = slots or []
    result = {}
    for slot in slots:
        result[_to_int(_get_any(slot, "slot_type", "slotType", default=-1))] = slot
    return result


def _slot_cd_ratio(slot: Any) -> float:
    if slot is None:
        return 1.0
    cd = float(_get_any(slot, "cooldown", default=0) or 0)
    cd_max = float(_get_any(slot, "cooldown_max", "cooldownMax", default=0) or 0)
    if cd_max > 0:
        return _clip(cd / max(cd_max, 1.0))
    return _clip(cd / MAX_SKILL_CD)


def _slot_usable(slot: Any) -> float:
    if slot is None:
        return 0.0
    usable = _get_any(slot, "usable", default=None)
    if usable is not None:
        return 1.0 if bool(usable) else 0.0
    return 1.0 if _slot_cd_ratio(slot) <= 1e-6 else 0.0


def _is_visible_to(hero: Any, camp: int) -> bool:
    visible = _get_any(hero, "camp_visible", "campVisible", default=None)
    if isinstance(visible, list) and len(visible) >= 2:
        # camp is 1/2 in the observed env. Index 0 means visible to camp1.
        idx = 0 if int(camp) == 1 else 1
        try:
            return bool(visible[idx])
        except Exception:
            return True
    loc = _loc(hero)
    # Enemy invisible often uses sentinel 100000 positions.
    if abs(loc["x"]) > 90000 or abs(loc["z"]) > 90000:
        return False
    return True


class FeatureProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.last_enemy_seen = {"x": 0.0, "z": 0.0, "frame": 0, "hp": 1.0}
        self.last_self_pos: Optional[Tuple[float, float]] = None
        self.same_position_steps = 0
        self.no_real_cmd_steps = 0
        self.no_effective_action_steps = 0
        self.grass_steps = 0
        self.grass_no_effective_steps = 0
        self.prev_slot_used = {}
        self.prev_slot_hit = {}
        self.prev_total_hurt = 0.0
        self.prev_enemy_total_hurt = 0.0
        self.last_own_tower_seen = {"x": 0.0, "z": 0.0, "frame": 0, "hp": 1.0}
        self.last_enemy_tower_seen = {"x": 0.0, "z": 0.0, "frame": 0, "hp": 1.0}

    def reset(self):
        self.__init__(self.main_camp)

    def _find_heroes(self, frame_state):
        main_hero, enemy_hero = None, None
        for hero in frame_state.get("hero_states", []):
            if _camp(hero) == self.main_camp:
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _split_npcs(self, frame_state):
        soldiers, towers, monsters = [], [], []
        for npc in frame_state.get("npc_states", []):
            if _is_soldier(npc):
                soldiers.append(npc)
            elif _is_tower(npc):
                towers.append(npc)
            elif _is_monster(npc):
                monsters.append(npc)
        return soldiers, towers, monsters

    def _find_towers(self, towers):
        own_tower, enemy_tower = None, None
        for t in towers:
            if _camp(t) == self.main_camp:
                own_tower = t
            else:
                enemy_tower = t
        return own_tower, enemy_tower

    def _split_soldiers(self, soldiers):
        friendly, enemy = [], []
        for s in soldiers:
            if _camp(s) == self.main_camp:
                friendly.append(s)
            else:
                enemy.append(s)
        return friendly, enemy

    def _split_cakes(self, frame_state, own_tower, enemy_tower):
        cakes = []
        for cake in frame_state.get("cakes", []) or []:
            cfg = _to_int(_get_any(cake, "configId", "config_id", default=-1))
            if cfg not in GameConfig.CAKE_CONFIG_IDS:
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
        own_item = min(cakes, key=lambda item: _dist_loc(item[1], own_pos))
        remaining = [item for item in cakes if item is not own_item]
        enemy_item = min(remaining, key=lambda item: _dist_loc(item[1], enemy_pos)) if remaining else None
        return own_item, enemy_item

    def _update_tower_cache(self, cache_name: str, tower: Any, frame_no: int):
        if tower is None:
            return
        pos = _loc(tower)
        cache = getattr(self, cache_name)
        cache.update({"x": pos["x"], "z": pos["z"], "frame": frame_no, "hp": _hp_rate(tower)})

    def _fallback_enemy_tower_pos(self, own_tower: Any) -> Dict[str, float]:
        if self.last_enemy_tower_seen.get("frame", 0) > 0:
            return {"x": self.last_enemy_tower_seen["x"], "z": self.last_enemy_tower_seen["z"]}
        if own_tower is not None:
            own_pos = _loc(own_tower)
            return {"x": -own_pos["x"], "z": -own_pos["z"]}
        if self.last_own_tower_seen.get("frame", 0) > 0:
            return {"x": -self.last_own_tower_seen["x"], "z": -self.last_own_tower_seen["z"]}
        return {"x": 0.0, "z": 0.0}

    def _tower_ref(self, tower: Any, cache: Dict[str, float], fallback_pos: Dict[str, float], camp: Any):
        if tower is not None:
            return tower
        return {
            "location": {"x": fallback_pos["x"], "z": fallback_pos["z"]},
            "hp": float(cache.get("hp", 1.0)),
            "max_hp": 1.0,
            "attack_range": 8800,
            "attack_target": 0,
            "runtime_id": None,
            "camp": camp,
        }

    def _append_pad(self, values: List[float], size: int) -> List[float]:
        values = [float(0.0 if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v) for v in values]
        if len(values) < size:
            values += [0.0] * (size - len(values))
        return values[:size]

    def _target_runtime_sets(self, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower, enemy_tower):
        enemy_ids = set(_runtime_id(x) for x in ([enemy_hero] + enemy_soldiers + [enemy_tower]) if x is not None)
        friendly_ids = set(_runtime_id(x) for x in ([main_hero] + friendly_soldiers + [own_tower]) if x is not None)
        enemy_soldier_ids = set(_runtime_id(x) for x in enemy_soldiers)
        friendly_soldier_ids = set(_runtime_id(x) for x in friendly_soldiers)
        return enemy_ids, friendly_ids, enemy_soldier_ids, friendly_soldier_ids

    def _extract_self_group(self, frame_no, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower, enemy_tower):
        cfg = _to_int(_config_id(main_hero))
        pos = _loc(main_hero)
        fw = _forward(main_hero)
        attack_target = _get_any(main_hero, "attack_target", "attackTarget", default=0)
        _, _, enemy_soldier_ids, _ = self._target_runtime_sets(main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower, enemy_tower)
        hit_info = _get_any(main_hero, "hit_target_info", "hitTargetInfo", default=[]) or []
        take_hurt = _get_any(main_hero, "take_hurt_infos", "takeHurtInfos", default=[]) or []
        passive = _get_any(main_hero, "passive_skill", default=[]) or []
        ready_passive = sum(1 for p in passive if float(_get(p, "cooldown", 0) or 0) <= 0)
        buff_state = _get_any(main_hero, "buff_state", default={}) or {}
        buff_count = len(_get(buff_state, "buff_skills", []) or []) + len(_get(buff_state, "buff_marks", []) or [])
        vals = [
            1.0 if cfg == GameConfig.HERO_LUBAN_ID else 0.0,
            1.0 if cfg == GameConfig.HERO_DIRENJIE_ID else 0.0,
            1.0 if _alive(main_hero) else 0.0,
            _hp_rate(main_hero),
            _ep_rate(main_hero),
            _clip(float(_get_any(main_hero, "level", default=0) or 0) / MAX_LEVEL),
            _clip(float(_get_any(main_hero, "exp", default=0) or 0) / MAX_EXP),
            _clip(float(_get_any(main_hero, "money", default=0) or 0) / MAX_MONEY),
            _clip(float(_get_any(main_hero, "money_cnt", default=0) or 0) / MAX_MONEY),
            _clip(float(_get_any(main_hero, "phy_atk", default=0) or 0) / 1000.0),
            _clip(float(_get_any(main_hero, "phy_def", default=0) or 0) / 1000.0),
            _clip(float(_get_any(main_hero, "mgc_def", default=0) or 0) / 1000.0),
            _clip(float(_get_any(main_hero, "mov_spd", default=0) or 0) / MAX_ATTR),
            _clip(float(_get_any(main_hero, "atk_spd", default=0) or 0) / MAX_ATTR),
            _clip(float(_get_any(main_hero, "attack_range", default=0) or 0) / 15000.0),
            _clip(float(_get_any(main_hero, "cd_reduce", default=0) or 0) / MAX_ATTR),
            _clip(float(_get_any(main_hero, "crit_rate", default=0) or 0) / MAX_ATTR),
            _clip(float(_get_any(main_hero, "phy_vamp", default=0) or 0) / MAX_ATTR),
            1.0 if bool(_get_any(main_hero, "is_in_grass", default=False)) else 0.0,
            _norm_pos(pos["x"]),
            _norm_pos(pos["z"]),
            _norm_signed(fw["x"], 1000.0),
            _norm_signed(fw["z"], 1000.0),
            _norm_dist(_dist_actor(main_hero, own_tower)),
            _norm_dist(_dist_actor(main_hero, enemy_tower)),
            1.0 if attack_target == _runtime_id(enemy_hero) else 0.0,
            1.0 if attack_target in enemy_soldier_ids else 0.0,
            1.0 if attack_target == _runtime_id(enemy_tower) else 0.0,
            1.0 if len(hit_info) > 0 else 0.0,
            1.0 if len(take_hurt) > 0 else 0.0,
            _clip(ready_passive / 3.0),
            _clip(buff_count / 10.0),
        ]
        return self._append_pad(vals, 32)

    def _extract_enemy_group(self, frame_no, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower, enemy_tower):
        cfg = _to_int(_config_id(enemy_hero))
        visible = enemy_hero is not None and _is_visible_to(enemy_hero, self.main_camp)
        main_pos = _loc(main_hero)
        if visible:
            epos = _loc(enemy_hero)
            self.last_enemy_seen = {"x": epos["x"], "z": epos["z"], "frame": frame_no, "hp": _hp_rate(enemy_hero)}
        else:
            epos = {"x": self.last_enemy_seen["x"], "z": self.last_enemy_seen["z"]}
        dx, dz = epos["x"] - main_pos["x"], epos["z"] - main_pos["z"]
        dist = math.sqrt(dx * dx + dz * dz) if visible else DIST_NORM
        attack_target = _get_any(enemy_hero, "attack_target", "attackTarget", default=0)
        friendly_soldier_ids = set(_runtime_id(x) for x in friendly_soldiers)
        hp_adv = _hp_rate(main_hero) - (_hp_rate(enemy_hero) if visible else self.last_enemy_seen.get("hp", 0.0))
        my_level = float(_get_any(main_hero, "level", default=0) or 0)
        en_level = float(_get_any(enemy_hero, "level", default=0) or 0) if enemy_hero is not None else 0.0
        my_money = float(_get_any(main_hero, "money_cnt", default=0) or 0)
        en_money = float(_get_any(enemy_hero, "money_cnt", default=0) or 0) if enemy_hero is not None else 0.0
        vals = [
            1.0 if cfg == GameConfig.HERO_LUBAN_ID else 0.0,
            1.0 if cfg == GameConfig.HERO_DIRENJIE_ID else 0.0,
            1.0 if visible else 0.0,
            1.0 if _alive(enemy_hero) and visible else 0.0,
            _hp_rate(enemy_hero) if visible else self.last_enemy_seen.get("hp", 0.0),
            _ep_rate(enemy_hero) if visible else 0.0,
            _clip(en_level / MAX_LEVEL),
            _clip(float(_get_any(enemy_hero, "exp", default=0) or 0) / MAX_EXP) if visible else 0.0,
            _clip(float(_get_any(enemy_hero, "money", default=0) or 0) / MAX_MONEY) if visible else 0.0,
            _clip(en_money / MAX_MONEY),
            _clip(float(_get_any(enemy_hero, "phy_atk", default=0) or 0) / 1000.0) if visible else 0.0,
            _clip(float(_get_any(enemy_hero, "phy_def", default=0) or 0) / 1000.0) if visible else 0.0,
            _clip(float(_get_any(enemy_hero, "mov_spd", default=0) or 0) / MAX_ATTR) if visible else 0.0,
            _clip(float(_get_any(enemy_hero, "atk_spd", default=0) or 0) / MAX_ATTR) if visible else 0.0,
            _clip(float(_get_any(enemy_hero, "attack_range", default=0) or 0) / 15000.0) if visible else 0.0,
            _norm_dist(dist),
            _norm_signed(dx),
            _norm_signed(dz),
            _norm_signed(self.last_enemy_seen["x"] - main_pos["x"]),
            _norm_signed(self.last_enemy_seen["z"] - main_pos["z"]),
            _clip((frame_no - self.last_enemy_seen.get("frame", frame_no)) / 500.0),
            1.0 if attack_target == _runtime_id(main_hero) else 0.0,
            1.0 if attack_target in friendly_soldier_ids else 0.0,
            1.0 if enemy_hero is not None and _dist_actor(enemy_hero, own_tower) < 12000 else 0.0,
            1.0 if enemy_hero is not None and _dist_actor(enemy_hero, enemy_tower) < 12000 else 0.0,
            _norm_signed(hp_adv, 1.0),
            _norm_signed(my_level - en_level, 15.0),
            _norm_signed(my_money - en_money, 30000.0),
            _clip(float(_get_any(enemy_hero, "total_hurt", default=0) or 0) / 60000.0) if visible else 0.0,
            _clip(float(_get_any(enemy_hero, "total_hurt_to_hero", default=0) or 0) / 20000.0) if visible else 0.0,
            _clip(float(_get_any(main_hero, "total_hurt_to_hero", default=0) or 0) / 20000.0),
            1.0 if visible and dist <= float(_get_any(main_hero, "attack_range", default=0) or 0) else 0.0,
        ]
        return self._append_pad(vals, 32)

    def _extract_skill_group(self, main_hero):
        slots = _slot_by_type(main_hero)
        vals = []
        for slot_type in [0, 1, 2, 3, 5, 6, 7]:
            slot = slots.get(slot_type)
            exists = 1.0 if slot is not None else 0.0
            key = (_to_int(_config_id(main_hero)), slot_type)
            used = float(_get_any(slot, "usedTimes", "used_times", default=0) or 0) if slot is not None else 0.0
            hit = float(_get_any(slot, "hitHeroTimes", "hit_hero_times", default=0) or 0) if slot is not None else 0.0
            used_delta = max(0.0, used - self.prev_slot_used.get(key, used))
            hit_delta = max(0.0, hit - self.prev_slot_hit.get(key, hit))
            self.prev_slot_used[key] = used
            self.prev_slot_hit[key] = hit
            vals.extend(
                [
                    exists,
                    _slot_usable(slot),
                    _slot_cd_ratio(slot),
                    _clip(float(_get_any(slot, "level", default=0) or 0) / 5.0) if slot is not None else 0.0,
                    _clip(used_delta / 5.0),
                    _clip(hit_delta / 5.0),
                    1.0 if float(_get_any(slot, "succUsedInFrame", "succ_used_in_frame", default=0) or 0) > 0 else 0.0,
                    _clip(float(_get_any(slot, "comboEffectTime", "combo_effect_time", default=0) or 0) / 5000.0) if slot is not None else 0.0,
                ]
            )
        return self._append_pad(vals, 56)

    def _soldier_kind(self, soldier):
        ar = float(_get_any(soldier, "attack_range", default=0) or 0)
        is_ranged = 1.0 if ar >= 5000 else 0.0
        is_melee = 1.0 - is_ranged
        return is_melee, is_ranged

    def _extract_lane_group(self, main_hero, friendly_soldiers, enemy_soldiers, own_tower, enemy_tower):
        fcnt, ecnt = len(friendly_soldiers), len(enemy_soldiers)
        fhp = sum(_hp_rate(s) for s in friendly_soldiers)
        ehp = sum(_hp_rate(s) for s in enemy_soldiers)
        main_pos = _loc(main_hero)
        nearest_enemy = min(enemy_soldiers, key=lambda s: _dist_actor(main_hero, s), default=None)
        lowest_enemy = min(enemy_soldiers, key=lambda s: _hp_rate(s), default=None)
        nearest_friendly = min(friendly_soldiers, key=lambda s: _dist_actor(main_hero, s), default=None)
        own_range = float(_get_any(own_tower, "attack_range", default=8800) or 8800)
        enemy_range = float(_get_any(enemy_tower, "attack_range", default=8800) or 8800)
        enemy_near_own = [s for s in enemy_soldiers if _dist_actor(s, own_tower) <= own_range * 1.15]
        friendly_near_enemy = [s for s in friendly_soldiers if _dist_actor(s, enemy_tower) <= enemy_range * 1.15]
        enemy_melee = enemy_ranged = friendly_melee = friendly_ranged = 0
        for s in enemy_soldiers:
            m, r = self._soldier_kind(s); enemy_melee += m; enemy_ranged += r
        for s in friendly_soldiers:
            m, r = self._soldier_kind(s); friendly_melee += m; friendly_ranged += r
        all_soldiers = friendly_soldiers + enemy_soldiers
        if all_soldiers:
            avg_x = sum(_loc(s)["x"] for s in all_soldiers) / len(all_soldiers)
            avg_z = sum(_loc(s)["z"] for s in all_soldiers) / len(all_soldiers)
        else:
            avg_x, avg_z = main_pos["x"], main_pos["z"]
        lane_push_adv = _clip((fcnt - ecnt + MAX_COUNT) / (2 * MAX_COUNT))
        enemy_ids = set(_runtime_id(s) for s in enemy_soldiers)
        friendly_ids = set(_runtime_id(s) for s in friendly_soldiers)
        own_tower_target = _get_any(own_tower, "attack_target", default=0)
        enemy_tower_target = _get_any(enemy_tower, "attack_target", default=0)
        n_melee, n_ranged = self._soldier_kind(nearest_enemy) if nearest_enemy is not None else (0.0, 0.0)
        vals = [
            _clip(fcnt / MAX_COUNT), _clip(ecnt / MAX_COUNT), _clip(fhp / MAX_COUNT), _clip(ehp / MAX_COUNT),
            _clip(fhp / max(1, fcnt)), _clip(ehp / max(1, ecnt)),
            _norm_dist(_dist_actor(main_hero, nearest_enemy)), _hp_rate(nearest_enemy), _hp_rate(lowest_enemy), _norm_dist(_dist_actor(main_hero, lowest_enemy)),
            _norm_dist(_dist_actor(main_hero, nearest_friendly)), _hp_rate(nearest_friendly),
            _clip(len(enemy_near_own) / MAX_COUNT), _clip(len(friendly_near_enemy) / MAX_COUNT),
            1.0 if len(enemy_near_own) > 0 else 0.0, 1.0 if len(friendly_near_enemy) > 0 else 0.0,
            _clip(enemy_melee / MAX_COUNT), _clip(enemy_ranged / MAX_COUNT), _clip(friendly_melee / MAX_COUNT), _clip(friendly_ranged / MAX_COUNT),
            _norm_pos(avg_x), _norm_pos(avg_z), lane_push_adv,
            _clip(len(enemy_near_own) / 4.0), _clip(len(friendly_near_enemy) / 4.0),
            1.0 if own_tower_target in enemy_ids else 0.0, 1.0 if enemy_tower_target in friendly_ids else 0.0,
            1.0 if enemy_tower_target in friendly_ids else 0.0, 1.0 if own_tower_target in enemy_ids else 0.0,
            1.0 if nearest_enemy is not None and _hp_rate(nearest_enemy) < 0.25 else 0.0,
            1.0 if lowest_enemy is not None and _hp_rate(lowest_enemy) < 0.25 else 0.0,
            n_ranged, n_melee,
            1.0 if any(_get_any(s, "attack_target", default=0) == _runtime_id(own_tower) for s in enemy_soldiers) else 0.0,
            1.0 if any(_get_any(s, "attack_target", default=0) in friendly_ids for s in enemy_soldiers) else 0.0,
            1.0 if any(_get_any(s, "attack_target", default=0) == _runtime_id(enemy_tower) for s in friendly_soldiers) else 0.0,
            1.0 if any(_get_any(s, "attack_target", default=0) in enemy_ids for s in friendly_soldiers) else 0.0,
            _clip((fcnt + ecnt) / (2 * MAX_COUNT)), 0.0, 1.0 if (fcnt + ecnt) > 0 else 0.0,
        ]
        return self._append_pad(vals, 40)

    def _extract_objective_group(self, frame_no, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower, enemy_tower, monsters, own_cake, enemy_cake):
        own_hp, enemy_hp = _hp_rate(own_tower), _hp_rate(enemy_tower)
        own_range = float(_get_any(own_tower, "attack_range", default=8800) or 8800)
        enemy_range = float(_get_any(enemy_tower, "attack_range", default=8800) or 8800)
        main_dist_enemy_tower = _dist_actor(main_hero, enemy_tower)
        friendly_ids = set(_runtime_id(s) for s in friendly_soldiers)
        enemy_tower_target = _get_any(enemy_tower, "attack_target", default=0)
        self_in_enemy_tower = main_dist_enemy_tower <= enemy_range
        tower_target_me = enemy_tower_target == _runtime_id(main_hero)
        friendly_tanking = enemy_tower_target in friendly_ids
        safe_push = friendly_tanking and main_dist_enemy_tower <= enemy_range * 1.20
        unsafe = self_in_enemy_tower and not friendly_tanking
        monster = monsters[0] if monsters else None
        mloc = _loc(monster) if monster is not None else {"x": 0.0, "z": 0.0}
        main_pos = _loc(main_hero)
        own_cake_exists, own_cake_dist, own_cake_dx, own_cake_dz = 0.0, DIST_NORM, 0.0, 0.0
        if own_cake is not None:
            own_cake_exists = 1.0
            loc = own_cake[1]
            own_cake_dist = _dist_loc(main_pos, loc)
            own_cake_dx = loc["x"] - main_pos["x"]
            own_cake_dz = loc["z"] - main_pos["z"]
        my_need_cake = 1.0 if _hp_rate(main_hero) < GameConfig.CAKE_LOW_HP_THRESHOLD else 0.0
        vals = [
            own_hp, enemy_hp, 0.0, 0.0,
            _norm_dist(_dist_actor(main_hero, own_tower)), _norm_dist(main_dist_enemy_tower),
            _norm_dist(own_range, 15000.0), _norm_dist(enemy_range, 15000.0),
            1.0 if self_in_enemy_tower else 0.0, 1.0 if tower_target_me else 0.0,
            1.0 if friendly_tanking else 0.0, 1.0 if safe_push else 0.0,
            1.0 if unsafe else 0.0, 1.0 if enemy_hp < 0.3 else 0.0,
            1.0 if monster is not None else 0.0, _hp_rate(monster), _norm_dist(_dist_actor(main_hero, monster)),
            _norm_signed(mloc["x"] - main_pos["x"]), _norm_signed(mloc["z"] - main_pos["z"]),
            1.0 if monster is not None and _dist_actor(main_hero, monster) <= 9000 else 0.0,
            0.0, own_cake_exists, _norm_dist(own_cake_dist), _norm_signed(own_cake_dx), _norm_signed(own_cake_dz),
            my_need_cake, 1.0 if my_need_cake and own_cake_exists and own_cake_dist < 12000 else 0.0,
            1.0 if enemy_cake is not None else 0.0,
            _clip(frame_no / MAX_FRAME),
            1.0 if frame_no < 3000 else 0.0,
            1.0 if 3000 <= frame_no < 12000 else 0.0,
            1.0 if frame_no >= 12000 else 0.0,
        ]
        return self._append_pad(vals, 32)

    def _extract_target_group(self, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower, enemy_tower, monsters):
        visible_enemy = enemy_hero is not None and _is_visible_to(enemy_hero, self.main_camp)
        enemy_dist = _dist_actor(main_hero, enemy_hero) if visible_enemy else DIST_NORM
        attack_range = float(_get_any(main_hero, "attack_range", default=8000) or 8000)
        vals = [
            1.0 if visible_enemy else 0.0,
            1.0 if visible_enemy else 0.0,
            _hp_rate(enemy_hero) if visible_enemy else self.last_enemy_seen.get("hp", 0.0),
            _norm_dist(enemy_dist),
            1.0 if visible_enemy and enemy_dist <= attack_range * 1.25 else 0.0,
            1.0,
            _hp_rate(main_hero),
            1.0 if _hp_rate(main_hero) < 0.5 else 0.0,
        ]
        all_soldiers = sorted(friendly_soldiers + enemy_soldiers, key=lambda s: _dist_actor(main_hero, s))[:4]
        own_range = float(_get_any(own_tower, "attack_range", default=8800) or 8800)
        enemy_range = float(_get_any(enemy_tower, "attack_range", default=8800) or 8800)
        for s in all_soldiers:
            is_enemy = 1.0 if _camp(s) != self.main_camp else 0.0
            under_tower = 1.0 if (_dist_actor(s, own_tower) <= own_range or _dist_actor(s, enemy_tower) <= enemy_range) else 0.0
            vals.extend([1.0, is_enemy, _hp_rate(s), _norm_dist(_dist_actor(main_hero, s)), 1.0 if is_enemy and _hp_rate(s) < 0.25 else 0.0, under_tower])
        while len(vals) < 8 + 24:
            vals.extend([0.0] * 6)
        vals.extend([
            1.0 if enemy_tower is not None else 0.0,
            _hp_rate(enemy_tower),
            _norm_dist(_dist_actor(main_hero, enemy_tower)),
            1.0 if _get_any(enemy_tower, "attack_target", default=0) in set(_runtime_id(s) for s in friendly_soldiers) else 0.0,
        ])
        monster = monsters[0] if monsters else None
        vals.extend([
            1.0 if monster is not None else 0.0,
            _hp_rate(monster),
            _norm_dist(_dist_actor(main_hero, monster)),
            1.0 if monster is not None and _dist_actor(main_hero, monster) <= 9000 else 0.0,
        ])
        return self._append_pad(vals, 40)

    def _extract_history_group(self, frame_no, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower, enemy_tower, lane_visible):
        real_cmd = _get_any(main_hero, "real_cmd", default=[]) or []
        has_real_cmd = len(real_cmd) > 0
        cmd_type = _to_int(_get(real_cmd[0], "command_type", -1), -1) if has_real_cmd else -1
        pos = _loc(main_hero)
        if self.last_self_pos is None:
            moved = DIST_NORM
        else:
            moved = math.sqrt((pos["x"] - self.last_self_pos[0]) ** 2 + (pos["z"] - self.last_self_pos[1]) ** 2)
        self.last_self_pos = (pos["x"], pos["z"])
        if moved < 100:
            self.same_position_steps += 1
        else:
            self.same_position_steps = 0
        if not has_real_cmd:
            self.no_real_cmd_steps += 1
        else:
            self.no_real_cmd_steps = 0
        hit_info = _get_any(main_hero, "hit_target_info", default=[]) or []
        hit_any = len(hit_info) > 0
        if _alive(main_hero) and moved < 100 and not has_real_cmd and not hit_any:
            self.no_effective_action_steps += 1
        else:
            self.no_effective_action_steps = 0
        in_grass = bool(_get_any(main_hero, "is_in_grass", default=False))
        if in_grass:
            self.grass_steps += 1
            if not hit_any and not has_real_cmd:
                self.grass_no_effective_steps += 1
            else:
                self.grass_no_effective_steps = 0
        else:
            self.grass_steps = 0
            self.grass_no_effective_steps = 0
        take_hurt = _get_any(main_hero, "take_hurt_infos", default=[]) or []
        own_range = float(_get_any(own_tower, "attack_range", default=8800) or 8800)
        enemy_ids = set(_runtime_id(s) for s in enemy_soldiers)
        defense_emergency = (
            _get_any(own_tower, "attack_target", default=0) in enemy_ids
            or any(_dist_actor(s, own_tower) <= own_range * 1.15 for s in enemy_soldiers)
        )
        vals = [
            1.0 if has_real_cmd else 0.0,
            _clip((cmd_type + 1) / 32.0),
            1.0 if cmd_type == 2 else 0.0,
            1.0 if cmd_type == 3 else 0.0,
            1.0 if cmd_type == 4 else 0.0,
            1.0 if cmd_type == 5 else 0.0,
            1.0 if cmd_type == 6 else 0.0,
            1.0 if cmd_type == 7 else 0.0,
            _clip(self.same_position_steps / 30.0),
            _clip(self.no_real_cmd_steps / 30.0),
            _clip(self.no_effective_action_steps / 30.0),
            _clip(self.no_effective_action_steps / 10.0),
            1.0 if hit_any else 0.0,
            1.0 if any(_get_any(info, "hit_target", default=0) == _runtime_id(enemy_hero) for info in hit_info) else 0.0,
            1.0 if any(_get_any(info, "hit_target", default=0) in enemy_ids for info in hit_info) else 0.0,
            1.0 if any(_get_any(info, "hit_target", default=0) == _runtime_id(enemy_tower) for info in hit_info) else 0.0,
            1.0 if len(take_hurt) > 0 else 0.0,
            0.0,
            1.0 if defense_emergency else 0.0,
            1.0 if lane_visible else 0.0,
            1.0 if in_grass else 0.0,
            _clip(self.grass_steps / 30.0),
            _clip(self.grass_no_effective_steps / 30.0),
            1.0 if in_grass and defense_emergency else 0.0,
        ]
        return self._append_pad(vals, 24)

    def process_feature(self, observation):
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = int(frame_state.get("frame_no", 0) or 0)
        main_hero, enemy_hero = self._find_heroes(frame_state)
        if main_hero is None:
            return [0.0] * FEATURE_DIM

        soldiers, towers, monsters = self._split_npcs(frame_state)
        own_tower, enemy_tower = self._find_towers(towers)
        friendly_soldiers, enemy_soldiers = self._split_soldiers(soldiers)
        self._update_tower_cache("last_own_tower_seen", own_tower, frame_no)
        self._update_tower_cache("last_enemy_tower_seen", enemy_tower, frame_no)
        own_tower_pos = (
            _loc(own_tower)
            if own_tower is not None
            else {"x": self.last_own_tower_seen["x"], "z": self.last_own_tower_seen["z"]}
        )
        enemy_tower_pos = _loc(enemy_tower) if enemy_tower is not None else self._fallback_enemy_tower_pos(own_tower)
        enemy_camp = 2 if self.main_camp == 1 else 1 if self.main_camp == 2 else None
        own_tower_ref = self._tower_ref(own_tower, self.last_own_tower_seen, own_tower_pos, self.main_camp)
        enemy_tower_ref = self._tower_ref(enemy_tower, self.last_enemy_tower_seen, enemy_tower_pos, enemy_camp)
        own_cake, enemy_cake = self._split_cakes(frame_state, own_tower_ref, enemy_tower_ref)
        lane_visible = len(friendly_soldiers) + len(enemy_soldiers) > 0
        enemy_tower_dist = _dist_actor(main_hero, enemy_tower_ref)
        enemy_tower_target = _get_any(enemy_tower_ref, "attack_target", default=0)
        friendly_ids = set(_runtime_id(s) for s in friendly_soldiers)
        friendly_tanking_enemy_tower = enemy_tower is not None and enemy_tower_target in friendly_ids
        enemy_tower_range = float(_get_any(enemy_tower_ref, "attack_range", default=8800) or 8800)
        in_enemy_tower = enemy_tower_dist <= enemy_tower_range
        unsafe_tower_entry = in_enemy_tower and not friendly_tanking_enemy_tower
        tower_target_me = enemy_tower is not None and enemy_tower_target == _runtime_id(main_hero)

        groups = [
            self._extract_self_group(frame_no, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower_ref, enemy_tower_ref),
            self._extract_enemy_group(frame_no, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower_ref, enemy_tower_ref),
            self._extract_skill_group(main_hero),
            self._extract_lane_group(main_hero, friendly_soldiers, enemy_soldiers, own_tower_ref, enemy_tower_ref),
            self._extract_objective_group(frame_no, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower_ref, enemy_tower_ref, monsters, own_cake, enemy_cake),
            self._extract_target_group(main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower_ref, enemy_tower_ref, monsters),
            self._extract_history_group(frame_no, main_hero, enemy_hero, friendly_soldiers, enemy_soldiers, own_tower_ref, enemy_tower_ref, lane_visible),
        ]
        feature = []
        for g in groups:
            feature.extend(g)
        if len(feature) != FEATURE_DIM:
            feature = self._append_pad(feature, FEATURE_DIM)
        return feature
