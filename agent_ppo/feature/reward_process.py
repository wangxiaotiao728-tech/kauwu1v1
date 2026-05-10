#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""D401 replica reward process.

This file intentionally follows the reward ideas in the provided D401 report:
hero_hurt, total_damage, hero_damage, crit, skill_hit, no_ops, in_grass,
under_tower_behavior, passive_skills, plus the original tower/forward baseline.
All field access is defensive because different environment versions expose
slightly different naming styles.
"""

import math
from agent_ppo.conf.conf import GameConfig


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


def _get(obj, key, default=0):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_any(obj, *keys, default=0):
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


def _loc(actor):
    return _get_any(actor, "location", default={"x": 0, "z": 0}) or {"x": 0, "z": 0}


def _dist(a, b):
    la, lb = _loc(a), _loc(b)
    dx = float(_get(la, "x", 0)) - float(_get(lb, "x", 0))
    dz = float(_get(la, "z", 0)) - float(_get(lb, "z", 0))
    return math.sqrt(dx * dx + dz * dz)


def _hp_rate(actor):
    hp = float(_get_any(actor, "hp", default=0))
    max_hp = max(1.0, float(_get_any(actor, "max_hp", default=1)))
    return hp / max_hp


def _is_alive(actor):
    return _get_any(actor, "hp", default=0) > 0


def _enum_str(value):
    return str(value).upper() if value is not None else ""


def _enum_matches(value, names, nums):
    s = _enum_str(value)
    if s in names:
        return True
    try:
        return int(value) in nums
    except Exception:
        return False


def _config_id(actor):
    return _get_any(actor, "config_id", "configId", default=None)


def _actor_type(actor):
    return _get_any(actor, "actor_type", "actorType", default=None)


def _sub_type(actor):
    return _get_any(actor, "sub_type", "subType", default=None)


def _is_tower(npc):
    # Official strict ordinary tower check: ACTOR_SUB_TOWER only.
    # Numeric fallback is configured in GameConfig.TOWER_SUB_TYPES.
    return _enum_matches(_sub_type(npc), {"ACTOR_SUB_TOWER"}, set(getattr(GameConfig, "TOWER_SUB_TYPES", {21})))


def _is_soldier(npc):
    # Official strict soldier check: ACTOR_SUB_SOLDIER only.
    # Unknown NPCs/monsters must not be counted as soldiers.
    if npc is None or not _is_alive(npc):
        return False
    if _enum_matches(_sub_type(npc), {"ACTOR_SUB_SOLDIER"}, set(getattr(GameConfig, "SOLDIER_SUB_TYPES", set()))):
        return True
    try:
        return int(_config_id(npc)) in set(getattr(GameConfig, "SOLDIER_CONFIG_IDS", set()))
    except Exception:
        return False


def _is_monster(npc):
    if npc is None or not _is_alive(npc):
        return False
    if _enum_matches(_actor_type(npc), {"ACTOR_TYPE_MONSTER"}, set(getattr(GameConfig, "MONSTER_ACTOR_TYPES", set()))):
        return True
    try:
        return int(_config_id(npc)) in set(getattr(GameConfig, "MONSTER_CONFIG_IDS", set()))
    except Exception:
        return False


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        for k, v in {
            1: 160,
            2: 298,
            3: 446,
            4: 524,
            5: 613,
            6: 713,
            7: 825,
            8: 950,
            9: 1088,
            10: 1240,
            11: 1406,
            12: 1585,
            13: 1778,
            14: 1984,
        }.items():
            self.m_each_level_max_exp[k] = v

    def _compose_weighted_reward_groups(self):
        """Return weighted group rewards and scalar sum from current m_reward_value.

        Each reward item value is already a delta/potential value. It is
        multiplied by its configured scale and accumulated into its D401 reward
        group. These group rewards are later used to compute separate GAE
        returns for the multi-critic heads.
        """
        reward_groups = {name: 0.0 for name in GameConfig.REWARD_GROUPS.keys()}
        assigned = set()
        for group_name, keys in GameConfig.REWARD_GROUPS.items():
            group_sum = 0.0
            for key in keys:
                assigned.add(key)
                if key not in self.m_cur_calc_frame_map:
                    continue
                weight = self.m_cur_calc_frame_map[key].weight
                group_sum += self.m_reward_value.get(key, 0.0) * weight
            reward_groups[group_name] = group_sum

        # Any ungrouped reward is kept in decay group by default, so no reward
        # silently disappears if users later add an item to REWARD_WEIGHT_DICT.
        default_group = next(iter(reward_groups.keys()), None)
        if default_group is not None:
            for key, reward_struct in self.m_cur_calc_frame_map.items():
                if key in assigned:
                    continue
                reward_groups[default_group] += self.m_reward_value.get(key, 0.0) * reward_struct.weight

        reward_sum = sum(reward_groups.values())
        return reward_groups, reward_sum

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        frame_no = frame_data.get("frame_no", 0)
        if self.time_scale_arg > 0:
            decay_factor = math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
            for key in list(self.m_reward_value.keys()):
                if key in ("reward_sum", "reward_groups") or key in GameConfig.NO_DECAY_REWARD_KEYS:
                    continue
                self.m_reward_value[key] *= decay_factor

        reward_groups, reward_sum = self._compose_weighted_reward_groups()
        self.m_reward_value["reward_groups"] = reward_groups
        self.m_reward_value["reward_sum"] = reward_sum
        return self.m_reward_value

    def _find_main_and_enemy(self, frame_data, camp):
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []):
            if _get_any(hero, "camp", default=None) == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _find_towers(self, frame_data, camp):
        main_tower, enemy_tower = None, None
        for npc in frame_data.get("npc_states", []):
            if not _is_tower(npc):
                continue
            if _get_any(npc, "camp", default=None) == camp:
                main_tower = npc
            else:
                enemy_tower = npc
        return main_tower, enemy_tower

    def _friendly_soldier_near_tower(self, frame_data, camp, tower):
        if tower is None:
            return False
        tower_range = float(_get_any(tower, "attack_range", default=0) or 0)
        for npc in frame_data.get("npc_states", []):
            if _get_any(npc, "camp", default=None) != camp:
                continue
            if not _is_soldier(npc):
                continue
            if _hp_rate(npc) <= 0:
                continue
            if tower_range > 0 and _dist(npc, tower) <= tower_range * 1.10:
                return True
        return False


    def _enemy_soldier_count(self, frame_data, camp):
        """Count alive enemy soldiers from the perspective of camp.

        This is intentionally coarse and result-style: reducing enemy soldier
        count while keeping friendly soldiers creates positive lane advantage.
        It does not reward every point of minion HP damage.
        """
        cnt = 0
        for npc in frame_data.get("npc_states", []):
            if _get_any(npc, "camp", default=None) == camp:
                continue
            if _is_tower(npc):
                continue
            if not _is_alive(npc):
                continue
            if not _is_soldier(npc):
                continue
            cnt += 1
        return cnt

    def _calc_bad_skill(self, hero):
        """Small miss proxy: usedTimes - hitHeroTimes.

        D401 already rewards skill_hit. This extra metric gives a weak negative
        signal when skill usage increases but hero hit count does not. Keep the
        configured weight small because some skills are valid for clearing lane.
        """
        skill_state = _get_any(hero, "skill_state", default={})
        slot_states = _get(skill_state, "slot_states", []) or []
        used_total = 0.0
        hit_total = 0.0
        for slot in slot_states:
            used_total += float(_get_any(slot, "usedTimes", "used_times", default=0) or 0)
            hit_total += float(_get_any(slot, "hitHeroTimes", "hit_hero_times", default=0) or 0)
        return max(0.0, used_total - hit_total) / 100.0

    def _calc_skill_hit(self, hero):
        skill_state = _get_any(hero, "skill_state", default={})
        slot_states = _get(skill_state, "slot_states", []) or []
        vals = []
        for slot in slot_states:
            used = float(_get_any(slot, "usedTimes", "used_times", default=0) or 0)
            if used <= 0:
                continue
            hit = float(_get_any(slot, "hitHeroTimes", "hit_hero_times", default=0) or 0)
            vals.append(hit / max(1.0, used))
        return sum(vals) / max(1, len(slot_states)) if slot_states else 0.0

    def _calc_in_grass(self, hero, enemy_hero):
        if not bool(_get_any(hero, "is_in_grass", default=False)):
            return 0.0
        val = 0.25
        main_visible = _get_any(hero, "camp_visible", default=[])
        enemy_visible = _get_any(enemy_hero, "camp_visible", default=[])
        try:
            main_vis_all = all(main_visible)
            enemy_vis_all = all(enemy_visible)
            if (not main_vis_all) and enemy_vis_all:
                val += 0.5
        except Exception:
            pass
        if enemy_hero is not None:
            attack_range = float(_get_any(hero, "attack_range", default=0) or 0)
            if attack_range > 0 and _dist(hero, enemy_hero) <= attack_range:
                val += 0.5
        return val

    def _calc_under_tower_behavior(self, frame_data, hero, enemy_hero, enemy_tower, camp):
        if hero is None or enemy_tower is None:
            return 0.0
        tower_range = float(_get_any(enemy_tower, "attack_range", default=0) or 0)
        in_tower_range = _dist(hero, enemy_tower) <= tower_range
        if not in_tower_range:
            return 0.0
        has_ally_soldier = self._friendly_soldier_near_tower(frame_data, camp, enemy_tower)
        reward = 0.0
        if not has_ally_soldier:
            reward -= 0.5
        attack_target = _get_any(hero, "attack_target", default=-1)
        if has_ally_soldier:
            if attack_target == _get_any(enemy_tower, "runtime_id", default=None):
                reward += 1.0
            elif enemy_hero is not None and attack_target == _get_any(enemy_hero, "runtime_id", default=None):
                reward -= 0.3
            else:
                for npc in frame_data.get("npc_states", []):
                    if _get_any(npc, "camp", default=None) == camp:
                        continue
                    if not _is_soldier(npc):
                        continue
                    if attack_target == _get_any(npc, "runtime_id", default=None):
                        reward += 0.2
                        break
        return reward

    def _calc_passive_skills(self, hero):
        """Official protocol exposes PassiveSkill(passive_skillid, cooldown).

        It does not expose D401-style level/triggered fields, so this reward is
        intentionally conservative. The default weight in conf.py is 0.0; this
        value is kept only for monitoring / future ablation.
        """
        passive_skills = _get_any(hero, "passive_skill", default=[]) or []
        if not passive_skills:
            return 0.0
        ready_cnt = 0
        for skill in passive_skills:
            cooldown = float(_get(skill, "cooldown", 0) or 0)
            if cooldown <= 0:
                ready_cnt += 1
        return ready_cnt / max(1.0, float(len(passive_skills)))

    def set_cur_calc_frame_vec(self, calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = self._find_main_and_enemy(frame_data, camp)
        main_tower, enemy_tower = self._find_towers(frame_data, camp)
        for reward_name, reward_struct in calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if main_hero is None:
                reward_struct.cur_frame_value = 0.0
                continue
            if reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = _hp_rate(main_tower)
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = _hp_rate(main_hero)
            elif reward_name == "money":
                reward_struct.cur_frame_value = float(_get_any(main_hero, "money_cnt", "money", default=0)) / 30000.0
            elif reward_name == "exp":
                level = int(_get_any(main_hero, "level", default=1) or 1)
                max_exp = max(1, self.m_each_level_max_exp.get(level, 2000))
                reward_struct.cur_frame_value = level / 15.0 + float(_get_any(main_hero, "exp", default=0)) / max_exp / 15.0
            elif reward_name == "kill":
                reward_struct.cur_frame_value = float(_get_any(main_hero, "kill_cnt", default=0))
            elif reward_name == "death":
                reward_struct.cur_frame_value = float(_get_any(main_hero, "dead_cnt", default=0))
            elif reward_name == "ep_rate":
                ep = float(_get_any(main_hero, "ep", default=0))
                max_ep = max(1.0, float(_get_any(main_hero, "max_ep", default=1)))
                reward_struct.cur_frame_value = ep / max_ep
            elif reward_name == "last_hit":
                # Baseline protocol does not expose an explicit last-hit event here.
                # Use money_cnt as a weak monotonically increasing proxy.
                reward_struct.cur_frame_value = float(_get_any(main_hero, "money_cnt", "money", default=0)) / 30000.0
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "lane_clear":
                # Potential is negative enemy soldier count. Through the standard
                # zero-sum delta, this rewards reducing enemy soldiers and keeping
                # friendly lane presence.
                reward_struct.cur_frame_value = -float(self._enemy_soldier_count(frame_data, camp)) / 10.0
            elif reward_name == "hero_hurt":
                reward_struct.cur_frame_value = float(
                    _get_any(main_hero, "total_be_hurt_by_hero", "totalBeHurtByHero", default=0)
                ) / 2e4
            elif reward_name == "hero_damage":
                reward_struct.cur_frame_value = float(
                    _get_any(main_hero, "total_hurt_to_hero", "totalHurtToHero", default=0)
                ) / 2e4
            elif reward_name == "total_damage":
                reward_struct.cur_frame_value = float(_get_any(main_hero, "total_hurt", "totalHurt", default=0)) / 6e4
            elif reward_name == "crit":
                crit_rate = float(_get_any(main_hero, "crit_rate", default=0))
                crit_effe = float(_get_any(main_hero, "crit_effe", default=0))
                reward_struct.cur_frame_value = crit_rate * crit_effe / 1e4
            elif reward_name == "skill_hit":
                reward_struct.cur_frame_value = self._calc_skill_hit(main_hero)
            elif reward_name == "bad_skill":
                reward_struct.cur_frame_value = self._calc_bad_skill(main_hero)
            elif reward_name == "no_ops":
                behav = _get_any(main_hero, "behav_mode", default="")
                reward_struct.cur_frame_value = 1.0 if str(behav) == "State_Idle" or behav == 0 else 0.0
            elif reward_name == "in_grass":
                reward_struct.cur_frame_value = self._calc_in_grass(main_hero, enemy_hero)
            elif reward_name == "under_tower_behavior":
                reward_struct.cur_frame_value = self._calc_under_tower_behavior(
                    frame_data, main_hero, enemy_hero, enemy_tower, camp
                )
            elif reward_name == "passive_skills":
                reward_struct.cur_frame_value = self._calc_passive_skills(main_hero)

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0
        dist_hero2emy = _dist(main_hero, enemy_tower)
        dist_main2emy = max(1.0, _dist(main_tower, enemy_tower))
        if _hp_rate(main_hero) > 0.99 and dist_hero2emy > dist_main2emy:
            return (dist_main2emy - dist_hero2emy) / dist_main2emy
        return 0.0

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data.get("hero_states", []):
            runtime_id = _get_any(hero, "runtime_id", default=None)
            player_id = _get_any(hero, "player_id", default=None)
            if runtime_id == self.main_hero_player_id or player_id == self.main_hero_player_id:
                main_camp = _get_any(hero, "camp", default=-1)
                self.main_hero_camp = main_camp
            else:
                enemy_camp = _get_any(hero, "camp", default=-1)
        if main_camp == -1 and frame_data.get("hero_states"):
            main_camp = _get_any(frame_data["hero_states"][0], "camp", default=-1)
        if enemy_camp == -1:
            for hero in frame_data.get("hero_states", []):
                if _get_any(hero, "camp", default=-1) != main_camp:
                    enemy_camp = _get_any(hero, "camp", default=-1)
                    break
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            reward_dict[reward_name] = reward_struct.value
        reward_groups, reward_sum = self._compose_weighted_reward_groups()
        reward_dict["reward_groups"] = reward_groups
        reward_dict["reward_sum"] = reward_sum
