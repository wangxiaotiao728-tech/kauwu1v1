#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Smooth reward manager for the 1v1 PPO agent.

The reward names follow the text方案, but each dense item is calculated from
small normalized deltas so PPO receives continuous learning signals.
"""

import math

from agent_ppo.conf.conf import GameConfig, RuleConfig


TOWER_SUB_TYPE = 21


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_last_scores = None
        self.m_last_terminal_frame = -1
        self.m_each_level_max_exp = {}
        self.m_last_known = {
            "my_hp": 1.0,
            "enemy_hp": 1.0,
            "own_tower_hp": 1.0,
            "enemy_tower_hp": 1.0,
            "blood_pack_dist": 100000.0,
        }

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp = {
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
        }

    def result(self, frame_data, observation=None, terminated=False, truncated=False):
        self.init_max_exp_of_each_hero()
        scores = self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value, scores, observation, terminated, truncated)
        return self.m_reward_value

    def frame_data_process(self, frame_data):
        my_hero, enemy_hero = self._split_heroes(frame_data)
        my_camp = my_hero.get("camp") if my_hero else None
        own_tower, enemy_tower, friendly_minions, enemy_minions, monsters = self._split_npcs(frame_data, my_camp)
        blood_pack_dist, safe_blood_pack = self._blood_pack_state(frame_data, my_hero, enemy_hero, enemy_tower)
        lane_score = self._lane_score(own_tower, enemy_tower, friendly_minions, enemy_minions)

        my_hp = self._stable_hp("my_hp", my_hero, 1.0)
        enemy_hp = self._stable_hp("enemy_hp", enemy_hero, 1.0)
        own_tower_hp = self._stable_hp("own_tower_hp", own_tower, 1.0)
        enemy_tower_hp = self._stable_hp("enemy_tower_hp", enemy_tower, 1.0)

        scores = {
            "my_hp": my_hp,
            "enemy_hp": enemy_hp,
            "own_tower_hp": own_tower_hp,
            "enemy_tower_hp": enemy_tower_hp,
            "my_money": self._money_value(my_hero),
            "my_exp": (my_hero or {}).get("exp", 0),
            "my_level": (my_hero or {}).get("level", 1),
            "my_kill_cnt": (my_hero or {}).get("kill_cnt", 0),
            "my_dead_cnt": (my_hero or {}).get("dead_cnt", 0),
            "enemy_dead": 1.0 if enemy_hero and not self._alive(enemy_hero) else 0.0,
            "lane_score": lane_score,
            "enemy_minion_count": len(enemy_minions),
            "enemy_minion_hp_sum": sum(self._hp_ratio(unit) for unit in enemy_minions),
            "friendly_minion_count": len(friendly_minions),
            "tower_risk": self._tower_risk(my_hero, enemy_tower),
            "tower_target_self": 1.0 if self._tower_targets_self(my_hero, enemy_tower) else 0.0,
            "safe_blood_pack": safe_blood_pack,
            "blood_pack_dist": blood_pack_dist,
            "enemy_visible": 1.0 if enemy_hero else 0.0,
            "enhanced_ready": self._enhanced_attack_ready(my_hero),
            "last_hit": self._last_hit_reward(frame_data),
            "monsters": len(monsters),
        }
        return scores

    def get_reward(self, frame_data, reward_dict, scores, observation=None, terminated=False, truncated=False):
        reward_dict.clear()
        terminal_items = self._terminal_items(frame_data, observation, terminated, truncated)
        if self.m_last_scores is None:
            reward_items = {key: 0.0 for key in self.m_cur_calc_frame_map}
            reward_items.update(terminal_items)
            self._write_reward(reward_dict, reward_items)
            self.m_last_scores = scores
            return

        last = self.m_last_scores
        enemy_tower_damage = max(0.0, last["enemy_tower_hp"] - scores["enemy_tower_hp"])
        own_tower_damage = max(0.0, last["own_tower_hp"] - scores["own_tower_hp"])
        enemy_hero_damage = max(0.0, last["enemy_hp"] - scores["enemy_hp"])
        self_damage_taken = max(0.0, last["my_hp"] - scores["my_hp"])
        lane_delta = scores["lane_score"] - last["lane_score"]
        enemy_minion_hp_drop = max(0.0, last["enemy_minion_hp_sum"] - scores["enemy_minion_hp_sum"])
        enemy_minion_count_drop = max(0.0, last["enemy_minion_count"] - scores["enemy_minion_count"])
        gold_gain = max(0.0, scores["my_money"] - last["my_money"])
        level_gain = max(0.0, scores["my_level"] - last["my_level"])
        exp_gain = max(0.0, scores["my_exp"] - last["my_exp"]) if level_gain <= 0 else level_gain * 300.0
        kill_gain = max(0.0, scores["my_kill_cnt"] - last["my_kill_cnt"])
        death_gain = max(0.0, scores["my_dead_cnt"] - last["my_dead_cnt"])
        hp_gain = max(0.0, scores["my_hp"] - last["my_hp"])
        blood_dist_not_improving = scores["blood_pack_dist"] >= last["blood_pack_dist"] - 250.0
        vision_risk = (1.0 - scores["enemy_visible"]) * self._sigmoid((RuleConfig.BLOOD_PACK_HP_RATIO - scores["my_hp"]) / 0.08) * max(0.3, scores["tower_risk"])

        reward_items = {
            "enemy_tower_damage": enemy_tower_damage,
            "own_tower_damage": own_tower_damage,
            "enhanced_tower_hit": enemy_tower_damage * scores["enhanced_ready"],
            "death_window_tower": scores["enemy_dead"] * max(enemy_tower_damage, max(0.0, lane_delta)),
            "tower_target_self": scores["tower_target_self"],
            "unsafe_tower_exposure": scores["tower_risk"] * self._sigmoid((RuleConfig.BLOOD_PACK_HP_RATIO - scores["my_hp"]) / 0.10),
            "lane_push": lane_delta,
            "enemy_minion_kill": enemy_minion_count_drop + 0.25 * enemy_minion_hp_drop,
            "last_hit": scores["last_hit"],
            "gold": gold_gain / 100.0,
            "exp": exp_gain / 100.0,
            "level_up": level_gain,
            "enemy_hero_damage": enemy_hero_damage,
            "self_damage_taken": self_damage_taken,
            "kill": kill_gain,
            "death": death_gain,
            "blood_pack_heal": hp_gain if last["safe_blood_pack"] > 0.0 or last["blood_pack_dist"] < 2500 else 0.0,
            "ignore_safe_blood_pack": scores["safe_blood_pack"] if blood_dist_not_improving else 0.0,
            "vision_risk": vision_risk,
        }
        reward_items.update(terminal_items)
        self._write_reward(reward_dict, reward_items)
        self.m_last_scores = scores
        self.m_last_known["blood_pack_dist"] = scores["blood_pack_dist"]

    def _write_reward(self, reward_dict, reward_items):
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            reward_struct.value = reward_items.get(reward_name, 0.0)
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = self._clamp(
            reward_sum,
            GameConfig.REWARD_SUM_CLIP_MIN,
            GameConfig.REWARD_SUM_CLIP_MAX,
        )

    def _terminal_items(self, frame_data, observation, terminated, truncated):
        items = {"win": 0.0, "lose": 0.0, "timeout": 0.0}
        if not terminated and not truncated:
            return items
        frame_no = frame_data.get("frame_no", 0)
        if self.m_last_terminal_frame == frame_no:
            return items
        self.m_last_terminal_frame = frame_no
        if truncated:
            items["timeout"] = 1.0
        elif observation and observation.get("win", 0):
            items["win"] = 1.0
        else:
            items["lose"] = 1.0
        return items

    def _stable_hp(self, key, obj, default):
        if obj:
            value = self._hp_ratio(obj)
            self.m_last_known[key] = value
            return value
        return self.m_last_known.get(key, default)

    def _lane_score(self, own_tower, enemy_tower, friendly_minions, enemy_minions):
        friendly_count = self._safe_div(len(friendly_minions), 8.0)
        enemy_count = self._safe_div(len(enemy_minions), 8.0)
        friendly_hp = self._safe_div(sum(self._hp_ratio(unit) for unit in friendly_minions), 8.0)
        enemy_hp = self._safe_div(sum(self._hp_ratio(unit) for unit in enemy_minions), 8.0)
        enemy_tower_pos = self._pos(enemy_tower)
        own_tower_pos = self._pos(own_tower)
        friendly_front = min((self._dist(self._pos(unit), enemy_tower_pos) for unit in friendly_minions), default=30000.0)
        enemy_front = min((self._dist(self._pos(unit), own_tower_pos) for unit in enemy_minions), default=30000.0)
        push_front = 1.0 - self._safe_div(friendly_front, 30000.0)
        defend_front = 1.0 - self._safe_div(enemy_front, 30000.0)
        return 0.30 * (friendly_count - enemy_count) + 0.25 * (friendly_hp - enemy_hp) + 0.30 * push_front - 0.15 * defend_front

    def _blood_pack_state(self, frame_data, my_hero, enemy_hero, enemy_tower):
        cakes = frame_data.get("cakes", []) or []
        if not cakes or not my_hero:
            return 100000.0, 0.0
        my_pos = self._pos(my_hero)
        cake = min(cakes, key=lambda item: self._dist(my_pos, self._cake_pos(item)))
        cake_pos = self._cake_pos(cake)
        dist = self._dist(my_pos, cake_pos)
        need = self._sigmoid((RuleConfig.BLOOD_PACK_HP_RATIO - self._hp_ratio(my_hero)) / 0.08)
        risk = self._tower_risk(my_hero, enemy_tower)
        enemy_dist = self._dist(self._pos(enemy_hero), cake_pos)
        closer = 1.0 if dist <= enemy_dist else 0.4
        safe = need * (1.0 - risk) * closer * math.exp(-min(dist, 30000.0) / 8000.0)
        return dist, safe

    def _last_hit_reward(self, frame_data):
        score = 0.0
        frame_action = frame_data.get("frame_action", {}) or {}
        for dead in frame_action.get("dead_action", []) or []:
            killer = dead.get("killer", {}) or {}
            death = dead.get("death", {}) or {}
            if killer.get("runtime_id") != self.main_hero_player_id:
                continue
            if death.get("actor_type") == "ACTOR_TYPE_HERO":
                continue
            if "TOWER" in str(death.get("sub_type", "")).upper():
                continue
            score += 1.0
        return score

    def _enhanced_attack_ready(self, my_hero):
        if not my_hero:
            return 0.0
        buffs = ((my_hero.get("buff_state") or {}).get("buff_skills", []) or [])
        marks = ((my_hero.get("buff_state") or {}).get("buff_marks", []) or [])
        passive = my_hero.get("passive_skill", []) or []
        if my_hero.get("config_id") in (112, 133):
            return 1.0 if buffs or marks or any(skill.get("cooldown", 0) <= 0 for skill in passive) else 0.0
        return 0.0

    def _tower_risk(self, my_hero, enemy_tower):
        if not my_hero or not enemy_tower or not self._alive(my_hero):
            return 0.0
        dist = self._dist(self._pos(my_hero), self._pos(enemy_tower))
        tower_range = enemy_tower.get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE)
        smooth = self._sigmoid((tower_range - dist) / RuleConfig.TOWER_RISK_SIGMOID_SCALE)
        target_self = 1.0 if enemy_tower.get("attack_target") == my_hero.get("runtime_id") else 0.0
        low_hp = self._sigmoid((RuleConfig.LOW_HP_RISK_RATIO - self._hp_ratio(my_hero)) / 0.08)
        return self._clamp(smooth * (0.35 + 0.45 * target_self + 0.20 * low_hp))

    def _tower_targets_self(self, my_hero, enemy_tower):
        return bool(my_hero and enemy_tower and enemy_tower.get("attack_target") == my_hero.get("runtime_id"))

    def _split_heroes(self, frame_data):
        heroes = frame_data.get("hero_states", []) or []
        my_hero, enemy_hero = None, None
        for hero in heroes:
            if hero.get("runtime_id") == self.main_hero_player_id:
                my_hero = hero
                break
        my_camp = my_hero.get("camp") if my_hero else None
        for hero in heroes:
            if hero is not my_hero and not self._same_camp(hero.get("camp"), my_camp):
                enemy_hero = hero
                break
        return my_hero, enemy_hero

    def _split_npcs(self, frame_data, my_camp):
        own_tower, enemy_tower = None, None
        friendly_minions, enemy_minions, monsters = [], [], []
        for npc in frame_data.get("npc_states", []) or []:
            npc_camp = npc.get("camp")
            if self._is_tower(npc):
                if self._same_camp(npc_camp, my_camp):
                    own_tower = npc
                else:
                    enemy_tower = npc
            elif self._is_monster(npc):
                monsters.append(npc)
            elif self._same_camp(npc_camp, my_camp):
                friendly_minions.append(npc)
            elif npc_camp not in (0, None, "0"):
                enemy_minions.append(npc)
        return own_tower, enemy_tower, friendly_minions, enemy_minions, monsters

    def _is_tower(self, npc):
        sub_type = npc.get("sub_type")
        return sub_type == TOWER_SUB_TYPE or "TOWER" in str(sub_type).upper()

    def _is_monster(self, npc):
        actor_type = str(npc.get("actor_type", "")).upper()
        sub_type = str(npc.get("sub_type", "")).upper()
        camp = self._camp_value(npc.get("camp"))
        return "MONSTER" in actor_type or "MONSTER" in sub_type or camp in (0, None, "0")

    def _hp_ratio(self, obj):
        return self._clamp(self._safe_div((obj or {}).get("hp", 0), (obj or {}).get("max_hp", 0)))

    def _alive(self, obj):
        return bool(obj and obj.get("hp", 0) > 0)

    def _money_value(self, hero):
        return (hero or {}).get("money_cnt", (hero or {}).get("money", 0))

    def _pos(self, obj):
        loc = obj.get("location", {}) if obj else {}
        return loc.get("x", 100000), loc.get("z", 100000)

    def _cake_pos(self, cake):
        loc = (cake.get("collider", {}) or {}).get("location", {})
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

    def _sigmoid(self, value):
        value = max(-30.0, min(30.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def _safe_div(self, num, den):
        try:
            return num / den if den else 0.0
        except (TypeError, ZeroDivisionError):
            return 0.0

    def _clamp(self, value, min_value=0.0, max_value=1.0):
        return max(min_value, min(max_value, value))
