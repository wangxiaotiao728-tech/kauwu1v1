#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""目标型 reward，禁止普通血量变化奖惩。"""

import math

from agent_ppo.conf.conf import CurriculumConfig, GameConfig, RuleConfig


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

    def init_max_exp_of_each_hero(self):
        pass

    def result(self, frame_data, observation=None, terminated=False, truncated=False):
        scores = self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value, scores, observation, terminated, truncated)
        return self.m_reward_value

    def frame_data_process(self, frame_data):
        my_hero, enemy_hero = self._split_heroes(frame_data)
        my_camp = (my_hero or {}).get("camp")
        own_tower, enemy_tower, friendly_minions, enemy_minions, neutral_units = self._split_npcs(frame_data, my_camp)
        lane_score = self._lane_score(own_tower, enemy_tower, friendly_minions, enemy_minions)
        direct_push_window = self._direct_push_window(my_hero, enemy_hero, enemy_tower, friendly_minions)
        defense_emergency = self._defense_emergency(own_tower, enemy_minions)
        resource_allowed = (not direct_push_window) and (not defense_emergency) and self._hp_ratio(my_hero) > 0.45

        return {
            "my_hp": self._hp_ratio(my_hero),
            "own_tower_hp": self._hp_ratio(own_tower),
            "enemy_tower_hp": self._hp_ratio(enemy_tower),
            "my_money": self._money_value(my_hero),
            "my_exp": (my_hero or {}).get("exp", 0),
            "my_level": (my_hero or {}).get("level", 1),
            "my_dead_cnt": (my_hero or {}).get("dead_cnt", 0),
            "my_kill_cnt": (my_hero or {}).get("kill_cnt", 0),
            "lane_score": lane_score,
            "last_hit": self._last_hit_reward(frame_data),
            "tower_risk": self._tower_risk(my_hero, enemy_tower),
            "enemy_dead": 1.0 if enemy_hero and not self._alive(enemy_hero) else 0.0,
            "enhanced_ready": self._enhanced_attack_ready(my_hero),
            "direct_push_window": direct_push_window,
            "defense_emergency": defense_emergency,
            "resource_allowed": resource_allowed,
            "neutral_taken": self._neutral_taken(frame_data),
            "safe_cake_pick": self._safe_cake_pick(frame_data, my_hero, enemy_hero, enemy_tower),
            "skill_result": self._skill_result(my_hero),
            "bad_resource": self._bad_resource_attempt(my_hero, neutral_units, resource_allowed),
        }

    def get_reward(self, frame_data, reward_dict, scores, observation=None, terminated=False, truncated=False):
        reward_dict.clear()
        terminal_items = self._terminal_items(frame_data, observation, terminated, truncated)
        if self.m_last_scores is None:
            reward_items = {key: 0.0 for key in self.m_cur_calc_frame_map}
            reward_items.update(terminal_items)
            self._write_reward(reward_dict, reward_items, self._channels(reward_items))
            self.m_last_scores = scores
            return

        last = self.m_last_scores
        enemy_tower_delta = max(0.0, last["enemy_tower_hp"] - scores["enemy_tower_hp"])
        own_tower_delta = max(0.0, last["own_tower_hp"] - scores["own_tower_hp"])
        lane_delta = scores["lane_score"] - last["lane_score"]
        growth = max(0.0, scores["my_money"] - last["my_money"]) / 100.0
        growth += max(0.0, scores["my_exp"] - last["my_exp"]) / 100.0
        growth += max(0.0, scores["my_level"] - last["my_level"])
        death_delta = max(0.0, scores["my_dead_cnt"] - last["my_dead_cnt"])

        objective_scale = 1.0
        defense_scale = 1.0
        growth_scale = 1.0
        resource_scale = 1.0
        cake_scale = 1.0
        if scores["direct_push_window"]:
            objective_scale = 1.5
            growth_scale = 0.3
            resource_scale = 0.0
            cake_scale = 0.0
        if scores["defense_emergency"]:
            defense_scale = 1.5
            resource_scale = 0.0
            growth_scale = min(growth_scale, 0.5)

        stage = self._stage()
        reward_items = {
            "terminal_win": terminal_items["terminal_win"],
            "terminal_lose": terminal_items["terminal_lose"],
            "timeout": terminal_items["timeout"],
            "enemy_tower_delta": enemy_tower_delta * objective_scale,
            "own_tower_delta": own_tower_delta * defense_scale,
            "lane": lane_delta,
            "growth": growth * growth_scale,
            "last_hit": scores["last_hit"],
            "death": death_delta,
            "tower_risk": scores["tower_risk"] * self._sigmoid((RuleConfig.LOW_HP_RISK_RATIO - scores["my_hp"]) / 0.10),
            "enhanced_tower": enemy_tower_delta * scores["enhanced_ready"] if stage.get("enable_enhanced_tower_reward", True) else 0.0,
            "death_window_tower_mult": enemy_tower_delta * scores["enemy_dead"] if stage.get("enable_death_window_reward", True) else 0.0,
            "neutral_result": scores["neutral_taken"] * resource_scale if stage.get("enable_resource_reward", False) else 0.0,
            "cake_safe_pick": scores["safe_cake_pick"] * cake_scale if stage.get("enable_cake_reward", False) else 0.0,
            "skill_result": scores["skill_result"] if stage.get("enable_skill_result_reward", False) else 0.0,
            "bad_resource": scores["bad_resource"] if stage.get("enable_resource_reward", False) else 0.0,
        }
        self._write_reward(reward_dict, reward_items, self._channels(reward_items))
        self.m_last_scores = scores

    def _write_reward(self, reward_dict, reward_items, channels):
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            reward_struct.value = reward_items.get(reward_name, 0.0)
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict.update(channels)
        reward_dict["reward_sum"] = self._clamp(reward_sum, GameConfig.REWARD_SUM_CLIP_MIN, GameConfig.REWARD_SUM_CLIP_MAX)

    def _channels(self, reward_items):
        weights = GameConfig.REWARD_WEIGHT_DICT
        return {
            "terminal": reward_items.get("terminal_win", 0.0) * weights["terminal_win"] + reward_items.get("terminal_lose", 0.0) * weights["terminal_lose"],
            "tower": reward_items.get("enemy_tower_delta", 0.0) * weights["enemy_tower_delta"],
            "tower_defense": reward_items.get("own_tower_delta", 0.0) * weights["own_tower_delta"],
            "lane": reward_items.get("lane", 0.0) * weights["lane"],
            "growth": reward_items.get("growth", 0.0) * weights["growth"],
            "last_hit": reward_items.get("last_hit", 0.0) * weights["last_hit"],
            "enhanced_tower": reward_items.get("enhanced_tower", 0.0) * weights["enhanced_tower"],
            "resource": reward_items.get("neutral_result", 0.0) * weights["neutral_result"],
            "cake": reward_items.get("cake_safe_pick", 0.0) * weights["cake_safe_pick"],
            "skill": reward_items.get("skill_result", 0.0) * weights["skill_result"],
            "death": reward_items.get("death", 0.0) * weights["death"],
            "tower_risk": reward_items.get("tower_risk", 0.0) * weights["tower_risk"],
        }

    def _terminal_items(self, frame_data, observation, terminated, truncated):
        items = {"terminal_win": 0.0, "terminal_lose": 0.0, "timeout": 0.0}
        if not terminated and not truncated:
            return items
        frame_no = frame_data.get("frame_no", 0)
        if self.m_last_terminal_frame == frame_no:
            return items
        self.m_last_terminal_frame = frame_no
        if truncated:
            items["timeout"] = 1.0
        elif observation and observation.get("win", 0):
            items["terminal_win"] = 1.0
        else:
            items["terminal_lose"] = 1.0
        return items

    def _stage(self):
        return CurriculumConfig.CURRICULUM_STAGES.get(CurriculumConfig.CURRENT_STAGE, CurriculumConfig.CURRICULUM_STAGES["S1_BASIC"])

    def _lane_score(self, own_tower, enemy_tower, friendly_minions, enemy_minions):
        friendly_count = self._safe_div(len(friendly_minions), 8.0)
        enemy_count = self._safe_div(len(enemy_minions), 8.0)
        friendly_hp = self._safe_div(sum(self._hp_ratio(unit) for unit in friendly_minions), 8.0)
        enemy_hp = self._safe_div(sum(self._hp_ratio(unit) for unit in enemy_minions), 8.0)
        enemy_tower_pos = self._pos(enemy_tower)
        own_tower_pos = self._pos(own_tower)
        friendly_front = min((self._dist(self._pos(unit), enemy_tower_pos) for unit in friendly_minions), default=30000.0)
        enemy_front = min((self._dist(self._pos(unit), own_tower_pos) for unit in enemy_minions), default=30000.0)
        return 0.30 * (friendly_count - enemy_count) + 0.25 * (friendly_hp - enemy_hp) + 0.30 * (1.0 - self._safe_div(friendly_front, 30000.0)) - 0.15 * (1.0 - self._safe_div(enemy_front, 30000.0))

    def _direct_push_window(self, my_hero, enemy_hero, enemy_tower, friendly_minions):
        enemy_dead = enemy_hero is not None and not self._alive(enemy_hero)
        tower_finish = enemy_tower is not None and self._hp_ratio(enemy_tower) < RuleConfig.FINISH_TOWER_HP_RATIO
        minion_tanking = self._friendly_minion_tanking(enemy_tower, friendly_minions)
        enhanced_can_hit = self._enhanced_attack_ready(my_hero) and self._can_attack_tower(my_hero, enemy_tower)
        return bool(enemy_dead or tower_finish or minion_tanking or enhanced_can_hit)

    def _defense_emergency(self, own_tower, enemy_minions):
        return bool(own_tower and (self._hp_ratio(own_tower) < RuleConfig.LOW_HP_RISK_RATIO or len(enemy_minions) >= 4))

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

    def _neutral_taken(self, frame_data):
        score = 0.0
        frame_action = frame_data.get("frame_action", {}) or {}
        for dead in frame_action.get("dead_action", []) or []:
            killer = dead.get("killer", {}) or {}
            death = dead.get("death", {}) or {}
            if killer.get("runtime_id") == self.main_hero_player_id and self._is_monster(death):
                score += 1.0
        return score

    def _safe_cake_pick(self, frame_data, my_hero, enemy_hero, enemy_tower):
        cakes = frame_data.get("cakes", []) or []
        if not cakes or not my_hero or self._hp_ratio(my_hero) >= 0.40:
            return 0.0
        my_pos = self._pos(my_hero)
        nearest = min(cakes, key=lambda item: self._dist(my_pos, self._cake_pos(item)))
        dist = self._dist(my_pos, self._cake_pos(nearest))
        enemy_safe = True if not enemy_hero else self._dist(self._pos(enemy_hero), self._cake_pos(nearest)) > RuleConfig.ENEMY_NEAR_DIST
        tower_safe = self._tower_risk(my_hero, enemy_tower) < RuleConfig.BLOOD_PACK_TOWER_RISK_LIMIT
        return 1.0 if dist < 2500 and enemy_safe and tower_safe else 0.0

    def _skill_result(self, my_hero):
        slots = (((my_hero or {}).get("skill_state") or {}).get("slot_states", []) or [])
        return 1.0 if any(slot.get("succUsedInFrame", 0) > 0 and slot.get("hitHeroTimes", 0) > 0 for slot in slots) else 0.0

    def _bad_resource_attempt(self, my_hero, neutral_units, resource_allowed):
        if resource_allowed or not neutral_units or not my_hero:
            return 0.0
        target_id = my_hero.get("attack_target")
        return 1.0 if target_id in {unit.get("runtime_id") for unit in neutral_units} else 0.0

    def _enhanced_attack_ready(self, my_hero):
        if not my_hero:
            return 0.0
        buffs = ((my_hero.get("buff_state") or {}).get("buff_skills", []) or [])
        marks = ((my_hero.get("buff_state") or {}).get("buff_marks", []) or [])
        return 1.0 if buffs or marks else 0.0

    def _tower_risk(self, my_hero, enemy_tower):
        if not my_hero or not enemy_tower or not self._alive(my_hero):
            return 0.0
        dist = self._dist(self._pos(my_hero), self._pos(enemy_tower))
        tower_range = enemy_tower.get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE)
        smooth = self._sigmoid((tower_range - dist) / RuleConfig.TOWER_RISK_SIGMOID_SCALE)
        target_self = 1.0 if enemy_tower.get("attack_target") == my_hero.get("runtime_id") else 0.0
        low_hp = self._sigmoid((RuleConfig.LOW_HP_RISK_RATIO - self._hp_ratio(my_hero)) / 0.08)
        return self._clamp(smooth * (0.35 + 0.45 * target_self + 0.20 * low_hp))

    def _friendly_minion_tanking(self, tower, minions):
        target = (tower or {}).get("attack_target")
        return target is not None and target in {unit.get("runtime_id") for unit in minions}

    def _can_attack_tower(self, my_hero, enemy_tower):
        return bool(my_hero and enemy_tower and self._dist(self._pos(my_hero), self._pos(enemy_tower)) <= my_hero.get("attack_range", 0) + RuleConfig.ATTACK_TOWER_EXTRA_RANGE)

    def _split_heroes(self, frame_data):
        heroes = frame_data.get("hero_states", []) or []
        my_hero, enemy_hero = None, None
        for hero in heroes:
            if hero.get("runtime_id") == self.main_hero_player_id:
                my_hero = hero
                break
        my_camp = (my_hero or {}).get("camp")
        for hero in heroes:
            if hero is not my_hero and not self._same_camp(hero.get("camp"), my_camp):
                enemy_hero = hero
                break
        return my_hero, enemy_hero

    def _split_npcs(self, frame_data, my_camp):
        own_tower, enemy_tower, friendly_minions, enemy_minions, neutral_units = None, None, [], [], []
        for npc in frame_data.get("npc_states", []) or []:
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
        return sub_type == TOWER_SUB_TYPE or "TOWER" in str(sub_type).upper()

    def _is_monster(self, npc):
        actor_type = str(npc.get("actor_type", "")).upper()
        sub_type = str(npc.get("sub_type", "")).upper()
        return "MONSTER" in actor_type or "MONSTER" in sub_type or self._camp_value(npc.get("camp")) in (0, None, "0")

    def _hp_ratio(self, obj):
        return self._clamp(self._safe_div((obj or {}).get("hp", 0), (obj or {}).get("max_hp", 0)))

    def _alive(self, obj):
        return bool(obj and obj.get("hp", 0) > 0)

    def _money_value(self, hero):
        return (hero or {}).get("money_cnt", (hero or {}).get("money", 0))

    def _pos(self, obj):
        loc = (obj or {}).get("location", {}) or {}
        return loc.get("x", 100000), loc.get("z", 100000)

    def _cake_pos(self, cake):
        loc = ((cake.get("collider", {}) or {}).get("location", {}) or {})
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
        try:
            if math.isnan(value) or math.isinf(value):
                return 0.0
        except TypeError:
            return 0.0
        return max(min_value, min(max_value, value))
