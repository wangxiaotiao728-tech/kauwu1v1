#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature processing for the PPO 1v1 agent.

The implementation keeps the original package layout, but folds the planned
FeatureBuilder and MemoryProcess responsibilities into this existing file.
"""

import math
import os

from agent_ppo.conf.conf import RuleConfig
from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess


FEATURE_DIM = 512
TOWER_SUB_TYPE = 21
SUMMONER_SKILL_IDS = (80102, 80103, 80104, 80105, 80107, 80108, 80109, 80110, 80115, 80121)
STATE_COUNT = 11


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.reset_memory()

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.reset_memory()

    def reset_memory(self):
        self.last_my_hp = 1.0
        self.last_enemy_hp = 1.0
        self.last_enemy_seen_hp = 1.0
        self.last_enemy_tower_hp = 1.0
        self.last_own_tower_hp = 1.0
        self.last_my_money = 0.0
        self.last_my_exp = 0.0
        self.last_my_level = 1.0
        self.last_my_kill_cnt = 0
        self.last_my_dead_cnt = 0
        self.last_enemy_minion_hp_sum = 0.0
        self.last_friendly_front = 30000.0
        self.last_my_pos = (100000, 100000)
        self.last_enemy_seen_pos = (100000, 100000)
        self.enemy_last_seen_frame = -1
        self.recent_attack_enemy_hero_frames = 0
        self.recent_attack_tower_frames = 0
        self.recent_damage_taken_frames = 0
        self.recent_skill_used_frames = 0
        self.recent_tower_damage = 0.0
        self.same_position_steps = 0
        self.no_op_steps = 0
        self.common_attack_count = 0
        self.luban_attack_count = 0
        self.direnjie_attack_mod2 = 0
        self.last_tactical_state = 0
        self.state_hold_steps = 0
        self.last_blood_pos = (100000, 100000)
        self.last_blood_dist = 100000.0
        self.near_blood_pack_last_step = 0.0
        self.moving_towards_blood_pack_last_step = 0.0
        self.debug_print_count = 0

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        frame_no = frame_state.get("frame_no", observation.get("frame_no", 0))
        player_id = observation.get("player_id")
        player_camp = observation.get("player_camp", observation.get("camp", self.camp))
        self.camp = player_camp

        my_hero, enemy_hero = self._split_heroes(frame_state, player_id, player_camp)
        my_camp = my_hero.get("camp") if my_hero else player_camp
        own_tower, enemy_tower, friendly_minions, enemy_minions, monsters = self._split_npcs(frame_state, my_camp)
        bullets = frame_state.get("bullets", []) or []
        cakes = frame_state.get("cakes", []) or []
        state_id = self._tactical_state(frame_state, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions, monsters)
        self._debug_frame_state(frame_state, frame_no)

        feature = [0.0] * FEATURE_DIM
        self._fill_global(feature, observation, frame_no, my_hero, enemy_tower, my_camp)
        self._fill_hero(feature, frame_no, my_hero, enemy_hero, my_camp)
        self._fill_equip(feature, my_hero, enemy_hero)
        self._fill_skills(feature, my_hero)
        self._fill_tower(feature, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions)
        self._fill_minions(feature, my_hero, own_tower, enemy_tower, friendly_minions, enemy_minions)
        self._fill_blood_pack(feature, cakes, my_hero, enemy_hero, enemy_tower)
        self._fill_monster(feature, monsters, my_hero, enemy_hero)
        self._fill_buff_passive(feature, my_hero, enemy_hero, enemy_tower)
        self._fill_bullets(feature, bullets, my_hero, my_camp)
        self._fill_vision(feature, frame_no, my_hero, enemy_hero, my_camp)
        self._fill_state(feature, state_id)
        self._fill_tactic(feature, frame_no, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions, monsters, cakes)
        self._update_memory(frame_no, frame_state, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions, state_id)
        return feature

    # 0 - 31: legacy-compatible and global features.
    def _fill_global(self, feature, observation, frame_no, my_hero, enemy_tower, my_camp):
        self._set(feature, 0, self._alive(my_hero))
        hero_x, hero_z = self._mirror_pos(self._pos(my_hero), my_camp)
        self._set(feature, 1, self._norm_pos(hero_x, 60000))
        self._set(feature, 2, self._norm_pos(hero_z, 60000))
        self._set(feature, 3, self._alive(enemy_tower))
        tower_x, tower_z = self._mirror_pos(self._pos(enemy_tower), my_camp)
        self._set(feature, 4, 0.0 if self._is_red(my_camp) else 1.0)
        self._set(feature, 5, self._norm_pos(tower_x, 41000))
        self._set(feature, 6, self._norm_pos(tower_z, 41000))
        raw_tower_x, raw_tower_z = self._pos(enemy_tower)
        raw_hero_x, raw_hero_z = self._pos(my_hero)
        if raw_tower_x != 100000 and raw_hero_x != 100000:
            dx, dz = raw_tower_x - raw_hero_x, raw_tower_z - raw_hero_z
            if self._is_red(my_camp):
                dx, dz = -dx, -dz
            self._set(feature, 7, (dx + 15000) / 30000.0)
            self._set(feature, 8, (dz + 15000) / 30000.0)
        self._set(feature, 9, self._hp_ratio(enemy_tower))

        self._set(feature, 10, self._safe_div(frame_no, 20000.0))
        self._set(feature, 11, 1.0 if observation.get("win", 0) else 0.0)
        self._set(feature, 12, self._safe_div((my_hero or {}).get("config_id", 0), 200.0))
        self._set(feature, 13, 1.0 if (my_hero or {}).get("config_id") == 112 else 0.0)
        self._set(feature, 14, 1.0 if (my_hero or {}).get("config_id") == 133 else 0.0)
        self._set(feature, 15, self._safe_div((my_hero or {}).get("sight_area", 0), 30000.0))
        forward = (my_hero or {}).get("forward", {}) or {}
        self._set_signed(feature, 16, forward.get("x", 0), 1000.0)
        self._set_signed(feature, 17, forward.get("z", 0), 1000.0)
        self._set(feature, 18, 1.0 if (my_hero or {}).get("is_in_grass", False) else 0.0)
        self._set(feature, 19, self._safe_div(len((my_hero or {}).get("take_hurt_infos", []) or []), 8.0))
        self._set(feature, 20, self._safe_div(len((my_hero or {}).get("hit_target_info", []) or []), 8.0))
        self._set(feature, 21, self._safe_div(len((my_hero or {}).get("real_cmd", []) or []), 8.0))

    # 32 - 95: hero state features.
    def _fill_hero(self, feature, frame_no, my_hero, enemy_hero, my_camp):
        start = 32
        my_pos = self._pos(my_hero)
        enemy_visible = 1.0 if enemy_hero else 0.0
        enemy_conf = self._enemy_confidence(frame_no)
        enemy_pos = self._pos(enemy_hero) if enemy_hero else self.last_enemy_seen_pos
        enemy_hp = self._hp_ratio(enemy_hero) if enemy_hero else self.last_enemy_seen_hp * enemy_conf

        self._set(feature, start + 0, self._hp_ratio(my_hero))
        self._set(feature, start + 1, enemy_hp)
        self._set(feature, start + 2, self._alive(my_hero))
        self._set(feature, start + 3, self._alive(enemy_hero) if enemy_hero else enemy_conf)
        self._set(feature, start + 4, self._level_norm(my_hero))
        self._set(feature, start + 5, self._level_norm(enemy_hero))
        self._set_signed(feature, start + 6, (my_hero or {}).get("level", 1) - (enemy_hero or {}).get("level", 1), 15.0)
        self._set(feature, start + 7, self._money_norm(my_hero))
        self._set(feature, start + 8, self._money_norm(enemy_hero))
        my_money = self._money_value(my_hero)
        enemy_money = self._money_value(enemy_hero)
        self._set_signed(feature, start + 9, my_money - enemy_money, 10000.0)
        self._set(feature, start + 10, self._exp_ratio(my_hero))
        self._set(feature, start + 11, self._exp_ratio(enemy_hero))
        self._set(feature, start + 12, self._safe_div((my_hero or {}).get("attack_range", 0), 12000.0))
        self._set(feature, start + 13, self._safe_div((enemy_hero or {}).get("attack_range", 0), 12000.0))
        self._set(feature, start + 14, self._safe_div(self._dist(my_pos, enemy_pos), 30000.0))
        self._set(feature, start + 15, enemy_visible)
        self._set(feature, start + 16, self._safe_div((my_hero or {}).get("revive_time", 0), 5000.0))
        self._set(feature, start + 17, self._safe_div((enemy_hero or {}).get("revive_time", 0), 5000.0))
        self._set_signed(feature, start + 18, (my_hero or {}).get("phy_atk", 0) - (enemy_hero or {}).get("phy_atk", 0), 1000.0)
        self._set_signed(feature, start + 19, (my_hero or {}).get("phy_def", 0) - (enemy_hero or {}).get("phy_def", 0), 1000.0)
        self._set_signed(feature, start + 20, (my_hero or {}).get("mgc_def", 0) - (enemy_hero or {}).get("mgc_def", 0), 1000.0)
        self._set(feature, start + 21, self._safe_div((my_hero or {}).get("mov_spd", 0), 1000.0))
        self._set(feature, start + 22, self._safe_div((enemy_hero or {}).get("mov_spd", 0), 1000.0))
        self._set(feature, start + 23, self._safe_div((my_hero or {}).get("atk_spd", 0), 1000.0))
        self._set(feature, start + 24, self._safe_div((enemy_hero or {}).get("atk_spd", 0), 1000.0))
        self._set(feature, start + 25, self._safe_div((my_hero or {}).get("phy_vamp", 0), 1000.0))
        self._set(feature, start + 26, self._safe_div((my_hero or {}).get("crit_rate", 0), 1000.0))
        self._set(feature, start + 27, self._safe_div((my_hero or {}).get("cd_reduce", 0), 1000.0))
        self._set(feature, start + 28, self._safe_div((my_hero or {}).get("ctrl_reduce", 0), 1000.0))
        self._set(feature, start + 29, self._safe_div((my_hero or {}).get("kill_cnt", 0), 10.0))
        self._set(feature, start + 30, self._safe_div((my_hero or {}).get("dead_cnt", 0), 10.0))
        self._set(feature, start + 31, self._safe_div((enemy_hero or {}).get("dead_cnt", 0), 10.0))
        self._set_signed(feature, start + 32, (my_hero or {}).get("total_hurt_to_hero", 0) - (enemy_hero or {}).get("total_hurt_to_hero", 0), 50000.0)
        self._set_signed(feature, start + 33, (enemy_hero or {}).get("total_be_hurt_by_hero", 0) - (my_hero or {}).get("total_be_hurt_by_hero", 0), 50000.0)
        my_x, my_z = self._mirror_pos(my_pos, my_camp)
        self._set_signed(feature, start + 34, my_x, 60000.0)
        self._set_signed(feature, start + 35, my_z, 60000.0)
        if enemy_pos[0] != 100000 and my_pos[0] != 100000:
            dx, dz = enemy_pos[0] - my_pos[0], enemy_pos[1] - my_pos[1]
            if self._is_red(my_camp):
                dx, dz = -dx, -dz
            self._set_signed(feature, start + 36, dx, 30000.0)
            self._set_signed(feature, start + 37, dz, 30000.0)
        self._set(feature, start + 38, enemy_conf)
        self._set(feature, start + 39, self._safe_div(frame_no, 20000.0))

    # 96 - 143: economy and equipment features.
    def _fill_equip(self, feature, my_hero, enemy_hero):
        start = 96
        my_equips = self._equips(my_hero)
        enemy_equips = self._equips(enemy_hero)
        my_price = sum(max(0, item.get("buyPrice", 0)) * max(1, item.get("amount", 1)) for item in my_equips)
        enemy_price = sum(max(0, item.get("buyPrice", 0)) * max(1, item.get("amount", 1)) for item in enemy_equips)
        self._set(feature, start + 0, self._safe_div(len(my_equips), 6.0))
        self._set(feature, start + 1, self._safe_div(len(enemy_equips), 6.0))
        self._set_signed(feature, start + 2, len(my_equips) - len(enemy_equips), 6.0)
        self._set(feature, start + 3, self._safe_div(my_price, 15000.0))
        self._set(feature, start + 4, self._safe_div(enemy_price, 15000.0))
        self._set_signed(feature, start + 5, my_price - enemy_price, 10000.0)
        self._set(feature, start + 6, 1.0 if any(item.get("active_skill") for item in my_equips) else 0.0)
        self._set(feature, start + 7, 1.0 if any(item.get("passive_skill") for item in my_equips) else 0.0)
        self._set(feature, start + 8, self._safe_div((my_hero or {}).get("money", 0), 3000.0))
        self._set(feature, start + 9, self._safe_div((my_hero or {}).get("money_cnt", 0), 20000.0))
        self._set_signed(feature, start + 10, self._money_value(my_hero) - self._money_value(enemy_hero), 10000.0)
        self._set(feature, start + 11, 1.0 if (my_hero or {}).get("money", 0) >= 250 else 0.0)
        for i in range(4):
            item = my_equips[i] if i < len(my_equips) else {}
            base = start + 16 + i * 4
            self._set(feature, base + 0, self._safe_div(item.get("configId", 0), 300000.0))
            self._set(feature, base + 1, self._safe_div(item.get("buyPrice", 0), 4000.0))
            self._set(feature, base + 2, self._safe_div(item.get("amount", 0), 3.0))
            active = item.get("active_skill", []) or []
            self._set(feature, base + 3, 1.0 if active and active[0].get("cooldown", 0) <= 0 else 0.0)

    # 144 - 191: active skills and summoner skill features.
    def _fill_skills(self, feature, my_hero):
        start = 144
        slots = ((my_hero or {}).get("skill_state") or {}).get("slot_states", []) or []
        normal_slots = [slot for slot in slots if slot.get("configId") not in SUMMONER_SKILL_IDS]
        normal_slots = sorted(normal_slots, key=lambda item: item.get("slot_type", 0))
        for i in range(4):
            slot = normal_slots[i] if i < len(normal_slots) else {}
            base = start + i * 8
            cooldown_max = max(1, slot.get("cooldown_max", 1))
            used_times = max(1, slot.get("usedTimes", 0))
            self._set(feature, base + 0, self._safe_div(slot.get("level", 0), 6.0))
            self._set(feature, base + 1, 1.0 if slot.get("usable", False) else 0.0)
            self._set(feature, base + 2, self._safe_div(slot.get("cooldown", 0), cooldown_max))
            self._set(feature, base + 3, 1.0 - self._safe_div(slot.get("cooldown", 0), cooldown_max))
            self._set(feature, base + 4, self._safe_div(slot.get("usedTimes", 0), 100.0))
            self._set(feature, base + 5, self._safe_div(slot.get("hitHeroTimes", 0), used_times))
            self._set(feature, base + 6, 1.0 if slot.get("succUsedInFrame", 0) > 0 else 0.0)
            self._set(feature, base + 7, self._safe_div(slot.get("comboEffectTime", 0), 5000.0))

        summoner = self._summoner_slot(slots)
        self._set(feature, start + 32, self._safe_div(summoner.get("configId", 0), 100000.0))
        self._set(feature, start + 33, 1.0 if summoner.get("usable", False) else 0.0)
        self._set(feature, start + 34, self._safe_div(summoner.get("cooldown", 0), max(1, summoner.get("cooldown_max", 1))))
        self._set(feature, start + 35, 1.0 if summoner.get("configId") == 80115 and summoner.get("usable", False) else 0.0)
        self._set(feature, start + 36, 1.0 if summoner.get("configId") == 80110 and summoner.get("usable", False) else 0.0)
        self._set(feature, start + 37, 1.0 if summoner.get("configId") == 80107 and summoner.get("usable", False) else 0.0)
        self._set(feature, start + 38, 1.0 if summoner.get("configId") == 80108 and summoner.get("usable", False) else 0.0)
        self._set(feature, start + 39, self._safe_div(self.recent_skill_used_frames, 30.0))

    # 192 - 255: tower and push safety features.
    def _fill_tower(self, feature, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions):
        start = 192
        my_pos, enemy_tower_pos, own_tower_pos = self._pos(my_hero), self._pos(enemy_tower), self._pos(own_tower)
        enemy_tower_dist = self._dist(my_pos, enemy_tower_pos)
        own_tower_dist = self._dist(my_pos, own_tower_pos)
        enemy_tower_range = (enemy_tower or {}).get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE)
        own_tower_range = (own_tower or {}).get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE)
        enemy_tower_target = (enemy_tower or {}).get("attack_target")
        own_tower_target = (own_tower or {}).get("attack_target")
        friendly_ids = {unit.get("runtime_id") for unit in friendly_minions}
        enemy_ids = {unit.get("runtime_id") for unit in enemy_minions}
        can_attack = enemy_tower and my_hero and enemy_tower_dist <= (my_hero.get("attack_range", 0) + RuleConfig.ATTACK_TOWER_EXTRA_RANGE)

        self._set(feature, start + 0, self._hp_ratio(enemy_tower))
        self._set(feature, start + 1, self._hp_ratio(own_tower))
        self._set_signed(feature, start + 2, self._hp_ratio(own_tower) - self._hp_ratio(enemy_tower), 1.0)
        self._set(feature, start + 3, self._safe_div(enemy_tower_dist, 30000.0))
        self._set(feature, start + 4, self._safe_div(own_tower_dist, 30000.0))
        self._set(feature, start + 5, 1.0 if enemy_tower_dist <= enemy_tower_range else 0.0)
        self._set(feature, start + 6, 1.0 if enemy_hero and self._dist(self._pos(enemy_hero), own_tower_pos) <= own_tower_range else 0.0)
        self._set(feature, start + 7, 1.0 if my_hero and enemy_tower_target == my_hero.get("runtime_id") else 0.0)
        self._set(feature, start + 8, 1.0 if enemy_tower_target in friendly_ids and enemy_tower_target is not None else 0.0)
        self._set(feature, start + 9, 1.0 if enemy_hero and enemy_tower_target == enemy_hero.get("runtime_id") else 0.0)
        self._set(feature, start + 10, 1.0 if can_attack else 0.0)
        self._set(feature, start + 11, 1.0 if self._friendly_minion_tanking(enemy_tower, friendly_minions) else 0.0)
        self._set(feature, start + 12, 1.0 if can_attack and not self._tower_targets_self(my_hero, enemy_tower) else 0.0)
        self._set(feature, start + 13, 1.0 if enemy_tower and self._hp_ratio(enemy_tower) < 0.18 else 0.0)
        self._set(feature, start + 14, 1.0 if own_tower and self._hp_ratio(own_tower) < RuleConfig.LOW_HP_RISK_RATIO else 0.0)
        self._set(feature, start + 15, 1.0 if enemy_tower and self._hp_ratio(enemy_tower) < RuleConfig.FINISH_TOWER_HP_RATIO else 0.0)
        self._set(feature, start + 16, self._tower_risk(my_hero, enemy_tower))
        self._set(feature, start + 17, self._safe_div(len(friendly_ids), 8.0))
        self._set(feature, start + 18, self._safe_div(len(enemy_ids), 8.0))
        self._set(feature, start + 19, 1.0 if own_tower_target in enemy_ids and own_tower_target is not None else 0.0)
        self._set(feature, start + 20, self._safe_div(self.recent_tower_damage, 1.0))

    # 256 - 319: minion and lane features.
    def _fill_minions(self, feature, my_hero, own_tower, enemy_tower, friendly_minions, enemy_minions):
        start = 256
        my_pos = self._pos(my_hero)
        enemy_sorted = sorted(enemy_minions, key=lambda unit: self._dist(self._pos(unit), my_pos))
        friendly_sorted = sorted(friendly_minions, key=lambda unit: self._dist(self._pos(unit), my_pos))
        friendly_hp_sum = sum(self._hp_ratio(unit) for unit in friendly_minions)
        enemy_hp_sum = sum(self._hp_ratio(unit) for unit in enemy_minions)
        lowest_enemy = min(enemy_minions, key=lambda unit: self._hp_ratio(unit), default=None)
        my_atk = (my_hero or {}).get("phy_atk", 0)
        enemy_tower_pos, own_tower_pos = self._pos(enemy_tower), self._pos(own_tower)
        friendly_front = min((self._dist(self._pos(unit), enemy_tower_pos) for unit in friendly_minions), default=30000.0)
        enemy_front = min((self._dist(self._pos(unit), own_tower_pos) for unit in enemy_minions), default=30000.0)
        enemy_tower_range = (enemy_tower or {}).get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE)
        own_tower_range = (own_tower or {}).get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE)

        self._set(feature, start + 0, self._safe_div(len(friendly_minions), 8.0))
        self._set(feature, start + 1, self._safe_div(len(enemy_minions), 8.0))
        self._set_signed(feature, start + 2, len(friendly_minions) - len(enemy_minions), 8.0)
        self._set(feature, start + 3, self._safe_div(friendly_hp_sum, 8.0))
        self._set(feature, start + 4, self._safe_div(enemy_hp_sum, 8.0))
        self._set(feature, start + 5, self._hp_ratio(enemy_sorted[0] if enemy_sorted else None))
        self._set(feature, start + 6, self._hp_ratio(friendly_sorted[0] if friendly_sorted else None))
        self._set(feature, start + 7, self._hp_ratio(lowest_enemy))
        self._set(feature, start + 8, 1.0 if lowest_enemy and lowest_enemy.get("hp", 0) <= max(1, my_atk * 1.2) else 0.0)
        self._set(feature, start + 9, 1.0 - self._safe_div(friendly_front, 30000.0))
        self._set(feature, start + 10, 1.0 - self._safe_div(enemy_front, 30000.0))
        self._set_signed(feature, start + 11, self.last_friendly_front - friendly_front, 12000.0)
        self._set(feature, start + 12, 1.0 if any(self._dist(self._pos(unit), enemy_tower_pos) <= enemy_tower_range for unit in friendly_minions) else 0.0)
        self._set(feature, start + 13, 1.0 if any(self._dist(self._pos(unit), own_tower_pos) <= own_tower_range for unit in enemy_minions) else 0.0)
        self._set(feature, start + 14, 1.0 if self._friendly_minion_tanking(enemy_tower, friendly_minions) else 0.0)
        self._set(feature, start + 15, 1.0 if self._friendly_minion_tanking(own_tower, enemy_minions) else 0.0)
        for i in range(4):
            unit = enemy_sorted[i] if i < len(enemy_sorted) else None
            base = start + 16 + i * 8
            self._set(feature, base + 0, self._hp_ratio(unit))
            self._set(feature, base + 1, self._safe_div(self._dist(my_pos, self._pos(unit)), 30000.0))
            self._set(feature, base + 2, 1.0 if unit else 0.0)
            self._set(feature, base + 3, 1.0 if unit and unit.get("hp", 0) <= max(1, my_atk * 1.2) else 0.0)
            self._set_signed(feature, base + 4, self._pos(unit)[0] - my_pos[0], 30000.0)
            self._set_signed(feature, base + 5, self._pos(unit)[1] - my_pos[1], 30000.0)
            self._set(feature, base + 6, self._safe_div(unit.get("kill_income", 0) if unit else 0, 200.0))
            self._set(feature, base + 7, 1.0 if unit and self._dist(self._pos(unit), enemy_tower_pos) <= enemy_tower_range else 0.0)

    # 320 - 351: blood pack features.
    def _fill_blood_pack(self, feature, cakes, my_hero, enemy_hero, enemy_tower):
        start = 320
        if not cakes or not my_hero:
            return
        my_pos = self._pos(my_hero)
        cake = min(cakes, key=lambda item: self._dist(my_pos, self._cake_pos(item)))
        cake_pos = self._cake_pos(cake)
        dist = self._dist(my_pos, cake_pos)
        enemy_dist = self._dist(self._pos(enemy_hero), cake_pos)
        need = self._sigmoid((RuleConfig.BLOOD_PACK_HP_RATIO - self._hp_ratio(my_hero)) / 0.08)
        enemy_need = self._sigmoid((RuleConfig.BLOOD_PACK_HP_RATIO - self._hp_ratio(enemy_hero)) / 0.08)
        risk = self._tower_risk(my_hero, enemy_tower)
        safe = need * (1.0 - risk)
        self._set(feature, start + 0, 1.0)
        self._set(feature, start + 1, self._safe_div(dist, 30000.0))
        self._set_signed(feature, start + 2, cake_pos[0] - my_pos[0], 30000.0)
        self._set_signed(feature, start + 3, cake_pos[1] - my_pos[1], 30000.0)
        self._set(feature, start + 4, self._safe_div((cake.get("collider", {}) or {}).get("radius", 0), 5000.0))
        self._set(feature, start + 5, need)
        self._set(feature, start + 6, enemy_need)
        self._set(feature, start + 7, safe)
        self._set(feature, start + 8, 1.0 if enemy_dist < dist else 0.0)
        self._set(feature, start + 9, 1.0 if dist <= enemy_dist else 0.0)
        self._set(feature, start + 10, 1.0 if self.last_blood_dist > dist else 0.0)
        self._set(feature, start + 11, self.near_blood_pack_last_step)
        self._set(feature, start + 12, self.moving_towards_blood_pack_last_step)
        self._set(feature, start + 13, 1.0 if self.last_my_hp < self._hp_ratio(my_hero) and self.last_blood_dist < 2500 else 0.0)
        self._set(feature, start + 14, safe * math.exp(-min(dist, 30000.0) / 8000.0))

    # 352 - 383: neutral monster features.
    def _fill_monster(self, feature, monsters, my_hero, enemy_hero):
        start = 352
        if not monsters or not my_hero:
            return
        my_pos = self._pos(my_hero)
        monster = min(monsters, key=lambda unit: self._dist(my_pos, self._pos(unit)))
        monster_pos = self._pos(monster)
        enemy_dist = self._dist(self._pos(enemy_hero), monster_pos)
        my_dist = self._dist(my_pos, monster_pos)
        self._set(feature, start + 0, 1.0)
        self._set(feature, start + 1, self._hp_ratio(monster))
        self._set(feature, start + 2, self._safe_div(my_dist, 30000.0))
        self._set_signed(feature, start + 3, monster_pos[0] - my_pos[0], 30000.0)
        self._set_signed(feature, start + 4, monster_pos[1] - my_pos[1], 30000.0)
        self._set(feature, start + 5, self._safe_div(monster.get("kill_income", 0), 500.0))
        self._set(feature, start + 6, 1.0 if monster.get("attack_target") == (my_hero or {}).get("runtime_id") else 0.0)
        self._set(feature, start + 7, 1.0 if self._hp_ratio(my_hero) > RuleConfig.FARM_MONSTER_HP_RATIO and my_dist < 9000 else 0.0)
        self._set(feature, start + 8, 1.0 if enemy_dist < my_dist else 0.0)
        self._set(feature, start + 9, 1.0 if enemy_dist < RuleConfig.DEFAULT_TOWER_RANGE and my_dist < RuleConfig.DEFAULT_TOWER_RANGE else 0.0)
        self._set(feature, start + 10, 1.0 if self._hp_ratio(monster) < 0.25 else 0.0)

    # 384 - 431: buff, passive, enhanced attack and recent command features.
    def _fill_buff_passive(self, feature, my_hero, enemy_hero, enemy_tower):
        start = 384
        buffs = ((my_hero or {}).get("buff_state") or {}).get("buff_skills", []) or []
        marks = ((my_hero or {}).get("buff_state") or {}).get("buff_marks", []) or []
        passive = (my_hero or {}).get("passive_skill", []) or []
        enemy_buffs = ((enemy_hero or {}).get("buff_state") or {}).get("buff_skills", []) or []
        hit_info = (my_hero or {}).get("hit_target_info", []) or []
        real_cmd = (my_hero or {}).get("real_cmd", []) or []
        config_id = (my_hero or {}).get("config_id")
        enhanced_ready = self._enhanced_attack_ready(my_hero)

        self._set(feature, start + 0, enhanced_ready)
        self._set(feature, start + 1, 1.0 if self.recent_attack_tower_frames > 0 else 0.0)
        self._set(feature, start + 2, 1.0 if self.recent_attack_enemy_hero_frames > 0 else 0.0)
        self._set(feature, start + 3, self._safe_div(self.common_attack_count % 5, 5.0))
        self._set(feature, start + 4, 1.0 if config_id == 112 and enhanced_ready else 0.0)
        self._set(feature, start + 5, self._safe_div(self.luban_attack_count % 5, 5.0))
        self._set(feature, start + 6, 1.0 if self.recent_skill_used_frames > 0 and config_id == 112 else 0.0)
        self._set(feature, start + 7, 1.0 if config_id == 112 and enhanced_ready and self._can_attack_tower(my_hero, enemy_tower) else 0.0)
        self._set(feature, start + 8, self._safe_div(max((mark.get("layer", 0) for mark in marks), default=0), 5.0))
        self._set(feature, start + 9, 1.0 if config_id == 133 and self.direnjie_attack_mod2 == 0 else 0.0)
        self._set(feature, start + 10, self._safe_div(self.direnjie_attack_mod2, 2.0))
        self._set(feature, start + 11, self._skill_usable(my_hero, 2))
        self._set(feature, start + 12, self._skill_usable(my_hero, 3))
        self._set(feature, start + 13, self._safe_div(len(buffs), 12.0))
        self._set(feature, start + 14, self._safe_div(len(marks), 12.0))
        self._set(feature, start + 15, self._safe_div(len(passive), 8.0))
        self._set(feature, start + 16, self._safe_div(len(enemy_buffs), 12.0))
        self._set(feature, start + 17, self._safe_div(len(hit_info), 8.0))
        self._set(feature, start + 18, self._safe_div(len(real_cmd), 8.0))
        self._set(feature, start + 19, 1.0 if any(cmd.get("attack_actor") for cmd in real_cmd) else 0.0)
        self._set(feature, start + 20, 1.0 if any(cmd.get("dir_skill") or cmd.get("pos_skill") or cmd.get("obj_skill") for cmd in real_cmd) else 0.0)
        for i in range(4):
            buff = buffs[i] if i < len(buffs) else {}
            base = start + 24 + i * 4
            self._set(feature, base + 0, self._safe_div(buff.get("configId", 0), 200000.0))
            self._set(feature, base + 1, self._safe_div(buff.get("times", 0), 20.0))
            self._set(feature, base + 2, self._safe_div((marks[i].get("layer", 0) if i < len(marks) else 0), 5.0))
            self._set(feature, base + 3, self._safe_div((passive[i].get("cooldown", 0) if i < len(passive) else 0), 10000.0))

    # 432 - 455: bullet and danger features.
    def _fill_bullets(self, feature, bullets, my_hero, my_camp):
        start = 432
        if not my_hero:
            return
        my_pos = self._pos(my_hero)
        enemy_bullets = [bullet for bullet in bullets if not self._same_camp(bullet.get("camp"), my_camp)]
        if not enemy_bullets:
            return
        bullet = min(enemy_bullets, key=lambda item: self._dist(my_pos, self._pos(item)))
        bullet_pos = self._pos(bullet)
        dist = self._dist(my_pos, bullet_pos)
        self._set(feature, start + 0, 1.0)
        self._set(feature, start + 1, self._safe_div(dist, 30000.0))
        self._set_signed(feature, start + 2, bullet_pos[0] - my_pos[0], 30000.0)
        self._set_signed(feature, start + 3, bullet_pos[1] - my_pos[1], 30000.0)
        self._set(feature, start + 4, self._safe_div(bullet.get("skill_id", 0), 200000.0))
        self._set(feature, start + 5, self._safe_div(bullet.get("slot_type", 0), 10.0))
        self._set(feature, start + 6, 1.0 if dist < 5000 else 0.0)
        self._set(feature, start + 7, 1.0 if dist < RuleConfig.BULLET_DANGER_DIST and self._hp_ratio(my_hero) < RuleConfig.BULLET_DANGER_HP_RATIO else 0.0)
        self._set(feature, start + 8, self._safe_div(len(enemy_bullets), 12.0))

    # 456 - 471: vision, grass and last-seen features.
    def _fill_vision(self, feature, frame_no, my_hero, enemy_hero, my_camp):
        start = 456
        enemy_visible = 1.0 if enemy_hero else 0.0
        enemy_conf = self._enemy_confidence(frame_no)
        my_pos = self._pos(my_hero)
        last_pos = self._pos(enemy_hero) if enemy_hero else self.last_enemy_seen_pos
        missing = 0 if self.enemy_last_seen_frame < 0 else max(0, frame_no - self.enemy_last_seen_frame)
        self._set(feature, start + 0, enemy_visible)
        self._set(feature, start + 1, 1.0 - enemy_visible)
        self._set(feature, start + 2, self._safe_div(missing, 20000.0))
        if last_pos[0] != 100000:
            dx, dz = last_pos[0] - my_pos[0], last_pos[1] - my_pos[1]
            if self._is_red(my_camp):
                dx, dz = -dx, -dz
            self._set_signed(feature, start + 3, dx, 30000.0)
            self._set_signed(feature, start + 4, dz, 30000.0)
        self._set(feature, start + 5, enemy_conf)
        self._set(feature, start + 6, 1.0 if (my_hero or {}).get("is_in_grass", False) else 0.0)
        self._set(feature, start + 7, 1.0 if (enemy_hero or {}).get("is_in_grass", False) else 0.0)
        self._set(feature, start + 8, 1.0 if not enemy_hero and self._dist(my_pos, self.last_enemy_seen_pos) < 9000 else 0.0)
        self._set(feature, start + 9, self._safe_div((my_hero or {}).get("sight_area", 0), 30000.0))
        self._set(feature, start + 10, self._visible_to_camp(enemy_hero, my_camp))

    # 472 - 487: tactical state one-hot and hold time.
    def _fill_state(self, feature, state_id):
        start = 472
        state_id = max(0, min(STATE_COUNT - 1, state_id))
        for i in range(STATE_COUNT):
            self._set(feature, start + i, 1.0 if i == state_id else 0.0)
        self._set(feature, start + 11, self._safe_div(self.state_hold_steps, 60.0))
        self._set(feature, start + 12, self._safe_div(self.last_tactical_state, 10.0))
        self._set(feature, start + 13, 1.0 if state_id != 0 else 0.0)

    # 488 - 503: risk and tactical score features.
    def _fill_tactic(self, feature, frame_no, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions, monsters, cakes):
        start = 488
        hp_adv = self._hp_ratio(my_hero) - (self._hp_ratio(enemy_hero) if enemy_hero else self.last_enemy_seen_hp * self._enemy_confidence(frame_no))
        lane_adv = self._clamp((len(friendly_minions) - len(enemy_minions) + 8) / 16.0)
        push_score = (1.0 - self._hp_ratio(enemy_tower)) * 0.45 + lane_adv * 0.30 + max(0.0, hp_adv) * 0.25
        defend_score = (1.0 - self._hp_ratio(own_tower)) * 0.65 + max(0.0, -hp_adv) * 0.20 + self._safe_div(len(enemy_minions), 8.0) * 0.15
        trade_score = max(0.0, hp_adv) * 0.55 + self._skill_ready_score(my_hero) * 0.25 + self._safe_div((my_hero or {}).get("phy_vamp", 0), 1000.0) * 0.20
        farm_monster_score = (1.0 if monsters else 0.0) * (0.6 if self._hp_ratio(my_hero) > RuleConfig.FARM_MONSTER_HP_RATIO else 0.2)
        blood_score = (1.0 if cakes else 0.0) * self._sigmoid((RuleConfig.BLOOD_PACK_HP_RATIO - self._hp_ratio(my_hero)) / 0.08)
        vision_risk = (1.0 - (1.0 if enemy_hero else 0.0)) * self._sigmoid((RuleConfig.BLOOD_PACK_HP_RATIO - self._hp_ratio(my_hero)) / 0.08)
        self._set(feature, start + 0, self._tower_risk(my_hero, enemy_tower))
        self._set(feature, start + 1, push_score)
        self._set(feature, start + 2, defend_score)
        self._set(feature, start + 3, trade_score)
        self._set(feature, start + 4, farm_monster_score)
        self._set(feature, start + 5, blood_score)
        self._set(feature, start + 6, vision_risk)
        self._set(feature, start + 7, self._safe_div(self.same_position_steps, 60.0))
        self._set(feature, start + 8, self._safe_div(self.no_op_steps, 60.0))
        self._set(feature, start + 9, self._safe_div(self.recent_damage_taken_frames, 30.0))
        self._set(feature, start + 10, self._safe_div(self.recent_attack_tower_frames, 30.0))
        self._set(feature, 504, self._safe_div(len(friendly_minions) + len(enemy_minions), 16.0))
        self._set(feature, 505, self._safe_div(len(monsters), 4.0))
        self._set(feature, 506, self._safe_div(len(cakes), 3.0))

    def _update_memory(self, frame_no, frame_state, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions, state_id):
        my_hp = self._hp_ratio(my_hero)
        enemy_hp = self._hp_ratio(enemy_hero)
        enemy_tower_hp = self._hp_ratio(enemy_tower)
        own_tower_hp = self._hp_ratio(own_tower)
        my_pos = self._pos(my_hero)

        if my_pos[0] != 100000 and self._dist(my_pos, self.last_my_pos) < 350:
            self.same_position_steps += 1
        else:
            self.same_position_steps = 0

        real_cmd = (my_hero or {}).get("real_cmd", []) or []
        if real_cmd:
            self.no_op_steps = 0
        else:
            self.no_op_steps += 1

        used_skill = self._real_cmd_has_skill(real_cmd)
        used_common = self._real_cmd_has_common_attack(real_cmd)
        target = (my_hero or {}).get("attack_target")
        if used_skill:
            self.recent_skill_used_frames = 30
            if (my_hero or {}).get("config_id") == 112:
                self.luban_attack_count = 4
        else:
            self.recent_skill_used_frames = max(0, self.recent_skill_used_frames - 1)

        if used_common:
            self.common_attack_count += 1
            if (my_hero or {}).get("config_id") == 112:
                self.luban_attack_count = (self.luban_attack_count + 1) % 5
            if (my_hero or {}).get("config_id") == 133:
                self.direnjie_attack_mod2 = (self.direnjie_attack_mod2 + 1) % 2

        if target == (enemy_tower or {}).get("runtime_id"):
            self.recent_attack_tower_frames = 30
        else:
            self.recent_attack_tower_frames = max(0, self.recent_attack_tower_frames - 1)
        if target == (enemy_hero or {}).get("runtime_id"):
            self.recent_attack_enemy_hero_frames = 30
        else:
            self.recent_attack_enemy_hero_frames = max(0, self.recent_attack_enemy_hero_frames - 1)

        if my_hero and my_hero.get("take_hurt_infos"):
            self.recent_damage_taken_frames = 30
        else:
            self.recent_damage_taken_frames = max(0, self.recent_damage_taken_frames - 1)

        self.recent_tower_damage = max(0.0, self.last_enemy_tower_hp - enemy_tower_hp)
        if enemy_hero:
            self.enemy_last_seen_frame = frame_no
            self.last_enemy_seen_pos = self._pos(enemy_hero)
            self.last_enemy_seen_hp = enemy_hp

        cakes = frame_state.get("cakes", []) or []
        if cakes and my_hero:
            cake = min(cakes, key=lambda item: self._dist(my_pos, self._cake_pos(item)))
            cake_pos = self._cake_pos(cake)
            dist = self._dist(my_pos, cake_pos)
            self.moving_towards_blood_pack_last_step = 1.0 if dist < self.last_blood_dist else 0.0
            self.near_blood_pack_last_step = 1.0 if dist < 2500 else 0.0
            self.last_blood_pos = cake_pos
            self.last_blood_dist = dist
        else:
            self.moving_towards_blood_pack_last_step = 0.0
            self.near_blood_pack_last_step = 0.0
            self.last_blood_dist = 100000.0

        if state_id == self.last_tactical_state:
            self.state_hold_steps += 1
        else:
            self.state_hold_steps = 0
        self.last_tactical_state = state_id
        self.last_my_hp = my_hp
        self.last_enemy_hp = enemy_hp
        self.last_enemy_tower_hp = enemy_tower_hp
        self.last_own_tower_hp = own_tower_hp
        self.last_my_money = self._money_value(my_hero)
        self.last_my_exp = (my_hero or {}).get("exp", 0)
        self.last_my_level = (my_hero or {}).get("level", 1)
        self.last_my_kill_cnt = (my_hero or {}).get("kill_cnt", 0)
        self.last_my_dead_cnt = (my_hero or {}).get("dead_cnt", 0)
        self.last_enemy_minion_hp_sum = sum(self._hp_ratio(unit) for unit in enemy_minions)
        self.last_friendly_front = min((self._dist(self._pos(unit), self._pos(enemy_tower)) for unit in friendly_minions), default=30000.0)
        self.last_my_pos = my_pos

    def _tactical_state(self, frame_state, my_hero, enemy_hero, own_tower, enemy_tower, friendly_minions, enemy_minions, monsters):
        if self._tower_targets_self(my_hero, enemy_tower):
            return 1
        if self._can_finish_tower(my_hero, enemy_tower):
            return 2
        if enemy_hero and not self._alive(enemy_hero) and enemy_tower:
            return 3
        if self._enhanced_attack_ready(my_hero) and self._can_attack_tower(my_hero, enemy_tower):
            return 4
        if self._friendly_minion_tanking(enemy_tower, friendly_minions):
            return 5
        if self._safe_blood_pack(frame_state.get("cakes", []) or [], my_hero, enemy_hero, enemy_tower):
            return 6
        if own_tower and self._hp_ratio(own_tower) < RuleConfig.LOW_HP_RISK_RATIO and enemy_minions:
            return 7
        if len(enemy_minions) > len(friendly_minions) + 1:
            return 8
        if enemy_hero and self._hp_ratio(my_hero) > self._hp_ratio(enemy_hero) + RuleConfig.TRADE_HP_ADVANTAGE:
            return 9
        if monsters and self._hp_ratio(my_hero) > RuleConfig.FARM_MONSTER_HP_RATIO:
            return 10
        return 0

    def _split_heroes(self, frame_state, player_id, player_camp):
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

    def _debug_frame_state(self, frame_state, frame_no):
        debug_flag = os.environ.get("HOK_FRAME_DEBUG", "0").lower()
        if debug_flag not in ("1", "true", "yes"):
            return
        if self.debug_print_count >= 5 or frame_no % 300 != 0:
            return
        npc_ids = [
            (npc.get("config_id"), npc.get("actor_type"), npc.get("sub_type"), npc.get("camp"))
            for npc in (frame_state.get("npc_states", []) or [])[:12]
        ]
        cake_ids = [
            (cake.get("configId"), (cake.get("collider", {}) or {}).get("radius", 0))
            for cake in (frame_state.get("cakes", []) or [])[:6]
        ]
        hero_skill_ids = []
        for hero in (frame_state.get("hero_states", []) or [])[:2]:
            slots = ((hero.get("skill_state") or {}).get("slot_states", []) or [])
            hero_skill_ids.append((hero.get("config_id"), [(slot.get("configId"), slot.get("slot_type")) for slot in slots]))
        print("[HOK_FRAME_DEBUG]", "frame", frame_no, "npc", npc_ids, "cakes", cake_ids, "skills", hero_skill_ids)
        self.debug_print_count += 1

    def _split_npcs(self, frame_state, my_camp):
        own_tower, enemy_tower = None, None
        friendly_minions, enemy_minions, monsters = [], [], []
        for npc in frame_state.get("npc_states", []) or []:
            npc_camp = npc.get("camp")
            if self._is_tower(npc):
                if self._same_camp(npc_camp, my_camp):
                    own_tower = npc
                else:
                    enemy_tower = npc
            elif self._is_monster(npc, my_camp):
                monsters.append(npc)
            elif self._same_camp(npc_camp, my_camp):
                friendly_minions.append(npc)
            elif npc_camp not in (0, None, "0"):
                enemy_minions.append(npc)
        return own_tower, enemy_tower, friendly_minions, enemy_minions, monsters

    def _is_tower(self, npc):
        sub_type = npc.get("sub_type")
        return sub_type == TOWER_SUB_TYPE or "TOWER" in str(sub_type).upper()

    def _is_monster(self, npc, my_camp):
        actor_type = str(npc.get("actor_type", "")).upper()
        sub_type = str(npc.get("sub_type", "")).upper()
        camp = self._camp_value(npc.get("camp"))
        return "MONSTER" in actor_type or "MONSTER" in sub_type or camp in (0, None, "0")

    def _tower_targets_self(self, my_hero, enemy_tower):
        return bool(my_hero and enemy_tower and enemy_tower.get("attack_target") == my_hero.get("runtime_id"))

    def _friendly_minion_tanking(self, tower, minions):
        target = (tower or {}).get("attack_target")
        return target is not None and target in {unit.get("runtime_id") for unit in minions}

    def _can_finish_tower(self, my_hero, enemy_tower):
        return bool(
            my_hero
            and enemy_tower
            and self._hp_ratio(enemy_tower) < RuleConfig.FINISH_TOWER_HP_RATIO
            and self._hp_ratio(my_hero) > RuleConfig.FINISH_TOWER_SELF_HP_RATIO
            and self._can_attack_tower(my_hero, enemy_tower)
        )

    def _can_attack_tower(self, my_hero, enemy_tower):
        if not my_hero or not enemy_tower:
            return False
        return self._dist(self._pos(my_hero), self._pos(enemy_tower)) <= my_hero.get("attack_range", 0) + RuleConfig.ATTACK_TOWER_EXTRA_RANGE

    def _safe_blood_pack(self, cakes, my_hero, enemy_hero, enemy_tower):
        if not cakes or not my_hero or self._hp_ratio(my_hero) >= RuleConfig.BLOOD_PACK_HP_RATIO:
            return False
        my_pos = self._pos(my_hero)
        cake = min(cakes, key=lambda item: self._dist(my_pos, self._cake_pos(item)))
        dist = self._dist(my_pos, self._cake_pos(cake))
        if dist > RuleConfig.BLOOD_PACK_MAX_DIST or self._tower_risk(my_hero, enemy_tower) > RuleConfig.BLOOD_PACK_TOWER_RISK_LIMIT:
            return False
        if enemy_hero and self._alive(enemy_hero) and self._dist(self._pos(enemy_hero), my_pos) < RuleConfig.ENEMY_NEAR_DIST:
            return self._hp_ratio(my_hero) >= self._hp_ratio(enemy_hero)
        return True

    def _tower_risk(self, my_hero, enemy_tower):
        if not my_hero or not enemy_tower:
            return 0.0
        dist = self._dist(self._pos(my_hero), self._pos(enemy_tower))
        tower_range = enemy_tower.get("attack_range", RuleConfig.DEFAULT_TOWER_RANGE)
        smooth = self._sigmoid((tower_range - dist) / RuleConfig.TOWER_RISK_SIGMOID_SCALE)
        target_self = 1.0 if enemy_tower.get("attack_target") == my_hero.get("runtime_id") else 0.0
        low_hp = self._sigmoid((RuleConfig.LOW_HP_RISK_RATIO - self._hp_ratio(my_hero)) / 0.08)
        return self._clamp(smooth * (0.35 + 0.45 * target_self + 0.20 * low_hp))

    def _enhanced_attack_ready(self, my_hero):
        if not my_hero:
            return 0.0
        config_id = my_hero.get("config_id")
        if config_id == 112:
            return 1.0 if self.luban_attack_count >= 4 or self.recent_skill_used_frames > 0 else 0.0
        if config_id == 133:
            marks = ((my_hero.get("buff_state") or {}).get("buff_marks", []) or [])
            max_layer = max((mark.get("layer", 0) for mark in marks), default=0)
            return 1.0 if self.direnjie_attack_mod2 == 0 or max_layer > 0 else 0.0
        buffs = ((my_hero.get("buff_state") or {}).get("buff_skills", []) or [])
        passive = my_hero.get("passive_skill", []) or []
        return 1.0 if buffs or any(skill.get("cooldown", 0) <= 0 for skill in passive) else 0.0

    def _skill_ready_score(self, my_hero):
        slots = ((my_hero or {}).get("skill_state") or {}).get("slot_states", []) or []
        usable = [slot for slot in slots if slot.get("configId") not in SUMMONER_SKILL_IDS and slot.get("usable", False)]
        return self._safe_div(len(usable), 3.0)

    def _skill_usable(self, my_hero, skill_index):
        slots = ((my_hero or {}).get("skill_state") or {}).get("slot_states", []) or []
        normal_slots = [slot for slot in slots if slot.get("configId") not in SUMMONER_SKILL_IDS]
        normal_slots = sorted(normal_slots, key=lambda item: item.get("slot_type", 0))
        if 0 <= skill_index - 1 < len(normal_slots):
            return 1.0 if normal_slots[skill_index - 1].get("usable", False) else 0.0
        return 0.0

    def _summoner_slot(self, slots):
        for slot in slots:
            if slot.get("configId") in SUMMONER_SKILL_IDS:
                return slot
        return slots[3] if len(slots) > 3 else {}

    def _equips(self, hero):
        return (((hero or {}).get("equip_state") or {}).get("equips", []) or [])

    def _real_cmd_has_skill(self, real_cmd):
        for cmd in real_cmd:
            if cmd.get("dir_skill") or cmd.get("pos_skill") or cmd.get("obj_skill") or cmd.get("charge_skill"):
                return True
            if cmd.get("command_type") in (5, 6, 7, 8, 9, 10, 11):
                return True
        return False

    def _real_cmd_has_common_attack(self, real_cmd):
        for cmd in real_cmd:
            if cmd.get("attack_common") or cmd.get("attack_actor"):
                return True
            if cmd.get("command_type") in (3, 4):
                return True
        return False

    def _enemy_confidence(self, frame_no):
        if self.enemy_last_seen_frame < 0:
            return 0.0
        missing = max(0, frame_no - self.enemy_last_seen_frame)
        return self._clamp(math.exp(-missing / 600.0))

    def _visible_to_camp(self, obj, camp):
        visible = (obj or {}).get("camp_visible", [])
        idx = self._camp_value(camp)
        if isinstance(visible, list) and idx in (1, 2) and len(visible) >= idx:
            return 1.0 if visible[idx - 1] else 0.0
        return 1.0 if obj else 0.0

    def _set(self, feature, index, value, min_value=0.0, max_value=1.0):
        if index < 0 or index >= len(feature):
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if math.isnan(value) or math.isinf(value):
            value = 0.0
        feature[index] = self._clamp(value, min_value, max_value)

    def _set_signed(self, feature, index, value, scale):
        self._set(feature, index, self._safe_div(value, scale), -1.0, 1.0)

    def _hp_ratio(self, obj):
        if not obj:
            return 0.0
        return self._clamp(self._safe_div(obj.get("hp", 0), obj.get("max_hp", 0)))

    def _alive(self, obj):
        return 1.0 if obj and obj.get("hp", 0) > 0 else 0.0

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

    def _level_norm(self, hero):
        return self._safe_div((hero or {}).get("level", 1), 15.0)

    def _money_value(self, hero):
        return (hero or {}).get("money_cnt", (hero or {}).get("money", 0))

    def _money_norm(self, hero):
        return self._safe_div(self._money_value(hero), 20000.0)

    def _exp_ratio(self, hero):
        if not hero:
            return 0.0
        max_exp = {1: 160, 2: 298, 3: 446, 4: 524, 5: 613, 6: 713, 7: 825, 8: 950, 9: 1088, 10: 1240, 11: 1406, 12: 1585, 13: 1778, 14: 1984}.get(hero.get("level", 1), 1984)
        return self._safe_div(hero.get("exp", 0), max_exp)

    def _same_camp(self, left, right):
        return self._camp_value(left) == self._camp_value(right)

    def _is_red(self, camp):
        return self._camp_value(camp) == 2

    def _camp_value(self, camp):
        if isinstance(camp, str):
            if camp.endswith("_1") or camp == "1":
                return 1
            if camp.endswith("_2") or camp == "2":
                return 2
        return camp

    def _mirror_pos(self, pos, camp):
        if self._is_red(camp) and pos[0] != 100000 and pos[1] != 100000:
            return -pos[0], -pos[1]
        return pos

    def _norm_pos(self, value, bound):
        if value == 100000:
            return 0.0
        return self._clamp((value + bound) / (2.0 * bound))

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
