#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""中立资源记忆。"""

import math

from agent_ppo.conf.conf import RuleConfig


class NeutralObjectiveMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_seen_frame = -1
        self.recently_killed_frame = -1
        self.last_disappear_frame = -1
        self.spawn_intervals = []
        self.last_export = {
            "neutral_observed": 0.0,
            "neutral_pos": (0.0, 0.0),
            "neutral_hp_ratio": 0.0,
            "neutral_dist_to_self": 100000.0,
            "neutral_dist_to_enemy": 100000.0,
            "neutral_low_hp": 0.0,
            "neutral_contested_score": 0.0,
            "neutral_safe_to_take": 0.0,
            "neutral_spawn_estimate": 0.0,
            "neutral_time_to_spawn": 0.0,
            "neutral_spawn_confidence": 0.0,
            "neutral_recently_killed": 0.0,
        }

    def sample(self, frame_no, neutral_units, self_pos, enemy_pos_if_obs, events):
        if isinstance(self_pos, dict):
            self_pos = self._pos(self_pos)
        neutral_units = neutral_units or []
        if not neutral_units:
            if self.last_seen_frame >= 0 and self.last_disappear_frame < self.last_seen_frame:
                self.last_disappear_frame = frame_no
                self.recently_killed_frame = frame_no
            recently = 1.0 if self.recently_killed_frame >= 0 and frame_no - self.recently_killed_frame < 300 else 0.0
            spawn_estimate = sum(self.spawn_intervals) / len(self.spawn_intervals) if self.spawn_intervals else 0.0
            time_to_spawn = max(0.0, spawn_estimate - (frame_no - self.last_disappear_frame)) if spawn_estimate > 0 and self.last_disappear_frame >= 0 else 0.0
            confidence = min(1.0, len(self.spawn_intervals) / 5.0)
            self.last_export.update(
                {
                    "neutral_observed": 0.0,
                    "neutral_recently_killed": recently,
                    "neutral_spawn_estimate": spawn_estimate,
                    "neutral_time_to_spawn": time_to_spawn,
                    "neutral_spawn_confidence": confidence,
                }
            )
            return
        neutral = min(neutral_units, key=lambda unit: self._dist(self_pos, self._pos(unit)))
        pos = self._pos(neutral)
        hp_ratio = self._hp_ratio(neutral)
        if self.last_disappear_frame >= 0 and self.last_seen_frame < self.last_disappear_frame:
            interval = frame_no - self.last_disappear_frame
            if interval > 0:
                self.spawn_intervals.append(interval)
                self.spawn_intervals = self.spawn_intervals[-10:]
        self.last_seen_frame = frame_no
        dist_self = self._dist(self_pos, pos)
        dist_enemy = self._dist(enemy_pos_if_obs, pos) if enemy_pos_if_obs else 100000.0
        contested = 1.0 if abs(dist_self - dist_enemy) < 5000 else 0.0
        self.last_export.update(
            {
                "neutral_observed": 1.0,
                "neutral_pos": pos,
                "neutral_hp_ratio": hp_ratio,
                "neutral_dist_to_self": dist_self,
                "neutral_dist_to_enemy": dist_enemy,
                "neutral_low_hp": 1.0 if hp_ratio < 0.25 else 0.0,
                "neutral_contested_score": contested,
                "neutral_safe_to_take": 1.0 if dist_self <= dist_enemy and hp_ratio < 0.6 else 0.0,
                "neutral_spawn_estimate": sum(self.spawn_intervals) / len(self.spawn_intervals) if self.spawn_intervals else 0.0,
                "neutral_spawn_confidence": min(1.0, len(self.spawn_intervals) / 5.0),
                "neutral_recently_killed": 0.0,
            }
        )

    def export(self):
        return dict(self.last_export)

    def _hp_ratio(self, obj):
        obj = obj or {}
        max_hp = 0
        for key in ("max_hp", "hp_max", "maxHp", "maxHP"):
            if obj.get(key):
                max_hp = obj.get(key)
                break
        if not max_hp:
            max_hp = RuleConfig.DEFAULT_NEUTRAL_MAX_HP
        return max(0.0, min(1.0, obj.get("hp", 0) / max(1.0, max_hp)))

    def _pos(self, obj):
        loc = (obj or {}).get("location", {}) or {}
        return loc.get("x", 100000), loc.get("z", 100000)

    def _dist(self, a, b):
        ax, az = a
        bx, bz = b
        if 100000 in (ax, az, bx, bz):
            return 100000.0
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)
