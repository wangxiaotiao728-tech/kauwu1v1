#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""防御塔范围探索器。"""

import math


class TowerRangeExplorer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.enemy_samples = []
        self.own_samples = []
        self.last_self_in_enemy_tower_range = 0.0
        self.last_safe_distance_to_enemy_tower = 8500.0

    def sample(
        self,
        frame_no,
        self_pos,
        enemy_tower_pos,
        own_tower_pos,
        enemy_tower_target,
        own_tower_target,
        friendly_minions,
        enemy_minions,
    ):
        self_id = None
        if isinstance(self_pos, dict):
            self_id = self_pos.get("runtime_id")
            self_pos = self._pos(self_pos)
        if enemy_tower_target is not None and (enemy_tower_target == self_id or enemy_tower_target in {m.get("runtime_id") for m in friendly_minions}):
            dist = self._dist(self_pos, enemy_tower_pos)
            if dist < 100000:
                self.enemy_samples.append(dist)
        if own_tower_target is not None and own_tower_target in {m.get("runtime_id") for m in enemy_minions}:
            dist = self._dist(self_pos, own_tower_pos)
            if dist < 100000:
                self.own_samples.append(dist)
        self.enemy_samples = self.enemy_samples[-200:]
        self.own_samples = self.own_samples[-200:]
        enemy_range = max(self.enemy_samples) if self.enemy_samples else 8500.0
        self_dist = self._dist(self_pos, enemy_tower_pos)
        self.last_self_in_enemy_tower_range = 1.0 if self_dist <= enemy_range else 0.0
        self.last_safe_distance_to_enemy_tower = max(0.0, self_dist - enemy_range)

    def export(self):
        enemy_range = max(self.enemy_samples) if self.enemy_samples else 8500.0
        own_range = max(self.own_samples) if self.own_samples else 8500.0
        valid_samples = len(self.enemy_samples) + len(self.own_samples)
        confidence = min(1.0, valid_samples / 200.0)
        return {
            "estimated_enemy_tower_range": enemy_range,
            "estimated_own_tower_range": own_range,
            "tower_range_confidence": confidence,
            "self_in_enemy_tower_range_estimated": self.last_self_in_enemy_tower_range,
            "safe_distance_to_enemy_tower": self.last_safe_distance_to_enemy_tower,
        }

    def _pos(self, obj):
        loc = (obj or {}).get("location", {}) or {}
        return loc.get("x", 100000), loc.get("z", 100000)

    def _dist(self, a, b):
        ax, az = a
        bx, bz = b
        if 100000 in (ax, az, bx, bz):
            return 100000.0
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)
