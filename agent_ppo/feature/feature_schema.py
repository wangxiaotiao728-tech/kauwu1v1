#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""固定 512 维特征协议。"""

from dataclasses import dataclass
import hashlib
import json


FEATURE_DIM = 512


@dataclass(frozen=True)
class FeatureRange:
    name: str
    start: int
    end: int
    priority: str


FEATURE_RANGES = [
    FeatureRange("global_phase", 0, 31, "P0"),
    FeatureRange("my_hero", 32, 79, "P0/P1"),
    FeatureRange("enemy_hero_visible_last_seen", 80, 143, "P0/P1"),
    FeatureRange("skill_combo", 144, 207, "P0/P1"),
    FeatureRange("tower", 208, 271, "P0/P1"),
    FeatureRange("lane_minion_slots", 272, 359, "P0/P1"),
    FeatureRange("cake_neutral_resource", 360, 407, "P0/P1/P2"),
    FeatureRange("vision_grass_geom", 408, 439, "P0/P1"),
    FeatureRange("opponent_behavior", 440, 471, "P1"),
    FeatureRange("rule_memory_debug", 472, 495, "P1/P2"),
    FeatureRange("reserved", 496, 511, "P2"),
]


IDX = {
    "frame_progress": 0,
    "timeout_progress": 1,
    "early_phase": 2,
    "mid_phase": 3,
    "late_phase": 4,
    "direct_push_window": 8,
    "defense_emergency": 9,
    "resource_allowed": 10,
    "my_hp_ratio": 16,
    "my_level_norm": 18,
    "my_gold_norm": 20,
    "my_alive": 21,
    "enemy_observed": 80,
    "enemy_not_observed_steps": 81,
    "enemy_alive": 82,
    "enemy_hp_ratio": 84,
    "enemy_last_seen_x": 96,
    "enemy_last_seen_z": 97,
    "enemy_last_seen_time_norm": 98,
    "enemy_tower_hp_ratio": 208,
    "own_tower_hp_ratio": 224,
    "friendly_minion_count": 272,
    "enemy_minion_count": 273,
    "cake_observed": 360,
    "neutral_observed": 376,
    "enemy_attack_hero_rate_30": 440,
    "enemy_attack_tower_rate_30": 441,
}


def get_feature_schema_hash():
    payload = [(r.name, r.start, r.end, r.priority) for r in FEATURE_RANGES]
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
