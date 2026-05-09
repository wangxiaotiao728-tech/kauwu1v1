#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""对手近期行为连续统计。"""

from collections import deque


class OpponentBehaviorMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.events_30 = deque(maxlen=30)
        self.events_60 = deque(maxlen=60)

    def update(self, frame_no, observation, enemy_observed, enemy_action_info):
        event = {
            "attack_hero": float(enemy_action_info.get("attack_hero", 0.0)),
            "attack_tower": float(enemy_action_info.get("attack_tower", 0.0)),
            "clear_wave": float(enemy_action_info.get("clear_wave", 0.0)),
            "near_neutral": float(enemy_action_info.get("near_neutral", 0.0)),
            "enemy_observed": 1.0 if enemy_observed else 0.0,
            "damage_to_my_tower": float(enemy_action_info.get("damage_to_my_tower", 0.0)),
        }
        self.events_30.append(event)
        self.events_60.append(event)

    def export(self):
        return {
            "enemy_attack_hero_rate_30": self._rate(self.events_30, "attack_hero"),
            "enemy_attack_tower_rate_30": self._rate(self.events_30, "attack_tower"),
            "enemy_clear_wave_rate_30": self._rate(self.events_30, "clear_wave"),
            "enemy_near_neutral_rate_60": self._rate(self.events_60, "near_neutral"),
            "enemy_missing_rate_60": 1.0 - self._rate(self.events_60, "enemy_observed"),
            "enemy_damage_to_my_tower_rate_60": self._rate(self.events_60, "damage_to_my_tower"),
            "enemy_behavior_change_score": abs(
                self._rate(self.events_30, "attack_tower") - self._rate(self.events_60, "attack_tower")
            ),
        }

    def _rate(self, events, key):
        if not events:
            return 0.0
        return sum(float(event.get(key, 0.0)) for event in events) / len(events)
