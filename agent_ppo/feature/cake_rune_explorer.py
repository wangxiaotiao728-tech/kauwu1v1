#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""cake 和神符探索器。"""

import math


class CakeRuneExplorer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_near_cake = False
        self.last_hp = 0.0
        self.last_speed = 0.0
        self.last_buff_count = 0
        self.last_cd_sum = 0.0
        self.last_export = {
            "cake_observed": 0.0,
            "cake_pos": (0.0, 0.0),
            "cake_dist": 100000.0,
            "safe_to_pick_cake": 0.0,
            "cake_effect_confidence": 0.0,
            "hp_delta_after_cake": 0.0,
            "speed_delta_after_cake": 0.0,
            "buff_added_after_cake": 0.0,
            "cd_change_after_cake": 0.0,
        }

    def sample(self, frame_no, cakes, self_pos, enemy_pos_if_obs, my_hp, my_speed, my_buffs, skill_cds):
        if isinstance(self_pos, dict):
            self_pos = self._pos(self_pos)
        cakes = cakes or []
        buff_count = len(my_buffs) if isinstance(my_buffs, (list, tuple)) else len(my_buffs.keys()) if isinstance(my_buffs, dict) else 0
        cd_sum = sum(float(cd) for cd in (skill_cds or []) if cd is not None)
        if not cakes:
            if self.last_near_cake:
                hp_delta = max(0.0, my_hp - self.last_hp)
                speed_delta = max(0.0, float(my_speed or 0.0) - self.last_speed)
                buff_added = 1.0 if buff_count > self.last_buff_count else 0.0
                cd_change = max(0.0, self.last_cd_sum - cd_sum)
                if hp_delta > 0 or speed_delta > 0 or buff_added > 0 or cd_change > 0:
                    self.last_export.update(
                        {
                            "cake_effect_confidence": min(1.0, self.last_export["cake_effect_confidence"] + 0.2),
                            "hp_delta_after_cake": hp_delta,
                            "speed_delta_after_cake": speed_delta,
                            "buff_added_after_cake": buff_added,
                            "cd_change_after_cake": cd_change,
                        }
                    )
            self.last_export.update({"cake_observed": 0.0, "safe_to_pick_cake": 0.0})
            self.last_near_cake = False
            self.last_hp = my_hp
            self.last_speed = float(my_speed or 0.0)
            self.last_buff_count = buff_count
            self.last_cd_sum = cd_sum
            return
        cake = min(cakes, key=lambda item: self._dist(self_pos, self._cake_pos(item)))
        cake_pos = self._cake_pos(cake)
        dist = self._dist(self_pos, cake_pos)
        enemy_dist = self._dist(enemy_pos_if_obs, cake_pos) if enemy_pos_if_obs else 100000.0
        safe = my_hp < 0.40 and dist <= enemy_dist
        self.last_export.update(
            {
                "cake_observed": 1.0,
                "cake_pos": cake_pos,
                "cake_dist": dist,
                "safe_to_pick_cake": 1.0 if safe else 0.0,
            }
        )
        self.last_near_cake = dist < 2500
        self.last_hp = my_hp
        self.last_speed = float(my_speed or 0.0)
        self.last_buff_count = buff_count
        self.last_cd_sum = cd_sum

    def export(self):
        return dict(self.last_export)

    def _pos(self, obj):
        loc = (obj or {}).get("location", {}) or {}
        return loc.get("x", 100000), loc.get("z", 100000)

    def _cake_pos(self, cake):
        loc = ((cake or {}).get("collider", {}) or {}).get("location", {}) or {}
        return loc.get("x", 100000), loc.get("z", 100000)

    def _dist(self, a, b):
        ax, az = a
        bx, bz = b
        if 100000 in (ax, az, bx, bz):
            return 100000.0
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)
