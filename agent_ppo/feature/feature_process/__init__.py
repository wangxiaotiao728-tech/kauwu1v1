#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Baseline feature processor used by the D401 replica branch.

The pure replica branch intentionally keeps the official baseline feature shape
at 10 dimensions: HeroProcess(3) + OrganProcess(7). This avoids changing the
input protocol while algorithm/reward/model changes are reproduced first.
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess


class FeatureProcess:
    def __init__(self, camp):
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        feature = []
        feature.extend(self.hero_process.process_vec_hero(frame_state))
        feature.extend(self.organ_process.process_vec_organ(frame_state))
        # Strictly keep baseline protocol: 10 features.
        if len(feature) < 10:
            feature.extend([0.0] * (10 - len(feature)))
        elif len(feature) > 10:
            feature = feature[:10]
        return feature
