#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Full D401 replication configuration.

This branch intentionally keeps the baseline 10-dim feature protocol to focus on
replicating the previous solution's late-stage algorithmic improvements:
- true LSTM sequence training
- grouped encoders + target attention
- D401 reward items
- full multi-critic/group-return GAE
- value clipping, optional dual-clip PPO
- LR / entropy / clip scheduling
- in-game reward decay and reward grouping
"""


class GameConfig:
    REWARD_WEIGHT_DICT = {
        # Baseline/objective rewards
        "tower_hp_point": 8.0,
        "hp_point": 1.0,
        "money": 0.05,
        "exp": 0.05,
        "kill": 1.0,
        "death": -1.0,
        "forward": 0.015,
        "ep_rate": 0.01,
        "last_hit": 0.06,
        # Targeted fixes for the observed issue: lane ignorance and random skill use.
        # lane_clear is a result-style lane advantage reward, not per-HP damage shaping.
        "lane_clear": 0.12,
        # bad_skill is a small penalty for increasing usedTimes without hitHeroTimes.
        "bad_skill": -0.005,
        # D401 replica rewards
        "hero_hurt": -0.10,
        "total_damage": 0.03,
        "hero_damage": 0.05,
        "crit": 0.01,
        "skill_hit": 0.03,
        "no_ops": -0.005,
        "in_grass": 0.001,
        "under_tower_behavior": 0.15,
        "passive_skills": 0.0,
    }

    # Full D401-style reward groups. Each group gets its own return/advantage
    # and value head. The policy advantage is a weighted sum of group advantages.
    REWARD_GROUPS = {
        "decay": [
            "money",
            "exp",
            "last_hit",
            "lane_clear",
            "forward",
            "ep_rate",
            "total_damage",
            "hero_hurt",
            "hero_damage",
            "crit",
            "skill_hit",
            "bad_skill",
            "in_grass",
            "no_ops",
            "under_tower_behavior",
            "passive_skills",
        ],
        "no_decay": ["death", "kill", "hp_point", "tower_hp_point"],
    }
    REWARD_GROUP_NAMES = list(REWARD_GROUPS.keys())
    REWARD_GROUP_ADV_WEIGHTS = {"decay": 1.0, "no_decay": 1.0}
    NO_DECAY_REWARD_KEYS = set(REWARD_GROUPS["no_decay"])

    # D401 report uses time decay for non-critical in-game rewards.
    # 0 disables it; 20000 matches the official max-frame horizon.
    TIME_SCALE_ARG = 20000
    MODEL_SAVE_INTERVAL = 1800

    # Official-protocol strict NPC type configuration.
    # String enum names are handled directly. Numeric enum values vary by environment
    # build, so only the known baseline tower value is enabled by default. Fill
    # SOLDIER_SUB_TYPES / MONSTER_ACTOR_TYPES after checking NPC_SCAN logs if the
    # environment provides integer enum values.
    TOWER_SUB_TYPES = {21}
    SOLDIER_SUB_TYPES = set()
    MONSTER_ACTOR_TYPES = set()
    SOLDIER_CONFIG_IDS = set()
    MONSTER_CONFIG_IDS = set()
    DEBUG_NPC_SCAN = False
    DEBUG_NPC_SCAN_MAX_FRAME = 200


class DimConfig:
    # Expanded D401-compatible feature dim.
    # 128 dims provide lane/skill/tower/target state needed by the current reward design.
    DIM_OF_FEATURE = [128]


class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512

    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    # Feature layout: self(16), enemy(24), skills(20), lane(28), tower(16), target(16), history(8)
    FEATURE_GROUP_SIZES = [16, 24, 20, 28, 16, 16, 8]
    LEGAL_ACTION_DIM = 85
    REWARD_GROUP_NAMES = GameConfig.REWARD_GROUP_NAMES
    REWARD_GROUP_NUM = len(REWARD_GROUP_NAMES)
    REWARD_GROUP_ADV_WEIGHTS = [GameConfig.REWARD_GROUP_ADV_WEIGHTS.get(name, 1.0) for name in REWARD_GROUP_NAMES]

    # One-frame layout before LSTM packing:
    # 0 seri_vec(feature+legal)
    # 1 global return, 2 global advantage
    # 3..3+G-1 group returns
    # next G group advantages
    # next 6 action labels
    # next 6 old action probabilities
    # next 6 branch weights/sub-action mask
    # next 1 old global value
    # next G old group values
    # next 1 is_train
    # final two entries are sequence initial lstm cell/hidden and are NOT repeated by T.
    DATA_SPLIT_SHAPE = (
        [FEATURE_DIM + LEGAL_ACTION_DIM, 1, 1]
        + [1] * REWARD_GROUP_NUM
        + [1] * REWARD_GROUP_NUM
        + [1] * len(LABEL_SIZE_LIST)
        + LABEL_SIZE_LIST
        + [1] * len(LABEL_SIZE_LIST)
        + [1]
        + [1] * REWARD_GROUP_NUM
        + [1]
        + [LSTM_UNIT_SIZE, LSTM_UNIT_SIZE]
    )
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]

    INIT_LEARNING_RATE_START = 2e-4
    TARGET_LR = 8e-5
    TARGET_STEP = 50000
    BETA_START = 0.010
    TARGET_BETA = 0.003
    LOG_EPSILON = 1e-6
    CLIP_PARAM = 0.18
    TARGET_CLIP_PARAM = 0.12
    PPO_EPOCH = 1

    USE_DUAL_CLIP = True
    DUAL_CLIP_C = 3.0
    USE_VALUE_CLIP = True
    VALUE_CLIP_PARAM = 0.2
    ADV_NORM = True

    VALUE_LOSS_COEF = 1.0
    GROUP_VALUE_LOSS_COEF = 1.0
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 64

    # Packed shapes over LSTM_TIME_STEPS for all non-LSTM-state fields.
    data_shapes = []
    for shape in DATA_SPLIT_SHAPE[:-2]:
        data_shapes.append([shape * LSTM_TIME_STEPS])
    data_shapes.extend([[LSTM_UNIT_SIZE], [LSTM_UNIT_SIZE]])

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
