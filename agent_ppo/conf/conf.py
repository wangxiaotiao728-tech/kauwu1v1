#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Final D401-256 configuration.

This version uses real protocol fields observed in the 1v1 environment:
- 256-dim feature vector grouped by hero/enemy/skill/lane/objective/target/history.
- 3 reward groups: objective, growth_combat, behavior_safety.
- LSTM384 grouped model.
- Conservative PPO schedules for the larger feature/network version.
"""


class GameConfig:
    # ===== Confirmed ids from real frame_state =====
    HERO_LUBAN_ID = 112
    HERO_DIRENJIE_ID = 133

    SOLDIER_SUB_TYPES = {11}
    SOLDIER_CONFIG_IDS = {6800, 6801, 6803, 6804}

    TOWER_SUB_TYPES = {21}
    TOWER_CONFIG_IDS = {1111, 1112}

    EXCLUDED_NPC_SUB_TYPES = {23, 24}
    BASE_OR_SPRING_CONFIG_IDS = {44, 46, 1113, 1114}

    MONSTER_CONFIG_IDS = {6827}
    CAKE_CONFIG_IDS = {5}
    OWN_CAKE_ONLY = True
    CAKE_LOW_HP_THRESHOLD = 0.40
    CAKE_MEDIUM_HP_THRESHOLD = 0.60

    # Button ids from the HoK 1v1 action protocol:
    # 0 invalid, 1 none, 2 move, 3 normal attack, 4-6 skills,
    # 7 recover, 8 summoner, 9 recall, 10 skill4, 11 equipment skill.
    BUTTON_INVALID = 0
    BUTTON_NONE = 1
    BUTTON_MOVE = 2
    BUTTON_ATTACK = 3
    BUTTON_SKILL1 = 4
    BUTTON_SKILL2 = 5
    BUTTON_SKILL3 = 6
    BUTTON_RECOVER = 7
    BUTTON_SUMMONER = 8
    BUTTON_RECALL = 9
    BUTTON_SKILL4 = 10
    BUTTON_EQUIPMENT = 11

    TARGET_NONE = 0
    TARGET_ENEMY = 1
    TARGET_SELF = 2
    TARGET_SOLDIER_0 = 3
    TARGET_SOLDIER_1 = 4
    TARGET_SOLDIER_2 = 5
    TARGET_SOLDIER_3 = 6
    TARGET_TOWER = 7
    TARGET_MONSTER = 8

    # Reward weights. Items with weight 0 are kept only for compatibility/monitor.
    REWARD_WEIGHT_DICT = {
        # objective
        "tower_hp_point": 8.0,
        "kill": 1.0,
        "death": -1.0,
        # growth / combat
        "hp_point": 1.0,
        "money": 0.02,
        "exp": 0.02,
        "last_hit": 0.06,
        "hero_hurt": -0.10,
        "total_damage": 0.03,
        "hero_damage": 0.05,
        "skill_hit": 0.02,
        # behavior / safety
        "lane_clear": 0.12,
        "defense": 0.08,
        "cake": 0.05,
        "tower_risk": 0.03,
        "stuck": -0.12,
        "no_ops": -0.03,
        "grass_behavior": 0.03,
        "forward": 0.20,
        # disabled / monitor-only
        "bad_skill": 0.0,
        "passive_skills": 0.0,
        "crit": 0.0,
        "in_grass": 0.0,
        # optional old baseline keys kept off
        "ep_rate": 0.0,
        "monster_resource": 0.0,
    }

    REWARD_GROUPS = {
        "objective": ["tower_hp_point", "kill", "death"],
        "growth_combat": [
            "hp_point",
            "money",
            "exp",
            "last_hit",
            "hero_hurt",
            "total_damage",
            "hero_damage",
            "skill_hit",
        ],
        "behavior_safety": [
            "lane_clear",
            "defense",
            "cake",
            "forward",
            "tower_risk",
            "stuck",
            "no_ops",
            "grass_behavior",
        ],
    }
    REWARD_GROUP_NAMES = list(REWARD_GROUPS.keys())
    REWARD_GROUP_ADV_WEIGHTS = {"objective": 1.0, "growth_combat": 1.0, "behavior_safety": 1.0}
    NO_DECAY_REWARD_KEYS = set(REWARD_GROUPS["objective"])

    # 0 disables decay. Keep a mild decay for shaping terms only.
    TIME_SCALE_ARG = 20000
    MODEL_SAVE_INTERVAL = 1800

    # Debug switches.
    DEBUG_NPC_SCAN = False
    DEBUG_NPC_SCAN_MAX_FRAME = 3000
    DEBUG_FEATURE_CHECK = False

    # Training lineup curriculum.
    # luban_priority: start with more Luban mirror games while still sampling all 4 matchups.
    # all_matchups: balanced 112/133 x 112/133.
    # luban_mirror / direnjie_mirror: single-hero debugging phases.
    TRAIN_LINEUP_MODE = "luban_priority"
    TRAIN_LINEUP_PAIRS = {
        "luban_mirror": [(112, 112)],
        "direnjie_mirror": [(133, 133)],
        "all_matchups": [(112, 112), (112, 133), (133, 112), (133, 133)],
        "luban_priority": [(112, 112), (112, 112), (112, 112), (133, 133), (112, 133), (133, 112)],
    }

    # Keep summoner skill fixed in the early curriculum to reduce exploration noise.
    SUMMONER_SKILL_POLICY = "fixed_flash"
    DEFAULT_SUMMONER_SKILL_ID = 80115
    HERO_SUMMONER_SKILL_IDS = {
        112: 80115,
        133: 80115,
    }
    SAFE_SUMMONER_SKILL_IDS = [80115, 80102, 80109]


class DimConfig:
    DIM_OF_FEATURE = [256]


class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 384
    FUSION_DIM = 512

    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    FEATURE_GROUP_SIZES = [32, 32, 56, 40, 32, 40, 24]
    LEGAL_ACTION_DIM = 85

    REWARD_GROUP_NAMES = GameConfig.REWARD_GROUP_NAMES
    REWARD_GROUP_NUM = len(REWARD_GROUP_NAMES)
    REWARD_GROUP_ADV_WEIGHTS = [GameConfig.REWARD_GROUP_ADV_WEIGHTS.get(name, 1.0) for name in REWARD_GROUP_NAMES]

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

    # LR schedule: warmup to BASE_LR, then cosine decay. The existing Agent uses
    # LambdaLR, so these values are consumed in Agent.lr_lambda.
    INIT_LEARNING_RATE_START = 8e-5
    WARMUP_LR = 2e-5
    WARMUP_STEPS = 5000
    MIN_LR = 2e-5
    TARGET_LR = MIN_LR
    TARGET_STEP = 180000

    BETA_START = 0.02
    TARGET_BETA = 0.003
    BETA_HOLD_STEPS = 10000
    LOG_EPSILON = 1e-6
    CLIP_PARAM = 0.18
    TARGET_CLIP_PARAM = 0.10
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
