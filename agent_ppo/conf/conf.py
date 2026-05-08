#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # ===== 模型与特征不变量 =====
    # 固定内部特征维度；训练开始后禁止修改，否则旧模型不能安全续训。
    FEATURE_DIM = 512
    # LSTM 隐状态维度；checkpoint 会校验该值。
    LSTM_HIDDEN_SIZE = 256
    # LSTM 层数；方案要求单层。
    LSTM_NUM_LAYERS = 1
    # learner 按 16 帧切成一个序列训练。
    LSTM_TIME_STEPS = 16
    # 官方六头动作协议：button、move_x、move_z、skill_x、skill_z、target。
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    # 当前模型版本；checkpoint 元信息使用。
    MODEL_VERSION = "ppo_lstm_512_v1"

    # ===== 功能模块开关 =====
    # 是否启用 LSTM 序列模型。
    USE_LSTM = True
    # 是否启用历史记忆模块。
    USE_MEMORY_PROCESS = True
    # 是否启用规则控制器，规则只产出 hard mask 和 logit bias。
    USE_RULE_CONTROLLER = True
    # 是否启用敌我防御塔范围探索。
    USE_TOWER_RANGE_EXPLORER = True
    # 是否启用血包 cake 和神符探索。
    USE_CAKE_RUNE_EXPLORER = True
    # 是否启用中立资源探索。
    USE_NEUTRAL_OBJECTIVE = True
    # 是否启用对手近期行为统计。
    USE_OPPONENT_BEHAVIOR_MEMORY = True
    # 是否启用技能和强化普攻连招记忆。
    USE_COMBO_MEMORY = True

    # ===== 禁止普通血量变化 reward =====
    # 禁止奖励敌方英雄普通掉血。
    USE_HERO_DAMAGE_REWARD = False
    # 禁止奖励敌方小兵普通掉血。
    USE_MINION_DAMAGE_REWARD = False
    # 禁止奖励精灵或野怪普通掉血。
    USE_NEUTRAL_DAMAGE_REWARD = False
    # 禁止对自己逐帧掉血惩罚。
    USE_SELF_HP_DELTA_PENALTY = False
    # 禁止对己方小兵逐帧掉血惩罚。
    USE_FRIENDLY_MINION_HP_DELTA_PENALTY = False

    # 奖励权重统一在这里调整，reward_process.py 只负责计算每一项原始值。
    # 注意：这里不包含普通英雄掉血、小兵掉血、野怪掉血、自身掉血逐帧惩罚。
    REWARD_WEIGHTS = {
        "terminal_win": 20.0,  # 获胜终局奖励
        "terminal_lose": -20.0,  # 失败终局惩罚
        "timeout": -6.0,  # 超时惩罚
        "enemy_tower_delta": 8.0,  # 敌方防御塔血量下降奖励
        "own_tower_delta": -10.0,  # 己方防御塔血量下降惩罚
        "lane": 1.0,  # 兵线推进和线权奖励
        "growth": 0.8,  # 金币、经验、等级成长奖励
        "last_hit": 0.2,  # 自己完成补刀奖励
        "death": -4.0,  # 自己死亡惩罚
        "tower_risk": -1.5,  # 危险塔下暴露惩罚
        "enhanced_tower": 1.0,  # 强化普攻点塔奖励
        "death_window_tower_mult": 1.5,  # 敌方死亡窗口推塔倍率
        "neutral_result": 0.5,  # 明确拿到中立资源奖励
        "cake_safe_pick": 0.2,  # 低血安全吃 cake 的小奖励
        "skill_result": 0.3,  # 技能明确结果奖励
        "bad_resource": -1.0,  # 错误争资源惩罚
    }
    # 兼容旧代码入口。
    REWARD_WEIGHT_DICT = REWARD_WEIGHTS

    # 单步总奖励裁剪范围，防止 reward_sum 过大导致 PPO 梯度震荡。
    REWARD_SUM_CLIP_MIN = -10.0
    REWARD_SUM_CLIP_MAX = 10.0

    # 模型保存间隔，单位：秒。
    MODEL_SAVE_INTERVAL = 1800

    # 预留时间衰减参数；当前 reward 主要使用相邻帧差分，暂不启用时间衰减。
    TIME_SCALE_ARG = 0


class RuleConfig:
    # 规则状态机阈值，统一放这里便于调参。
    # 这些阈值只用于高确定性的兜底动作，不改变官方动作空间。

    # 敌塔血量低于该比例，且自己血量足够时，优先终结防御塔。
    FINISH_TOWER_HP_RATIO = 0.08

    # 终结防御塔时，自己的最低安全血量。
    FINISH_TOWER_SELF_HP_RATIO = 0.30

    # 血量低于该比例才考虑主动吃血包。
    BLOOD_PACK_HP_RATIO = 0.40

    # 主动吃血包的最大搜索距离。
    BLOOD_PACK_MAX_DIST = 12000

    # 塔风险超过该值时，不主动吃血包，优先撤离危险区域。
    BLOOD_PACK_TOWER_RISK_LIMIT = 0.45

    # 敌方英雄小于该距离时，认为血包路线可能被威胁。
    ENEMY_NEAR_DIST = 6500

    # 普攻防御塔时，在攻击范围外额外放宽的距离。
    ATTACK_TOWER_EXTRA_RANGE = 1200

    # 把目标方向映射到 16 档离散移动方向时使用的距离尺度。
    MOVE_DIR_SCALE = 12000.0

    # 默认防御塔攻击范围，用于环境暂时未给 attack_range 的兜底。
    DEFAULT_TOWER_RANGE = 8500

    # 塔风险 sigmoid 平滑尺度，越小越敏感。
    TOWER_RISK_SIGMOID_SCALE = 1500.0

    # 低血风险阈值，用于塔下风险和视野风险。
    LOW_HP_RISK_RATIO = 0.35

    # 规则层认为可以主动换血的血量优势。
    TRADE_HP_ADVANTAGE = 0.12

    # 血量高于该比例时，才把野怪/精灵作为可考虑目标。
    FARM_MONSTER_HP_RATIO = 0.55

    # 子弹危险判断：低血且敌方子弹靠近时，视为高风险。
    BULLET_DANGER_DIST = 4500
    BULLET_DANGER_HP_RATIO = 0.45

    # 规则 bias 的绝对上限；禁止 +5、+10 这种过强偏置。
    LOGIT_BIAS_ABS_MAX = 2.0

    # hard mask 比例超过该值时，应把非关键屏蔽降级为 bias。
    HARD_MASK_RATE_LIMIT = 0.20

    # fallback 动作，所有 head 被屏蔽时回退到 noop。
    NOOP_ACTION = [0, 15, 15, 15, 15, 0]


class CurriculumConfig:
    # 课程训练开关：
    # True 表示训练局自动在 common_ai 和 selfplay 之间切换；
    # False 表示完全使用 train_env_conf.toml 里的 opponent_agent。
    ENABLE_CURRICULUM = True

    # 评估局是否始终使用 common_ai。
    # 建议保持 True，这样 win_rate_common_ai / enemy_tower_hp_common_ai 曲线可作为稳定标尺。
    EVAL_ALWAYS_COMMON_AI = True

    # 早期固定 common_ai 的局数，先学习基础清线、防守、推塔。
    COMMON_AI_WARMUP_EPISODES = 800

    # 最近多少局 common_ai 训练结果用于判断是否可以混入 selfplay。
    COMMON_AI_GATE_WINDOW = 20

    # 进入 selfplay 混合训练的门槛：敌塔平均剩余血量低于该值。
    # 防御塔满血约 12000，6000 表示至少能稳定打掉一半。
    COMMON_AI_GATE_ENEMY_TOWER_HP = 6000

    # 进入 selfplay 混合训练的门槛：最近 common_ai 局平均死亡数不能太高。
    COMMON_AI_GATE_DEATH = 1.5

    # 如果训练很久仍未过门槛，允许少量 selfplay 提供多样性，避免只过拟合 common_ai。
    FORCE_SELFPLAY_AFTER_EPISODES = 1200

    # 通过门槛后，selfplay 的初始采样概率。
    SELFPLAY_PROB_START = 0.20

    # selfplay 最大采样概率。不要一口气拉到 1.0，否则 common_ai 基础能力可能退化。
    SELFPLAY_PROB_MAX = 0.60

    # selfplay 概率从 START 增长到 MAX 的训练局数。
    SELFPLAY_RAMP_EPISODES = 800

    # 未过门槛但超过 FORCE_SELFPLAY_AFTER_EPISODES 后的探索性 selfplay 概率。
    FALLBACK_SELFPLAY_PROB = 0.10

    # 当前课程阶段名；可通过配置或环境变量切换，不需要改训练代码。
    CURRENT_STAGE = "S1_BASIC"

    # 四阶段课程配置。只允许阶段间调整 reward、rule bias、对手和超参。
    CURRICULUM_STAGES = {
        "S1_BASIC": {
            "opponent": "common_ai",  # 基础阶段，对手固定 common_ai
            "enable_rule_bias": False,  # 关闭复杂规则偏置
            "enable_resource_reward": False,  # 关闭资源奖励
            "enable_skill_result_reward": False,  # 关闭技能结果奖励
            "enable_cake_reward": False,  # 关闭 cake 奖励
            "lr_start": 3e-4,  # 阶段起始学习率
            "lr_end": 2.3e-4,  # 阶段结束学习率
            "entropy_start": 0.020,  # 阶段起始熵系数
            "entropy_end": 0.015,  # 阶段结束熵系数
            "clip_start": 0.20,  # 阶段起始 PPO clip
            "clip_end": 0.18,  # 阶段结束 PPO clip
            "ppo_epoch": 1,  # 每批样本更新轮数
        },
        "S2_TOWER_WINDOW": {
            "opponent": "common_ai",
            "enable_rule_bias": True,
            "enable_enhanced_tower_reward": True,
            "enable_death_window_reward": True,
            "lr_start": 2.3e-4,
            "lr_end": 1.7e-4,
            "entropy_start": 0.015,
            "entropy_end": 0.011,
            "clip_start": 0.18,
            "clip_end": 0.16,
            "ppo_epoch": 1,
        },
        "S3_MECHANISM": {
            "opponent": "common_ai_plus_short_selfplay",
            "enable_resource_reward": True,
            "enable_skill_result_reward": True,
            "enable_cake_reward": True,
            "lr_start": 1.7e-4,
            "lr_end": 1.2e-4,
            "entropy_start": 0.011,
            "entropy_end": 0.008,
            "clip_start": 0.16,
            "clip_end": 0.15,
            "ppo_epoch": 1,
        },
        "S4_GENERALIZATION": {
            "opponent": "selfplay_history_pool_common_ai_eval",
            "enable_opponent_behavior": True,
            "enable_resource_bias": True,
            "lr_start": 1.2e-4,
            "lr_end": 8e-5,
            "entropy_start": 0.008,
            "entropy_end": 0.005,
            "clip_start": 0.15,
            "clip_end": 0.12,
            "ppo_epoch": 1,  # 稳定后可切为 2
        },
    }


class DimConfig:
    # 模型输入特征维度。当前采用 512 维大特征协议，未使用维度填 0。
    FEATURE_DIM = GameConfig.FEATURE_DIM

    # 合法动作 mask 维度：12 + 16 + 16 + 16 + 16 + 9 = 85。
    LEGAL_ACTION_DIM = 85

    # 框架读取的特征维度列表。
    DIM_OF_FEATURE = [FEATURE_DIM]


class Config:
    # 网络名称，框架侧识别用，一般不需要改。
    NETWORK_NAME = "network"

    # PPO 按 16 帧拼成一个训练样本；环境每 step 对应 6 frame。
    LSTM_TIME_STEPS = GameConfig.LSTM_TIME_STEPS

    # LSTM hidden/cell 维度；episode reset 后必须清零，checkpoint 不保存单局隐状态。
    LSTM_UNIT_SIZE = GameConfig.LSTM_HIDDEN_SIZE

    # 输入维度。
    FEATURE_DIM = DimConfig.FEATURE_DIM
    LEGAL_ACTION_DIM = DimConfig.LEGAL_ACTION_DIM

    # Residual MLP 网络超参数。
    MODEL_HIDDEN_DIM = 512  # FrameEncoder 第一层隐藏维度
    MODEL_EMBED_DIM = 256  # FrameEncoder 输出维度，同时作为 LSTM 输入维度
    MODEL_RESIDUAL_BLOCK_NUM = 0  # LSTM 版不再使用残差块
    POLICY_HIDDEN_DIM = 256  # policy encoder 隐藏维度
    VALUE_HIDDEN_DIM = 256  # value encoder 隐藏维度

    # learner 侧样本切分协议，必须和 feature、动作、概率、LSTM 状态拼接方式一致。
    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_DIM,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]

    # observation 中 feature 和 legal_action 的切分形状。
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]

    # 学习率调度：从 INIT_LEARNING_RATE_START 线性衰减到 TARGET_LR。
    INIT_LEARNING_RATE_START = 3e-4
    TARGET_LR = 8e-5
    TARGET_STEP = 10000

    # PPO 熵正则权重，越大探索越强，但过大可能导致策略不收敛。
    BETA_START = 0.020

    # log 计算稳定项。
    LOG_EPSILON = 1e-6

    # 动作空间维度：button、move_x、move_z、skill_x、skill_z、target。
    # 不要随意修改，否则 action mask、样本格式和环境动作都会不兼容。
    LABEL_SIZE_LIST = GameConfig.LABEL_SIZE_LIST

    # 每个动作头是否参与 PPO policy loss。
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    # PPO clip 参数，常用范围 0.1 - 0.3。
    CLIP_PARAM = 0.20

    # 合法动作 softmax 的最小概率，防止 log(0)。
    MIN_POLICY = 0.00001

    # 预留 target embedding 维度，保持 baseline 协议兼容。
    TARGET_EMBED_DIM = 32

    # reverb 样本各段的实际扁平长度。
    data_shapes = [
        [(FEATURE_DIM + LEGAL_ACTION_DIM) * LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[0] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[1] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[2] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[3] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[4] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[5] * LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_UNIT_SIZE],
        [LSTM_UNIT_SIZE],
    ]

    # legal_action 在环境原始协议里 target 可能按 button 展开，采样时会压回 9 维。
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    # GAE 折扣参数。GAMMA 越高越重视长期收益，LAMDA 控制 bias/variance。
    GAMMA = 0.997
    LAMDA = 0.95

    # 梯度裁剪，防止 reward 或 advantage 波动导致梯度爆炸。
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # PPO value loss 系数。
    VALUE_LOSS_COEF = 0.5

    # PPO 熵系数初始值和最终值，由课程调度覆盖。
    ENTROPY_COEF_INIT = 0.020
    ENTROPY_COEF_FINAL = 0.005

    # PPO clip 初始值和最终值，由课程调度覆盖。
    PPO_CLIP_INIT = 0.20
    PPO_CLIP_FINAL = 0.12

    # 学习率初始值和最终值，由课程调度覆盖。
    LR_INIT = 3e-4
    LR_FINAL = 8e-5

    # PPO 更新轮数；课程后期稳定后可增至 2。
    PPO_EPOCH = 1
    PPO_EPOCH_FINAL = 2

    # learner mini batch 大小。
    MINI_BATCH_SIZE = 512

    # learner 从 Reverb 接收的单条样本总维度。
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
