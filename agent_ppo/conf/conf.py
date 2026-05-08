#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # 奖励权重统一在这里调整，reward_process.py 只负责计算每一项的原始奖励值。
    # 调参原则：
    # 1. 推塔相关权重最高，因为本任务胜利条件是摧毁敌方防御塔。
    # 2. 发育、补刀、打英雄是辅助目标，权重不能高过推塔。
    # 3. 血包奖励要小，避免模型为了吃血包放弃推塔窗口。
    # 4. 惩罚项保持温和但持续，帮助模型学会少送死、少乱进塔。
    REWARD_WEIGHT_DICT = {
        # 终局奖励：只有对局结束时触发。
        "win": 16.0,  # 获胜奖励
        "lose": -12.0,  # 失败惩罚
        "timeout": -4.0,  # 超时惩罚

        # 推塔主线：最重要的 dense reward。
        "enemy_tower_damage": 10.0,  # 敌方防御塔掉血
        "own_tower_damage": -6.0,  # 己方防御塔掉血

        # 推塔窗口：鼓励抓住关键推塔机会。
        "enhanced_tower_hit": 5.0,  # 强化普攻命中防御塔
        "death_window_tower": 7.0,  # 敌方死亡期间推进/点塔

        # 塔下安全：减少乱进塔和被塔锁定。
        "tower_target_self": -0.25,  # 敌塔正在攻击自己
        "unsafe_tower_exposure": -0.12,  # 低血或危险情况下暴露在敌塔范围

        # 兵线与补刀：帮助模型理解清线是为了推塔。
        "lane_push": 0.45,  # 兵线向敌塔推进
        "enemy_minion_kill": 0.25,  # 击杀/压低敌方小兵
        "last_hit": 0.2,  # 自己完成补刀

        # 发育奖励：低权重辅助信号。
        "gold": 0.05,  # 金币增长
        "exp": 0.05,  # 经验增长
        "level_up": 0.3,  # 升级

        # 英雄战斗：鼓励优势换血，但不让目标偏离推塔。
        "enemy_hero_damage": 0.25,  # 对敌方英雄造成伤害
        "self_damage_taken": -0.25,  # 自己受到伤害
        "kill": 0.8,  # 击杀敌方英雄
        "death": -2.5,  # 自己死亡

        # 血包：只作为低血保命辅助。
        "blood_pack_heal": 0.25,  # 安全吃血包后回血
        "ignore_safe_blood_pack": -0.01,  # 低血且安全血包在附近却不靠近

        # 视野风险：敌方丢视野且自己处于危险状态时轻微惩罚。
        "vision_risk": -0.03,
    }

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


class DimConfig:
    # 模型输入特征维度。当前采用 512 维大特征协议，未使用维度填 0。
    FEATURE_DIM = 512

    # 合法动作 mask 维度：12 + 16 + 16 + 16 + 16 + 9 = 85。
    LEGAL_ACTION_DIM = 85

    # 框架读取的特征维度列表。
    DIM_OF_FEATURE = [FEATURE_DIM]


class Config:
    # 网络名称，框架侧识别用，一般不需要改。
    NETWORK_NAME = "network"

    # PPO 按 16 帧拼成一个训练样本；环境每 step 对应 6 frame。
    LSTM_TIME_STEPS = 16

    # 保留官方 LSTM 状态协议维度。当前模型不真正使用 LSTM 计算，只透传状态以兼容框架。
    LSTM_UNIT_SIZE = 512

    # 输入维度。
    FEATURE_DIM = DimConfig.FEATURE_DIM
    LEGAL_ACTION_DIM = DimConfig.LEGAL_ACTION_DIM

    # Residual MLP 网络超参数。
    MODEL_HIDDEN_DIM = 512  # LayerNorm 后第一层隐藏维度
    MODEL_EMBED_DIM = 256  # Residual MLP 输出到 policy/value 前的公共 embedding
    MODEL_RESIDUAL_BLOCK_NUM = 2  # 残差块数量
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
    TARGET_LR = 5e-5
    TARGET_STEP = 10000

    # PPO 熵正则权重，越大探索越强，但过大可能导致策略不收敛。
    BETA_START = 0.025

    # log 计算稳定项。
    LOG_EPSILON = 1e-6

    # 动作空间维度：button、move_x、move_z、skill_x、skill_z、target。
    # 不要随意修改，否则 action mask、样本格式和环境动作都会不兼容。
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]

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
    CLIP_PARAM = 0.2

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
    GAMMA = 0.995
    LAMDA = 0.95

    # 梯度裁剪，防止 reward 或 advantage 波动导致梯度爆炸。
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # learner 从 Reverb 接收的单条样本总维度。
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
