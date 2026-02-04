"""
DCGAN Configuration Module
すべてのハイパーパラメータとグローバル設定を管理
"""
import random
import torch


class Config:
    """DCGAN学習の設定クラス"""
    
    # ランダムシード（再現性のため）
    MANUAL_SEED = 999
    
    # データセット設定
    DATAROOT = "data/fashion-mnist"
    WORKERS = 2  # DataLoaderで使用するワーカープロセス数
    
    # 学習設定
    BATCH_SIZE = 128
    IMAGE_SIZE = 32  # 学習安定化のために32x32にリサイズ
    NUM_EPOCHS = 50
    
    # 画像設定
    NC = 1  # 画像のチャンネル数（グレースケール）
    
    # ネットワーク設定
    NZ = 100  # 潜在変数zの次元数
    NGF = 64  # Generatorの特徴マップ数の基準値
    NDF = 64  # Discriminatorの特徴マップ数の基準値
    
    # 最適化設定
    LR = 0.0002  # 学習率
    BETA1 = 0.5  # Adam最適化のbeta1パラメータ
    
    # GPU設定
    NGPU = 1  # 使用するGPU数（0の場合はCPUのみ）
    
    # ラベル設定
    REAL_LABEL = 1.0
    FAKE_LABEL = 0.0
    
    @staticmethod
    def set_seed():
        """ランダムシードを設定"""
        random.seed(Config.MANUAL_SEED)
        torch.manual_seed(Config.MANUAL_SEED)
    
    @staticmethod
    def get_device():
        """使用するデバイス（GPU/CPU）を取得"""
        return torch.device(
            "cuda:0" if (torch.cuda.is_available() and Config.NGPU > 0) else "cpu"
        )
