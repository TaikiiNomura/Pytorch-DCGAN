"""
DCGAN Models Module
Generator と Discriminator の定義
"""
import torch.nn as nn


def weights_init(m):
    """
    カスタム重み初期化関数
    netG（Generator）と netD（Discriminator）に適用
    
    Args:
        m: ニューラルネットワークのレイヤー
    """
    # レイヤー（m）のクラス名を文字列として取得
    # 例: "Conv2d", "ConvTranspose2d", "BatchNorm2d"
    classname = m.__class__.__name__
    
    # レイヤー名に "Conv" が含まれている場合
    # → 畳み込み層（Conv / ConvTranspose）であると判定
    if classname.find('Conv') != -1:
        # 重みを平均0、標準偏差0.02の正規分布で初期化
        # DCGANで推奨されている初期化方法
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    # レイヤー名に "BatchNorm" が含まれている場合
    # → バッチ正規化層であると判定
    elif classname.find('BatchNorm') != -1:
        # BatchNormの重み（γ）を
        # 平均1、標準偏差0.02の正規分布で初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        
        # BatchNormのバイアス（β）を0で初期化
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Generator（生成器）
    潜在変数 z から Fashion-MNIST（32×32, グレースケール）の画像を生成する
    """
    
    def __init__(self, ngpu, nz=100, ngf=64, nc=1):
        """
        Args:
            ngpu: 使用するGPU数
            nz: 潜在変数zの次元数
            ngf: 特徴マップ数の基準値
            nc: 出力画像のチャンネル数
        """
        # 親クラス（nn.Module）の初期化
        super(Generator, self).__init__()
        # 使用するGPU数（DataParallel用）
        self.ngpu = ngpu
        
        # ネットワーク本体
        self.main = nn.Sequential(
            # 入力は潜在変数 z: (nz) × 1 × 1
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # 出力サイズ: (ngf*4) × 4 × 4
            # ----------------------------------
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 出力サイズ: (ngf*2) × 8 × 8
            # ----------------------------------
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # 出力サイズ: (ngf) × 16 × 16
            # ----------------------------------
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            
            # 最終出力サイズ: (nc) × 32 × 32
        )
    
    def forward(self, input):
        """
        順伝播
        
        Args:
            input: 潜在変数 z
            
        Returns:
            生成された画像
        """
        return self.main(input)


class Discriminator(nn.Module):
    """
    Discriminator（識別器）
    入力画像が「本物か偽物か」を判別するネットワーク
    """
    
    def __init__(self, ngpu, ndf=64, nc=1):
        """
        Args:
            ngpu: 使用するGPU数
            ndf: 特徴マップ数の基準値
            nc: 入力画像のチャンネル数
        """
        # 親クラス（nn.Module）の初期化
        super(Discriminator, self).__init__()
        
        # 使用するGPU数（DataParallel用）
        self.ngpu = ngpu
        
        # Discriminatorのネットワーク本体
        self.main = nn.Sequential(
            # 入力画像: (nc) × 32 × 32
            # 畳み込みにより特徴を抽出しつつ解像度を半分にする
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 出力サイズ: (ndf) × 16 × 16
            # ----------------------------------
            
            # 特徴マップ数を増やし、解像度をさらに縮小
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 出力サイズ: (ndf*2) × 8 × 8
            # ----------------------------------
            
            # 空間情報を集約し、1次元の判別結果へ変換
            nn.Conv2d(ndf * 2, 1, 8, 1, 0, bias=False),
            
            # 出力を [0, 1] の確率として解釈できるようにする
            nn.Sigmoid()
            
            # 最終出力サイズ: 1 × 1 × 1
        )
    
    def forward(self, input):
        """
        順伝播
        
        Args:
            input: 入力画像
            
        Returns:
            判別結果（0-1の確率値）
        """
        return self.main(input)
