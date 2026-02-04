import random
import torch.nn as nn
import torch.optim as optim
from matplotlib import animation
from IPython.display import HTML

from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# torch.use_deterministic_algorithms(True)

# データセットのルートディレクトリ
dataroot = "data/fashion-mnist"

# DataLoaderで使用するワーカープロセス数
workers = 2

# 学習時のバッチサイズ
batch_size = 128

# 学習に使用する画像サイズ（Fashion-MNISTは28×28）
# 学習安定化のために32x32にリサイズ
image_size = 32

# 画像のチャンネル数（グレースケールなので1）
nc = 1

# 潜在変数 z（Generatorへの入力ノイズ）の次元数
nz = 100

# Generatorの特徴マップ数の基準値（小さめで十分）
ngf = 64

# Discriminatorの特徴マップ数の基準値
ndf = 64

# 学習エポック数（データセットを何周するか）
num_epochs = 50

# 最適化手法の学習率
lr = 0.0002

# Adam最適化手法のbeta1パラメータ（GANでよく使われる値）
beta1 = 0.5

# 使用するGPUの数（0の場合はCPUのみ）
ngpu = 1


# Fashion-MNIST用の前処理を定義
transform = transforms.Compose([
    transforms.Resize(image_size),          # 画像サイズを指定サイズにリサイズ
    transforms.ToTensor(),                   # Tensorに変換（[0,1]）
    transforms.Normalize((0.5,), (0.5,))     # 正規化（グレースケールなので1チャンネル）
])

# データセットの作成（Fashion-MNIST）
dataset = datasets.FashionMNIST(
    root=dataroot,   # データ保存先
    train=True,      # 学習用データ
    download=True,   # 未ダウンロードなら取得
    transform=transform
)

# DataLoaderの作成
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,  # バッチサイズ
    shuffle=True,           # データをシャッフル
    num_workers=workers,    # ワーカープロセス数
    pin_memory=True,
)

# 使用するデバイス（GPUがあればGPU、なければCPU）
device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
)

# 学習データの一部を表示
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0][:64],  # 画像データ（ラベルは real_batch[1]）
            padding=2,
            normalize=True
        ),
        (1, 2, 0)
    ),
    cmap="gray"  # グレースケール表示
)
plt.show()

# netG（Generator）と netD（Discriminator）に適用する
# カスタム重み初期化関数
def weights_init(m):

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

# Generator（生成器）の定義
# 潜在変数 z から Fashion-MNIST（28×28, グレースケール）の画像を生成する
class Generator(nn.Module):

    def __init__(self, ngpu):
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
        # 潜在変数 z を入力として画像を生成
        return self.main(input)

# Generator（生成器）のインスタンスを作成し、指定したデバイス（CPU / GPU）へ転送
netG = Generator(ngpu).to(device)

# GPUが利用可能かつ複数GPUを使用する場合の処理
# DataParallelを用いて複数GPUで並列計算を行う
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 事前に定義した weights_init 関数を用いて
# Generator内のすべての層の重みを初期化する
# ・畳み込み層：平均0、標準偏差0.02の正規分布
# ・BatchNorm層：平均1、標準偏差0.02（バイアスは0）
netG.apply(weights_init)

# Generatorのネットワーク構造を表示
print(netG)

# Discriminator（識別器）の定義
# 入力画像が「本物か偽物か」を判別するネットワーク
class Discriminator(nn.Module):

    def __init__(self, ngpu):
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
        # 入力画像をDiscriminator本体に通して判別結果を出力
        return self.main(input)

# Discriminator（識別器）のインスタンスを作成し、
# 指定したデバイス（CPU / GPU）へ転送
netD = Discriminator(ngpu).to(device)

# GPUが利用可能で、かつ複数GPUを使用する場合の処理
# DataParallelを用いてDiscriminatorを並列実行する
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 事前に定義した weights_init 関数を用いて
# Discriminator内のすべての層の重みを初期化する
# ・畳み込み層：平均0、標準偏差0.02の正規分布
# ・BatchNorm層：平均1、標準偏差0.02（バイアスは0）
netD.apply(weights_init)

# Discriminatorのネットワーク構造を表示
print(netD)

# Binary Cross Entropy（2値交差エントロピー）損失関数を定義
# Discriminatorが「本物 / 偽物」を正しく判別できているかを評価する
criterion = nn.BCELoss()

# Generatorの学習過程を可視化するための固定ノイズを作成
# 毎回同じノイズを使うことで、生成画像の変化を比較できる
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 学習中に使用するラベルの定義
# 本物画像には 1、偽物画像には 0 を割り当てる
real_label = 1.
fake_label = 0.

# Discriminator用の最適化手法（Adam）を設定
# netD.parameters()：Discriminatorの全パラメータを最適化対象とする
optimizerD = optim.Adam(
    netD.parameters(),
    lr=lr,
    betas=(beta1, 0.999)
)

# Generator用の最適化手法（Adam）を設定
# netG.parameters()：Generatorの全パラメータを最適化対象とする
optimizerG = optim.Adam(
    netG.parameters(),
    lr=lr,
    betas=(beta1, 0.999)
)

# =========================
# 学習ループ（Training Loop）
# =========================

# 生成画像の履歴を保存するリスト（可視化用）
img_list = []

# Generator / Discriminator の損失を記録するリスト
G_losses = []
D_losses = []

# 学習ステップ数のカウンタ
iters = 0

print("Starting Training Loop...")

# エポック数分、学習を繰り返す
for epoch in range(num_epochs):

    # DataLoaderからバッチ単位でデータを取得
    for i, data in enumerate(dataloader, 0):

        #############################################
        # (1) Discriminator の更新
        #     log(D(x)) + log(1 - D(G(z))) を最大化
        #############################################

        # 勾配をリセット
        netD.zero_grad()

        # --- 本物画像での学習 ---
        # data[0] が画像、data[1] はラベル（今回は使わない）
        real_cpu = data[0].to(device)

        # バッチサイズを取得
        b_size = real_cpu.size(0)

        # 本物画像用のラベル（すべて 1）
        label = torch.full(
            (b_size,),
            real_label,
            dtype=torch.float,
            device=device
        )

        # 本物画像を Discriminator に入力
        output = netD(real_cpu).view(-1)

        # 本物画像に対する損失を計算
        errD_real = criterion(output, label)

        # 勾配を計算（逆伝播）
        errD_real.backward()

        # D(x)：本物を本物と判定できた確率の平均
        D_x = output.mean().item()

        # --- 偽画像での学習 ---
        # 潜在変数 z をランダムに生成
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generator により偽画像を生成
        fake = netG(noise)

        # 偽画像用のラベル（すべて 0）
        label.fill_(fake_label)

        # 偽画像を Discriminator に入力
        # detach() により Generator への勾配伝播を防ぐ
        output = netD(fake.detach()).view(-1)

        # 偽画像に対する損失を計算
        errD_fake = criterion(output, label)

        # 勾配を計算（本物分と合算される）
        errD_fake.backward()

        # D(G(z))：偽画像を本物と誤認した確率の平均
        D_G_z1 = output.mean().item()

        # Discriminator の総損失
        errD = errD_real + errD_fake

        # Discriminator のパラメータを更新
        optimizerD.step()

        #############################################
        # (2) Generator の更新
        #     log(D(G(z))) を最大化
        #############################################

        # Generator の勾配をリセット
        netG.zero_grad()

        # Generator の目的は「偽画像を本物だと判定させる」こと
        # そのためラベルを本物（1）に設定
        label.fill_(real_label)

        # 更新後の Discriminator で偽画像を再評価
        output = netD(fake).view(-1)

        # Generator の損失を計算
        errG = criterion(output, label)

        # 勾配を計算（逆伝播）
        errG.backward()

        # D(G(z))：Generator 更新後の判定確率
        D_G_z2 = output.mean().item()

        # Generator のパラメータを更新
        optimizerG.step()

        #############################################
        # 学習状況の表示
        #############################################

        if i % 50 == 0:
            print(
                '[%d/%d][%d/%d]\t'
                'Loss_D: %.4f\tLoss_G: %.4f\t'
                'D(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (
                    epoch, num_epochs,
                    i, len(dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2
                )
            )

        # 損失を保存（後でグラフ化するため）
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        #############################################
        # Generator の出力を定期的に保存（可視化用）
        #############################################

        if (iters % 500 == 0) or (
            (epoch == num_epochs - 1) and
            (i == len(dataloader) - 1)
        ):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()

            # 生成画像をグリッド化して保存
            img_list.append(
                vutils.make_grid(fake, padding=2, normalize=True)
            )

        # イテレーション数を更新
        iters += 1

# =========================
# Generator / Discriminator の損失推移を可視化
# =========================

plt.figure(figsize=(10, 5))

# グラフのタイトル
plt.title("Generator and Discriminator Loss During Training")

# Generator の損失をプロット
plt.plot(G_losses, label="G")

# Discriminator の損失をプロット
plt.plot(D_losses, label="D")

# x軸・y軸のラベル
plt.xlabel("iterations")
plt.ylabel("Loss")

# 凡例を表示
plt.legend()

# グラフを表示
plt.show()

# =========================
# Generator が生成した画像の変化をアニメーション表示
# =========================

# 図のサイズを指定
fig = plt.figure(figsize=(8, 8))

# 軸を非表示
plt.axis("off")

# img_list に保存された生成画像を順番に表示するためのリストを作成
# 各画像は (C, H, W) → (H, W, C) に変換
ims = [
    [plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
    for i in img_list
]

# アニメーションを作成
# interval: フレーム間隔（ms）
ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=1000,
    repeat_delay=1000,
    blit=True
)

# Jupyter Notebook 上でアニメーションを表示
HTML(ani.to_jshtml())

# =========================
# 本物画像と生成画像の比較表示
# =========================

# DataLoaderから本物画像のバッチを1つ取得
real_batch = next(iter(dataloader))

# 図のサイズを指定
plt.figure(figsize=(15, 15))

# -------- 本物画像 --------
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")

# 本物画像をグリッド状に表示
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[:64],
            padding=5,
            normalize=True
        ).cpu(),
        (1, 2, 0)
    )
)

# -------- 生成画像 --------
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")

# 学習終了時点のGeneratorが生成した画像を表示
plt.imshow(
    np.transpose(
        img_list[-1],
        (1, 2, 0)
    )
)

# 表示
plt.show()
