"""
Visualization Module
学習結果の可視化
"""
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from matplotlib import animation
from IPython.display import HTML


def plot_losses(G_losses, D_losses):
    """
    Generator / Discriminator の損失推移を可視化
    
    Args:
        G_losses: Generatorの損失リスト
        D_losses: Discriminatorの損失リスト
    """
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


def create_animation(img_list):
    """
    Generator が生成した画像の変化をアニメーション表示
    
    Args:
        img_list: 生成画像のリスト
        
    Returns:
        HTML: Jupyter Notebook用のHTMLアニメーション
    """
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
    return HTML(ani.to_jshtml())


def compare_real_fake(dataloader, img_list, device):
    """
    本物画像と生成画像の比較表示
    
    Args:
        dataloader: DataLoader
        img_list: 生成画像のリスト
        device: 使用デバイス
    """
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
