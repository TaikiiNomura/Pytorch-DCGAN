"""
Data Loader Module
データセットの準備とDataLoaderの作成
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torchvision import datasets, transforms


def get_transform(image_size):
    """
    Fashion-MNIST用の前処理を定義
    
    Args:
        image_size: リサイズ後の画像サイズ
        
    Returns:
        transforms.Compose: 前処理のパイプライン
    """
    return transforms.Compose([
        transforms.Resize(image_size),          # 画像サイズを指定サイズにリサイズ
        transforms.ToTensor(),                   # Tensorに変換（[0,1]）
        transforms.Normalize((0.5,), (0.5,))     # 正規化（グレースケールなので1チャンネル）
    ])


def create_dataloader(dataroot, batch_size, image_size, workers):
    """
    Fashion-MNISTのDataLoaderを作成
    
    Args:
        dataroot: データ保存先のパス
        batch_size: バッチサイズ
        image_size: 画像サイズ
        workers: ワーカープロセス数
        
    Returns:
        DataLoader: 作成されたDataLoader
    """
    # 前処理の定義
    transform = get_transform(image_size)
    
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
    
    return dataloader


def visualize_batch(dataloader, num_images=64):
    """
    学習データの一部を可視化
    
    Args:
        dataloader: DataLoader
        num_images: 表示する画像数
    """
    # DataLoaderから1バッチ取得
    real_batch = next(iter(dataloader))
    
    # 図の作成
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0][:num_images],  # 画像データ（ラベルは real_batch[1]）
                padding=2,
                normalize=True
            ),
            (1, 2, 0)
        ),
        cmap="gray"  # グレースケール表示
    )
    plt.show()
