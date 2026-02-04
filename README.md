# Pytorch-DCGAN

PyTorchを使用したDCGAN（Deep Convolutional GAN）の実装です。Fashion-MNISTデータセットで学習を行います。

## プロジェクト構成

```
Pytorch-DCGAN/
├── src/
│   ├── config.py          # 設定・ハイパーパラメータ
│   ├── models.py          # Generator/Discriminator定義
│   ├── data_loader.py     # データセット準備
│   ├── train.py           # 学習ループ
│   └── visualize.py       # 可視化関数
├── notebooks/
│   └── train_dcgan.ipynb  # Jupyter notebook実行環境
└── README.md
```

## 特徴

- ✅ **モジュール化されたコード**: 各機能が独立したモジュールに分割
- ✅ **Google Colab対応**: Colabで直接実行可能
- ✅ **対話的な実行**: Jupyter notebookで段階的に実行
- ✅ **可視化機能**: 損失グラフ、アニメーション、比較画像

## Google Colabでの実行方法

1. Google Colabを開く: https://colab.research.google.com/

2. 以下のリンクからnotebookを開く:
   ```
   https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/Pytorch-DCGAN/blob/main/notebooks/train_dcgan.ipynb
   ```

3. ランタイムタイプをGPUに変更:
   - メニュー: `ランタイム` → `ランタイムのタイプを変更`
   - ハードウェアアクセラレータ: `GPU`

4. セルを順番に実行

## ローカルでの実行方法

### 必要な環境

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/YOUR_GITHUB_USERNAME/Pytorch-DCGAN.git
cd Pytorch-DCGAN

# 依存パッケージをインストール
pip install torch torchvision matplotlib numpy jupyter
```

### 実行

```bash
# Jupyter Notebookを起動
jupyter notebook notebooks/train_dcgan.ipynb
```

## モジュールの説明

### config.py
すべての設定とハイパーパラメータを管理するモジュール。

### models.py
Generator（生成器）とDiscriminator（識別器）のニューラルネットワーク定義。

### data_loader.py
Fashion-MNISTデータセットの読み込みと前処理。

### train.py
DCGAN学習ループの実装。Discriminator/Generatorの更新ロジックを含む。

### visualize.py
学習結果の可視化（損失グラフ、アニメーション、比較画像）。

## ハイパーパラメータ

主要なハイパーパラメータは`src/config.py`で設定できます:

- `BATCH_SIZE`: 128
- `IMAGE_SIZE`: 32
- `NUM_EPOCHS`: 50
- `LR`: 0.0002（学習率）
- `NZ`: 100（潜在変数の次元）

## ライセンス

MIT License

## 参考文献

- [DCGAN論文](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
