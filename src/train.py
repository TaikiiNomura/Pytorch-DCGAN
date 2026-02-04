"""
Training Module
DCGAN学習ループの実装
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils


class DCGANTrainer:
    """DCGAN学習クラス"""
    
    def __init__(self, netG, netD, device, config):
        """
        Args:
            netG: Generator
            netD: Discriminator
            device: 使用するデバイス（CPU/GPU）
            config: 設定クラス
        """
        self.netG = netG
        self.netD = netD
        self.device = device
        self.config = config
        
        # Binary Cross Entropy（2値交差エントロピー）損失関数を定義
        self.criterion = nn.BCELoss()
        
        # Generatorの学習過程を可視化するための固定ノイズを作成
        self.fixed_noise = torch.randn(64, config.NZ, 1, 1, device=device)
        
        # Discriminator用の最適化手法（Adam）を設定
        self.optimizerD = optim.Adam(
            netD.parameters(),
            lr=config.LR,
            betas=(config.BETA1, 0.999)
        )
        
        # Generator用の最適化手法（Adam）を設定
        self.optimizerG = optim.Adam(
            netG.parameters(),
            lr=config.LR,
            betas=(config.BETA1, 0.999)
        )
        
        # 学習履歴
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
    
    def train_discriminator(self, real_data, batch_size):
        """
        Discriminatorの更新
        log(D(x)) + log(1 - D(G(z))) を最大化
        
        Args:
            real_data: 本物画像のバッチ
            batch_size: バッチサイズ
            
        Returns:
            tuple: (errD, D_x, D_G_z1)
                - errD: Discriminatorの損失
                - D_x: 本物を本物と判定できた確率
                - D_G_z1: 偽物を本物と誤認した確率
        """
        # 勾配をリセット
        self.netD.zero_grad()
        
        # --- 本物画像での学習 ---
        real_cpu = real_data.to(self.device)
        
        # 本物画像用のラベル（すべて 1）
        label = torch.full(
            (batch_size,),
            self.config.REAL_LABEL,
            dtype=torch.float,
            device=self.device
        )
        
        # 本物画像を Discriminator に入力
        output = self.netD(real_cpu).view(-1)
        
        # 本物画像に対する損失を計算
        errD_real = self.criterion(output, label)
        
        # 勾配を計算（逆伝播）
        errD_real.backward()
        
        # D(x)：本物を本物と判定できた確率の平均
        D_x = output.mean().item()
        
        # --- 偽画像での学習 ---
        # 潜在変数 z をランダムに生成
        noise = torch.randn(batch_size, self.config.NZ, 1, 1, device=self.device)
        
        # Generator により偽画像を生成
        fake = self.netG(noise)
        
        # 偽画像用のラベル（すべて 0）
        label.fill_(self.config.FAKE_LABEL)
        
        # 偽画像を Discriminator に入力
        # detach() により Generator への勾配伝播を防ぐ
        output = self.netD(fake.detach()).view(-1)
        
        # 偽画像に対する損失を計算
        errD_fake = self.criterion(output, label)
        
        # 勾配を計算（本物分と合算される）
        errD_fake.backward()
        
        # D(G(z))：偽画像を本物と誤認した確率の平均
        D_G_z1 = output.mean().item()
        
        # Discriminator の総損失
        errD = errD_real + errD_fake
        
        # Discriminator のパラメータを更新
        self.optimizerD.step()
        
        return errD, D_x, D_G_z1, fake
    
    def train_generator(self, fake):
        """
        Generator の更新
        log(D(G(z))) を最大化
        
        Args:
            fake: Generatorが生成した偽画像
            
        Returns:
            tuple: (errG, D_G_z2)
                - errG: Generatorの損失
                - D_G_z2: Generator更新後の判定確率
        """
        # Generator の勾配をリセット
        self.netG.zero_grad()
        
        # Generator の目的は「偽画像を本物だと判定させる」こと
        # そのためラベルを本物（1）に設定
        label = torch.full(
            (fake.size(0),),
            self.config.REAL_LABEL,
            dtype=torch.float,
            device=self.device
        )
        
        # 更新後の Discriminator で偽画像を再評価
        output = self.netD(fake).view(-1)
        
        # Generator の損失を計算
        errG = self.criterion(output, label)
        
        # 勾配を計算（逆伝播）
        errG.backward()
        
        # D(G(z))：Generator 更新後の判定確率
        D_G_z2 = output.mean().item()
        
        # Generator のパラメータを更新
        self.optimizerG.step()
        
        return errG, D_G_z2
    
    def train(self, dataloader, num_epochs):
        """
        学習ループ全体
        
        Args:
            dataloader: DataLoader
            num_epochs: エポック数
        """
        print("Starting Training Loop...")
        
        # 学習ステップ数のカウンタ
        iters = 0
        
        # エポック数分、学習を繰り返す
        for epoch in range(num_epochs):
            # DataLoaderからバッチ単位でデータを取得
            for i, data in enumerate(dataloader, 0):
                # バッチサイズを取得
                b_size = data[0].size(0)
                
                # (1) Discriminator の更新
                errD, D_x, D_G_z1, fake = self.train_discriminator(data[0], b_size)
                
                # (2) Generator の更新
                errG, D_G_z2 = self.train_generator(fake)
                
                # 学習状況の表示
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
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                
                # Generator の出力を定期的に保存（可視化用）
                if (iters % 500 == 0) or (
                    (epoch == num_epochs - 1) and
                    (i == len(dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    
                    # 生成画像をグリッド化して保存
                    self.img_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )
                
                # イテレーション数を更新
                iters += 1
        
        print("Training Complete!")
    
    def get_results(self):
        """
        学習結果を取得
        
        Returns:
            dict: 学習結果
        """
        return {
            'img_list': self.img_list,
            'G_losses': self.G_losses,
            'D_losses': self.D_losses
        }
