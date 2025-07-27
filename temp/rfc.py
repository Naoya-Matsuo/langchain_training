import torch
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np

device='mps'
# image_sizeやmean, stdはデータに合わせて設定してください。
image_size = 25

# trainデータとvalidationデータが入っているディレクトリのパスを指定
train_image_dir = './data/ramen_db-1.0.0/data/train'
val_image_dir = './data/ramen_db-1.0.0/data/valid'

# trainデータ向けとvalidationデータ向けに、transformを用意します。
# 皆さんのやりたいことに合わせて適宜変更してください。
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.5, 1.0)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=[-15, 15]),
        transforms.ToTensor(),
        transforms.RandomErasing(0.5),
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
}

# torchvision.datasets.ImageFolderでデータの入っているディレクトリのパスと
# transformを指定してあげるだけ。
train_dataset = torchvision.datasets.ImageFolder(root=train_image_dir, transform=data_transform['train'])
val_dataset = torchvision.datasets.ImageFolder(root=val_image_dir, transform=data_transform['val'])

# Datasetができたら、dataloaderに渡してあげればOK
batch_size = 32
train_dataLoader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataLoader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 4)

print(model)
model.to(device)
lr = 1e-4
epoch = 40
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().to(device) 

def train_loop(model , criterion , optimizer , num_epoch=epoch):
    best_val_accuracy = 0.0
    best_epoch = -1
    for epoch in range(num_epoch):

        loss_train = []
        acc_train = 0
        loss_val = []
        acc_val = 0

        # Train Loop
        model.train()
        for x,t in train_dataLoader:
            model.zero_grad()
            x = x.to(device)

            # t_hotは不要な場合が多い。CrossEntropyLossはone-hotではなくクラスIDを受け取る
            # t_hot = torch.eye(4)[t]
            # t_hot = t_hot.to(device) # t_hotを使うならこれもdeviceに移動

            # CrossEntropyLoss はターゲットとしてクラスID (long型) を期待します
            # t はDataLoaderからロードされるときに通常はCPUテンソルなので、それをdeviceに移動
            t = t.to(device) # <- ここでtもdeviceに移動させる

            y = model.forward(x)

            # 修正: criterion(出力, ターゲット)
            # ターゲット t はクラスID（one-hotではない）である必要があります
            loss = criterion(y, t) # <- t_hotではなくtを使う (tはdevice上にあること)
            loss.backward()
            optimizer.step()

            pred = y.argmax(1) # predはMPSデバイス上

            # 修正箇所: accuracy_scoreに渡す前にpredとtをCPUに移動
            acc_train += accuracy_score(pred.cpu(), t.cpu())
            loss_train.append(loss.item()) # .item()はテンソルからPythonの数値を取得する安全な方法

        # Validation Loop
        model.eval()
        with torch.no_grad(): # 推論時は勾配計算を無効化
            for x,t in val_dataLoader:
                x = x.to(device)
                t = t.to(device) # <- ここでもtをdeviceに移動

                # t_hotは不要な場合が多い
                # t_hot = torch.eye(4)[t]
                # t_hot = t_hot.to(device)

                y = model.forward(x)

                loss = criterion(y, t) # <- t_hotではなくtを使う

                pred = y.argmax(1) # predはMPSデバイス

                # 修正箇所: accuracy_scoreに渡す前にpredとtをCPUに移動
                acc_val += accuracy_score(pred.cpu(), t.cpu())
                loss_val.append(loss.item()) # .item()を使用

        current_val_accuracy = acc_val/len(val_dataLoader)
        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch,
            np.mean(loss_train),
            acc_train/len(train_dataLoader),
            np.mean(loss_val),
            current_val_accuracy
    ))
        
        # --- 最も精度の高かったモデルの保存 ---
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_epoch = epoch
            # モデルのstate_dictを保存
            torch.save(model.state_dict(), "/Users/maoya23/webtraining/temp/checkpoint/ramen.pth")
            print(f"--> 最高精度を更新しました！モデルを {'/Users/maoya23/webtraining/temp/checkpoint/ramen.pth'} に保存しました。 (精度: {best_val_accuracy:.3f}, エポック: {best_epoch})")


model_fit = train_loop(model,criterion,optim)



