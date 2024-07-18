# data_augmentation_torch
<div align="center">
  <img src="https://github.com/user-attachments/assets/ce8f4685-fdbe-46f4-a722-ace41159b836" width="700">
</div>

## Description
data_augmentation_torchは、バウンディングボックスを考慮したデータ拡張機能を提供するリポジトリです。本リポジトリに含まれる機能は以下の通りです。
* バウンディングボックスを考慮したデータ拡張
* 画像、ラベル、拡張後の画像の保存
* 拡張枚数の指定

> [!WARNING]
> バウンディングボックスのフォーマットはYOLOのみ対応しています。

</br>

## Requirement
開発時の各バージョンは以下の通りですが、変更してもかまいません。</br>
ただし、`torchvision`==0.15系での動作は確認できませんでした。

| 項目 | バージョン |
| --- | --- |
| CUDA | 11.8 |
| Ubuntu | 22.04 |
| Python | 3.10.11 |
| torch | 2.3.1 |
| torchvision | 0.18.1 |
| tqdm | 4.66.4 |

</br>

## Installation

Python3系とCUDAのインストールは完了している前提です。

本リポジトリをクローンします。

```bash
git clone https://github.com/HappyYusuke/data_augmentation_torch.git
```

必要なパッケージをインストールします。

```bash
pip install -r ~/data_augmentation_torch/requirements.txt
```

> [!TIP]
> <details><summary> 仮想環境下での実行を推奨します。</summary>
>  
> pipenvのインストール
> ```bash
> pip install pipenv
> ```
>  
> ディレクトリの作成
> ```bash
> mkdir ~/Project1
> cd ~/Project1
> ```
>  
> 仮想環境を生成する
> ```bash
> pipenv
> ```
>  
> 仮想環境の中に入る
> ```bash
> pipenv shell
> ```
> </details>

</br>

## Usage
### Step1. 各種設定

パス、GPU、データ拡張の設定を`data_augmentation_torch/main.py`に記述してください。 </br>
データ拡張の設定はPytorchを使用できます。</br>
拡張機能についてはこちら 👉 https://pytorch.org/vision/main/transforms.html#v2-api-reference-recommended

> [!NOTE]
> <details><summary>設定の例</summary>
>  
>  ```py
>  # 保存するファイル名
>  SAVE_NAME = "laser_img_aug"
>  # 読み込むディレクトリまでのパス
>  IMAGES_PATH = "/home/demulab/follow_me_dataset_origin/train_val/images"
>  LABELS_PATH = "/home/demulab/follow_me_dataset_origin/train_val/labels"
>  # GPUの設定
>  DEVICE = "cuda:0"
>  # 何枚拡張するか
>  AUGMENTATION_NUM = 125000 - 11923
>  # 拡張後のデータを確認するか
>  DATA_CHECK = True
>  # データ拡張の設定
>  DATA_AUGMENTATION_TRANSFORMS = [
>          T.Compose([
>              T.ToImage(),
>
>              # 切り取って指定されたサイズに変更する
>              #T.RandomResizedCrop(size=(700, 700), antialias=True),
>              # 水平に反転
>              T.RandomHorizontalFlip(p=0.5),
>              # 鮮鋭化
>              T.RandomAdjustSharpness(sharpness_factor=0 ,p=0.2),
>              T.RandomAdjustSharpness(sharpness_factor=3, p=0.2),
>              T.RandomAdjustSharpness(sharpness_factor=5, p=0.2),
>              # アフィン変換
>              T.RandomAffine(degrees=[-10, 10], translate=(0.2, 0.2), scale=(0.7, 1.5)),
>
>              T.ToDtype(torch.uint8, scale=True)
>              ]),
>
>          T.Compose([
>              T.ToImage(),
>
>              # 射影変換(pは確率)
>              T.RandomPerspective(p=0.3),
>              # 鮮鋭化
>              T.RandomAdjustSharpness(sharpness_factor=0 ,p=0.2),
>              T.RandomAdjustSharpness(sharpness_factor=3, p=0.2),
>              T.RandomAdjustSharpness(sharpness_factor=5, p=0.2),
>              # 水平に反転
>              T.RandomHorizontalFlip(p=0.5),
>            
>              T.ToDtype(torch.uint8, scale=True)
>              ]),
>
>          T.Compose([
>              T.ToImage(),
>
>              # 回転
>              T.RandomRotation(degrees=20),
>              # アフィン変換
>              T.RandomAffine(degrees=[-10, 10], translate=(0.2, 0.2), scale=(0.7, 1.5)),
>              # 水平に反転
>              T.RandomHorizontalFlip(p=0.5),
>
>              T.ToDtype(torch.uint8, scale=True)
>              ]),
>          ]
>  ```
>
> </details>

### Step2. 実行
設定が完了したら実行します。

```bash
python3 ~/data_augmentation_torch/main.py
```

### Step3. 結果の確認

実行した結果は以下のパスに保存されます。</br>
bbox_imagesの画像は確認用にご使用ください。

<pre>
data_augmentation_torch
└── results
     ├── bbox_images  # 拡張後のバウンディングボックスを描画した画像が保存されます
     ├── images       # 拡張後の画像データのみが保存されます
     ├── labels       # 拡張後のラベルデータのみが保存されます
</pre>


