# data_augmentation_torch

## Installation


## Usage
### Step1. 各種設定
パス、GPU、データ拡張の設定を`data_augmentation_torch/main.py`に記述してください。 </br>
データ拡張の設定はPytorchを使用できます。</br>
拡張機能についてはこちら 👉 https://pytorch.org/vision/main/transforms.html#v2-api-reference-recommended
<details>
<summary>設定の例</summary>
  
  ```py
  # 保存するファイル名
  SAVE_NAME = "laser_img_aug"
  # 読み込むディレクトリまでのパス
  IMAGES_PATH = "/home/demulab/follow_me_dataset_origin/train_val/images"
  LABELS_PATH = "/home/demulab/follow_me_dataset_origin/train_val/labels"
  # GPUの設定
  DEVICE = "cuda:0"
  # 何枚拡張するか
  AUGMENTATION_NUM = 125000 - 11923
  # 拡張後のデータを確認するか
  DATA_CHECK = True
  # データ拡張の設定
  DATA_AUGMENTATION_TRANSFORMS = [
          T.Compose([
              T.ToImage(),

              # 切り取って指定されたサイズに変更する
              #T.RandomResizedCrop(size=(700, 700), antialias=True),
              # 水平に反転
              T.RandomHorizontalFlip(p=0.5),
              # 鮮鋭化
              T.RandomAdjustSharpness(sharpness_factor=0 ,p=0.2),
              T.RandomAdjustSharpness(sharpness_factor=3, p=0.2),
              T.RandomAdjustSharpness(sharpness_factor=5, p=0.2),
              # アフィン変換
              T.RandomAffine(degrees=[-10, 10], translate=(0.2, 0.2), scale=(0.7, 1.5)),

              T.ToDtype(torch.uint8, scale=True)
              ]),

          T.Compose([
              T.ToImage(),

              # 射影変換(pは確率)
              T.RandomPerspective(p=0.3),
              # 鮮鋭化
              T.RandomAdjustSharpness(sharpness_factor=0 ,p=0.2),
              T.RandomAdjustSharpness(sharpness_factor=3, p=0.2),
              T.RandomAdjustSharpness(sharpness_factor=5, p=0.2),
              # 水平に反転
              T.RandomHorizontalFlip(p=0.5),
            
              T.ToDtype(torch.uint8, scale=True)
              ]),

          T.Compose([
              T.ToImage(),

              # 回転
              T.RandomRotation(degrees=20),
              # アフィン変換
              T.RandomAffine(degrees=[-10, 10], translate=(0.2, 0.2), scale=(0.7, 1.5)),
              # 水平に反転
              T.RandomHorizontalFlip(p=0.5),

              T.ToDtype(torch.uint8, scale=True)
              ]),
          ]
  ```

</details>

### Step2. 実行
設定が完了したら実行します。

```bash
python3 ~/data_augmentation_torch/main.py
```


## 試作時の各バージョン
python3
```sh
Python 3.10.11
```
Pytorch
```sh
torch       2.3.1+cu118
torchvision 0.18.1+cu118
```

## TODO
* ~~フォルダを指定できるようにする~~
* ~~フォルダ内のデータをすべて変換できるようにする~~
* ~~変換したデータをファイルに出力する~~
* ~~ファイルをフォルダに保存できるようにする~~
* ~~ファイル名を指定できるようにする~~
* ~~フォルダの有無を確認して自動でフォルダ作成する~~
* ~~変換後のbboxの座標が0.0の場合はファイルを出力しない~~
* ~~進捗状況を出力するようにする~~
* ~~データ拡張の合計変換数を終了時に出力する~~
* ~~プログラムとしてまとめる（可読性を上げる）~~
* ~~cudaを使用するようにする~~
* 未来の自分のためにREADMEを作成する（特にpython3とtorchのバージョン）
* READMEにUsage書く
* クラスの機能をREADMEに書く
* メソッドの機能をREADMEに書く
