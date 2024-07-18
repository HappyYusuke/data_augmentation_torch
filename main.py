from data_augmentator import *


# 保存するファイル名
SAVE_NAME = "laser_img_aug"
# 読み込むディレクトリまでのパス
IMAGES_PATH = "/home/user/path_to/images"
LABELS_PATH = "/home/user/path_to/labels"
# GPUの設定
DEVICE = "cuda:0"
# 何枚拡張するか
AUGMENTATION_NUM = 1000
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


if __name__ == '__main__':
    a = Augmentator(
            transforms=DATA_AUGMENTATION_TRANSFORMS,
            save_name=SAVE_NAME,
            images_path=IMAGES_PATH,
            labels_path=LABELS_PATH,
            augmentation_num=AUGMENTATION_NUM,
            device=DEVICE,
            data_check=DATA_CHECK
            )
    try:
        a.run()
    except KeyboardInterrupt:
        a.interrupted()
