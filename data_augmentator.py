import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from torchvision.io import read_image, ImageReadMode


# 　データ拡張の設定
transforms = T.Compose([
    T.ToImage(),

    T.RandomRotation(degrees=30),
    T.RandomResizedCrop(size=(600, 600), antialias=True),
    T.RandomPerspective(),

    T.ToDtype(torch.float32, scale=True)
])


# 画像の読み込み
img = read_image('images/laser_img_0.jpg', ImageReadMode.RGB)


# bboxの座標の読み込み
#yoloのlabelフォーマットについて：https://qiita.com/yarakigit/items/4d4044bc2740cecba92a
yolo_format = []

with open('labels/laser_img_0.txt') as f:

    # 情報をすべて少数にして子リストに格納する
    for bbox in f.read().split('\n'):
        yolo_format.append([float(i) for i in bbox.split(' ') if i != ''])

    # for文が一回多いので空のリストを削除
    for index in range(len(yolo_format)):
        if not yolo_format[index]:
            del yolo_format[index]

    # 0番目はクラス番号なので削除
    for index in range(len(yolo_format)):
        del yolo_format[index][0]

    # 各座標をピクセル値に変換する
    px_list = []
    for child in yolo_format:
        bbox_px = list(map(lambda x: x*700, child))
        px_list.append(bbox_px)

    # torchvitionのxmin, ymin, xmax, ymaxに合わせる
    boxes_list = []
    for child in px_list:
        cx = child[0]
        cy = child[1]
        w = child[2]
        h = child[3]

        xmin = cx - (w/2)
        ymin = cy - (h/2)
        xmax = cx + (w/2)
        ymax = cy + (h/2)
        
        boxes_list.append([xmin, ymin, xmax, ymax])

print(yolo_format)  # 確認用
print(boxes_list)   # 使うのはこっち


# bboxの座標をtorchで使用できるように変換
boxes = tv_tensors.BoundingBoxes(
    boxes_list,
    format=tv_tensors.BoundingBoxFormat.XYXY,
    canvas_size=img.shape[-2:]
)


# 変換
img_ts, boxes_ts = transforms(img, boxes)