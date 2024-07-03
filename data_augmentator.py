import argparse
# データ拡張系
import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, ImageReadMode, write_jpeg

# データ拡張の設定
transforms = T.Compose([
    T.ToImage(),

    T.RandomRotation(degrees=30),
    T.RandomResizedCrop(size=(700, 700), antialias=True),
    T.RandomPerspective(),

    T.ToDtype(torch.uint8, scale=True)
    ])

class Augmentator():
    def __init__(self, config_path):
        self.config_path = config_path
        self.yolo = []

    def read_arg(self):
        parser = argparse.ArgumentParser(
                prog="sample",
                usage="python3 data_augmentator.py <images_path> <labels_path>",
                description="Specify the paths to the images and labels directories you wish to expand.",
                )

    def yolobox_to_xyxy(self, boxes, img_size=700):
        # 各座標をピクセル値に変換する
        px_list = []
        for value in boxes:
            bbox_px = list(map(lambda x: x*img_size, value))
            px_list.append(bbox_px)
        # xmin, ymin, xmax, ymaxに変換する
        xyxy_boxes = []
        for cxcywh_px in px_list:
            cx = cxcywh_px[0]
            cy = cxcywh_px[1]
            w = cxcywh_px[2]
            h = cxcywh_px[3]

            xmin = cx - (w/2)
            ymin = cy - (h/2)
            xmax = cx + (w/2)
            ymax = cy + (h/2)

            xyxy_boxes.append([xmin, ymin, xmax, ymax])

        return xyxy_boxes

    def xyxy_to_yolobox(self, boxes, img_size=700):
        # yoloのフォーマットに変換する
        yolo_boxes = []
        for xyxy in boxes:
            xmin, ymin, xmax, ymax = list(map(lambda value1: float(value1), xyxy))
            # CXCYWHに変換する
            w = xmax - xmin
            h = ymax - ymin
            cx = xmax - (w/2)
            cy = ymax - (h/2)
            cxcywh_box = [cx, cy, w, h]
            # 画像サイズで割る
            yolo_box = list(map(lambda value2: round(float(value2)/img_size, 6), cxcywh_box)) 
            yolo_boxes.append(yolo_box)

        return yolo_boxes


if __name__ == '__main__':
    a = Augmentator()
    tmp = a.xyxy_to_yolobox([[350.0, 200.0, 395.0, 233.0]])
    print(tmp)
