import json
import os

import pandas as pd
from PIL import Image

import re
import unicodedata
from dataset import KProductsDataset
import matplotlib.pyplot as plt


class KProductsExtra:
    def __init__(self, dataset, root, seed=7777):
        self.dataset = dataset
        self.root = root
        self.seed = seed

    def merge_annotation(self, target_path, multiply=2, beta=2, plot_result=False):
        paths = self.get_paths()
        annotations = [self.convert_annot(file_root, file_name, label)
                       for file_root, file_name, label in paths]
        annotations = [annot for annot in annotations if annot is not None]

        annotations = pd.DataFrame(annotations)

        annot_by_class = [annotations.query("{} == '{}'".format(self.dataset.config['class_key'], label))
                          for label in self.dataset.unique_labels]

        max_sample = max([annot.shape[0] for annot in annot_by_class])
        annotations = [annot.sample(n=min(annot.shape[0]*multiply, max_sample*beta), replace=True, random_state=self.seed)
                       for annot in annot_by_class]
        annotations = pd.concat(annotations)
        merged_annotations = pd.concat([self.dataset.annotations, annotations])
        merged_annotations.reset_index(drop=True)

        merged_annotations.to_csv(target_path, index=False)

        if plot_result:
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            ori_annot = dataset.annotations
            dataset.plot_class_distributions(title_prefix="Original ", ax=axes[0])
            dataset.annotations = annotations
            dataset.plot_class_distributions(title_prefix="Extra Only ", ax=axes[1])
            dataset.annotations = merged_annotations
            dataset.plot_class_distributions(title_prefix="Merged ", ax=axes[2])
            dataset.annotations = ori_annot

            fig.suptitle(f"Multiply extra data by: {multiply}, Beta: {beta}")
            fig.tight_layout()
            fig.show()

        return merged_annotations, annotations

    def get_paths(self):
        _, dirs, _ = next(os.walk(self.root))
        labels = self.dataset.config['label_dict'].values()
        paths = []

        for dir in dirs:
            label = unicodedata.normalize('NFC', re.sub("[0-9]+_", "", dir))
            if label in labels:
                _, _, files = next(os.walk(f"{self.root}/{dir}"))
                for file in files:
                    paths.append((dir, file, label))

        return paths

    def convert_annot(self, file_root, file_name, label):
        new_annot = dict()
        try:
            img = Image.open(f"{self.root}/{file_root}/{file_name}")
        except:
            print("Open Image failed on {}/{}/{}!!".format(self.root, file_root, file_name))
            return None

        new_annot['root'] = self.root
        new_annot['file_name'] = file_name
        new_annot['file_root'] = file_root
        new_annot['img_width'] = img.width
        new_annot['img_height'] = img.height

        new_annot['bbox_x1'] = -1
        new_annot['bbox_y1'] = -1
        new_annot['bbox_x2'] = -1
        new_annot['bbox_y2'] = -1
        new_annot['class_name'] = label

        new_annot['truncated'] = 0
        new_annot['종ID'] = label
        new_annot['대분류'] = label
        new_annot['중분류'] = label
        new_annot['소분류'] = label
        new_annot['Instance'] = label

        return new_annot


if __name__ == "__main__":
    root = "/Users/jeikei/Dropbox/사업자료/2020/2020_0515_인공지능_그랜드_챌린지/Projects/extra_data_crawling"

    with open("/Users/jeikei/Dropbox/사업자료/2020/2020_0515_인공지능_그랜드_챌린지/Projects/aihub_dataset_process/conf.json", 'r', encoding='UTF8') as f:
        config = json.load(f)

    dataset = KProductsDataset(config, skip_save_config=False)
    extra = KProductsExtra(dataset, root)

    m_annot, annot = extra.merge_annotation("test.csv", multiply=5, beta=2, plot_result=True)
    ori_annot = dataset.annotations

