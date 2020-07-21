import json
from util import prettyjson
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from p_tqdm import p_umap, p_map
from functools import partial


class KProductsDataset:
    """
    Reading KProduct Dataset on AI Hub
    """
    def __init__(self, conf_or_path, refresh_annot=False, refresh_multi_process=False):
        """
        Args:
            conf_or_path (dict, str): Dataset Configuration json dict or path
            refresh_annot (bool): Refresh Converted Annotation File
        """
        if type(conf_or_path) == dict:
            self.config = conf_or_path
        else:
            with open(conf_or_path, 'r') as f:
                self.config = json.load(conf_or_path)

        self.annotations, self.unique_labels = self.read_annotations(refresh=refresh_annot, multiprocess=refresh_multi_process)

    def get_annotation_path_list(self):
        annot_path_list = [(root, file_name)
                           for root, dirs, files in os.walk(self.config['dataset_root'])
                           if len(files) > 1
                           for file_name in files if file_name.endswith("json")]

        return annot_path_list

    def read_annotations(self, refresh=False, multiprocess=False):
        """
        Read Annotations and Convert into one csv file.
        If converted annotation file already exists, it directly reads from it unless refresh is set to True.

        Args:
            refresh (bool): True - Reload all annotations.

        Returns:

        """
        if os.path.isfile(self.config['annotation_path']) and refresh is False:
            annotations = pd.read_csv(self.config['annotation_path'])
        else:
            annot_path_list = self.get_annotation_path_list()
            if multiprocess:
                annotations = p_map(convert_annotation, annot_path_list, desc="Converting Annotations ...")
            else:
                annotations = [convert_annotation([root, file_name]) for root, file_name in tqdm(annot_path_list, desc="Converting Annotations ...")]

            annotations = [annotation for annotation in annotations if annotation is not None]
            annotations = pd.DataFrame(annotations)
            annotations.to_csv(self.config['annotation_path'], index=False)

        unique_labels = annotations[self.config['class_key']].unique()

        return annotations, unique_labels

    @staticmethod
    def resize_image(args, target_w=320, target_root="./export", skip_exists=True):
        """
        Resize Image and save to the target_root
        Args:
            args (list): [dataset_root, file_root, file_name].
                dataset_root (str): Dataset root from dataset configuration file.
                file_root (str): Image file root from dataset_root.
                file_name (str): Image file name
            target_w (int): Target width for resizing. Height is automatically set by ratio.
            target_root: Target dataset root to save resized images.
        """
        dataset_root, file_root, file_name = args

        target_file_root = f"{target_root}/{file_root}"
        os.makedirs(target_file_root, exist_ok=True)
        target_path = f"{target_file_root}/{file_name}"

        if skip_exists and os.path.isfile(target_path):
            return

        img_path = f"{dataset_root}/{file_root}/{file_name}"
        try:
            img = Image.open(img_path)
            target_h = int((target_w / img.size[0]) * img.size[1])
            img = img.resize((target_w, target_h))
            img.save(target_path)
        except FileNotFoundError:
            print("Open file failed on {} -> {}".format(img_path, target_path))

    def resize_dataset(self, target_w=320, target_root="./export", skip_exists=True, multiprocess=True):
        """
        Resize images from entires dataset.
        This functions uses multi-cores. Be aware that it will slow down your computer.

        Args:
            target_w (int): Target width for resizing. Height is automatically set by ratio.
            target_root: Target dataset root to save resized images.
        """
        mp_args = self.annotations[['file_root', 'file_name']].values.tolist()
        mp_args = [[self.config['dataset_root']] + arg for arg in mp_args]

        if multiprocess:
            p_umap(partial(KProductsDataset.resize_image, target_w=target_w, target_root=target_root, skip_exists=skip_exists),
                   mp_args, desc="Resizing Images ...")
        else:
            for arg in tqdm(mp_args, desc="Resizing Images ..."):
                KProductsDataset.resize_image(arg, target_w=target_w, target_root=target_root, skip_exists=skip_exists)

    def get_distribution(self, key='소분류'):
        """
        Get distributions of dataset by key
        Args:
            key (str): key value to get distribution

        Returns:
            (list): Length of each unique value from 'key'

        """
        return {u_id: self.annotations.query(f"{key} == '{u_id}'").shape[0]
                for u_id in self.annotations[key].unique()}

    def get_data(self, idx):
        """
        Get image and annotation

        Args:
            idx (int): Index number of data.

        Returns:
            (PIL.Image): Image
            (pd.DataFrame): Annotation
        """
        target = self.annotations.iloc[idx]

        img_path = f"{self.config['dataset_root']}/{target['file_root']}/{target['file_name']}"
        img = Image.open(img_path)

        return img, target

    def plot_class_images(self):
        """
        Plot Every class images by randomly choosing within the class
        """
        n_label = len(self.unique_labels)
        subplot_w = np.ceil(np.sqrt(n_label)).astype(np.int32)
        subplot_h = subplot_w-1 if subplot_w*(subplot_w-1) > n_label else subplot_w

        fontprop = fm.FontProperties(fname="./res/NanumSquareRoundR.ttf")

        plt.figure(figsize=(15, 15))
        for i, label in enumerate(self.unique_labels):
            class_index = self.annotations.query("{} == '{}'".format(self.config['class_key'], label)).index.values
            np.random.shuffle(class_index)
            img, annot = self.get_data(class_index[0])

            plt.subplot(subplot_w, subplot_h, i+1)
            plt.imshow(img)
            plt.title(f"{label} :: {annot['file_name']}", fontproperties=fontprop)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def convert_annotation(args):
    """
    Convert Original annotation json file to python dict type

    Args:
        args (list, tuple): Contains two arguments. (root, annot_or_filename)
            root (str): Dataset root directory
            annot_or_filename (dict, str): Annotation dict or path
    Returns:
        (dict): Converted annotation dict type

    """
    root, annot_or_filename = args

    if type(annot_or_filename) == dict:
        annot = annot_or_filename
    else:
        try:
            with open(f"{root}/{annot_or_filename}") as fp:
                annot = json.load(fp)
        except json.decoder.JSONDecodeError:
            print("Something went wrong on {}/{}!!".format(root, annot_or_filename))
            return None

    if len(annot['regions']) > 1:
        print(f"!!! More than Two Annotation Found ({len(annot['regions']):02d}) at {root}/{annot['image']['identifier']}")

    roots = root.split("/")
    file_root = "/".join(roots[-2:])
    new_annot = dict()
    new_annot['file_name'] = annot['image']['identifier']
    new_annot['file_root'] = file_root
    new_annot['img_width'] = annot['image']['imsize'][0]
    new_annot['img_height'] = annot['image']['imsize'][1]

    region_annot = annot['regions'][0]
    new_annot['bbox_x1'] = region_annot['boxcorners'][0]
    new_annot['bbox_y1'] = region_annot['boxcorners'][1]
    new_annot['bbox_x2'] = region_annot['boxcorners'][2]
    new_annot['bbox_y2'] = region_annot['boxcorners'][3]
    new_annot['obj_name'] = region_annot['class']

    for tag in region_annot['tags']:
        tag_name, tag_value = tag.split(":")
        if tag_value == '':
            continue
        new_annot[tag_name] = tag_value

    return new_annot
