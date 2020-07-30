import json
import os
from sys import platform

from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from p_tqdm import p_umap, p_map
from functools import partial
import shutil
from util import prettyjson
import platform


class KProductsDataset:
    """
    Reading KProduct Dataset on AI Hub
    """

    path_key_list = [
        "dataset_root",
        "annotation_path",
        "train_annotation",
        "test_annotation",
        "self_path"
    ]

    def __init__(self, conf_or_path, refresh_annot=False,
                 refresh_multi_process=False, seed=7777, encoding='UTF8', skip_save_config=False):
        """
        Args:
            conf_or_path (dict, str): Dataset Configuration json dict or path
            refresh_annot (bool): Refresh Converted Annotation File
            encoding (str): Config Encoding type
        """
        self.encoding = encoding

        if type(conf_or_path) == dict:
            self.config = conf_or_path
        else:
            with open(conf_or_path, 'r', encoding=self.encoding) as f:
                self.config = json.load(f)
            self.config['self_path'] = conf_or_path

        for fix_key in KProductsDataset.path_key_list:
            if fix_key in self.config.keys():
                self.config[fix_key] = os.path.abspath(self.config[fix_key])

                if platform.system().find("Windows") >= 0:
                    self.config[fix_key] = self.config[fix_key].replace("\\", "\\\\")

        np.random.seed(seed)

        self.annotations, self.unique_labels = self.read_annotations(refresh=refresh_annot, multiprocess=refresh_multi_process)
        if "label_dict" not in self.config.keys():
            self.config['label_dict'] = {str(i): label for i, label in enumerate(self.unique_labels)}
        else:
            self.config['label_dict'] = {i: label for i, label in self.config['label_dict'].items()}

        if not skip_save_config:
            with open(self.config['self_path'], 'w', encoding=self.encoding) as f:
                f.write(prettyjson(self.config))

        self.n_classes = len(self.config['label_dict'])
        self.seed = seed

        # For Korean Label
        font_list = [font.name for font in fm.fontManager.ttflist]
        possible_fonts = [font_name for font_name in font_list if font_name.find("CJK") >=0]
        if len(possible_fonts) > 0:
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = possible_fonts[0]

    def split_train_test(self, train_ratio=0.7, balance_type='min', alpha=0.3, beta=1.5, plot_distribution=False, skip_n=2):
        """

        Args:
            train_ratio (float): train ratio respect to dataset image number
            balance_type (str): ('min', 'over', 'over2', 'none')
                                min: Limit class image numbers by minimum sample class.
                                over: Reduce class image numbers by min((number of class images) * alpha, (minimum sample class)*beta)
            alpha:

        Returns:

        """
        class_distribution = self.get_distribution(key=self.config['class_key'])
        median_sample = np.median(list(class_distribution.values()))

        original_annotations = self.annotations

        if skip_n > 1:
            annot_by_class = [self.annotations.query("{} == '{}'".format(self.config['class_key'], label))
                              for label in self.unique_labels]
            annot_by_class = [annot.iloc[range(0, annot.shape[0], skip_n if annot.shape[0] > median_sample else 1)]
                               for annot in annot_by_class]
            skip_annotation = pd.concat(annot_by_class)
        else:
            skip_annotation = original_annotations

        self.annotations = skip_annotation
        class_distribution = self.get_distribution(key=self.config['class_key'])

        min_sample = min(class_distribution.values())

        if balance_type == 'min':
            annotations1 = [self.annotations.query("{} == '{}'".format(self.config['class_key'], label)).sample(n=min_sample, random_state=self.seed)
                           for label in self.unique_labels]

            annotations = pd.concat(annotations1)
            annotations = annotations.sample(n=annotations.shape[0], random_state=self.seed).reset_index(drop=True)
        ## jason
        elif balance_type == 'over':
            sample_cnt = {}
            for i in self.config['label_dict'].values():
                sample_cnt[i] = int(min(max(class_distribution[i]*alpha, min_sample*beta), class_distribution[i]))

            annotations1 = [self.annotations.query("{} == '{}'".format(self.config['class_key'], label)).sample(n=sample_cnt[label], random_state=self.seed)
                           for label in self.unique_labels]

            annotations = pd.concat(annotations1)
            annotations = annotations.sample(n=annotations.shape[0], random_state=self.seed).reset_index(drop=True)
        elif balance_type == 'over2':
            limit_sample = lambda x: max(x*alpha, min_sample * beta)
            # limit_sample = lambda x: min(max(x, max_limit)*alpha, max(min_sample*beta, x))
            target_size = {key: limit_sample(v) for key, v in class_distribution.items()}
            annotations1 = [self.annotations.query("{} == '{}'".format(self.config['class_key'], label)).sample(n=int(target_size[label]),
                                                                                                                random_state=self.seed,
                                                                                                                replace=True)
                            for label in self.unique_labels]
            annotations = pd.concat(annotations1)
            annotations = annotations.sample(n=annotations.shape[0], random_state=self.seed).reset_index(drop=True)
        else:
            annotations = self.annotations.sample(n=self.annotations.shape[0], random_state=self.seed).reset_index(drop=True)

        self.annotations = original_annotations
        n_train = int(annotations.shape[0]*train_ratio)

        train_annotation = annotations.iloc[:n_train]
        test_annotation = annotations.iloc[n_train:]

        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        root = self.config['annotation_path'].split(seperator)
        root, file_name = seperator.join(root[:-1]), root[-1]
        ext_idx = file_name.rfind('.')
        file_name = file_name[:ext_idx] if ext_idx > 0 else file_name

        self.config['train_annotation'] = os.path.abspath(f"{root}{seperator}{file_name}_train.csv").replace("\\", "\\\\")
        self.config['test_annotation'] = os.path.abspath(f"{root}{seperator}{file_name}_test.csv").replace("\\", "\\\\")

        train_annotation.to_csv(self.config['train_annotation'], index=False)
        test_annotation.to_csv(self.config['test_annotation'], index=False)

        with open(self.config['self_path'], 'w', encoding=self.encoding) as f:
            f.write(prettyjson(self.config))

        print("Train Annotation({:,}) saved to {}".format(train_annotation.shape[0], self.config['train_annotation']))
        print("Test Annotation({:,}) saved to {}".format(test_annotation.shape[0], self.config['test_annotation']))

        if plot_distribution:
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))

            annotations = self.annotations
            self.plot_class_distributions(title_prefix="Entire ", ax=axes[0])
            self.annotations = train_annotation
            self.plot_class_distributions(title_prefix="Train ", ax=axes[1])
            self.annotations = test_annotation
            self.plot_class_distributions(title_prefix="Test ", ax=axes[2])
            self.annotations = annotations

            fig.suptitle(f"Balance: {balance_type}, train_ratio: {train_ratio}, alpha: {alpha}, beta: {beta}, Split from {annotations.shape[0]:,} to ({train_annotation.shape[0]:,} / {test_annotation.shape[0]:,})")
            fig.tight_layout()
            fig.show()

    def get_annotation_path_list(self, multiprocess=False, sort=True):
        annot_path_list = [(root, file_name)
                           for root, dirs, files in tqdm(os.walk(self.config['dataset_root']), desc="Searching annotation .json files ...")
                           if len(files) > 1
                           for file_name in files if file_name.endswith("json")]
        print("Annotation list: {}".format(len(annot_path_list)))

        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        file_checker = lambda x: x if os.path.isfile(f"{x[0]}{seperator}{x[1]}") and \
                                      (os.path.isfile(f"{x[0]}{seperator}{x[1][:-5]}.JPG") or
                                       os.path.isfile(f"{x[0]}{seperator}{x[1][:-5]}.jpg")) else None
        if multiprocess:
            annot_path_list = p_map(file_checker, annot_path_list, desc="Double-Check File List ...")
            annot_path_list = [x for x in annot_path_list if x is not None]
        else:
            annot_path_list = [(root, file_name) for root, file_name in tqdm(annot_path_list, desc="Double-Check File List ...")
                               if os.path.isfile(f"{root}{seperator}{file_name}") and
                               (os.path.isfile(f"{root}{seperator}{file_name[:-5]}.JPG") or
                                os.path.isfile(f"{root}{seperator}{file_name[:-5]}.jpg"))
                               ]

        if sort:
            annot_path_list.sort()
        print("Annotation list: {}".format(len(annot_path_list)))
        return annot_path_list

    def read_annotations(self, refresh=False, multiprocess=False):
        """
        Read Annotations and Convert into one csv file.
        If converted annotation file already exists, it directly reads from it unless refresh is set to True.

        Args:
            refresh (bool): True - Reload all annotations.

        Returns:

        """
        convert_annot_func = partial(convert_annotation, dataset_root=self.config['dataset_root'], encoding=self.encoding)
        if os.path.isfile(self.config['annotation_path']) and refresh is False:
            annotations = pd.read_csv(self.config['annotation_path'])
        else:
            annot_path_list = self.get_annotation_path_list(multiprocess=multiprocess)
            if multiprocess:
                annotations = p_map(convert_annot_func, annot_path_list, desc="Converting Annotations ...")
            else:
                annotations = [convert_annot_func([root, file_name]) for root, file_name in tqdm(annot_path_list, desc="Converting Annotations ...")]

            annotations = [annotation for annotation in annotations if annotation is not None]
            annotations = pd.DataFrame(annotations)
            annotations.to_csv(self.config['annotation_path'], index=False)

        unique_labels = annotations[self.config['class_key']].unique()

        return annotations, unique_labels

    @staticmethod
    def resize_image(args, target_w=320, target_root="./export", skip_exists=True, copy_annotation=True):
        """
        Resize Image and save to the target_root
        Args:
            args (list): [dataset_root, file_root, file_name].
                dataset_root (str): Dataset root from dataset configuration file.
                file_root (str): Image file root from dataset_root.
                file_name (str): Image file name
            target_w (int): Target width for resizing. Height is automatically set by ratio.
            target_root (str): Target dataset root to save resized images.
            skip_exists (bool): Skip if the image already exists
            copy_annotation (bool): Copy annotation
        """
        dataset_root, file_root, file_name = args

        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"
        target_root = target_root.replace("/", "\\") if seperator == "\\" else target_root

        target_file_root = f"{target_root}{seperator}{file_root}"
        os.makedirs(target_file_root, exist_ok=True)
        target_path = f"{target_file_root}{seperator}{file_name}"

        if copy_annotation:
            annot_path = f"{dataset_root}{seperator}{file_root}{seperator}{file_name[:-4]}.json"
            target_annot_path = f"{target_path[:-4]}.json"
            try:
                shutil.copy2(annot_path, target_annot_path)
            except:
                print("Copy failed from {} to {}".format(annot_path, target_annot_path))

        if skip_exists and os.path.isfile(target_path):
            return

        img_path = f"{dataset_root}{seperator}{file_root}{seperator}{file_name}"
        try:
            img = Image.open(img_path)
            target_h = int((target_w / img.size[0]) * img.size[1])
            img = img.resize((target_w, target_h))
            img.save(target_path)
        except FileNotFoundError:
            print("Open file failed on {} -> {}".format(img_path, target_path))

    def resize_dataset(self, target_w=320, target_root="./export", skip_exists=True, multiprocess=True, num_cpus=1.0, copy_annotation=True):
        """
        Resize images from entires dataset.
        This functions uses multi-cores. Be aware that it will slow down your computer.

        Args:
            target_w (int): Target width for resizing. Height is automatically set by ratio.
            target_root (str): Target dataset root to save resized images.
            skip_exists (bool): True: Skip resizing if resized file already exists.
            multiprocess (bool): Use multi process.
            num_cpus (int, float): Number(int) or proportion(float) of cpus to utilize in multiprocess.
        """

        target_root = target_root.replace("/", "\\") if platform.system().find("Windows") >= 0 else target_root

        mp_args = self.annotations[['root', 'file_root', 'file_name']].values.tolist()

        if multiprocess:
            p_umap(partial(KProductsDataset.resize_image, target_w=target_w, target_root=target_root, skip_exists=skip_exists, copy_annotation=copy_annotation),
                   mp_args, desc="Resizing Images ...", num_cpus=num_cpus)
        else:
            for arg in tqdm(mp_args, desc="Resizing Images ..."):
                KProductsDataset.resize_image(arg, target_w=target_w, target_root=target_root, skip_exists=skip_exists, copy_annotation=copy_annotation)

    @staticmethod
    def copy_image(args, target_root="./export", reverse_dict=None):
        assert reverse_dict is not None

        dataset_root, file_root, file_name, class_name = args
        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        src = f"{dataset_root}{seperator}{file_root}{seperator}{file_name}"
        dest = f"{target_root}{seperator}{reverse_dict[class_name]:02d}_{class_name}"
        os.makedirs(dest, exist_ok=True)
        dest = f"{dest}{seperator}{file_name}"

        shutil.copy2(src, dest)

    def rebuild_dataset_by_dir(self, annotation=None, target_root="./export", multiprocess=True, num_cpus=1.0):
        target_root = target_root.replace("/", "\\") if platform.system().find("Windows") >= 0 else target_root

        if annotation is None:
            annotation = self.annotations

        mp_args = annotation[['file_root', 'file_name', "class_name"]].values.tolist()
        mp_args = [[self.config['dataset_root']] + arg for arg in mp_args]

        reverse_dict = {v: int(k) for k, v in self.config['label_dict'].items()}

        if multiprocess:
            p_umap(partial(KProductsDataset.copy_image, target_root=target_root, reverse_dict=reverse_dict),
                   mp_args, desc="Rebuilding Dataset by directory ...", num_cpus=num_cpus)
        else:
            for arg in tqdm(mp_args, desc="Rebuilding Dataset by directory ..."):
                KProductsDataset.copy_image(arg, target_root=target_root, reverse_dict=reverse_dict)

    def get_distribution(self, key='class'):
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
        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        target = self.annotations.iloc[idx]

        img_path = f"{target['root']}{seperator}{target['file_root']}{seperator}{target['file_name']}"
        try:
            img = Image.open(img_path)
        except:
            print("Error Opening Image file {}".format(img_path))
            return None, target

        return img, target

    def plot_all_class_images(self, title="", figsize=(30, 30), save_path=""):
        """
        Plot Every class images by randomly choosing within the class
        """
        n_label = len(self.unique_labels)
        subplot_w = np.ceil(np.sqrt(n_label)).astype(np.int32)
        subplot_h = subplot_w-1 if subplot_w*(subplot_w-1) > n_label else subplot_w

        plt.figure(figsize=figsize)
        for i, label in self.config['label_dict'].items():
            i = int(i)
            class_index = self.annotations.query("{} == '{}'".format(self.config['class_key'], label)).index.values
            np.random.shuffle(class_index)
            img, annot = self.get_data(class_index[0])

            plt.subplot(subplot_w, subplot_h, i+1)
            plt.imshow(img)
            plt.title(f"{i:02d}: {label} :: {annot['file_name']}")
            plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()

        if save_path == "":
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def plot_class_images(self, class_key_id, title="", figsize=(30, 30), save_path=""):

        reverse_label = {value: int(key) for key, value in self.config['label_dict'].items()}

        class_name = self.config['label_dict'][str(class_key_id)] if type(class_key_id) == int else class_key_id
        class_id = reverse_label[class_name]

        class_annot = self.annotations.query("{} == '{}'".format(
            self.config['class_key'], class_name
        ))

        unique_obj_id = class_annot['종ID'].unique()

        class_annot_by_obj_id = [class_annot.query("종ID == '{}'".format(unique_obj_id[i])) for i in range(len(unique_obj_id))]
        class_index_by_obj_id = [annot.index.values for annot in class_annot_by_obj_id]

        n_plot = len(class_annot_by_obj_id)

        subplot_w = np.ceil(np.sqrt(n_plot)).astype(np.int32)
        subplot_h = subplot_w - 1 if subplot_w * (subplot_w - 1) > n_plot else subplot_w
        plt.figure(figsize=figsize)

        for i, obj_index in enumerate(class_index_by_obj_id):
            np.random.shuffle(obj_index)

            img, annot = self.get_data(obj_index[0])

            plt.subplot(subplot_w, subplot_h, i+1)
            plt.imshow(img)
            plt.title(f"{i:02d} - {class_id:02d}:{class_name} :: {annot['file_root'].split('/')[0]}/{annot['file_name']} - {len(obj_index):,}")
            plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()

        if save_path == "":
            plt.show()
        else:
            plt.savefig(save_path)

    def plot_class_distributions(self, title_prefix="", figsize=(12, 8), save_path="", ax=None):
        """
        Plot class data number distribution
        """

        n_label = len(self.unique_labels)

        dists = [self.annotations.query("{} == '{}'".format(self.config['class_key'], self.config['label_dict'][str(i)])).shape[0]
                 for i in range(n_label)]
        class_names = list(self.config['label_dict'].values())
        class_names = [f"{i:02d}: {name}" for i, name in enumerate(class_names)]

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.barh(class_names, dists)
        for i, v in enumerate(dists):
            ax.text(v+0.05, i-.25, f"{v:,}", color='k', fontweight='bold')

        ax.set_title(f"{title_prefix}Class Distribution")
        if fig is not None:
            fig.tight_layout()

        if fig is not None:
            if save_path == "":
                plt.show()
            else:
                fig.savefig(save_path)
                plt.close(fig)

        return fig, ax


def convert_annotation(args, dataset_root="", encoding='UTF8'):
    """
    Convert Original annotation json file to python dict type

    Args:
        args (list, tuple): Contains two arguments. (root, annot_or_filename)
            root (str): Dataset root directory
            annot_or_filename (dict, str): Annotation dict or path
        encoding (str): Annotation Encoding Type
    Returns:
        (dict): Converted annotation dict type

    """
    root, annot_or_filename = args

    seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

    if type(annot_or_filename) == dict:
        annot = annot_or_filename
    else:
        try:
            with open(f"{root}{seperator}{annot_or_filename}", encoding=encoding) as fp:
                annot = json.load(fp)
        except json.decoder.JSONDecodeError:
            print("Something went wrong on {}/{}!!".format(root, annot_or_filename))
            return None

    if len(annot['regions']) > 1:
        print(f"!!! More than Two Annotation Found ({len(annot['regions']):02d}) at {root}{seperator}{annot['image']['identifier']}")

    roots = root.split(seperator)
    file_root = seperator.join(roots[-2:])
    new_annot = dict()
    new_annot['root'] = dataset_root
    new_annot['file_name'] = annot['image']['identifier']
    new_annot['file_root'] = file_root
    new_annot['img_width'] = annot['image']['imsize'][0]
    new_annot['img_height'] = annot['image']['imsize'][1]

    region_annot = annot['regions'][0]
    new_annot['bbox_x1'] = region_annot['boxcorners'][0]
    new_annot['bbox_y1'] = region_annot['boxcorners'][1]
    new_annot['bbox_x2'] = region_annot['boxcorners'][2]
    new_annot['bbox_y2'] = region_annot['boxcorners'][3]
    new_annot['class_name'] = region_annot['class']

    for tag in region_annot['tags']:
        tag_name, tag_value = tag.split(":")
        if tag_value == '':
            continue
        new_annot[tag_name] = tag_value

    return new_annot
