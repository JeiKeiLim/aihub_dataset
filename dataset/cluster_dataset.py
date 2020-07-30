from dataset.kproducts_dataset import KProductsDataset
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from p_tqdm import p_umap
import pandas as pd
from functools import partial
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import platform


class ClusterData:
    """
    Cluster dataset on KProductsDataset
    Attributes:
        preprocess_dict (dict): Contains key as (tf.keras.applications.Models) and value as (pre-processing method).
                                Supports every Models in tf.keras.applications.

    """
    name_to_model_dict = {
        "densenet121": tf.keras.applications.DenseNet121,
        "densenet169": tf.keras.applications.DenseNet169,
        "densenet201": tf.keras.applications.DenseNet201,
        "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "mobilenet": tf.keras.applications.MobileNet,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "nasnetlarge": tf.keras.applications.NASNetLarge,
        "nasnetmobile": tf.keras.applications.NASNetMobile,
        "resnet101": tf.keras.applications.ResNet101,
        "resnet101v2": tf.keras.applications.ResNet101V2,
        "resnet152": tf.keras.applications.ResNet152,
        "resnet152v2": tf.keras.applications.ResNet152V2,
        "resnet50": tf.keras.applications.ResNet50,
        "resnet50v2": tf.keras.applications.ResNet50V2,
        "vgg16": tf.keras.applications.VGG16,
        "vgg19": tf.keras.applications.VGG19,
        "xception": tf.keras.applications.Xception
    }

    preprocess_dict = {
        tf.keras.applications.DenseNet121: tf.keras.applications.densenet.preprocess_input,
        tf.keras.applications.DenseNet169: tf.keras.applications.densenet.preprocess_input,
        tf.keras.applications.DenseNet201: tf.keras.applications.densenet.preprocess_input,
        tf.keras.applications.InceptionResNetV2: tf.keras.applications.inception_resnet_v2.preprocess_input,
        tf.keras.applications.InceptionV3: tf.keras.applications.inception_v3.preprocess_input,
        tf.keras.applications.MobileNet: tf.keras.applications.mobilenet.preprocess_input,
        tf.keras.applications.MobileNetV2: tf.keras.applications.mobilenet_v2.preprocess_input,
        tf.keras.applications.NASNetLarge: tf.keras.applications.nasnet.preprocess_input,
        tf.keras.applications.NASNetMobile: tf.keras.applications.nasnet.preprocess_input,
        tf.keras.applications.ResNet101: tf.keras.applications.resnet.preprocess_input,
        tf.keras.applications.ResNet101V2: tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.applications.ResNet152: tf.keras.applications.resnet.preprocess_input,
        tf.keras.applications.ResNet152V2: tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.applications.ResNet50: tf.keras.applications.resnet50.preprocess_input,
        tf.keras.applications.ResNet50V2: tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.applications.VGG16: tf.keras.applications.vgg16.preprocess_input,
        tf.keras.applications.VGG19: tf.keras.applications.vgg19.preprocess_input,
        tf.keras.applications.Xception: tf.keras.applications.xception.preprocess_input,
    }

    def __init__(self, dataset, model_input_size=(224, 224), BaseModel=tf.keras.applications.ResNet152V2, preprocess_func=None, seed=7777):
        """

        Args:
            dataset (KProductsDataset): dataset
            model_input_size (tuple): (height, width)
            BaseModel (tf.keras.models.Model): Base model to vectorize the images.
            preprocess_func (lambda, func): Custom pre-processing function in case custom model is given in BaseModel
        """

        np.random.seed(seed)

        self.seed = seed
        self.dataset = dataset
        self.model_input_size = model_input_size
        self.model = BaseModel(input_shape=self.model_input_size + (3, ), weights='imagenet')
        self.model_n_out = self.model.output.shape[1]

        if BaseModel in ClusterData.preprocess_dict.keys():
            self.preprocess_func = ClusterData.preprocess_dict[BaseModel]
        elif preprocess_func is not None:
            self.preprocess_func = preprocess_func
        else:
            self.preprocess_func = lambda x: (x / 127.5)-1

    def vectorize_images(self, annotation, batch_size=32):
        total = annotation.shape[0]
        annot_index = annotation.index

        feature_vector = np.empty((0, self.model_n_out))
        drop_index = []
        for b in tqdm(range(0, total, batch_size), desc="Vectorize Images ..."):
            batch_index = [annot_index[i] for i in range(b, min(b+batch_size, total))]
            batch_imgs = [self.dataset.get_data(idx)[0] for idx in batch_index]

            batch_imgs = np.array([np.array(img.resize(self.model_input_size[::-1]), dtype=np.uint8)
                                   for img in batch_imgs if img is not None])
            batch_imgs = self.preprocess_func(batch_imgs)

            predict_result = self.model.predict(batch_imgs)

            feature_vector = np.concatenate([feature_vector, predict_result])

            drop_index += [batch_index[i] for i in range(len(batch_imgs)) if batch_imgs[i] is None]

        if len(drop_index) > 0:
            drop_list = [f"{r}/{f}" for r, f in annotation.iloc[drop_index][['file_root', 'file_name']].values]
            print("Dropping annotation...")
            for drop_file in drop_list:
                print("..... {}".format(drop_file))

            annotation.drop(drop_index)

        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        save_root = f"{self.dataset.config['dataset_root']}{seperator}{annotation['file_root'].iloc[0]}"
        pd_feat_vector = pd.DataFrame(feature_vector)

        pd_feat_vector.to_csv(f"{save_root}{seperator}feature_vector.csv", header=False, index=False)
        annotation.to_csv(f"{save_root}{seperator}annotation.csv", index=False)

        return feature_vector

    def get_annotation_by_file_root(self):
        unique_file_root = self.dataset.annotations['file_root'].unique()
        annot_by_file_root = [self.dataset.annotations.query("file_root == '{}'".format(f_root)) for f_root in
                              unique_file_root]

        return annot_by_file_root

    def vectorize_dataset(self, multiprocess=False, batch_size=32):
        annot_by_file_root = self.get_annotation_by_file_root()
        if multiprocess:
            p_umap(partial(self.vectorize_images, batch_size=batch_size),
                   annot_by_file_root, desc="Vectorization Dataset ...")
        else:
            for annotation in tqdm(annot_by_file_root, "Vectorization Dataset ..."):
                self.vectorize_images(annotation, batch_size=batch_size)

    def reconstruct_from_cluster_result(self, target_root="./export", target_annotation_name="converted_annotation.csv",
                                        include_non_core=True, ignore_under_median=True):
        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"
        target_root = target_root.replace("/", "\\") if seperator == "\\" else target_root

        unique_file_root = self.dataset.annotations['file_root'].unique()
        paths = [f"{self.dataset.config['dataset_root']}{seperator}{file_root}{seperator}annotation_cluster_core.csv"
                 for file_root in unique_file_root]
        if include_non_core:
            paths += [f"{self.dataset.config['dataset_root']}{seperator}{file_root}{seperator}annotation_cluster_non_core.csv"
                     for file_root in unique_file_root]

        for path in tqdm(paths, desc="Copying files and annotations ..."):
            self.copy_to_target_directory(path, target_root, target_annotation_name=target_annotation_name)

    def copy_to_target_directory(self, annotation_path, target_root, target_annotation_name="converted_annotation.csv"):
        try:
            annotation = pd.read_csv(annotation_path)
        except:
            print(f"Reading Annotation Error at {annotation_path}")
            return

        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        target_annot_path = f"{target_root}{seperator}{target_annotation_name}"

        fail_index = []
        for i, row in annotation.iterrows():
            src_img_path = f"{self.dataset.config['dataset_root']}{seperator}{row['file_root']}{seperator}{row['file_name']}"
            dest_img_path = f"{target_root}{seperator}{row['file_root']}"
            os.makedirs(dest_img_path, exist_ok=True)
            dest_img_path += f"{seperator}{row['file_name']}"
            try:
                shutil.copy2(src_img_path, dest_img_path)
            except:
                print("File could not copy from {} to {}".format(src_img_path, dest_img_path))
                fail_index.append(i)

        annotation.drop(fail_index)

        if os.path.isfile(target_annot_path):
            target_annotation = pd.read_csv(target_annot_path)
            target_annotation = pd.concat([target_annotation, annotation], ignore_index=True)
        else:
            target_annotation = annotation
        target_annotation.drop_duplicates(inplace=True, ignore_index=True)

        target_annotation.to_csv(f"{target_root}{seperator}{target_annotation_name}", index=False)

    def cluster_dataset(self, **kwargs):
        """

        Args:
            **kwargs (cluster_from_vectorization): Keyword Arguments
        """
        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        unique_file_root = self.dataset.annotations['file_root'].unique()
        paths = [(f"{self.dataset.config['dataset_root']}{seperator}{file_root}{seperator}feature_vector.csv",
                  f"{self.dataset.config['dataset_root']}{seperator}{file_root}{seperator}annotation.csv")
                  for file_root in unique_file_root]
        for path in tqdm(paths, "Clustering ..."):
            self.cluster_from_vectorization_path(path, **kwargs)

    def cluster_from_vectorization_path(self, args, **kwargs):
        feat_vec_path, annot_path = args

        try:
            feat_vec = pd.read_csv(feat_vec_path, header=None)
            annot = pd.read_csv(annot_path)
        except:
            print("Openning file error at {}, {}".format(feat_vec_path, annot_path))
            return None

        core, non_core = self.cluster_from_vectorization(feat_vec, annot, **kwargs)
        core[1].to_csv(f"{annot_path[:-4]}_cluster_core.csv", index=False)
        non_core[1].to_csv(f"{annot_path[:-4]}_cluster_non_core.csv", index=False)

    def cluster_from_vectorization(self, feature_vector, annotation, eps=0.1, plot=False, plot_img_size=(50, 50),
                                   canvas_size_alpha=50, figure_size=(30, 30), save_plot=False):
        dbscan = DBSCAN(eps=eps, n_jobs=-1).fit(feature_vector.values)
        core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True

        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        if annotation.shape[0] != core_samples_mask.shape[0]:
            print("Something went wrong here! annotation: {:,}, feature_vector: {:,}, {:,}, core_samples_mask: {:,}".format(annotation.shape[0], feature_vector.shape[0], feature_vector.values.shape[0], core_samples_mask.shape[0]))
            print(f"{self.dataset.config['dataset_root']}/{annotation.iloc[0]['file_root']}")
            raise IndexError

        labels = dbscan.labels_

        non_core_samples_mask = np.logical_and(~core_samples_mask, labels != -1)

        core_feat_vector = feature_vector.iloc[core_samples_mask]
        core_annotation = annotation.iloc[core_samples_mask]

        non_core_feat_vector = feature_vector.iloc[non_core_samples_mask]
        non_core_annotation = annotation.iloc[non_core_samples_mask]

        print("Label Number: {:02d}, Core Vectors: {:03d}, Non-core Vectors: {:03d}, Data Number: {:03d}".format(
            len(set(labels)), core_annotation.shape[0], non_core_annotation.shape[0], annotation.shape[0]
        ))

        if plot is False and save_plot is False:
            return (core_feat_vector, core_annotation), (non_core_feat_vector, non_core_annotation)
        else:
            x_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(feature_vector.values)

            unique_labels = set(labels)

            colors = [np.array(plt.cm.Spectral(each))[:3] * 0.2 + 1.0
                      for each in np.linspace(0, 1, len(unique_labels))]

            canvas = np.ones((plot_img_size[1] * canvas_size_alpha, plot_img_size[0] * canvas_size_alpha, 3), dtype=np.uint8) * 255

            if plot:
                plt.figure(figsize=figure_size)

            x_embedded[:, 0] -= x_embedded[:, 0].min()
            x_embedded[:, 1] -= x_embedded[:, 1].min()

            x_embedded[:, 0] /= x_embedded[:, 0].max()
            x_embedded[:, 1] /= x_embedded[:, 1].max()

            for i in tqdm(range(x_embedded.shape[0]), "Generating T-SNE Image ..."):
                if labels[i] == -1:
                    color = [0.7, 0.7, 0.7]
                else:
                    color = colors[labels[i]]

                if core_samples_mask[i]:
                    resize_size = (int(plot_img_size[0] * 2.0), int(plot_img_size[1] * 2.0))
                elif labels[i] == -1:
                    resize_size = (int(plot_img_size[0] * 0.5), int(plot_img_size[1] * 0.5))
                else:
                    resize_size = plot_img_size

                img_path = f"{annotation.iloc[i]['root']}{seperator}{annotation.iloc[i]['file_root']}{seperator}{annotation.iloc[i]['file_name']}"
                img = Image.open(img_path)
                img = img.resize(resize_size)

                img = np.maximum(np.minimum(np.array(img) * color, 255), 0)

                x1, y1 = (x_embedded[i, :] * canvas.shape[:2]).astype(np.int)
                x2, y2 = x1 + resize_size[0], y1 + resize_size[1]
                x2, y2 = min(x2, canvas.shape[1]), min(y2, canvas.shape[0])

                canvas[y1:y2, x1:x2, :] = np.array(img, np.uint8)[:(y2 - y1), :(x2 - x1), :]

            if save_plot != "":
                path = f"{self.dataset.config['dataset_root']}{seperator}{annotation.iloc[i]['file_root']}{seperator}cluster_result.jpg"
                Image.fromarray(canvas).save(path)
            if plot:
                plt.imshow(canvas)
                plt.show()

            return (core_feat_vector, core_annotation), (non_core_feat_vector, non_core_annotation)
