from kproducts_dataset import KProductsDataset
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from p_tqdm import p_umap
import pandas as pd
from functools import partial


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

    def __init__(self, dataset, model_input_size=(224, 224), BaseModel=tf.keras.applications.ResNet152V2, preprocess_func=None):
        """

        Args:
            dataset (KProductsDataset): dataset
            model_input_size (tuple): (height, width)
            BaseModel (tf.keras.models.Model): Base model to vectorize the images.
            preprocess_func (lambda, func): Custom pre-processing function in case custom model is given in BaseModel
        """
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
        for b in tqdm(range(0, total, batch_size), desc="Vectorize Images ..."):
            batch_imgs = [self.dataset.get_data(annot_index[i])[0] for i in range(b, min(b+batch_size, total))]
            batch_imgs = np.array([np.array(img.resize(self.model_input_size[::-1]), dtype=np.uint8)
                                   for img in batch_imgs])
            batch_imgs = self.preprocess_func(batch_imgs)

            predict_result = self.model.predict(batch_imgs)

            feature_vector = np.concatenate([feature_vector, predict_result])

        save_root = f"{self.dataset.config['dataset_root']}/{annotation['file_root'].iloc[0]}"
        pd_feat_vector = pd.DataFrame(feature_vector)

        pd_feat_vector.to_csv(f"{save_root}/feature_vector.csv", header=False, index=False)
        annotation.to_csv(f"{save_root}/annotation.csv", index=False)

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
