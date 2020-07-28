import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from dataset.tfkeras import preprocessing
import platform
import PIL
from tqdm import tqdm
from p_tqdm import p_map
from functools import partial


class KProductsTFGenerator:
    def __init__(self, annotation, label_dict, dataset_root, shuffle=False, class_key='class_name', image_size=(224, 224),
                 augment_func=None, augment_in_dtype="numpy", preprocess_func=preprocessing.preprocess_default,
                 dtype=np.float32, seed=7777, load_all=False, load_all_image_size=(147, 196), load_all_multiprocess=True):
        """

        Args:
            annotation (pd.DataFrame, str): Annotation Data or Path
            label_dict (dict): Label Dictionary.
            dataset_root (str): Dataset Root directory
            shuffle (bool): Shuffle Data Order
            class_key (str): Class Key Name. ('소분류', '중분류')
            image_size (tuple): (Height, Width)
            augment_func (function, None): Augmentation Function that takes one input of NumPy Array (height, width, channel)
            augment_in_dtype (str): Augmentation Function Input Data Type. Possible value: ('numpy', 'pil')
            preprocess_func (dataset.tfkeras.preprocessing function): Pre Processing function that takes one input of PIL.Image or Numpy Array
            dtype (np.dtype): Data type for target model.
            seed (int): Random seed.
            load_all (bool): Load all images into memory. It reuiqres large memory size.
            load_all_image_size (tuple): Target resizing image size when load all. (height, width)
        """
        assert augment_in_dtype in ['numpy', 'pil']

        if type(annotation) == pd.DataFrame:
            self.annotation = annotation
        else:
            self.annotation = pd.read_csv(annotation)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.label_dict = label_dict
        self.reverse_label = {value: int(key) for key, value in self.label_dict.items()}
        self.shuffle = shuffle
        self.class_key = class_key
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.augment_func = augment_func
        self.preprocess_func = preprocess_func
        self.dtype = dtype
        self.seed = seed
        self.augment_in_dtype = augment_in_dtype

        self.load_all_image_size = load_all_image_size
        self.dataset = None
        
        if load_all:
            self.load_all(load_all_multiprocess)

    @staticmethod
    def read_image(path, image_size=(128, 96)):
        try:
            img = Image.open(path)
            img = img.resize(image_size)
            img = np.array(img, dtype=np.uint8)
        except:
            return None

        return img

    def load_all(self, multiprocess=False):
        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"

        labels = np.array([self.reverse_label[key] for key in self.annotation[self.class_key].values], dtype=np.int32)

        img_paths = [f"{self.dataset_root}{seperator}{file_root}{seperator}{file_name}"
                     for file_root, file_name in self.annotation[["file_root", "file_name"]].values]

        img_loader = partial(KProductsTFGenerator.read_image, image_size=(self.load_all_image_size[1], self.load_all_image_size[0]))

        if multiprocess:
            images = p_map(img_loader, img_paths, num_cpus=8, desc="Loading datasets into memory ...")
        else:
            images = [img_loader(path) for path in tqdm(img_paths, desc="Loading datasets into memory ...")]

        error_idx = [i for i in range(len(images)) if images[i] is None]

        images = np.array([image for image in images if image is not None], dtype=np.uint8)
        labels = np.delete(labels, error_idx)

        self.dataset = (images, labels)

    def get_data(self, annotation, i):
        if self.dataset is not None:
            img, label = self.dataset[0][i], self.dataset[1][i]
        else:
            seperator = "\\" if platform.system().find("Windows") >= 0 else "/"
            annot = annotation.iloc[i]
            img_path = f"{self.dataset_root}{seperator}{annot['file_root']}{seperator}{annot['file_name']}"

            label = self.reverse_label[annot[self.class_key]]

            try:
                img = Image.open(img_path)
            except:
                print("Error Opening Image File at {}".format(img_path))
                return None, None

        return img, label

    def apply_augment(self, img):
        if self.augment_func is not None:
            if type(img) != np.ndarray and self.augment_in_dtype == 'numpy':
                img = np.array(img, dtype=np.uint8)
            if type(img) == np.ndarray and self.augment_in_dtype == 'pil':
                img = Image.fromarray(img)

            img = self.augment_func(img)

        if type(img) == np.ndarray:
            img = Image.fromarray(img)

        return img

    def __call__(self):
        if self.shuffle:
            annotation = self.annotation.sample(n=self.annotation.shape[0]).reset_index(drop=True)
        else:
            annotation = self.annotation

        for i in range(annotation.shape[0]):
            img, label = self.get_data(annotation, i)

            if img is None:
                continue

            img = self.apply_augment(img)

            img = img.resize((self.image_size[1], self.image_size[0]))
            img = self.preprocess_func(img, dtype=self.dtype)

            yield img, label

    def get_tf_dataset(self, batch_size):

        dataset = tf.data.Dataset.from_generator(self,
                                                 ((tf.as_dtype(self.dtype)), tf.int32),
                                                 (tf.TensorShape([self.image_size[0], self.image_size[1], 3]),
                                                  tf.TensorShape([]))).batch(batch_size)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

        return dataset


