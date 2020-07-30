import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from dataset.tfkeras import KProductsTFGenerator
from functools import partial
from p_tqdm import p_map
import multiprocess as mp
import platform


class KProductsTFGeneratorTTA(KProductsTFGenerator):
    def __init__(self, *args, n_tta=3, multiprocess=True, **kwargs):
        """

        Args:
            annotation (pd.DataFrame, str): Annotation Data or Path
            label_dict (dict): Label Dictionary.
            dataset_root (str): Dataset Root directory
            n_tta (int): Number of test time augmentation
            shuffle (bool): Shuffle Data Order
            class_key (str): Class Key Name. ('소분류', '중분류')
            image_size (tuple): (Height, Width)
            augment_func (function, None): Augmentation Function that takes one input of NumPy Array (height, width, channel)
            augment_in_dtype (str): Augmentation Function Input Data Type. Possible value: ('numpy', 'pil', 'tensor')
            preprocess_func (dataset.tfkeras.preprocessing function): Pre Processing function that takes one input of PIL.Image or Numpy Array
            dtype (np.dtype): Data type for target model.
            seed (int): Random seed.
            load_all (bool): Load all images into memory. It reuiqres large memory size.
            load_all_image_size (tuple): Target resizing image size when load all. (height, width)
        """
        super(KProductsTFGeneratorTTA, self).__init__(*args, **kwargs)
        self.n_tta = n_tta
        assert self.augment_func is not None, "Augmentation function must be defined!"

        self.multiprocess = multiprocess
    
    @staticmethod
    def _tta_generator(dataset=None, apply_augment=None, shuffle=False):
        if shuffle:
            np.random.shuffle(dataset)
    
        for i in range(len(dataset)):
            try:
                img = Image.open(dataset[i][0]).convert("RGB")
            except:
                print("Error Opening Image at {}".format(dataset[i][0]))
                continue
    
            img = apply_augment(img)
    
            yield img, dataset[i][1]

    @staticmethod
    def augment_pipeline(img, preprocess_func=None, augment_func=None, data_format="channels_first",
                         image_size=(112, 112), dtype=np.float32):
        ori_img = img
        img = augment_func(ori_img)

        if type(img) == np.ndarray:
            img = Image.fromarray(img)

        img = img.resize((image_size[1], image_size[0]))

        img = preprocess_func(img, dtype=dtype)
        img = np.swapaxes(img.T, 1, 2) if data_format == "channels_first" else img

        return img

    @staticmethod
    def apply_augment(img, pipeline=None, augment_in_dtype="pil", multiprocess=False, n_tta=3):
        if type(img) != np.ndarray and augment_in_dtype == 'numpy':
            img = np.array(img, dtype=np.uint8)
        if type(img) == np.ndarray and augment_in_dtype == 'pil':
            img = Image.fromarray(img)

        if multiprocess:
            imgs = p_map(pipeline, [img for _ in range(n_tta)],
                         disable=True, num_cpus=min(mp.cpu_count()-1, n_tta))
        else:
            imgs = [pipeline(img) for _ in range(n_tta)]

        imgs = np.concatenate(imgs, axis=-1)
        return imgs

    def __call__(self):
        if self.shuffle:
            annotation = self.annotation.sample(n=self.annotation.shape[0]).reset_index(drop=True)
        else:
            annotation = self.annotation

        for i in range(annotation.shape[0]):
            img, label = self.get_data(annotation, i)

            if img is None:
                continue

            # if self.augment_in_dtype == "tensor":
            #     img = self.apply_tf_augment(img)
            # else:
            if self.augment_in_dtype != "tensor":
                img = apply_augment(img)
            else:
                img = img.resize((self.image_size[1], self.image_size[0]))

            yield img, label

    def get_tf_dataset(self, batch_size):
        seperator = "\\" if platform.system().find("Windows") >= 0 else "/"
        dataset = [(f"{root}{seperator}{file_root}{seperator}{file_name}", self.reverse_label[class_name])
                     for root, file_root, file_name, class_name in self.annotation[['root', 'file_root', 'file_name', "class_name"]].values]

        pipeline = partial(KProductsTFGeneratorTTA.augment_pipeline, preprocess_func=self.preprocess_func,
                           augment_func=self.augment_func, data_format=self.data_format,
                           image_size=self.image_size, dtype=self.dtype)

        apply_augment = partial(KProductsTFGeneratorTTA.apply_augment, pipeline=pipeline, augment_in_dtype=self.augment_in_dtype,
                                multiprocess=self.multiprocess, n_tta=self.n_tta)
        generator = partial(KProductsTFGeneratorTTA._tta_generator, dataset=dataset, apply_augment=apply_augment, shuffle=self.shuffle)

        img_shape = tf.TensorShape([self.image_size[0], self.image_size[1],
                                    3 if self.augment_in_dtype == "tensor" else 3*self.n_tta])

        img_shape = img_shape if self.data_format == "channels_last" else (img_shape[-1],) + img_shape[:2]

        dataset = tf.data.Dataset.from_generator(generator,
                                                 ((tf.as_dtype(self.dtype)), tf.int32),
                                                 (img_shape, tf.TensorShape([])))
        if self.use_cache:
            dataset = dataset.cache()

        # if self.augment_in_dtype == "tensor":
        #     dataset = dataset.map(self.apply_tf_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(batch_size)

        if self.prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


