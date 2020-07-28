import numpy as np

"""
    Image Pre-processing described as in TensorFlow Documents
    https://www.tensorflow.org/api_docs/python/tf/keras/applications
    >>> tf.keras.applications.imagenet_utils.preprocess_input
    >>> tf.keras.applications.imagenet_utils._preprocess_numpy_input
"""


def preprocess_default(x, dtype=np.float32):
    """
    MobileNet, ResNetV2, Inception, Xception
    -1 to 1
    """
    return (np.array(x, dtype=dtype) / 127.5) - 1


def preprocess_zero2one(x, dtype=np.float32):
    """
    0 to 1
    """
    return np.array(x, dtype=dtype) / 255.0


def preprocess_resnet(x, dtype=np.float32):
    """
    ResNet, VGG
    RGB -> BGR then zero-centered(ImageNet) without scaling
    """
    return np.array(x, dtype=dtype)[..., ::-1] - np.array([103.939, 116.779, 123.67])


def preprocess_effnet(x, dtype=np.float32):
    """
    Scale 0 to 1 then zero-centered(ImageNet) with Standard Deviation Normalization(ImageNet)
    """
    return ((np.array(x, dtype=dtype) / 255.0) - np.array([0.485, 0.456, 0.406], dtype=dtype)) / np.array([0.229, 0.224, 0.225], dtype=dtype)


def get_preprocess_by_model_name(model_name):
    if model_name.startswith("logistic"):
        return preprocess_zero2one
    elif model_name.startswith("mobilenet") or \
            (model_name.startswith("resnet") and model_name.endswith("v2")) or \
            model_name.startswith("inception") or \
            model_name.startswith("xception") or \
            model_name.endswith("custom"):
        return preprocess_default
    elif model_name.startswith("resnet") or model_name.startswith("VGG"):
        return preprocess_resnet
    elif model_name.startswith("effnet"):
        return preprocess_effnet
    else:
        return preprocess_default
