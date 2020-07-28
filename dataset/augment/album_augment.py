import PIL
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np


class DeepInAirPolicy:
    def __init__(self):
        self.policy = A.Compose([
            A.OneOf([
                A.Rotate(180),
                A.Flip(),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=0, p=0.2),
            A.OneOf([
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16, p=0.3),
                A.GridDropout(ratio=0.3, p=0.3),
            ]),
            A.OneOf([
                A.ElasticTransform(sigma=10, alpha_affine=25, p=0.3),
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.7, p=0.2),
            ], p=0.2),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
                A.ISONoise()
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.3),
                A.MedianBlur(blur_limit=5, p=0.3),
                A.Blur(blur_limit=5, p=0.3),
                A.GaussianBlur(p=0.3)
            ], p=0.2),
            A.OneOf([
                A.ChannelShuffle(p=.3),
                A.HueSaturationValue(p=0.3),
                A.ToGray(p=0.3),
                A.ChannelDropout(p=0.3),
                A.InvertImg(p=0.1)
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.2),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
            ], p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
            A.Solarize(p=0.2),
        ])

    def __call__(self, img):
        return self.policy(image=img)['image']

    def __repr__(self):
        return "DeepInAir AIHub KProducts Policy"


if __name__ == "__main__":
    img_paths = [
                "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-8/HF020126_상품_가방_여행용가방_여행용가방_보스톤가방/HF020126_0101_0003.JPG",
                 "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-11/HF020238_상품_잡화_기타잡화_우산양산_장우산/HF020238_0101_0008.JPG",
                 "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-18/HF021148_상품_지갑_기타지갑_기타지갑_카드지갑/HF021148_0101_0004.JPG",
                 "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-13/HF021003_상품_신발_운동화캐주얼화_운동화_키높이운동화/HF021003_0101_0002.JPG",
                 "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-13/HF020538_상품_화장품_클렌징마스크팩_클렌징필링_필링젤스크럽/HF020538_0111_0002.jpg",
                 "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-6/HF020088_상품_신발_아동신발_아동운동화_브랜드운동화/HF020088_0111_0050.jpg",
                "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-9/HF020149_상품_지갑_기타지갑_기타지갑_여권지갑/HF020149_0101_0001.jpg",
                "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-18/HF021180_상품_아이웨어_안경테_안경테_뿔테/HF021180_0111_0002.jpg",
                "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-19/HF021279_상품_시계_패션시계_패션시계_가죽시계/HF021279_0101_0008.JPG",
                "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-7/HF020104_상품_가방_패션_캐쥬얼가방_힙색/HF020104_0101_0015.jpg",
                "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-12/HF020518_상품_화장품_선케어메이크업_에어쿠션팩트_에어쿠션/HF020518_0111_0015.jpg",
                "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-12/HF020519_상품_화장품_선케어메이크업_에어쿠션팩트_루스팩트파우더/HF020519_0111_0006.jpg",
                "/Users/jeikei/Documents/datasets/aihub_kproducts_resized_320/1-13/HF020525_상품_화장품_선케어메이크업_립케어블러셔_립틴트/HF020525_0111_0087.jpg"
                 ]

    imgs = [PIL.Image.open(path) for path in img_paths]
    augment_func = DeepInAirPolicy()

    for target_img in imgs:
        plt.figure(figsize=(20, 20))
        plt.subplot(4, 4, 1)
        plt.imshow(target_img)
        plt.axis('off')

        for i in range(2, 17):
            plt.subplot(4, 4, i)
            plt.imshow(augment_func(np.array(target_img)))
            plt.axis('off')

        plt.show()