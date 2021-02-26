import os
import random

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

import albumentations as A
import albumentations.pytorch as AT
import cv2


class ImgTransformer():
    def __init__(self, img_size, color_aug=False):
        self.img_size = img_size
        self.color_aug = color_aug

    def transform(self, image):
        transformer = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.img_size)
        ] + ([
            transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=.5),
            transforms.RandomGrayscale(p=0.1)
        ] if self.color_aug else []) + [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        transform_image = transformer(image)

        return transform_image

    def load(self, path):
        image = Image.open(path).convert('RGB')
        # print('image: ', image.size)
        return self.transform(image)

class ImgTransformerAlbu1():
    def __init__(self, img_size, color_aug=False):
        self.img_size = img_size
        self.color_aug = color_aug

    def transform(self, image):
        if self.color_aug:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.ColorJitter(brightness=1, contrast=1, saturation=1, hue=.5, p=1),
                A.ToGray(p=0.1),
                A.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
                AT.ToTensorV2(),
            ])
        else:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
                AT.ToTensorV2(),
            ])

        transform_image = transforms(image=image)['image']

        return transform_image

    def load(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print('image: ', image.size)
        return self.transform(image)

class ImgTransformerAlbu2():
    def __init__(self, img_size, color_aug=False):
        self.img_size = img_size
        self.color_aug = color_aug

    def transform(self, image, mask):
        if self.color_aug:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(var_limit=(100,200)),
                    A.JpegCompression(quality_lower=75, p=0.2)
                ], p=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=10, p=.2),
                    A.MedianBlur(blur_limit=11, p=0.1),
                    A.Blur(blur_limit=11, p=0.1),
                ], p=0.3),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.1, 0.4), rotate_limit=5, p=0.6),
                A.Cutout(num_holes=8, max_h_size=48, max_w_size=48, p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=4),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                    A.RandomBrightnessContrast(),            
                ], p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.5),
            ])
        else:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
            ])

        image_norm_and_to_tensor = A.Compose([
                A.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
                AT.ToTensorV2(transpose_mask=True),
            ])
        mask_norm_and_to_tensor = A.Compose([
                A.Normalize(mean=[0.0, 0.0, 0.0],
                            std=[1, 1, 1]),
                AT.ToTensorV2(transpose_mask=True),
            ])

        transform_pair = transforms(image=image, mask=mask)
        transform_image = transform_pair['image']
        transform_mask = transform_pair['mask']

        transform_image = image_norm_and_to_tensor(image=transform_image)['image']
        transform_mask = mask_norm_and_to_tensor(image=transform_mask)['image']

        return transform_image, transform_mask

    def load(self, impath, maskpath):
        image = cv2.imread(impath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(maskpath)
        return self.transform(image, mask)


class HairDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, image_size=448, color_aug=False):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception("%s  not exists." % self.data_folder)

        self.imagedir_path = os.path.join(data_folder, "images")
        self.maskdir_path = os.path.join(data_folder, "masks")
        self.image_names = os.listdir(self.imagedir_path)

        self.image_size = image_size
        self.transformer = ImgTransformerAlbu2(image_size, color_aug)

    def __getitem__(self, index):
        img_path = os.path.join(self.imagedir_path, self.image_names[index])
        maskfilename = os.path.join(self.maskdir_path, self.image_names[index].split('.')[0] + '.png')

        transform_image, transform_mask = self.transformer.load(img_path, maskfilename)

        transform_mask = transform_mask[0:1, :, :]

        # random horizontal flip
        hflip = random.choice([True, False])
        if hflip:
            transform_image = transform_image.flip([2])
            transform_mask = transform_mask.flip([2])
        

        return transform_image, transform_mask

    def __len__(self):
        return len(self.image_names)

# class HairDataset(torch.utils.data.Dataset):
#     def __init__(self, data_folder, image_size=448, color_aug=False):
#         self.data_folder = data_folder
#         if not os.path.exists(self.data_folder):
#             raise Exception("%s  not exists." % self.data_folder)

#         self.imagedir_path = os.path.join(data_folder, "images")
#         self.maskdir_path = os.path.join(data_folder, "masks")
#         self.image_names = os.listdir(self.imagedir_path)

#         self.image_size = image_size

#         # self.transformer = ImgTransformer(image_size, color_aug)
#         self.transformer = ImgTransformerAlbu1(image_size, color_aug)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.imagedir_path, self.image_names[index])
#         transform_image = self.transformer.load(img_path)

#         # maskfilename = self.image_names[index].split('-')[0] + '-gt.pbm'
#         maskfilename = self.image_names[index].split('.')[0] + '.png'
#         mask = Image.open(os.path.join(self.maskdir_path, maskfilename))
#         # print('mask: ', mask.size)

#         transform_mask = transforms.Compose([
#             transforms.CenterCrop(min(mask.size[0], mask.size[1])),
#             transforms.Resize(self.image_size),
#             transforms.ToTensor()
#         ])(mask)
#         transform_mask = transform_mask[0:1, :, :]
#         # print('transform_mask: ', transform_mask.size())

#         # random horizontal flip
#         hflip = random.choice([True, False])
#         if hflip:
#             transform_image = transform_image.flip([2])
#             transform_mask = transform_mask.flip([2])
        

#         return transform_image, transform_mask

#     def __len__(self):
#         return len(self.image_names)

