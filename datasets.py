import glob
import random
import os
import numpy as np

from jittor.dataset.dataset import Dataset
from PIL import Image
import jittor.transform as transform
import jittor as jt

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def get_params(new_w, new_h, w, h):
    x = random.randint(0, np.maximum(0, new_w - w))
    y = random.randint(0, np.maximum(0, new_h - h))

    flip = random.random() > 0.5

    return {'top': y, 'left': x, 'crop_h': h, 'crop_w': w, 'load_h': new_h, 'load_w': new_w, 'flip': flip}

def get_transform(params, gray = False, mask = False):
    transform_ = []
    # resize
    transform_.append(transform.Resize((params['load_h'], params['load_w']), Image.BICUBIC))
    # crop
    transform_.append(transform.Lambda(lambda img: transform.crop(img, params['top'], params['left'], params['crop_h'], params['crop_w'])))
    # flip
    if params['flip']:
        transform_.append(transform.Lambda(lambda img: transform.hflip(img)))
    if gray:
        transform_.append(transform.Gray())
    if mask:
        transform_.append(transform.ImageNormalize([0.,], [1.,]))
    else:
        if not gray:
            transform_.append(transform.ImageNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        else:
            transform_.append(transform.ImageNormalize([0.5,], [0.5,]))
    return transform.Compose(transform_)


class ImageDataset(Dataset):
    def __init__(self, root, mode="train", load_h=572, load_w=572, crop_h=512, crop_w=512):
        super().__init__()

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

        self.auxdir_A = os.path.join(root, "%s/A" % mode)
        self.auxdir_B = os.path.join(root, "%s/B" % mode)

        self.total_len = max(len(self.files_A), len(self.files_B))
        self.batch_size = None
        self.shuffle = False
        self.drop_last = False
        self.num_workers = None
        self.buffer_size = 512*1024*1024
        self.load_h = load_h
        self.load_w = load_w
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __getitem__(self, index):
        A_path = self.files_A[index % len(self.files_A)]
        B_path = self.files_B[random.randint(0, len(self.files_B) - 1)]
        basenA = os.path.basename(A_path)
        basenB = os.path.basename(B_path)

        image_A = Image.open(A_path).convert('RGB')
        image_B = Image.open(B_path).convert('RGB')
        mask_A_ln = Image.open(os.path.join(self.auxdir_A+'_nose',basenA))
        mask_A_le = Image.open(os.path.join(self.auxdir_A+'_eyes',basenA))
        mask_A_ll = Image.open(os.path.join(self.auxdir_A+'_lips',basenA))
        mask_B_ln = Image.open(os.path.join(self.auxdir_B+'_nose',basenB))
        mask_B_le = Image.open(os.path.join(self.auxdir_B+'_eyes',basenB))
        mask_B_ll = Image.open(os.path.join(self.auxdir_B+'_lips',basenB))
        
        # Image transformations
        params_A = get_params(self.load_h, self.load_w, self.crop_h, self.crop_w)
        params_B = get_params(self.load_h, self.load_w, self.crop_h, self.crop_w)
        
        transform_A = get_transform(params_A)
        transform_A_mask = get_transform(params_A, gray=True, mask=True)
        transform_B = get_transform(params_B, gray=True)
        transform_B_mask = get_transform(params_B, gray=True, mask=True)

        item_A = transform_A(image_A)
        item_A_mask_ln = transform_A_mask(mask_A_ln)
        item_A_mask_le = transform_A_mask(mask_A_le)
        item_A_mask_ll = transform_A_mask(mask_A_ll)
        item_B = transform_B(image_B)
        item_B_mask_ln = transform_B_mask(mask_B_ln)
        item_B_mask_le = transform_B_mask(mask_B_le)
        item_B_mask_ll = transform_B_mask(mask_B_ll)

        B_feat = np.load(os.path.join(self.auxdir_B+'_feat',basenB[:-4]+'.npy'))
        item_B_label = jt.array(np.argmax(B_feat))
        item_B_style = jt.array(B_feat).view(3, 1, 1).repeat(1, 128, 128)
        item_B_style0 = jt.array(B_feat)

        return item_A, item_B, item_A_mask_ln, item_A_mask_le, item_A_mask_ll, item_B_mask_ln, item_B_mask_le, item_B_mask_ll, item_B_label, item_B_style, item_B_style0

class TestDataset(Dataset):
    def __init__(self, root, mode="test", load_h=512, load_w=512):
        super().__init__()
        transform_ = [
            transform.Resize((load_h, load_w), Image.BICUBIC),
            transform.ImageNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = transform.Compose(transform_)

        self.files_A = sorted(glob.glob(root + "/*.*"))

        self.total_len = len(self.files_A)
        self.batch_size = None
        self.shuffle = False
        self.drop_last = False
        self.num_workers = None
        self.buffer_size = 512*1024*1024

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        width = image_A.size[0]
        height = image_A.size[1]

        item_A = self.transform(image_A)
        return item_A, width, height