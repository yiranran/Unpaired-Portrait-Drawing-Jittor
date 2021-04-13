import argparse
import os
import numpy as np
import math
import datetime
import time

from models import *
from datasets import *
from utils import *

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default="./samples", help="the folder of input photos")
parser.add_argument("--model_name", type=str, default="pretrained", help="the load folder of model")
parser.add_argument("--which_epoch", type=int, default=200, help="number of epoch to load")
parser.add_argument("--dataset_name", type=str, default="portrait_drawing", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--in_channels", type=int, default=3, help="number of input channels")
parser.add_argument("--out_channels", type=int, default=1, help="number of output channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--extra_channel", type=int, default=3, help="extra channel for style feature")
opt = parser.parse_args()
print(opt)

# Create save directories
save_folder = "results/%s/%s_%d" % (opt.dataset_name, opt.model_name, opt.which_epoch)
os.makedirs(save_folder, exist_ok=True)

input_shape = (opt.in_channels, opt.img_height, opt.img_width)
output_shape = (opt.out_channels, opt.img_height, opt.img_width)

# Initialize generator
G_AB = GeneratorResStyleNet(input_shape, output_shape, opt.n_residual_blocks, opt.extra_channel)

# Load weight
model_path = os.path.join("checkpoints", opt.model_name, "{}_net_G_A.pkl".format(opt.which_epoch))
if not os.path.exists(model_path):
    model_path = os.path.join("checkpoints", opt.model_name, "{}_net_G_A.pth".format(opt.which_epoch))
G_AB.load(model_path)

# Test data loader
test_dataloader = TestDataset(opt.input_folder, mode="test", load_h=opt.img_height, load_w=opt.img_width).set_attrs(batch_size=1, shuffle=False, num_workers=1)
import cv2
def save_single_image(img, path, width, height):
    N,C,W,H = img.shape
    img = img[0]
    min_ = -1
    max_ = 1
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(path,img)


# ----------
#  Testing
# ----------
G_AB.eval()

prev_time = time.time()
for i, (real_A, width, height) in enumerate(test_dataloader):

    style1 = jt.float32([1, 0, 0]).view(3, 1, 1).repeat(1, 1, 128, 128)
    style2 = jt.float32([0, 1, 0]).view(3, 1, 1).repeat(1, 1, 128, 128)
    style3 = jt.float32([0, 0, 1]).view(3, 1, 1).repeat(1, 1, 128, 128)

    fake_B1 = G_AB(real_A, style1)
    fake_B2 = G_AB(real_A, style2)
    fake_B3 = G_AB(real_A, style3)

    save_single_image(real_A.numpy(), "%s/%d_real.png" % (save_folder, i), width, height)
    save_single_image(fake_B1.numpy(), "%s/%d_fake1.png" % (save_folder, i), width, height)
    save_single_image(fake_B2.numpy(), "%s/%d_fake2.png" % (save_folder, i), width, height)
    save_single_image(fake_B3.numpy(), "%s/%d_fake3.png" % (save_folder, i), width, height)
print("Test time: %.2f" % (time.time() - prev_time))