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
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="portrait_drawing", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.000015, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--nepoch', type=int, default=100, help='# of epoch at starting learning rate')
parser.add_argument('--nepoch_decay', type=int, default=200, help='# of epoch to linearly decay learning rate to zero')
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--in_channels", type=int, default=3, help="number of input channels")
parser.add_argument("--out_channels", type=int, default=1, help="number of output channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument('--lambda_A', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=5.0, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_A_trunc', type=float, default=5.0, help='weight for cycle loss for trunc')
parser.add_argument('--lambda_G_A_l', type=float, default=0.5, help='weight for local GAN loss in G')
parser.add_argument("--extra_channel", type=int, default=3, help="extra channel for style feature")
parser.add_argument("--n_class", type=int, default=3, help="number of style classes")
parser.add_argument('--trunc_a', type=float, default=31.875, help='multiply which value to round when trunc')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("checkpoints/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_Cls = CrossEntropyLossNoReduction()

input_shape = (opt.in_channels, opt.img_height, opt.img_width)
output_shape = (opt.out_channels, opt.img_height, opt.img_width)
output_shape1 = (opt.out_channels + 1, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResStyleNet(input_shape, output_shape, opt.n_residual_blocks, opt.extra_channel)
G_BA = GeneratorResNet(output_shape, input_shape, opt.n_residual_blocks)
D_A = DiscriminatorCls(output_shape, opt.n_class) # discriminate drawing
D_A_ln = Discriminator(output_shape1)
D_A_le = Discriminator(output_shape1)
D_A_ll = Discriminator(output_shape1)
D_B = Discriminator(input_shape) # discriminate photo

hed = HED()
lpips = PNetLin()
hed.load('checkpoints/hed.pth')
lpips.load('checkpoints/weights/v0.1/alex.pth')
lpips.eval()

# Optimizers
optimizer_G = nn.Adam(
    G_AB.parameters() + G_BA.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = nn.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A_ln = nn.Adam(D_A_ln.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A_le = nn.Adam(D_A_le.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A_ll = nn.Adam(D_A_ll.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = nn.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizers = [optimizer_G, optimizer_D_A, optimizer_D_A_ln, optimizer_D_A_le, optimizer_D_A_ll, optimizer_D_B]
def lambda_rule(epoch):
	lr_l = 1.0 - max(0, epoch - opt.nepoch) / float(opt.nepoch_decay + 1)
	return lr_l
schedulers = [jt.optim.LambdaLR(optimizer, lr_lambda=lambda_rule) for optimizer in optimizers]

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Training data loader
dataloader = ImageDataset("data/%s" % opt.dataset_name, load_h=int(1.12*opt.img_height), load_w=int(1.12*opt.img_width), crop_h=opt.img_height, crop_w=opt.img_width).set_attrs(batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_dataloader = TestDataset("./samples", mode="test", load_h=opt.img_height, load_w=opt.img_width).set_attrs(batch_size=5, shuffle=True, num_workers=1)
import cv2
def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    cv2.imwrite(path,img)

def masked(A, mask):
    masked = ((A/2+0.5)*mask+(1-mask))*2-1
    return jt.contrib.concat((masked, mask), dim=1)

def truncate(fake_B, a=127.5):#[-1,1]
    return ((fake_B+1)*a).int().float()/a-1

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()

    style1 = jt.float32([1, 0, 0]).view(3, 1, 1).repeat(5, 1, 128, 128)
    style2 = jt.float32([0, 1, 0]).view(3, 1, 1).repeat(5, 1, 128, 128)
    style3 = jt.float32([0, 0, 1]).view(3, 1, 1).repeat(5, 1, 128, 128)

    real_A = imgs[0].stop_grad()
    fake_B1 = G_AB(real_A, style1).repeat(1, 3, 1, 1)
    fake_B2 = G_AB(real_A, style2).repeat(1, 3, 1, 1)
    fake_B3 = G_AB(real_A, style3).repeat(1, 3, 1, 1)
    # Arange images along x-axis
    real_A_ = []
    for i in range(5): real_A_.append(real_A.numpy()[i])
    real_A = np.concatenate(real_A_, -1)[np.newaxis,:,:,:]
    fake_B1_ = []
    for i in range(5): fake_B1_.append(fake_B1.numpy()[i])
    fake_B1 = np.concatenate(fake_B1_, -1)[np.newaxis,:,:,:]
    fake_B2_ = []
    for i in range(5): fake_B2_.append(fake_B2.numpy()[i])
    fake_B2 = np.concatenate(fake_B2_, -1)[np.newaxis,:,:,:]
    fake_B3_ = []
    for i in range(5): fake_B3_.append(fake_B3.numpy()[i])
    fake_B3 = np.concatenate(fake_B3_, -1)[np.newaxis,:,:,:]
    # Arange images along y-axis
    image_grid = np.concatenate((real_A, fake_B1, fake_B2, fake_B3), 0)
    save_image(image_grid, "images/%s/%s.jpg" % (opt.dataset_name, batches_done), 1)


# ----------
#  Training
# ----------

start = time.time()
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    epoch_start_time = time.time()
    process = (epoch - opt.epoch) / float(opt.n_epochs - opt.epoch)

    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = batch[0]
        real_B = batch[1]
        real_A_mask_ln = batch[2]
        real_A_mask_le = batch[3]
        real_A_mask_ll = batch[4]
        real_B_mask_ln = batch[5]
        real_B_mask_le = batch[6]
        real_B_mask_ll = batch[7]
        real_B_label = batch[8]
        real_B_style = batch[9]
        real_B_style0 = batch[10]
        zero = jt.float32([0.]).expand(real_B_label.shape)
        one = jt.float32([1.]).expand(real_B_label.shape)
        two = jt.float32([2.]).expand(real_B_label.shape)

        # Adversarial ground truths
        valid = jt.array(np.ones((real_A.size(0), *D_A.output_shape))).float32().stop_grad()
        fake = jt.array(np.zeros((real_A.size(0), *D_A.output_shape))).float32().stop_grad()

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        # GAN loss
        fake_B = G_AB(real_A, real_B_style)
        pred_fake_B, pred_fake_B_cls = D_A(fake_B)
        loss_GAN_A = criterion_GAN(pred_fake_B, valid)

        fake_A = G_BA(real_B)
        pred_fake_A = D_B(fake_A)
        loss_GAN_B = criterion_GAN(pred_fake_A, valid)

        fake_B_ln = masked(fake_B, real_A_mask_ln)
        fake_B_le = masked(fake_B, real_A_mask_le)
        fake_B_ll = masked(fake_B, real_A_mask_ll)
        loss_GAN_A_ln = criterion_GAN(D_A_ln(fake_B_ln), valid) * opt.lambda_G_A_l
        loss_GAN_A_le = criterion_GAN(D_A_le(fake_B_le), valid) * opt.lambda_G_A_l
        loss_GAN_A_ll = criterion_GAN(D_A_ll(fake_B_ll), valid) * opt.lambda_G_A_l

        loss_GAN = loss_GAN_A + loss_GAN_B + loss_GAN_A_ln + loss_GAN_A_le + loss_GAN_A_ll

        # Style loss
        loss_GAN_cls = jt.mean(real_B_style0[:,0] * criterion_Cls(pred_fake_B_cls, zero) + real_B_style0[:,1] * criterion_Cls(pred_fake_B_cls, one) + real_B_style0[:,2] * criterion_Cls(pred_fake_B_cls, two))

        # Forward cycle loss  LPIPS(HED(G_B(G_A(A))), HED(A))
        recov_A = G_BA(fake_B)
        recov_A_hed = (hed(recov_A / 2 + 0.5) - 0.5) * 2
        real_A_hed = (hed(real_A / 2 + 0.5) - 0.5) * 2
        ts = real_A.shape
        loss_cycle_A = lpips(recov_A_hed.expand(ts), real_A_hed.expand(ts)).mean() * opt.lambda_A * (1 - process * 0.9)

        # Truncation loss
        recov_At = G_BA(truncate(fake_B, opt.trunc_a))
        recov_At_hed = (hed(recov_At / 2 + 0.5) - 0.5) * 2
        loss_cycle_A_trunc = lpips(recov_At_hed.expand(ts), real_A_hed.expand(ts)).mean() * opt.lambda_A_trunc * process * 0.9

        # Backward cycle loss || G_A(G_B(B)) - B||
        recov_B = G_AB(fake_A, real_B_style)
        loss_cycle_B = criterion_cycle(recov_B, real_B) * opt.lambda_B

        # Total loss
        loss_G = loss_GAN + loss_GAN_cls + loss_cycle_A + loss_cycle_A_trunc + loss_cycle_B

        optimizer_G.step(loss_G)

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        # Real loss
        pred_real_B_, pred_real_B_cls_ = D_A(real_B)
        loss_real = criterion_GAN(pred_real_B_, valid)
        loss_real_cls = jt.mean(real_B_style0[:,0] * criterion_Cls(pred_real_B_cls_, zero) + real_B_style0[:,1] * criterion_Cls(pred_real_B_cls_, one) + real_B_style0[:,2] * criterion_Cls(pred_real_B_cls_, two))
        
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        pred_fake_B_, pred_fake_B_cls_ = D_A(fake_B_.stop_grad())
        loss_fake = criterion_GAN(pred_fake_B_, fake)
        loss_fake_cls = jt.mean(real_B_style0[:,0] * criterion_Cls(pred_fake_B_cls_, zero) + real_B_style0[:,1] * criterion_Cls(pred_fake_B_cls_, one) + real_B_style0[:,2] * criterion_Cls(pred_fake_B_cls_, two))
        
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2 + (loss_real_cls + loss_fake_cls) / 2

        optimizer_D_A.step(loss_D_A)

        # ----------------------------------
        #  Train Discriminator A ln, le, ll
        # ----------------------------------

        real_B_ln = masked(real_B, real_B_mask_ln)
        real_B_le = masked(real_B, real_B_mask_le)
        real_B_ll = masked(real_B, real_B_mask_ll)
        fake_B_ln_ = masked(fake_B_, real_A_mask_ln)
        fake_B_le_ = masked(fake_B_, real_A_mask_le)
        fake_B_ll_ = masked(fake_B_, real_A_mask_ll)

        loss_real_ln = criterion_GAN(D_A_ln(real_B_ln), valid)
        loss_fake_ln = criterion_GAN(D_A_ln(fake_B_ln_.stop_grad()), fake)
        loss_D_A_ln = (loss_real_ln + loss_fake_ln) / 2
        optimizer_D_A_ln.step(loss_D_A_ln)

        loss_real_le = criterion_GAN(D_A_le(real_B_le), valid)
        loss_fake_le = criterion_GAN(D_A_le(fake_B_le_.stop_grad()), fake)
        loss_D_A_le = (loss_real_le + loss_fake_le) / 2
        optimizer_D_A_le.step(loss_D_A_le)

        loss_real_ll = criterion_GAN(D_A_ll(real_B_ll), valid)
        loss_fake_ll = criterion_GAN(D_A_ll(fake_B_ll_.stop_grad()), fake)
        loss_D_A_ll = (loss_real_ll + loss_fake_ll) / 2
        optimizer_D_A_ll.step(loss_D_A_ll)

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        # Real loss
        loss_real = criterion_GAN(D_B(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        fake_A_.sync()
        loss_fake = criterion_GAN(D_B(fake_A_.stop_grad()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        optimizer_D_B.step(loss_D_B)

        loss_D = loss_D_A + loss_D_B + loss_D_A_ln + loss_D_A_le + loss_D_A_ll

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        if i % 50 == 0:
            # Print log
            sys.stdout.write(
                "\n[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, style: %f, cycle_A: %f, cycle_A_trunc: %f, cycle_B: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.data[0],
                    loss_G.data[0],
                    loss_GAN.data[0],
                    loss_GAN_cls.data[0],
                    loss_cycle_A.data[0],
                    loss_cycle_A_trunc.data[0],
                    loss_cycle_B.data[0],
                    time_left,
                )
            )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)
    
    # save checkpoints
    if (epoch + 1) % opt.checkpoint_interval == 0:
        G_AB.save("checkpoints/%s/%d_net_G_A.pkl" % (opt.dataset_name, epoch+1))
        G_BA.save("checkpoints/%s/%d_net_G_B.pkl" % (opt.dataset_name, epoch+1))
        D_A.save("checkpoints/%s/%d_net_D_A.pkl" % (opt.dataset_name, epoch+1))
        D_A_ln.save("checkpoints/%s/%d_net_D_A_ln.pkl" % (opt.dataset_name, epoch+1))
        D_A_le.save("checkpoints/%s/%d_net_D_A_le.pkl" % (opt.dataset_name, epoch+1))
        D_A_ll.save("checkpoints/%s/%d_net_D_A_ll.pkl" % (opt.dataset_name, epoch+1))
        D_B.save("checkpoints/%s/%d_net_D_B.pkl" % (opt.dataset_name, epoch+1))
    
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch+1, opt.n_epochs, time.time() - epoch_start_time))
    for scheduler in schedulers:
        scheduler.step()
        lr = optimizers[0].param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
print('Total Time Taken: %d sec' % (time.time() - start))
