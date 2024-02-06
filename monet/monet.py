from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import init
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import gc
import torch
from itertools import chain
from collections import OrderedDict
from types import SimpleNamespace
import time
from typing import Optional
import polars as pl
import wandb
from dotenv import load_dotenv
load_dotenv()

from utils import MONetModel

opt = SimpleNamespace(
    input_nc=3, # num of channels
    num_slots=7, # Number of supported slots
    z_dim=16, # Dimension of individual z latent per slot
    beta=0.5, # weight for the encoder KLD
    gamma=0.5, # weight for the mask KLD
    isTrain=True,
    lr=0.0002,
    lr_policy="linear", # learning rate policy. [linear | step | plateau | cosine]
    epoch_count=1, # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
    niter= 5, # 100, # num of iter at starting learning rate
    niter_decay= 5, # 100, # num of iter to linearly decay learning rate to zero
    continue_train=False, # continue training: load the latest model
    verbose=True,
    print_freq=10,
    batch_size= 128,
    save_latest_freq=5000,
    save_epoch_freq=5,
    save_by_iter=False, # "whether saves model by iteration"
)

label_path = "./data/label_updated.csv"
img_path = "./data/clock_drawing"
checkpoint_path = "./models"

class ClockDrawingsDataset(Dataset):
    def __init__(self, csv: str, img_root_dir: str, transform: Optional[callable]=None, flags: Optional[int]=None) -> None:
        """
        Arguments:
            csv (str): path to csv file containing label info
            img_root_dir (str): path to the root directory of images
            transform: the function to transform images, currently unavailable
            flags: the format for reading images

        Returns:
            None
        """
        self.label_df = pl.read_csv(csv)
        self.img_dir = Path(img_root_dir)
        self.transform = transform
        self.flags = flags

    def __len__(self) -> int:
        num_images = len(list(self.img_dir.glob("*/*.png")))
        assert len(self.label_df) == num_images, f"The length of the label, {len(self.label_df)} does not match the number of images, {num_images}."
        return len(self.label_df)

    def __getitem__(self, index) -> tuple:
        row = self.label_df.row(index, named=True)
        img_name, round, label = row["img_name"], row["round"], row["label"]
        img = Image.open(str(self.img_dir / f"round{round}" / f"{img_name.split('.')[0]}.png"))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # return img, int(label <= 3)
        return img

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = ClockDrawingsDataset(label_path, img_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
model = MONetModel(opt)
model.setup(opt)
total_iters = 0                # the total number of training iterations

# wandb
wandb.login(key=os.environ.get("WANDB_API_KEY"))
wandb.init(
    project="world-model-monet"
)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

    for i, data in enumerate(tqdm(data_loader)):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            wandb.log({"loss": losses})

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
    gc.collect()

