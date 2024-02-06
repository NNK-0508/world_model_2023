import os
from utils.visualizer import save_images
from utils import html, MONetModel
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional
import polars as pl
from pathlib import Path
import torch

label_path = "./data/label_updated.csv"
img_path = "./data/clock_drawing"

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

    results_dir = "./result",
    name = "test",
    phase = 2,
    epoch = 0,
    num_test = 50,
    eval = True,
    aspect_ratio = 1,
    display_winsize = 2000
)

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
        self.img_path = None

    def __len__(self) -> int:
        num_images = len(list(self.img_dir.glob("*/*.png")))
        assert len(self.label_df) == num_images, f"The length of the label, {len(self.label_df)} does not match the number of images, {num_images}."
        return len(self.label_df)

    def __getitem__(self, index) -> tuple:
        row = self.label_df.row(index, named=True)
        img_name, round, label = row["img_name"], row["round"], row["label"]
        self.img_path = str(self.img_dir / f"round{round}" / f"{img_name.split('.')[0]}.png")
        img = Image.open(self.img_path)
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
# data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

if __name__ == '__main__':
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = MONetModel(opt)      # create a model given opt.model and other options
    state_dict = torch.load("./models/latest_net_Attn.pth")
    new_state_dict = {}
    for k, v in state_dict.items():
        name = 'module.' + k  # 新しいキー名
        new_state_dict[name] = v
    model.netAttn.load_state_dict(new_state_dict)
    state_dict = torch.load("./models/latest_net_CVAE.pth")
    new_state_dict = {}
    for k, v in state_dict.items():
        name = 'module.' + k  # 新しいキー名
        new_state_dict[name] = v
    model.netCVAE.load_state_dict(new_state_dict)
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(torch.unsqueeze(data, 0))  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, dataset.img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML