from argparse import Namespace

import matplotlib.pyplot as plt 
import sys, os
sys.path.append(os.getcwd() + '/CLIPstyler')

from pathlib import Path
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms

import fast_stylenet


from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True

import utils


def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def hr_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def fast_clip(img, text_cond="anime", output_path=None):
    source = "a Photo"
    # "acrylic", "desert_sand", "inkwash_painting", "oil_bluered_brush",
    # "sketch_blackpencil", "stonewall", "water_purple_brush", "anime",
    # "blue_wool", "cyberpunk", "mondrian", "papyrus"

    vgg_dir = "./CLIPstyler/models/vgg_normalized.pth"
    test_dir = "./CLIPstyler/test_set" #@param {type: "string"}
    use_hr = False #@param {type:"boolean"}   
    if use_hr:
        hr_dir = "./CLIPstyler/hr_set" #@param {type: "string"}
    else:
        hr_dir=None

    decoder_dir = "./CLIPstyler/models/clip_decoder_"+text_cond+".pth.tar"

    training_args = {
        "test_dir": test_dir,
        "hr_dir":hr_dir,
        "vgg": vgg_dir,
        "n_threads":2,
        "num_test":1,
        "decoder":decoder_dir
    }

    args = Namespace(**training_args)

    device = torch.device('cuda')

    decoder = fast_stylenet.decoder
    vgg = fast_stylenet.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    decoder.load_state_dict(torch.load(args.decoder))

    network = fast_stylenet.Net(vgg, decoder)
    network.eval()
    network.to(device)

    test_tf = test_transform()
    
    img = test_tf(img)
    img = img.unsqueeze(0)
    img = img.cuda()

    # test_dataset = FlatFolderDataset(args.test_dir, test_tf)
    # test_iter = iter(data.DataLoader(
    #     test_dataset, batch_size=args.num_test,
    #     num_workers=args.n_threads))
    
    

    # test_images1 = next(test_iter)
    # print(test_images1.shape)
    # test_images1 = test_images1.cuda()
    # print(test_images1.shape)

    if args.hr_dir is not None:
        hr_tf = hr_transform()
        hr_dataset = FlatFolderDataset(args.hr_dir, hr_tf)
        hr_iter = iter(data.DataLoader(
        hr_dataset, batch_size=1,
        num_workers=args.n_threads))

        hr_images = next(hr_iter)
        hr_images = hr_images.cuda()

    with torch.no_grad():
        _, test_out1 = network(img)
        test_out1 = adjust_contrast(test_out1,1.5)
        # output_test = torch.cat([test_images1,test_out1],dim=0)
        save_image(test_out1, output_path, nrow=test_out1.size(0),normalize=True,scale_each=True)

        # plt.imshow(utils.im_convert2(test_out1[0].unsqueeze(0)))
        # plt.show()
        if args.hr_dir is not None:
            _, test_hr = network(hr_images)
            test_hr = adjust_contrast(test_hr,1.5)
            save_image(test_hr, output_path, nrow=test_hr.size(0),normalize=True,scale_each=True)
    
    return test_out1.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    image_path = "./output.jpg"
    img = Image.open(image_path).convert('RGB')
    print("image shape: ", np.array(img).shape)
    output_path = "./clip_output.png"
    fast_clip(img, text_cond="oil_bluered_brush", output_path=output_path)
    # "acrylic", "desert_sand", "inkwash_painting", "oil_bluered_brush",
    # "sketch_blackpencil", "stonewall", "water_purple_brush", "anime",
    # "blue_wool", "cyberpunk", "mondrian", "papyrus"