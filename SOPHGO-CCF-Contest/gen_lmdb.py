import sys
sys.path.append('./NWPU-Crowd-Sample-Code/')

import ufwio
from PIL import Image
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms

import argparse

parser = argparse.ArgumentParser(description='Trace Pytorch Model.')
parser.add_argument('--img-path', type=str, default='./',
                    help='image path')
parser.add_argument('--txt-path', type=str,
                    help='txt file for data split')
parser.add_argument('--out-path', type=str,
                    help='out put path')

args = parser.parse_args()


txn = ufwio.LMDB_Dataset(args.out_path)


mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()
LOG_PARA = 100.0

with open(args.txt_path) as f:
    for infos in f.readlines():
        filename = infos.split()[0]
        imgname = os.path.join(args.img_path, filename + '.jpg')
        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        with torch.no_grad():
            img = Variable(img)
            crop_imgs, crop_masks = [], []
            b, c, h, w = img.shape
            rh, rw = 576, 768
            for i in range(0, h, rh):
                gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                    crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros(b, 1, h, w)
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

            crop_preds = []
            nz, bz = crop_imgs.size(0), 1
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i+bz)
                txn.put(crop_imgs[gs:gt].numpy())
txn.close()


