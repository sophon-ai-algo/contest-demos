
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import time
import argparse
import sys
sys.path.append('/root/data/bmnnsdk2/bmnnsdk2-bm1684_v2.7.0/lib/sail/python3/soc/py37')
import sophon.sail as sail


parser = argparse.ArgumentParser(description='Model inference')
parser.add_argument('--data', type=str, default='',
                    help='root path of data')
parser.add_argument('--model', type=str, default='',
                    help='model path')
parser.add_argument('--result', type=str, default='',
                    help='result path')

args = parser.parse_args()


class Tester:
    """Model inference and testing using SAIL

    Attributes:
        engine: sail engine
        graph_name: graph name of loaded model in sail engine
        input_tensor_name: input tensor name in graph
        output_tensor_name: output tensor name in graph
        rh: croped image height
        rw: croped image width
    """
    def __init__(self) -> None:
        """sail engine initialization"""
        self.LOG_PARA = 100.0
        self.engine = sail.Engine(args.model, 0, sail.IOMode.SYSIO)
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_tensor_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_tensor_name = self.engine.get_output_names(self.graph_name)[0]
        self.rh, self.rw = 576, 768

    def pre_process(self, img):
        """image preprocession"""
        mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
        img_transform = standard_transforms.Compose([
                standard_transforms.ToTensor(),
                standard_transforms.Normalize(*mean_std)
            ])
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        return img


    def test_one(self, img_pth):
        """predict number of people in the given image
        Args:
            img_pth: image path
        Returns:
            predicted number
        """
        img = Image.open(img_pth)
        img = self.pre_process(img)
        crop_imgs, crop_masks = [], []
        b, c, h, w = img.shape
        # crop image to the fixed shape
        for i in range(0, h, self.rh):
            gis, gie = max(min(h-self.rh, i), 0), min(h, i+self.rh)
            for j in range(0, w, self.rw):
                gjs, gje = max(min(w-self.rw, j), 0), min(w, j+self.rw)
                crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                mask = torch.zeros(b, 1, h, w)
                mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                crop_masks.append(mask)
        crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
        crop_preds = []
        nz, bz = crop_imgs.size(0), 1
        
        # infernece each croped image
        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i+bz)
            input_data = {self.input_tensor_name: crop_imgs[gs:gt]}
            out = self.engine.process(self.graph_name, input_data)
            crop_pred = torch.tensor(out[self.output_tensor_name])
            crop_preds.append(crop_pred)
        crop_preds = torch.cat(crop_preds, dim=0)

        # calculate the predicted density map
        idx = 0
        pred_map = torch.zeros(b, 1, h, w)
        for i in range(0, h, self.rh):
            gis, gie = max(min(h-self.rh, i), 0), min(h, i+self.rh)
            for j in range(0, w, self.rw):
                gjs, gje = max(min(w-self.rw, j), 0), min(w, j+self.rw)
                pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                idx += 1

        # calculate the predicted crowd count
        mask = crop_masks.sum(dim=0).unsqueeze(0)
        pred_map = pred_map / mask

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
        pred = np.sum(pred_map) / self.LOG_PARA

        return pred

    def test_all(self):
        """test all images and save results"""
        with open(args.result, 'w') as out:
            with open(os.path.join('TestDataSet.txt')) as f:
                for line in f.readlines():
                    img_id = line.split()[0]
                    img_pth = os.path.join(args.data, 'img_' + img_id + '.jpg')
                    time_start = time.time()
                    pred = self.test_one(img_pth)
                    time_cost = time.time() - time_start
                    print('{} {:.4f} {:.9f}'.format(img_id, pred, time_cost), file=out)
                    print('{} {:.4f}'.format(img_id, pred))
                    print('time cost {}'.format(time_cost))
         
tester = Tester()
tester.test_all()
