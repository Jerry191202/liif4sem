import argparse
import os
from PIL import Image
from tqdm import trange

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--numsamp', default=10)
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    print(img.shape)

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    preds = []
    model.train()
    for i in trange(args.numsamp):
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
                               coord.unsqueeze(0),
                               cell.unsqueeze(0),
                               bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0,
                                                                    1).cpu()
        preds.append(pred.clone().detach())
    preds = torch.stack(preds)
    uncer = preds.var(dim=0)
    print(uncer.shape, uncer.min(), uncer.max())
    transforms.ToPILImage()(uncer*20000).save(args.output)