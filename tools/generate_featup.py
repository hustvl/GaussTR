# Refered to https://github.com/mhamilton723/FeatUp/blob/main/example_usage.ipynb
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from featup.util import norm
from PIL import Image
from tqdm import tqdm

image_dir = 'data/nuscenes/samples/'
save_dir = 'data/nuscenes_featup/'


def main():
    device = torch.device('cuda')
    upsampler = torch.hub.load(
        'mhamilton723/FeatUp', 'maskclip', use_norm=False).to(device)
    upsampler.eval()
    transform = T.Compose([T.Resize((432, 768)), T.ToTensor(), norm])

    for view_dir in os.listdir(image_dir):
        for image_name in tqdm(os.listdir(osp.join(image_dir, view_dir))):

            image_path = osp.join(image_dir, view_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                hr_feats = upsampler(image_tensor)

            save_path = osp.join(save_dir, image_name.split('.')[0])
            np.save(save_path, F.avg_pool2d(hr_feats, 16)[0].cpu().numpy())


if __name__ == '__main__':
    main()
