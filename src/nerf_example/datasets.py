import torch
import numpy as np

from utils import get_rays


def create_dataset(path="tiny_nerf_data.npz", hold_out=False):
    data = np.load('tiny_nerf_data.npz')
    images = torch.Tensor(data['images'][..., :3]) #ignore transparency layer
    poses = torch.Tensor(data['poses'])
    focal = data['focal']
    H, W = images.shape[1:3]
    hwf = [H, W, focal]
    if hold_out:
        testimg, testpose = images[101], poses[101]
        images = images[:100,...,:3]
        poses = poses[:100, ...]
        training_dataset = TinyLegoDataset(images, poses, hwf)
        test_dataset = TinyLegoDataset(testimg[None, ...], testpose[None, ...], hwf)
        return training_dataset, test_dataset

    else:
        training_dataset = TinyLegoDataset(images, poses, hwf)
        return training_dataset

class TinyLegoDataset(torch.utils.data.Dataset):

    def __init__(self, images, poses, hwf):
        super().__init__()
        [H, W, focal] = hwf
        self.images = images
        rays = [tuple(get_rays(H, W, focal, poses[i])) for i in range(poses.shape[0])]
        #self.rays_o, self.rays_d = get_rays(H, W, focal, poses)
        self.rays_o = torch.stack([i[0] for i in rays], dim=0)
        self.rays_d = torch.stack([i[1] for i in rays], dim=0)

    def __len__(self):
        return self.rays_o.shape[0]

    def __getitem__(self, index):
        return self.rays_o[index, ...], self.rays_d[index, ...], self.images[index, ...]