import torch

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], dim=-1)
    #THIS IS MATMUL
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = torch.broadcast_to(c2w[:3,-1], rays_d.shape)
    return rays_o, rays_d


def exclusive_cum_prod(A):
    """exclusive cumlative product
        cumlative product except the value at an index 
        excludes the value from the cumlative prod
    """
    return torch.concat((torch.ones(size=(*A.shape[:-1], 1)).to(A), torch.cumprod(A[..., :-1], dim=-1)), dim=-1)