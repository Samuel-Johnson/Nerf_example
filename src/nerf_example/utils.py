import torch

def get_rays(H, W, focal, c2w):
    """
    Return a set of orgins and directions for camera rays for each pixel.
    
    Args:
        H (int): number of pixels along the verticle axis of the camera.
        W (int): number of pixels along the horizonal axis of the camera.
        focal (float): the focal ratio of the camera.
        c2w (torch.Tensor): the camera to world transformation matrix.
            size: (4, 4, N) #TODO: check these dimensions
    Results:
        torch.tensor: the ray origins.
        
        torch.tensor: the unit direction (magnitude of 1) of the rays.
    """
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], dim=-1)
    #THIS IS MATMUL
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = torch.broadcast_to(c2w[:3,-1], rays_d.shape)
    return rays_o, rays_d


def exclusive_cum_prod(A):
    """
    Exclusive cumlative product along the last dimension. same as cumlative product except the value at an index 
        is excluded excluded from the product. Similar to tensorflow's tf.math.cumprod(..., axis=-1, exclusive=True).

    Args:
        A (torch.Tensor): the tensor on which to proform the cumprod. Can be anysize.
    
    Returns:
        torch.Tensor: the cumlative product.
    """
    return torch.concat((torch.ones(size=(*A.shape[:-1], 1)).to(A), torch.cumprod(A[..., :-1], dim=-1)), dim=-1)