import torch
from nerf_example.utils import exclusive_cum_prod

def stratified_sampling(N, near, far, noise=False):
    """stratified sampling

    Args:
        N (int): number of a bins 
        near (torch.Tensor): near bound
            size: (batch_size, H, W)
        far (torch.Tensor): far bound. bound_near < bound_far for all elements
            size: (batch_size, H, W)

    Returns:
        torch.Tensor: ray lengths
            size: (batch_size, H, W, N)
    """
    batch_size = near.shape[0]
    t = torch.linspace(0.0, 1.0, N + 1)[:-1]
    t = t.to(near)
    #t = torch.stack([t for _ in range(batch_size)], dim=0)
    t = torch.broadcast_to(t[None, None, None, :] , size=(*near.shape, N))
    bound_diff = far - near
    t = bound_diff[..., None] * t + near[..., None]
    noi = torch.rand(size=t.shape)
    noi = noi.to(near)
    if noise:
        return t + (bound_diff / N)[..., None] * noi
    else:
        return t

def render(model, rays_o, rays_d, bounds, N_c=64 ):
    """

    Args:
        model (torch.nn.Module): the model to query points for color and density
        rays_o (torch.Tensor): ray origins
            size: (batch_size, H, W, 3)
        rays_d (torch.Tensor): ray directions
            size: (batch_size, H, W, 3)
        bounds: [near, far] each a tensor of shape [batch_size, H, W]
        N_c (int): the number of points to query along each ray

    Returns:
        torch.Tensor: 
            size:  (batch_size, H, W)
    """
    near = bounds[0]
    far = bounds[1]
    t = stratified_sampling(N_c, near, far, noise=True) #[batch_size, H, W, N_c]
    d = torch.broadcast_to(rays_d[..., None], size=(*rays_d.shape, N_c))
    #points
    x = rays_d[..., None, :] * t[..., None] + rays_o[..., None, :]
    
    colors, densities = model(x, d)

    C_hat, w_i, alpha_i = volume_render(t, colors, densities)

    return C_hat
    

def volume_render(t, c_i, sigma_i):
    """volume rendering routine

    Parameters:
        c_i (torch.Tensor): the ith rgb color vector with components on interval [-1.0, 1.0]
            size: (batch_size, N, 3)
        sigma_i (torch.Tensor): the ith density
            size: (batch_size, N, 3)
        t (torch.Tensor): the ith length of the ray passing through the volume. 
            These are in accending order of length.
            size: (batch_size, N)

    Returns:
        C_hat: the rendered rgb color vector.
            size: (batch_size, 3)
        w_i: weight of each color along the ray.
            this ends up being the product T_i * alpha_i
            size: (batch_size, N)
        alpha_i: the ith transparency.
            size: (batch_size, N)
    """

    #take the last distance be from infinity (which apparently is 1e10 ¯\_(ツ)_/¯)
    delta_i = torch.concat((t[..., 1:], torch.full(size=(*t.shape[:-1], 1), fill_value=1e10).to(t)), dim=-1) - t
    # [btach_size, N?]

    alpha_i = 1.0 - torch.exp(-sigma_i * delta_i) #[batch_size, H, W, N]

    #add 1e-10 for numerical stability
    
    w_i = alpha_i * exclusive_cum_prod(1.0 - alpha_i + 1e-10) #[batch_size, H, W, N]

    #TODO: can this be a matmul? 
    # C_hat = w_i @ c_i.t()
    #I think you can use torch.nn.Function.Linear lol
    C_hat = torch.sum(w_i[..., None] * c_i, dim=-2) #[batch_size, H, W, N]
    
    return C_hat, w_i, alpha_i