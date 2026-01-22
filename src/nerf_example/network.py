import torch
import numpy as np
from torch import nn
from collections import OrderedDict

class TinyNet(nn.Module):

    def __init__(self, L_x=6, L_d=4, real=False):
        super().__init__()
        self.net_1 = nn.Sequential(
            OrderedDict(
                    [
                        ("linear1", nn.Linear(36 + 3 , 256)),
                        ("relu1", nn.ReLU()),
                        ("linear2", nn.Linear(256, 256)),
                        ("relu2", nn.ReLU()),
                        ("linear3", nn.Linear(256, 256)),
                        ("relu3", nn.ReLU()),
                        ("linear4", nn.Linear(256, 256)),
                        ("relu4", nn.ReLU()),
                    ]
                )
            )

        self.net_2 = nn.Sequential(
            OrderedDict(
                    [
                        ("linear5", nn.Linear(256 + 36 + 3, 256)),
                        ("relu5", nn.ReLU()),
                        ("linear6", nn.Linear(256, 256)),
                        ("relu6", nn.ReLU()),
                        ("linear7", nn.Linear(256, 256)),
                        ("relu7", nn.ReLU()),
                        ("linear8", nn.Linear(256, 256)),
                        ("relu8", nn.ReLU()),
                    ]
                )
            )
        
        
        self.net_3 = nn.Linear(256, 4)


    def gamma(self, p, L):
        """Positional encoding
        
        Paramters:
            p : [batch_size, N_c, 3] 
            l (int):
        """
        #assert len(p.shape) == 3, f"input to gamma is of shape {p.shape}"
        powers = torch.full(size=(L,), fill_value=2.0) ** torch.arange(L)
        powers = powers.to(p)
        acc = np.pi * p[..., None] * powers[None, None, None, None, None, :] # [batch_size, N_c, 3, L]
        #acc = acc.flatten(start_dim=-2, end_dim=-1)
        #acc = torch.flatten(acc, start_dim=-2, end_dim=-1)
        #acc = torch.reshape(acc, shape=(acc.shape[0] * acc.shape[1] * acc.shape[2] * 3 , 3))
        acc = torch.reshape(acc, shape=(*p.shape[:-1], 3 * L))

        #out_val = torch.empty((*p.shape[:2], 3 * 2 * L), device=device)
        out_val = torch.empty((*acc.shape[:-1], 3 * 2 * L))
        out_val = out_val.to(p)
        out_val[..., 0::2] = torch.sin(acc)
        out_val[..., 1::2] = torch.cos(acc)
        #try adding the regular
        out_val = torch.concat((p, out_val), dim=-1)
        return out_val


    def forward(self, x, d):
        gamma_x = self.gamma(x, L=6)
        
        output = self.net_1(gamma_x)
        output = torch.concat((output, gamma_x), dim=-1)
        output = self.net_2(output)
        output = self.net_3(output)
        
        color = output[..., :3]
        color = torch.nn.functional.sigmoid(color)
        density = output[..., 3]
        density = torch.nn.functional.relu(density)

        return color, density
