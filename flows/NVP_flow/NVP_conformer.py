import torch
from torch import nn

class RealNVP(nn.Module):
    def __init__(self, scaling_network, translation_network, mask_d, mask_da, mask_dat, base_dist, device):
        super(RealNVP, self).__init__()
        
        self.base_distribution = base_dist
        self.mask_d = nn.Parameter(mask_d, requires_grad=False)
        self.mask_da = nn.Parameter(mask_da, requires_grad=False)
        self.mask_dat = nn.Parameter(mask_dat, requires_grad=False)
        self.t = torch.nn.ModuleList([translation_network() for _ in range(len(mask_dat))]) #networks that scale the transformed variables
        self.s = torch.nn.ModuleList([scaling_network() for _ in range(len(mask_dat))])     #networks that translate the transformed variables
        self.device = device
        
    def pushforward(self, forward_input, mask):
        '''
        Input:
            - forward_input (torch.tensor) ([batch_size, 1, dim(previous) + dim(atom_rep) + dim(mol_rep) + 3]): Last dim is previous, atom_rep, mol_rep, latent space dim.
              this procedure.
        Output:
            - x (torch.tensor) ([batch_size, 1, dim(previous) + dim(atom_rep) + dim(mol_rep) + 3])
            - log_det_J (torch.tensor) (batch_size)
        '''
        log_det_J, x = forward_input.new_zeros(forward_input.shape[0]), forward_input
        for i in range(len(self.t)):
            if torch.any( 1- mask[i].type(torch.uint8)):
                x_ = x*mask[i] # Note that the coordinates with 0 are the ones that are transformed
                s = self.s[i](x_) * (1-mask[i])
                t = self.t[i](x_) * (1-mask[i])
                x = x_ + (1 - mask[i]) * (x * torch.exp(s) + t) # pushforward transformation z->x (latenspace to data)
                log_det_J += s.squeeze(dim=1).sum(dim=1)

        x = x[:, :, -3:]
        return x, log_det_J

    def pushback(self, back_input, mask):
        '''
        Input:
            - back_input (torch.tensor) ([batch_size, 1, dim(previous) + dim(atom_rep) + dim(mol_rep) + 3]): Last dim is previous, atom_rep, mol_rep, coordinates.
        Output:
            - z (torch.tensor) ([batch_size, 1, 3])
            - log_det_inv (torch.tensor) (batch_size)
        '''
        log_det_invJ, z = back_input.new_zeros(back_input.shape[0]), back_input
        for i in reversed(range(len(self.t))):
            if torch.any(1 - mask[i].type(torch.uint8)):
                z_ = mask[i] * z
                s = self.s[i](z_) * (1-mask[i])
                t = self.t[i](z_) * (1-mask[i])
                z = (1 - mask[i]) * (z - t) * torch.exp(-s) + z_ #pushback transformation x -> z (Data to latent space)
                log_det_invJ -= s.squeeze(dim=1).sum(dim=1)

        z = z[:, :, -3:]
        return z, log_det_invJ
    
    def log_prob(self, x):
        z, log_det_invJ = self.pushback(x)
        return self.base_distribution.log_prob(z) + log_det_invJ
