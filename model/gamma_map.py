import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class GammaMap(nn.Module):
    '''
    Following the original paper, we set gamme to a nonlinear map (for image based experiments)
    '''
    def __init__(self, latent_dim, device = 'cuda:0'):
        super(GammaMap, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.gamma = nn.Parameter(torch.zeros(4, latent_dim), requires_grad=True) # [ 4, latent_dim ]
    
    def forward(self, z : torch.tensor, y : torch.tensor):
        '''
        Args:
        - z : [ B, latent_dim ]
        - y : [ B, 2 ]
        '''
        indexs  = y[:, 0].long()                        # [ B ]
        gamma   = self.gamma[indexs]                    # [ B, latent_dim ]
        out     = torch.bmm(gamma.unsqueeze(1), z.unsqueeze(2)).squeeze()   # [ B ]
        return out

    def loss(self, z : torch.tensor, y : torch.tensor):
        indexs  = y[:, 0].long()                        # [ B ]
        y_hat   = torch.rand(self.latent_dim)[indexs].to(self.device)       # [ B ]
        out     = self.forward(z, y)
        loss    = torch.sum((out - y_hat)**2)           # scalar
        return loss
    
    def transform(self, z_pred):
        '''
        Args:
        - z_pred : [ B, latent_dim ]
        '''
        # return torch.mm(z_pred, self.gamma)     # [ B, latent_dim ]
        return torch.mm(self.gamma, z_pred.T).T   # [ B, 4 ]
    
    
class GammaLoader:

    def __init__(self, z_pred : torch.tensor, y : np.array):
        self.z_pred = z_pred
        self.y      = torch.tensor(y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.z_pred[idx], self.y[idx]

    
def train_gammamap(z_pred : torch.tensor, y : np.array, model : GammaMap, train_config : dict):
    '''
    Args:
    - z_pred : [ B, latent_dim ]
    '''
    # parameters
    lr              = train_config['lr']
    batch_size      = train_config['batch_size']
    num_epoch       = train_config['num_epoch']
    weight_decay    = train_config['weight_decay']
    device          = train_config['device']
    log_int         = train_config['log_int']

    model = model.to(device)
    opt = optim.Adam(model.parameters(),
                    lr= lr,
                    weight_decay = weight_decay)
    
    dl              = GammaLoader(z_pred, y)
    train_dataset   = DataLoader(dl, batch_size=batch_size, shuffle=True)
    train_losses    = []
    batch_losses    = []
    model.train()
    for epoch in range(num_epoch):
        try:
            for z, y in train_dataset:
                opt.zero_grad()
                loss = model.loss(z, y)
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
                batch_losses.append(loss.item())
            if epoch % log_int == 0:
                print(f'Epoch: {epoch}\t Loss: {np.mean(batch_losses)}')
                batch_losses = []
        except KeyboardInterrupt:
            break
    return train_losses
    
