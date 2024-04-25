from model.encoder import ImageEncoder
from model.decoder import ImageDecoder
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms


# '''model hyperparameter'''
# latent_dim = 4

# '''train hyperparameters'''
# num_epochs = 1000
# batch_size = 64
# lr = 5e-4
# weight_decay = 5e-4
# gamma = 0.5
# device = 'cuda'
# # log_int = 1000

train_config = {
    'lr'            : 1e-3,
    'batch_size'    : 64,
    'num_epoch'     : 100,
    'weight_decay'  : 5e-4,
    'device'        : 'cpu',
    'log_int'       : 5
}

class BallsDataLoader:

    def __init__(self, train_x, device):
        '''
        Args:
        - train_x : [ B, H, W, C ]
        '''
        x = torch.tensor(train_x).to(device).float()
        self.data = x.permute(0, 3, 1, 2) # [ B, C, H, W ]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        data_transform =  transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return data_transform(x)
    

def train(train_x : np.array, encoder : ImageEncoder, decoder : ImageDecoder, train_config : dict):

    # parameters
    lr              = train_config['lr']
    batch_size      = train_config['batch_size']
    num_epoch       = train_config['num_epoch']
    weight_decay    = train_config['weight_decay']
    device          = train_config['device']
    log_int         = train_config['log_int']

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    opt = optim.Adam([{'params': filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(decoder.parameters()) )}, ],
                    lr= lr,
                    weight_decay = weight_decay)
    dataloader      = BallsDataLoader(train_x, device)
    train_dataset   = DataLoader(dataloader, batch_size=batch_size, shuffle=True)
    train_losses    = []
    encoder.train()
    decoder.train()
    for epoch in range(num_epoch):
        try:
            for x in train_dataset: # [ batch_size, 64, 64, 3 ]
                opt.zero_grad()
                z_pred  = encoder(x)
                x_pred  = decoder(z_pred)
                loss = torch.mean(((x - x_pred)**2))
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
            if epoch % log_int == 0:
                print(f'Epoch: {epoch}\t Loss: {loss.item()}')
        except KeyboardInterrupt:
            break
    return train_losses

def reconstruct(test_x : np.array, encoder : ImageEncoder, decoder : ImageDecoder, train_config : dict):
    '''
    Args:
    - test_x    : [ B, H, W, C ] raw input
    Returns:
    - x, x_pred : [ B, C, H, W ] transformed and unnormalized inputs
    '''
    def Unnormalize(img : torch.Tensor, device):
        '''
        unnormalize output tensor for visualization
        Args:
        - img    : [ B, C, H, W ]
        Returns:
        - un_img : [ B, H, W, C ] np.array
        '''
        B, _, H, W = img.shape
        means   = torch.tensor([ 0.485, 0.456, 0.406 ]).to(device)      # [ C ]
        stds    = torch.tensor([ 0.229, 0.224, 0.225 ]).to(device)      # [ C ]
        means   = means.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W)   # [ B, C, H, W ]
        stds    = stds.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W)    # [ B, C, H, W ]
        un_img  = img * stds + means # [ B, C, H, W ]
        un_img  = un_img.permute(0, 2, 3, 1) # [ B, H, W, C ]
        un_img  = un_img.cpu().long().detach().numpy()
        return un_img
    
    device          = train_config['device']
    dataloader      = BallsDataLoader(test_x, device = device)
    test_dataset    = DataLoader(dataloader, batch_size = len(test_x), shuffle=False)
    encoder.train()
    decoder.train()
    for x in test_dataset:
        z_pred  = encoder(x)
        x_pred  = decoder(z_pred)
    x      = Unnormalize(x, device)
    x_pred = Unnormalize(x_pred, device)
    return x, x_pred, z_pred
    
