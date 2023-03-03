import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from .layers import Norm, EncoderLayer, DecoderLayer, get_clones
from .attention import PositionalEncoder
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout, d_ff):
        super().__init__()
        self.N_layers = N_layers
        self.pe = PositionalEncoder(d_model, window, device)
        self.layers = get_clones(EncoderLayer(d_model, attention, device, dropout, d_ff), N_layers)
        self.norm = Norm(d_model, device)
    def forward(self, src):
        # En la variable x se almacena src, el batch de datos de entrada en cada iteración,
        # pero con el positional encoding aplicado mediante una suma
        x = self.pe(src)
        # Seguidamente se pasa x por las N capas del encoder (que son todas idénticas)
        for i in range(self.N_layers):
            x = self.layers[i](x)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout, d_ff):
        super().__init__()
        self.N_layers = N_layers
        self.pe = PositionalEncoder(d_model, window, device)
        self.layers = get_clones(DecoderLayer(d_model, attention, device, dropout, d_ff), N_layers)
        self.norm = Norm(d_model, device)
    def forward(self, trg, e_outputs, mask):
        # x = self.embed(trg)
        x = self.pe(trg)
        for i in range(self.N_layers):
            x = self.layers[i](x, e_outputs, mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout=0.1, d_ff = 512):
        super().__init__()
        self.encoder = Encoder(d_model, N_layers, attention, window, device, dropout, d_ff)
        self.decoder = Decoder(d_model, N_layers, attention, window, device, dropout, d_ff)
        self.out = nn.Linear(d_model*(window-1), d_model).to(torch.device(device))
        self.window = window
        self.device = device
    def nopeak_mask(self, size):
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        np_mask = Variable(torch.from_numpy(np_mask) == 0).to(torch.device(self.device))
        return np_mask
    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # this code is very important! It initialises the parameters with a
                # range of values that stops the signal fading or getting too big.
                # See https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization 
                # for a mathematical explanation.
    def forward(self, src, trg):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, self.nopeak_mask(trg.size(1)))
        output = self.out(d_output.view(d_output.size(0), -1))
        # No se hace softmax a la salida porque no se buscan probabilidades, sino los valores predichos
        return output
    
    def train_model(self, data_loader, epochs=10, print_every=1, return_evo=False):
        optim = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.train()    
        total_loss = 0
        loss_evo = []
        for epoch in range(epochs):
            print(f'\nEpoch: {epoch+1} of {epochs}')
            with tqdm(total = data_loader.__len__()) as pbar:
                for i, batch in enumerate(data_loader):   
                    pbar.update(1)
                    # Al decoder se le pasa como entrada el batch de datos de entrada menos el último dato,
                    # que es el que se quiere predecir
                    d_input = batch[:, :-1]
                    trg_output = batch[:, -1]
                    preds = self.forward(batch, d_input)
                    optim.zero_grad()
                    criterion = nn.MSELoss() 
                    loss = criterion(preds, trg_output)           
                    loss.backward()
                    optim.step()
                    
                    total_loss += loss.item()
                    loss_evo.append(loss.item())

                    if (i + 1) % print_every == 0:
                        loss_avg = total_loss / print_every
                        pbar.set_postfix({'Last mean loss': loss_avg})
                        total_loss = 0
        if return_evo:
            return loss_evo

    def detect(self, data_loader):
        self.eval()    
        ano_scores = []
        criterion = nn.MSELoss(reduction = 'none')
        with tqdm(total = data_loader.__len__()) as pbar:
            for batch in data_loader:  
                pbar.update(1)
                d_input = batch[:, :-1]
                trg_output = batch[:, -1]
                preds = self.forward(batch, d_input)
                loss = criterion(preds, trg_output)
                ano_scores = ano_scores + loss.mean(dim=-1).tolist()
        return ano_scores