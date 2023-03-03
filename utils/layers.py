import torch
import copy
from torch.autograd import Variable
from torch import nn
from .attention import SingleHeadAttention, MultiHeadAttention, GeneralAttention

# Esta es el último módulo por el que pasan los datos en el decoder y encoder 
# Cnstituye una red profunda normal, con un mogollón de arcos de conexión
class FeedForward(nn.Module):
    # La dimensión de las capas lineales del módulo ff (d_ff) es un hiperparámetro
    # que se puede ajustar, si bien se ha fijado a 512 neuronas por defecto
    def __init__(self, d_model, device, dropout, d_ff):
        super().__init__() 
        # Se definen las capas lineales y de dropout
        self.linear_1 = nn.Linear(d_model, d_ff).to(torch.device(device))
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        self.linear_2 = nn.Linear(d_ff, d_model).to(torch.device(device))
        self.device = device
    def forward(self, x):
        # Las capas se conectan mediante una función de activación ReLU
        x = self.dropout(nn.functional.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

# Capa de normalización de batch
class Norm(nn.Module):
    def __init__(self, d_model, device, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # Se definen los parámetros de normalización, alpha y bias, como variables de aprendizaje
        self.alpha = nn.Parameter(torch.ones(self.size).to(torch.device(device)))
        self.bias = nn.Parameter(torch.zeros(self.size).to(torch.device(device)))
        self.eps = eps
    def forward(self, x):
        # El input x se normaliza con respecto a las dimensiones 0 y 1, que corresponden al batch y a la secuencia,
        # de forma que para cada atributo quede calculada su media en la muestra dada por la secuencia
        x_mean = x.mean(dim=(0,1), keepdim=True)
        x_std = x.std(dim=(0,1), keepdim=True)
        norm = self.alpha * (x - x_mean) / (x_std + self.eps) + self.bias
        return norm
    
# Construcción de un módulo (capa) encoder, que consta de una subcapa de atención y 
# una subcapa de feed-forward, también aplica normalización por batch entre cada subcapa
class EncoderLayer(nn.Module):
    def __init__(self, d_model, attention, device, dropout, d_ff):
        super().__init__()
        self.norm_1 = Norm(d_model, device).to(torch.device(device))
        self.norm_2 = Norm(d_model, device).to(torch.device(device))

        if attention == 'general':
            self.attn = GeneralAttention(d_model, device).to(torch.device(device))
        elif attention == 'single':
            self.attn = SingleHeadAttention(d_model, device).to(torch.device(device))
        elif isinstance(attention, int):
            heads = attention
            self.attn = MultiHeadAttention(heads, d_model, device).to(torch.device(device))
        else:
            raise ValueError('Attention type not recognized')
            
        self.ff = FeedForward(d_model, device, dropout, d_ff).to(torch.device(device))
        self.dropout_1 = nn.Dropout(dropout).to(torch.device(device))
        self.dropout_2 = nn.Dropout(dropout).to(torch.device(device))
        
    def forward(self, x):
        # Se normalizan los inputs
        x_norm_1 = self.norm_1(x)
        # Se calcula la atención sobre la entrada normalizada (que es q, k y v) y se aplica dropout
        # Nótese que la entrada x se suma a la salida de la capa de atención, formando una conexión residual
        x_dropout_1 = x + self.dropout_1(self.attn(x_norm_1, x_norm_1, x_norm_1))
        # Se renormalizan los inputs
        x_norm_2 = self.norm_2(x_dropout_1)
        # Se propaga sobre la entrada renormalizada y se aplica dropout
        # De nuevo, la entrada x se suma a la salida de la capa de atención
        x_dropout_2 = x_dropout_1 + self.dropout_2(self.ff(x_norm_2))
        return x_dropout_2
    
# Construcción de un módulo (capa) encoder, que consta de dos subcapas de atención y 
# una subcapa de feed-forward, también aplica normalización por batch entre cada subcapa
class DecoderLayer(nn.Module):
    def __init__(self, d_model, attention, device, dropout, d_ff):
        super().__init__()
        self.norm_1 = Norm(d_model, device).to(torch.device(device))
        self.norm_2 = Norm(d_model, device).to(torch.device(device))
        self.norm_3 = Norm(d_model, device).to(torch.device(device))
        
        self.dropout_1 = nn.Dropout(dropout).to(torch.device(device))
        self.dropout_2 = nn.Dropout(dropout).to(torch.device(device))
        self.dropout_3 = nn.Dropout(dropout).to(torch.device(device))
        
        if attention == 'general':
            self.attn_1 = GeneralAttention(d_model, device).to(torch.device(device))
            self.attn_2 = GeneralAttention(d_model, device).to(torch.device(device))
        elif attention == 'single':
            self.attn_1 = SingleHeadAttention(d_model, device).to(torch.device(device))
            self.attn_2 = SingleHeadAttention(d_model, device).to(torch.device(device))
        elif isinstance(attention, int):
            heads = attention
            self.attn_1 = MultiHeadAttention(heads, d_model, device).to(torch.device(device))
            self.attn_2 = MultiHeadAttention(heads, d_model, device).to(torch.device(device))
        else:
            raise ValueError('Attention type not recognized')

        self.ff = FeedForward(d_model, device, dropout, d_ff).to(torch.device(device))

    def forward(self, x, e_outputs, trg_mask):
        x2 = self.norm_1(x)
        # TODO: probar sin máscara
        # x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x = x + self.dropout_1(self.attn_1(x2, x2, x2))
        x2 = self.norm_2(x)
        # De acuerdo con "Attention is all you need", el input del decoder entra a esta atención
        # como q, y la salida del encoder como los valores v y las claves k.
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs)) 
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
    
# Función que genera múltiples copias del módulo que se indique
def get_clones(module, N_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N_layers)])