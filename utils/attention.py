import torch
import math
from torch import nn

# Clase para aplicar codificación posicional sobre un dato dado
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, window, device, encoding_length = 10000):
        super().__init__()
        # Dimensión de los datos de entrada, es decir, el número de atributos que lo describen
        self.d_model = d_model
        # Tamaño de la ventana deslizante bajo la que se definen las secuencias
        self.window = window
        # Creación de la matriz de codificación posicional, pe, cuyos elementos dependen de la posición y la dimensión
        pe = torch.zeros(window, d_model).to(torch.device(device))
        for pos in range(window):
            # La codificación posicional se aplica a cada dimensión de los datos de entrada
            # Se aplica una codificación seno y coseno, con una frecuencia que depende de la dimensión
            # El resultado es un vector de dimensión d_model (la misma dimensión que los datos de entrada)
            # que es determinista para cada posición de la secuencia
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (encoding_length ** ((2 * i)/d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (encoding_length ** ((2 * (i + 1))/d_model))) 
        # Se formatea como tensor para poder sumarse a los datos de entrada
        pe = pe.unsqueeze(0)
        self.pe = pe
    
    def forward(self, x):
        # Se suma la codificación posicional a los datos de entrada
        # Sólo se toman las posiciones existentes en la secuencia de entrada
        x = x + self.pe[:, :x.size(1), :]
        return x
    
def scaled_DPattention(q, k, v, d_attention, mask=None, dropout=None):
    # Scaled dot-product attention, definida en "Attention is all you need" (Vaswani et al., 2017)
    # Al trasponer k, su producto con q proporciona una matriz cuadrada de orden igual al número de 
    # elementos de la secuencia para cada input del batch
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_attention)
    # Aplica una máscara, si es necesario, haciendo infinitamente baja la atención en los elementos
    # que no se deben considerar
    if mask is not None:
        # En el código original se hacía una operación de unsqueezing, pero la he quitado
        # para que me cuadren las dimensiones.
        # mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    # Normalización de los scores de acuerdo a una capa softmax
    scores = nn.functional.softmax(scores, dim=-1)
    # Aplica una capa de dropout, si es necesario
    if dropout is not None:
        scores = dropout(scores)
    # Multiplica los scores por los valores para tasarlos por la atención
    output = torch.matmul(scores, v)
    return output

class GeneralAttention(nn.Module):
    def __init__(self, d_model, device, dropout = 0.1):
        # La atención se inicializa como un modelo de NN
        super().__init__()
        # Dimensión de los datos de entrada
        self.d_model = d_model
        # Definición de la capa de dropout
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
    
    def forward(self, q, k, v, mask=None):
        # Se calcula la atención usando la función de atención producto interno reescalada
        scores = scaled_DPattention(q, k, v, self.d_model, mask, self.dropout)    
        # Se aplica la capa lineal de salida y se devuelve la atención calculada
        output = scores
        return output
    
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, device, dropout = 0.1):
        # La atención se inicializa como un modelo de NN
        super().__init__()
        # Dimensión de los datos de entrada
        self.d_model = d_model
        # Definición de las capas lineales para la atención, que aplican a
        # los datos de entrada una transformación lineal del tipo y = Ax + b
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        # Definición de la capa de dropout
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        # Definición de la capa lineal de salida
        self.out = nn.Linear(d_model, d_model).to(torch.device(device))
    
    def forward(self, q, k, v, mask=None):
        # Se calcula la atención usando la función de atención producto interno reescalada
        q_linear = self.q_linear(q)
        k_linear = self.k_linear(k)
        v_linear = self.v_linear(v)
        scores = scaled_DPattention(q_linear, k_linear, v_linear, self.d_model, mask, self.dropout)    
        # Se aplica la capa lineal de salida y se devuelve la atención calculada
        output = self.out(scores)
        return output

# TODO: temrinar de programar, que por ahora no funciona
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, device, dropout = 0.1):
        # La atención multicabeza se inicializa como un modelo de NN
        super().__init__()
        # Dimensión de los datos de entrada
        self.d_model = d_model
        # Número de cabezas paralelas para la atención
        self.N_heads = heads
        # Dimensión de la vista por cada cabeza, calculada como la mínima necesaria
        # para reconstruir sin pérdida la dimensionalidad de los datos de entrada
        self.d_head = (d_model // heads)+1
        # Definición de las capas lineales para la atención, que aplican a
        # los datos de entrada una transformación lineal del tipo y = Ax + b
        # Pasan de la dimensión original de los datos de entrada a la dimensión
        # total de las vistas para cada cabeza
        self.q_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        self.v_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        self.k_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        # Definición de la capa de dropout
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        # Definición de la capa lineal de salida
        self.out = nn.Linear(self.N_heads * self.d_head, d_model).to(torch.device(device))
    
    def forward(self, q, k, v, mask=None):
        # Tamaño del batch utilizado
        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)

        # Se transforman los datos y se les aplica una vista, que reduce
        # la dimensionalidad para evaluarlos en cada cabeza
        # Llegados a este punto ya no tienen que ser la misma cosa porque
        # cada uno ha pasado por un transformador lineal distinto
        q_lin = self.q_linear(q)
        k_lin = self.k_linear(k)
        v_lin = self.v_linear(v)

        q_view = q_lin.view(batch_size, q_len, self.N_heads, self.d_head)
        k_view = k_lin.view(batch_size, k_len, self.N_heads, self.d_head)
        v_view = v_lin.view(batch_size, v_len, self.N_heads, self.d_head)
        
        # Se transponen q, k y v para que tengan dimensión
        # batch_size * N_heads * seq_len * d_head
        q_heads = q_view.transpose(1,2)
        k_heads = k_view.transpose(1,2)
        v_heads = v_view.transpose(1,2)
        
        # calculate attention using attention function
        scores = scaled_DPattention(q_heads, k_heads, v_heads, self.d_head, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.N_heads * self.d_head)
        
        output = self.out(concat)
    
        return output