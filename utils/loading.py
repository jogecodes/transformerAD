import torch
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from .transformer import Transformer

# Estructura de datos de pyTorch, es necesario para poder iterar sobre los datos
class seq_Loader(torch.utils.data.Dataset):
    # Define la inicialización a partir de la del padre
    def __init__(self, init_data, window, device):
        super(seq_Loader, self).__init__() 
        # Inicializa la clase padre
        self.dataset = init_data
        # Tamaño de la ventana deslizante bajo la que se definen las secuencias
        self.window = window
        # Dispositivo en el que se guarda el tensor
        self.device = device
    
    def __len__(self):
        # Número de secuencias que se pueden generar, teniendo en cuenta el tamaño de la ventana
        return (len(self.dataset)-(self.window-1))
    
    def __getitem__(self, idx):
        # Obtiene una secuencia de datos desde idx y con tamaño window
        row = self.dataset.iloc[idx:idx+self.window]
        # Convierte los datos a un tensor de pyTorch
        data = torch.from_numpy(np.array(row)).float().to(torch.device(self.device))
        return data
    
def get_dataLoader(data, window_size, device, batch_size = 1):
    train_set = seq_Loader(data, window_size, device)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=os.cpu_count(), # Uses all the CPU cores, but doesn't work on Jupyter running on Windows
        num_workers=0,
        drop_last=False # Ignores the last batch when it is not complete
    )
    return train_loader

def prepare_data(data, dataset, scaling = 'gaussian', columns = 'normal'):
    if dataset == 'UNSW-NB15':
        pred_columns = ['dur', 'sbytes',
        'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'sload', 'dload',
        'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
        'dmeansz', 'sjit', 'djit', 'stime',
        'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat']
        extra_columns = ['is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
            'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'trans_depth', 'res_bdy_len']
        min_columns = ['dur', 'sbytes', 'dbytes', 'sload', 'dload']
        if columns == 'normal':
            columned_data = data[pred_columns]
        elif columns == 'minimal':
            columned_data = data[min_columns]
        elif columns == 'extended':
            columned_data = data[pred_columns+extra_columns]
        else:
            columned_data = data
    elif dataset == 'WUSTL-IIoT':
        pred_columns = ['Mean', 'Sport', 'Dport',
       'SrcPkts', 'DstPkts', 'TotPkts', 'DstBytes', 'SrcBytes', 'TotBytes',
       'SrcLoad', 'DstLoad', 'Load', 'SrcRate', 'DstRate', 'Rate', 'SrcLoss',
       'DstLoss', 'Loss', 'pLoss', 'SrcJitter', 'DstJitter', 'SIntPkt',
       'DIntPkt', 'Proto', 'Dur', 'TcpRtt', 'IdleTime', 'Sum', 'Min', 'Max',
       'sDSb', 'sTtl', 'dTtl', 'sIpId', 'dIpId', 'SAppBytes', 'DAppBytes',
       'TotAppByte', 'SynAck', 'RunTime', 'sTos', 'SrcJitAct', 'DstJitAct']
        columned_data = data[pred_columns]
    else:
        raise ValueError('Dataset not implemented')
    if scaling == 'gaussian':
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(columned_data.values), columns=columned_data.columns, index=columned_data.index)
        return scaled_data, scaler
    elif os.path.isfile(scaling+'scaler.pkl'):
        scaler = pickle.load(open(scaling+'scaler.pkl', 'rb'))
        scaled_data = pd.DataFrame(scaler.transform(columned_data.values), columns=columned_data.columns, index=columned_data.index)
        return scaled_data
    else:
        raise ValueError('Scaling method not implemented')

def load_model(model_path, device, return_info = True):
    model_info = pickle.load(open(model_path+'model_info.pkl', 'rb'))
    data_columns = model_info['data_columns']
    d_model = len(data_columns)
    attention = model_info['attention']
    if attention[:10] == 'multi-head':
        attention = int(attention.split(' ')[1])
    model = Transformer(d_model, model_info['N_layers'], attention, model_info['window_size'], device, model_info['dropout'], model_info['ff_neurons'])
    model.load_state_dict(torch.load(model_path+'model_state.pt', map_location=torch.device(device)))
    model_info = {
        'data_columns': data_columns,
        'window_size': model_info['window_size'],
        'batch_size': model_info['batch_size'],
        'dataset': model_info['dataset'],
        'dropout' : model_info['dropout'],
        'ff_neurons' : model_info['ff_neurons']
    }
    if return_info:
        return model, model_info
    else:
        return model