import torch
import pandas as pd
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

dataset = 'WUSTL-IIoT'
normal_data = pd.read_pickle('data/'+dataset+'/train_'+dataset+'.pkl')
train_data, train_scaler = utils.prepare_data(normal_data, dataset)

d_model = int(len(train_data.columns))

attention = 'single'

N_layers = 6
window_size = 50
batch_size = 1024
epochs = 25
dropout = 0
ff_neurons = 512

if __name__ == '__main__':

    train_loader = utils.get_dataLoader(train_data, window_size, device, batch_size = batch_size)

    model = utils.Transformer(d_model, N_layers, attention, window_size, device, dropout, ff_neurons)
    model.initialize()
    model.train_model(train_loader, epochs=epochs, print_every=1)

    if isinstance(attention, int):
        attention = f"multi-head {attention}"

    info_dict = {
        'data_columns' : list(train_data.columns),
        'attention' : attention,
        'N_layers' : N_layers,
        'window_size' : window_size,
        'batch_size' : batch_size,
        'model_info' : str(model),
        'epochs' : epochs,
        'dataset' : dataset,
        'dropout' : dropout,
        'ff_neurons' : ff_neurons
    }

    utils.save_model(model, info_dict, train_scaler)