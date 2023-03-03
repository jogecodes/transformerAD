import torch
import pandas as pd
import utils
from pickle import dump
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

test_batch_size = 256

dataset = 'WUSTL-IIoT'
model_name = 'model_1'
model_path = f'models/{dataset}/{model_name}/'
model, data_info = utils.load_model(model_path, device)
assert(data_info['dataset'] == dataset)

dos_data = pd.read_pickle(f'data/{dataset}/dos_{dataset}.pkl')
test_dos = utils.prepare_data(dos_data, dataset, model_path)
assert(data_info['data_columns'] == list(test_dos.columns))

recon_data = pd.read_pickle(f'data/{dataset}/recon_{dataset}.pkl')
test_recon = utils.prepare_data(recon_data, dataset, model_path)
assert(data_info['data_columns'] == list(test_recon.columns))

comm_data = pd.read_pickle(f'data/{dataset}/comm_{dataset}.pkl')
test_comm = utils.prepare_data(comm_data, dataset, model_path)
assert(data_info['data_columns'] == list(test_comm.columns))

if __name__ == '__main__':

    dos_loader = utils.get_dataLoader(test_dos, data_info['window_size'], device, batch_size = test_batch_size)
    dos_scores = model.detect(dos_loader)
    recon_loader = utils.get_dataLoader(test_recon, data_info['window_size'], device, batch_size = test_batch_size)
    recon_scores = model.detect(recon_loader)
    comm_loader = utils.get_dataLoader(test_comm, data_info['window_size'], device, batch_size = test_batch_size)
    comm_scores = model.detect(comm_loader)
    
    output_dir = f"output/{dataset}/output_{model_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dump(dos_scores, open(f'{output_dir}dos_scores.pkl', 'wb'))
    dump(recon_scores, open(f'{output_dir}recon_scores.pkl', 'wb'))
    dump(comm_scores, open(f'{output_dir}comm_scores.pkl', 'wb'))
    dump(data_info, open(f'{output_dir}data_info.pkl', 'wb'))