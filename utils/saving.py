import torch
import os
import json
from pickle import dump

def save_model(model, info_dict, scaler, path = 'models/', loss = None):

    models_path = path
    dataset = info_dict['dataset']
    models_path = f'{models_path}{dataset}/'
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    arr = os.listdir(models_path)
    if len(arr) != 0:
        folder_nums = []
        for folder_name in arr:
            if folder_name.split("_")[0] == "model":
                folder_nums.append(int(folder_name.split("_")[1]))
        if len(folder_nums) == 0:
            new_folder_num = 1
        else:
            try:
                new_folder_num = max(folder_nums) + 1
            except:
                new_folder_num = 666
    else:
        new_folder_num = 1
    model_dir = f"{models_path}/model_{new_folder_num}"
    os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir+'/model_state.pt')

    dump(info_dict, open(f'{model_dir}/model_info.pkl', 'wb'))
    dump(scaler, open(f'{model_dir}/scaler.pkl', 'wb'))
    if loss is not None:
        dump(loss, open(f'{model_dir}/loss.pkl', 'wb'))

    return True