import numpy as np
import os
import subprocess
import argparse


# Parse optional command-line arguments
parser = argparse.ArgumentParser(description="Run EMG2QWERTY training or inference")
parser.add_argument('--user', type=str, default='user0', help='User id [user0 to user7 or generic]')
parser.add_argument('--model', type=str, default='S4_large', help='Model name: S4_large, TC_S4_medium, TC_S4_small')
parser.add_argument('--root_dataset', type=str, default='./data', help='Path to dataset root')
parser.add_argument('--train', action='store_true', help='Set this flag to run training')

args = parser.parse_args()

user = args.user
model = args.model
train = args.train
dataset_path = args.root_dataset


if model not in ['S4_large', 'TC_S4_medium', 'TC_S4_small' ]:
    print(f'Model should be one of ["S4_large", "TC_S4_medium", "TC_S4_small"], but got {model}')
    exit()
if not os.path.exists(dataset_path):
    print('Please provide a valid dataset root path via --root_dataset')
    exit()
if user not in ['user0', 'user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'generic']:
    print(f'User should be one of ["user0" to "user7" or "generic"], but got {user}')
    exit()

cmd = f'python3 -m emg2qwerty.train user={user}'

if model == 'S4_large':
        
    model_HF = 'model_HF=identity'
    model_LF = 'model_LF=LF_S4_pooled_model'

    model_norm = 'model.norm=layernorm'
    model_mlp_feat = 'model.mlp_features="[60]"'
    model_input_mean_pooling = 'model.input_mean_pooling=4'

    model_dimension = 'model_LF.d_model=256'

elif model == 'TC_S4_small':

    model_HF = 'model_HF=HF_tempConv_model'
    model_LF = 'model_LF=LF_S4_model'

    model_norm = 'model.norm=layernorm'
    model_mlp_feat = 'model.mlp_features="[36]"'
    model_input_mean_pooling = 'model.input_mean_pooling=4'

    model_dimension = 'model_HF.out_channels="[96,192,150]"'

elif model == 'TC_S4_medium':

    model_HF = 'model_HF=HF_tempConv_model'
    model_LF = 'model_LF=LF_S4_model'

    model_norm = 'model.norm=layernorm'
    model_mlp_feat = 'model.mlp_features="[72]"'
    model_input_mean_pooling = 'model.input_mean_pooling=4'

    model_dimension = 'model_HF.out_channels="[192,384,150]"'


if train:

    log_str = 'training'
    ckp_string = f'checkpoint=trained_checkpoints/{model}_generic/generic_checkpoint.ckpt'
    wandb_name = f'wandb.name={model}_{user}'

    cmd = f'{cmd} {model_HF} {model_LF} {model_norm} {model_mlp_feat} {model_input_mean_pooling} {model_dimension} {ckp_string} {wandb_name} dataset.root={dataset_path}'

else:
    if user=='generic':
        ckp_string = f'checkpoint=trained_checkpoints/{model}_generic/generic_checkpoint.ckpt'
    else:
        ckp_string = f'checkpoint=trained_checkpoints/{model}_ft/{user}_checkpoint.ckpt'
    
    cmd = f'{cmd} {ckp_string} {model_HF} {model_LF} train=false dataset.root={dataset_path} wandb.mode=disabled'
    log_str = 'inference'

if os.path.exists('run_logs') is False:
    os.makedirs('run_logs')


log_file = f'run_logs/log_{user}_{model}_{log_str}.txt'
with open(log_file, 'w') as f:
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end='')  
        f.write(line)      

    process.wait() 

print('Ended Job')

