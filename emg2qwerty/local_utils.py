import numpy as np
import matplotlib.pyplot as plt
import re


TARGET_TO_MODEL_dict = {

        'emg2qwerty.modules.GRULayer': 'GRU',
        'emg2qwerty.modules.S4Model':  'S4',
        'emg2qwerty.modules.SJ_SNN':   'SJ_SNN',
        'emg2qwerty.modules.TemporalConvolution': 'TC',

        'identity': 'identity'

}


def update_config(cfg, model_HF, model_LF):

    if cfg == None:
        return None

    duplicate_model_keys = ['non_linearity', 'in_features', 'mlp_features', 'pooling', 'offsets', 'sparsity_RotInvMLP_after_non_linearity', 'norm', 'input_mean_pooling' ]
    for key in duplicate_model_keys:
        if cfg['checkpoint']==None:
            if key in cfg:
                assert cfg[key] == cfg[f'model/{key}'], f'{key} and model/{key} should be the same but are: {cfg[key]} and {cfg[f"model/{key}"]}'
                cfg.pop(key)

    if 'model/input_mean_pooling' not in cfg:
        cfg['model/input_mean_pooling'] = 1

    if 'optimizer/weight_decay' not in cfg:
        cfg['optimizer/weight_decay'] = 0.0

    if 'monitor_metric2' not in cfg:
        cfg['monitor_metric2'] = 'val_continuous/CER'
    if 'monitor_mode2' not in cfg:
        cfg['monitor_mode2'] = 'min'
    if 'autoTestOnly' not in cfg:
        cfg['autoTestOnly'] = ''
    
    if 'starting_seed' not in cfg:
        cfg['starting_seed'] = cfg['seed']
    else:
        if cfg['checkpoint'] is None:
            assert cfg['starting_seed'] == cfg['seed'], 'starting_seed should be the same as seed (see local_utils.py)'

    if model_LF == 'emg2qwerty.modules.S4Model':
        if 'model_LF/activation' not in cfg:
            cfg['model_LF/activation'] = 'relu'
        if 'model_LF/d_latent_space' not in cfg:
            cfg['model_LF/d_latent_space'] = 64
        if 'model_LF/reg_timescale_mode' not in cfg:
            cfg['model_LF/reg_timescale_mode'] = 'real'
        if 'model_LF/reg_timescale' not in cfg:
            cfg['model_LF/reg_timescale'] = 0.
        if 'model_LF/dropout_tie' not in cfg:
            cfg['model_LF/dropout_tie'] = 'True'
        if 'model_LF/delta_pre_activation' not in cfg:
            cfg['model_LF/delta_pre_activation'] = False
        if 'model_LF/delta_post_activation' not in cfg:
            cfg['model_LF/delta_post_activation'] = False
        if 'model_LF/des' not in cfg:
            cfg['model_LF/des'] = ''
        if 'model_LF/end_layer_pool_mode' not in cfg:
            cfg['model_LF/end_layer_pool_mode'] = 'none'
        if 'model_LF/end_layer_pools' not in cfg:
            cfg['model_LF/end_layer_pools'] = [0]
        if 'model_LF/mid_layer_pool_mode' not in cfg:
            cfg['model_LF/mid_layer_pool_mode'] = 'none'
        if 'model_LF/mid_layer_pools' not in cfg:
            cfg['model_LF/mid_layer_pools'] = [0]
        if 'model_LF/sparsity_post_actv_S4D' not in cfg:
            cfg['model_LF/sparsity_post_actv_S4D'] = 0
    elif model_LF == 'emg2qwerty.modules.TransformerEncoder':
        if 'model_LF/rotary_base' not in cfg:
            cfg['model_LF/rotary_base'] = 10000
        if 'model_LF/end_layer_pools' not in cfg:
            cfg['model_LF/end_layer_pools'] = [0]
        if 'model_LF/mid_layer_pool_mode' not in cfg:
            cfg['model_LF/end_layer_pool_mode'] = 'none'

    if model_HF == 'emg2qwerty.modules.S4Model':
        if 'model_HF/delta_pre_activation' not in cfg:
            cfg['model_HF/delta_pre_activation'] = False
        if 'model_HF/delta_post_activation' not in cfg:
            cfg['model_HF/delta_post_activation'] = False
        if 'model_HF/des' not in cfg:
            cfg['model_HF/des'] = '_HF'
        if 'model_HF/end_layer_pool_mode' not in cfg:
            cfg['model_HF/end_layer_pool_mode'] = 'none'
        if 'model_HF/end_layer_pools' not in cfg:
            cfg['model_HF/end_layer_pools'] = [0]
        if 'model_HF/mid_layer_pool_mode' not in cfg:
            cfg['model_HF/mid_layer_pool_mode'] = 'none'
        if 'model_HF/mid_layer_pools' not in cfg:
            cfg['model_HF/mid_layer_pools'] = [0]
        if 'model_HF/sparsity_post_actv_S4D' not in cfg:
            cfg['model_HF/sparsity_post_actv_S4D'] = 0
        
    if model_HF == 'emg2qwerty.modules.SJ_SNN':
        if 'model_HF/include_neurons_in_last_layer' not in cfg:
            cfg['model_HF/include_neurons_in_last_layer'] = True
        if 'model_HF/activation_post_reduction' not in cfg:
            cfg['model_HF/activation_post_reduction'] = 'none'
        
        if 'model_HF/tau_synapse_filter' not in cfg:
            cfg['model_HF/tau_synapse_filter'] = -1.


    if model_HF == 'emg2qwerty.modules.GRULayer':
        if 'model_HF/dropout_tie' not in cfg:
            cfg['model_HF/dropout_tie'] = 'True'
        if 'model_HF/sparsity_after_non_linearity' not in cfg:
            cfg['model_HF/sparsity_after_non_linearity'] = 0.
        if 'model_HF/mean_pooling' not in cfg:
            cfg['model_HF/mean_pooling'] = 1

    
        if 'model_HF/gru_model' not in cfg:
            cfg['model_HF/gru_model'] = 'GRU'

        if cfg['model_HF/gru_model'] == 'GRU':
            cfg['model_HF/binary_output_EGRU'] = 'none'
            cfg['model_HF/dropout_EGRU'] = 'none'
            cfg['model_HF/zoneout_EGRU'] = 'none'
            cfg['model_HF/dampening_factor_EGRU'] = 'none'
            cfg['model_HF/pseudo_derivative_support_EGRU'] = 'none'
            cfg['model_HF/thr_mean_EGRU'] = 'none'
            cfg['model_HF/weight_initialization_gain_EGRU'] = 'none'
            cfg['model_HF/grad_clip_EGRU'] = 'none'


    if 'train_on_longer_every' not in cfg:
        cfg['train_on_longer_every'] = -1
        cfg['train_on_longer_factor'] = -1
    
    if 'model/norm' not in cfg:
        cfg['model/norm'] = 'none'

    if 'model/offsets' not in cfg:
        cfg['model/offsets'] = '[-1, 0, 1]'
    
    if 'model/sparsity_RotInvMLP_after_non_linearity' not in cfg:
        cfg['model/sparsity_RotInvMLP_after_non_linearity'] = 0


    if 'model/non_linearity' not in cfg:
        cfg['model/non_linearity'] = 'relu'
    if 'model/pooling' not in cfg:
        cfg['model/pooling'] = 'mean'

    return cfg



def compare_configs(cfg1, cfg2, return_string_format=False, exclude_diff=['trainer/default_root_dir']):
    if cfg1 == None or cfg2 == None:
        return {}
    
    diff = {}
    # Recursively compare two configurations
    def _compare(d1, d2, path=""):
        for key in d1:
            if key not in d2:
                if key not in exclude_diff:
                    if return_string_format:
                        diff[path + key] = f"Only in cfg1: {d1[key]}"
                    else:
                        diff[path + key] = [ d1[key], 'Absent' ]
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                # If both values are dictionaries, recurse deeper
                _compare(d1[key], d2[key], path + key + ".")
            elif d1[key] != d2[key]:
                if key not in exclude_diff:
                    # If values are different, record the difference
                    if return_string_format:
                        diff[path + key] = f"cfg1: {d1[key]} != cfg2: {d2[key]}"
                    else:
                        diff[path + key] = [ d1[key], d2[key] ]

        for key in d2:
            if key not in d1:
                if key not in exclude_diff:
                    if return_string_format:
                        diff[path + key] = f"Only in cfg2: {d2[key]}"
                    else:
                        diff[path + key] = [ 'Absent', d2[key] ]

    _compare(cfg1, cfg2)
    return diff



def expanded_name(p, run_names, extra_names, SUBSTITUTIONS={}):

    if ' extra' in p:
        p = p.replace(' extra', '')
        extra = True
    else:
        extra = False

    if ' to ' not in p:
        if p in run_names:
            p_ret = [p]
        else:
            p_ret = [name for name in run_names if name.startswith(f'{p}_')]
            assert len(p_ret) == 1, f'Expected exactly one run name starting with {p}, but found {len(p_ret)}: {p_ret}'
        if extra:
            extra_names.append(p)
        return p_ret, extra_names
    else:
        # print(p)
        start, end = p.split(' to ')
        prenameID = re.findall(r'[A-Za-z]+(?=\d)', start)[0]
        start_id = int(start[len(prenameID):])
        end_id = int(end[len(prenameID):])
        # print(start, end, prenameID, start_id, end_id)
        
        expanded_names = []
        for name in run_names:
            if name.startswith(prenameID) and name[len(prenameID)].isdigit():
                name_id = int(name.split('_')[0][len(prenameID):])
                if 'rerun' in name:
                    continue
                if start_id<=name_id and name_id<end_id:

                    if f'{prenameID}{name_id}' in SUBSTITUTIONS:
                        if SUBSTITUTIONS[f'{prenameID}{name_id}'] in run_names:
                            name = SUBSTITUTIONS[f'{prenameID}{name_id}']
                        else:
                            sub_name_id = SUBSTITUTIONS[f'{prenameID}{name_id}']
                            sub_names = [ sub_name for sub_name in run_names if sub_name.startswith(f'{sub_name_id}_') ]
                            if len(sub_names) == 1:
                                print('MAKING SUBSTITUTION', f'{prenameID}{name_id} --> {sub_name_id}')
                                name = sub_names[0]
                            else:
                                print(f'Warning: multiple or none substitutions found for {prenameID}{name_id}: {sub_names}')
                                exit()

                    expanded_names.append(name)
                    if extra:
                        extra_names.append(name)
        if extra:
            print('extra', expanded_names)
        return expanded_names, extra_names
