from collections.abc import Sequence

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
import platform
try:
    import evnn_pytorch as evnn
except ImportError:
    pass
import emg2qwerty.ssm_models as ssm_models
from spikingjelly.activation_based import neuron, surrogate, layer, functional


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C, f"Expected {self.channels} channels, but got {bands} * {C}"

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        non_linearity: str = "relu",
        sparsity_after_non_linearity: float = 0.,
        norm: str = "none",
    ) -> None:
        super().__init__()

        # assert len(mlp_features) > 0 
        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"

        self.sg = False      
        self.linear_layers = nn.ModuleList()
        self.non_linearities = nn.ModuleList()
        self.norms = nn.ModuleList()
    
        self.num_layers = len(mlp_features)
        self.normalize = norm != 'none'

        for out_features in mlp_features:

            self.linear_layers.append(nn.Linear(in_features, out_features))
            if norm == 'batchnorm':
                self.norms.append(nn.BatchNorm1d(out_features))
            elif norm == 'layernorm':
                self.norms.append(nn.LayerNorm(out_features))
            elif norm == 'none':
                self.norms.append(nn.Identity())

            if non_linearity == 'relu':
                self.non_linearities.append( nn.ReLU() )
            elif non_linearity == 'gelu':
                self.non_linearities.append( nn.GELU() )
            elif 'binary' in non_linearity:
                self.sg_smoothstep = float(non_linearity.split('_')[1])
                self.sg = True
                self.non_linearities.append( SmoothStepModule(self.sg_smoothstep) )
            elif 'sparse_relu' in non_linearity:
                if 'tie' in non_linearity:
                    tie_thr = True                  # sparse_relu_tie_X
                else:
                    tie_thr = False                 # sparse_relu_X
                self.sg_smoothstep = float(non_linearity.split('_')[-1])
                self.non_linearities.append( SparseReLu(out_features, tie_thr, self.sg_smoothstep) )
            elif 'sparse_sugar' in non_linearity:
                self.sg_smoothstep = float(non_linearity.split('_')[-1])
                self.non_linearities.append( SparseSugar(self.sg_smoothstep) )
            else:
                raise NotImplementedError(f"non_linearity {non_linearity} not implemented")  
            in_features = out_features
        
        self.pooling = pooling
        self.offsets = offsets if len(offsets) > 0 else (0,)
        self.sparsity_after_non_linearity = sparsity_after_non_linearity

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)        # C is input channels per single band

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, input_channels)
        x = x.flatten(start_dim=3)      # flattens over frequency features, if present

        sparsity_after_non_linearity_val = {}
        sparsity_after_non_linearity_size = {}
        sparsity_after_non_linearity_zeros = {}
        i_layer = 0

        for lin, norm, non_lin in zip(self.linear_layers, self.norms, self.non_linearities):
            x = lin(x)
            if self.normalize:
                # x has shape (T, N, rotation, out_features)
                _x = norm(x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))        # normalization, considering batch, time and rotations as a single dim: 
                                                                                              # batch norm average across time also, layernorm average across features, separately for each time, rotation and batch
                x = _x.reshape(x.shape[0], x.shape[1],x.shape[2], x.shape[3])

            x = non_lin(x)
            if type(x) == tuple:  # for SparseReLu or SparseSugar
                x, o = x
            else:
                o = x

            if self.sparsity_after_non_linearity>0:
                sparsity_after_non_linearity_val[i_layer] = torch.mean(o)
            else:
                sparsity_after_non_linearity_val[i_layer] = torch.mean(o).detach().cpu().item()
            sparsity_after_non_linearity_size[i_layer] = o.numel()
            sparsity_after_non_linearity_zeros[i_layer] = sparsity_after_non_linearity_size[i_layer] - torch.count_nonzero(o).item()
            i_layer += 1
        
        sparsity_dict = {}
        for i_layer in range(self.num_layers):
            sparsity_dict[f'sparsity_RotInvMLP_after_non_linearity_{i_layer}'] = [ sparsity_after_non_linearity_val[i_layer], sparsity_after_non_linearity_size[i_layer], sparsity_after_non_linearity_zeros[i_layer] ]
        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values, sparsity_dict
        else:
            return x.mean(dim=2), sparsity_dict


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
        non_linearity: str = "relu",

        sparsity_after_non_linearity: float = 0.,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim
        self.num_layers = len(mlp_features)

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                    non_linearity=non_linearity,
                    sparsity_after_non_linearity=sparsity_after_non_linearity,
                    norm=norm,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands, f'Expected {self.num_bands} bands, but got {inputs.shape[self.stack_dim]}'

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]

        outputs = [ outputs_per_band[i][0] for i in range(self.num_bands) ]
        sparsity_dicts = [ outputs_per_band[i][1] for i in range(self.num_bands) ]
        
        output = torch.stack(outputs, dim=self.stack_dim)

        sparsity_dict = {}
        for i_layer in range(self.num_layers):
            val   = 0.
            size  = 0
            zeros = 0
            for i_band in range(self.num_bands):
                val   = val   + sparsity_dicts[i_band][f'sparsity_RotInvMLP_after_non_linearity_{i_layer}'][0]
                size  = size  + sparsity_dicts[i_band][f'sparsity_RotInvMLP_after_non_linearity_{i_layer}'][1]
                zeros = zeros + sparsity_dicts[i_band][f'sparsity_RotInvMLP_after_non_linearity_{i_layer}'][2]
            sparsity_dict[f'sparsity_RotInvMLP_after_non_linearity_{i_layer}'] = [ val/self.num_bands, size, zeros ]
        return output, sparsity_dict



# adapted from the "official" implementation of S4: https://github.com/state-spaces/s4/blob/main/example.py (Albert Gu)
class S4Model(nn.Module):
    ''' note: requires specific setup of the optimizer '''
    def __init__(
        self,
        d_input,
        lr,
        d_output,
        d_model,
        n_layers,
        dropout_inner,      # within the S4D module
        dropout_outer,      # after the S4D module
        dt_min,
        dt_max,
        prenorm,
        decoder_non_linearity,
        reduce_time_dimension,
        permuteLBHtoBLH,
        S4_model,
        d_latent_space=64,
        activation='relu',

        reg_timescale=0.,
        reg_timescale_mode='real',      # real or complex

        dropout_tie=True,

        end_layer_pools = [0],
        end_layer_pool_mode = 'none',  # 'mean' 'last' 'max' or 'none

        mid_layer_pools = [0],
        mid_layer_pool_mode = 'none',  # 'mean' 'last' 'max' or 'none

        sparsity_post_actv_S4D = 0.,
        delta_pre_activation=False,
        delta_post_activation=False,

        des = '',

        upscale_last_layer = 1,

    ) -> None:
        assert delta_pre_activation == False
        assert delta_post_activation == False

        super().__init__()

        self.reg_timescale = reg_timescale
        self.reg_timescale_mode = reg_timescale_mode

        self.sparsity_post_actv_S4D = sparsity_post_actv_S4D

        self.end_layer_pools = end_layer_pools
        self.end_layer_pool_mode = end_layer_pool_mode
        if end_layer_pool_mode != 'none':
            assert end_layer_pool_mode in ['mean', 'last', 'max'], f'Unsupported end_layer_pool_mode: {end_layer_pool_mode}'
            assert len(end_layer_pools) == n_layers, f'Expected end_layer_pools to have {n_layers} elements, but got {len(end_layer_pools)}'

        self.mid_layer_pools = mid_layer_pools
        self.mid_layer_pool_mode = mid_layer_pool_mode
        if mid_layer_pool_mode != 'none':
            assert mid_layer_pool_mode in ['mean', 'last', 'max'], f'Unsupported mid_layer_pool_mode: {mid_layer_pool_mode}'
            assert len(mid_layer_pools) == n_layers, f'Expected mid_layer_pools to have {n_layers} elements, but got {len(mid_layer_pools)}'
        
        self.des = des

        self.HLfreq_ratio = 1
        if self.mid_layer_pool_mode != 'none':
            self.HLfreq_ratio = np.prod(mid_layer_pools)
        if self.end_layer_pool_mode != 'none':
            self.HLfreq_ratio *= np.prod(end_layer_pools)

        self.prenorm = prenorm
        self.reduce_time_dimension = reduce_time_dimension
        self.permuteLBH2BLH = permuteLBHtoBLH

        self.d_input = d_input
        self.d_output = d_output
        self.num_layers = n_layers

        # Linear encoder 
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i_layer in range(n_layers):
            if S4_model == 'S4D':
                self.s4_layers.append(
                    ssm_models.S4D(d_model, d_state=d_latent_space, activation=activation, dropout=dropout_inner, dropout_tie=dropout_tie, transposed=True, lr=min(0.001, lr), dt_min=dt_min, dt_max=dt_max, 
                                   mid_layer_pool_mode=mid_layer_pool_mode, mid_layer_pool=mid_layer_pools[i_layer] if mid_layer_pool_mode!='none' else 0, sparsity_post_actv_S4D=sparsity_post_actv_S4D )
                )
            else:
                print(f'{S4_model} not implemented')
                raise NotImplementedError
            
            self.norms.append(nn.LayerNorm(d_model))
            # self.dropouts.append(nn.Dropout1d(dropout))
            self.dropouts.append(ssm_models.DropoutNd(dropout_outer, tie=dropout_tie)) # expects input with shape B,d_model,L

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

        if decoder_non_linearity == 'relu':
            self.decoder_non_linearity = nn.ReLU()
        elif decoder_non_linearity == 'gelu':
            self.decoder_non_linearity = nn.GELU()
        elif decoder_non_linearity == 'none':
            self.decoder_non_linearity = nn.Identity()
        else:
            raise NotImplementedError(f"decoder_non_linearity {decoder_non_linearity} not implemented")
        
        self.upsample_last = upscale_last_layer
        self.upsample_causal = True
        
    def forward(self, x):
        if self.permuteLBH2BLH:
            x = x.transpose(0,1) # L,B,H -> B,L,H

        ''' x has shape (B, L, d_input) '''
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        i_layer = 0

        sparsity_dict = {}
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            if not self.upsample_causal:
                if self.upsample_last>1 and i_layer == self.num_layers-1:
                    x = x.repeat_interleave(self.upsample_last, dim=2)
                    layer.kernel.log_dt.data = layer.kernel.log_dt.data - math.log(self.upsample_last)
            else:
                if self.upsample_last>1 and i_layer == self.num_layers-1:
                    layer.upsample_last = self.upsample_last

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _, sparsity_dict_layer = layer(z)
            for key in sparsity_dict_layer.keys():
                sparsity_dict[f'{key}{self.des}_{i_layer}'] = sparsity_dict_layer[key]

            if not self.upsample_causal:
                if self.upsample_last>1 and i_layer == self.num_layers-1:
                    layer.kernel.log_dt.data = layer.kernel.log_dt.data + math.log(self.upsample_last)
            else:
                if self.upsample_last>1 and i_layer == self.num_layers-1:              
                    x = x.repeat_interleave(self.upsample_last, dim=2)      # useful for residual connection
                    x[:,:,self.upsample_last-1:] = x[:,:,:-(self.upsample_last-1)].clone()      # shift the sequence to the right, to avoid using future information and align with z
                    x[:,:,:self.upsample_last-1] = 0.  
                    
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection

            if self.mid_layer_pool_mode!='none' and self.mid_layer_pools[i_layer] > 0:
                if x.shape[2] % self.mid_layer_pools[i_layer] != 0:
                    x = x[:,:,:-(x.shape[2] % self.mid_layer_pools[i_layer])]  # remove the last elements to make it divisible by mid_layer_pools[i_layer]
                if self.mid_layer_pool_mode == 'mean':
                    x = x.reshape(x.shape[0], x.shape[1], -1, self.mid_layer_pools[i_layer]).mean(dim=-1)
                elif self.mid_layer_pool_mode == 'max':
                    x = x.reshape(x.shape[0], x.shape[1], -1, self.mid_layer_pools[i_layer]).max(dim=-1).values
                elif self.mid_layer_pool_mode == 'last':
                    x = x.reshape(x.shape[0], x.shape[1], -1, self.mid_layer_pools[i_layer])[:,:,:,-1]

            x = z + x                   # B, d_model, L 
            
            if self.end_layer_pool_mode!='none' and self.end_layer_pools[i_layer] > 0:
                if x.shape[2] % self.end_layer_pools[i_layer] != 0:
                    x = x[:,:,:-(x.shape[2] % self.end_layer_pools[i_layer])] 
                if self.end_layer_pool_mode == 'mean':
                    x = x.reshape(x.shape[0], x.shape[1], -1, self.end_layer_pools[i_layer]).mean(dim=-1)
                elif self.end_layer_pool_mode == 'max':
                    x = x.reshape(x.shape[0], x.shape[1], -1, self.end_layer_pools[i_layer]).max(dim=-1).values
                elif self.end_layer_pool_mode == 'last':
                    x = x.reshape(x.shape[0], x.shape[1], -1, self.end_layer_pools[i_layer])[:,:,:,-1]

            i_layer += 1


            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)        # (B, d_model, L) -> (B, L, d_model)

        ''' x has shape (B, L, d_input) '''
        # Pooling: average pooling over the sequence length
        if self.reduce_time_dimension:
            x = x.mean(dim=1)          # (B, L, d_model) -> (B, d_model)      

        # Decode the outputs
        x = self.decoder(x)  # (B, (L), d_model) -> (B, (L), d_output)
        x = self.decoder_non_linearity(x)  # (B, (L), d_output)

        if not self.reduce_time_dimension:
            if self.permuteLBH2BLH:
                x = x.transpose(0,1)    # B,L,H -> L,B,H
        
        return x, sparsity_dict
    

    def get_timescales(self):
        """
        Returns the timescales of the S4D model
        """
        if self.reg_timescale == 0.:
            return 0.
        
        else:
            S4D_kernel_dtA = []
            for i_layer, layer in enumerate(self.s4_layers):
                dt = torch.exp(layer.kernel.log_dt)              # (H)    a dt for each input channel
                log_A_real = layer.kernel.log_A_real
                if self.reg_timescale_mode == 'real':
                    A = -torch.exp(log_A_real)      # this is the real part of the A matrix (strictly negative). tau = 1/(A_real*dt) = 1/(-real(A)*dt)  because A = -A_real+1j*A_imag  
                    A = -A # this is A_real
                else:
                    A_imag = layer.kernel.A_imag
                    A = -torch.exp(log_A_real) + 1j * A_imag
                S4D_kernel_dtA.append( A * dt[:, None] )   # (H N)  a complex A diagonal matrix for each input channel      # np.exp(layer.kernel.log_dt.data.detach().numpy())
            
            S4D_kernel_dtA = torch.stack(S4D_kernel_dtA, dim=0)  # (n_layer, H, N)
            return S4D_kernel_dtA



class SparseReLu(nn.Module):
    def __init__(self, dim, tie_thr, sg_smoothstep):
        super().__init__()
        
        self.sg_smoothstep = sg_smoothstep  
        self.thr = 0
        self.step_func = SmoothStepModule(self.sg_smoothstep)
        self.relu = nn.ReLU()

    def forward(self, x):
        o = self.step_func(x-self.thr)
        x = self.relu(x-self.thr)
        return x, o



class SparseSugar(nn.Module):
    def __init__(self, sg_smoothstep):
        super().__init__()

        self.sg_smoothstep = sg_smoothstep
        self.step_func = SmoothStepModule(self.sg_smoothstep)
        self.sugar_func = Sugar().apply

    def forward(self, x):   
        o = self.step_func(x)
        x = self.sugar_func(x)
        return x, o
    
class Sugar(torch.autograd.Function):
    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return nn.functional.relu(x) 

    def backward(aux, grad_output): 
        inp, = aux.saved_tensors
        alpha = 1.67
        surrogate = nn.functional.sigmoid(inp) * ( 1 + (inp+alpha)*(1-nn.functional.sigmoid(inp)) )  # Sigmoid surrogate gradient
        grad_input = grad_output * surrogate   
        return grad_input


class GRULayer(nn.Module):
    def __init__(
            self,
            d_input,    # input_size
            d_hidden,   # hidden_size

            norm,
            num_layers,
            bidirectional,

            prenorm,
            include_encoder,        # d_input to d_model
            dropout,
            include_decoder,        # d_model to d_model
            include_residual,

            HLfreq_ratio,
            reduction_mode,
            detach_hidden_on_chunks,
            include_linear_layer,
            non_linearity,

            dropout_tie=True,
            sparsity_after_non_linearity=0.,

            gru_model='GRU', 
            sparsity_hidden_EGRU=0.,
            binary_output_EGRU=False,
            dropout_EGRU=0.0,
            zoneout_EGRU=0.0,
            dampening_factor_EGRU=0.7,
            pseudo_derivative_support_EGRU=1.0,
            thr_mean_EGRU=0.3,                       # mean of threshold values
            weight_initialization_gain_EGRU=1.0,
            grad_clip_EGRU='none',

            mean_pooling = 1,  # if >1, it will perform mean pooling over the time dimension before everything AND will divide HLfreq_ratio by this value
    ):
        super().__init__()
        self.ONNX_export = False
        self.ONNX_export_perform_norm = False

        if grad_clip_EGRU == 'none':
            grad_clip_EGRU = None

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.prenorm = prenorm
        self.include_encoder = include_encoder
        self.dropout = dropout
        self.dropout_tie = dropout_tie
        self.include_decoder = include_decoder
        self.include_residual = include_residual

        self.d_input = d_input
        self.d_hidden = d_hidden
        if include_linear_layer>0:
            self.d_output = include_linear_layer
        else:
            self.d_output = d_hidden

        self.mean_pooling = mean_pooling
        if self.mean_pooling>1:
            assert HLfreq_ratio % self.mean_pooling == 0, f"HLfreq_ratio ({HLfreq_ratio}) must be divisible by mean_pooling ({mean_pooling})"
        self.HLfreq_ratio = HLfreq_ratio

        self.reduction_mode = reduction_mode
        self.detach_hidden_on_chunks = detach_hidden_on_chunks

        assert gru_model in [ 'GRU', 'EventBasedGRU' ], f"not implemented gru_model: {gru_model}"
        self.gru_model = gru_model
        self.binary_output_EGRU = binary_output_EGRU

        if self.HLfreq_ratio//self.mean_pooling == 1:
            assert not self.detach_hidden_on_chunks, "STRONG WARNING: you should not detach with HLfreq_ratio=1..."
            assert reduction_mode in [ "none" ], f"reduction_mode: {reduction_mode} must be none with HLfreq_ratio//mean_pooling=1"  
        else:
            assert self.num_layers==1, f"HLfreq_ratio ({self.HLfreq_ratio}) > 1 is not compatible with n_layers>1"
            assert self.include_residual==False, f"Lfreq_ratio ({self.HLfreq_ratio}) > 1 is not compatible with residual_connections"   # this can be made compatible performing reduction on the input

            assert reduction_mode in [ "last", "mean" ], f"not implemented reduction_mode: {reduction_mode}"

        if self.include_encoder:
            # Linear encoder 
            self.encoder = nn.Linear(d_input, self.d_hidden)
            self.d_input_gru = self.d_hidden
        else:
            self.encoder = nn.Identity()
            self.d_input_gru = self.d_input
        
        if self.num_layers>1 or self.include_residual:
            # these must be verified ehither with and without residual connections
            if include_linear_layer>0:
                assert include_linear_layer == self.d_input_gru, f"include_linear_layer ({include_linear_layer}) must be equal to d_input_gru ({self.d_input_gru}) when num_layers>1"
            else:
                assert self.d_hidden == self.d_input_gru, f"d_hidden ({self.d_hidden}) must be equal to d_input_gru ({self.d_input_gru}) when num_layers>1"
        
        assert prenorm == False, "prenorm not implemented in GRU yet"

        ''' sparsity '''
        self.sparsity_after_non_linearity = sparsity_after_non_linearity
        self.sparsity_hidden_EGRU = sparsity_hidden_EGRU

        self.pooling = nn.AvgPool1d(mean_pooling) if mean_pooling>1 else nn.Identity()      # perform pooling on the last dimension...
        self.gru_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.non_linearities = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(self.num_layers):
            # Define GRU
            if gru_model == 'GRU':
                self.gru_layers.append( nn.GRU(
                    input_size=self.d_input_gru,
                    hidden_size=self.d_hidden,
                    num_layers=1,
                    bidirectional=self.bidirectional,
                    batch_first=True  # Assuming input tensor is (batch, seq, feature)
                ))
            elif gru_model == 'EventBasedGRU':
                self.gru_layers.append( evnn.EGRU(
                input_size=self.d_input_gru,
                hidden_size=self.d_hidden,
                batch_first=True,
                return_state_sequence=True,
                use_custom_cuda=True,
                
                dropout=dropout_EGRU,
                zoneout=zoneout_EGRU,
                dampening_factor=dampening_factor_EGRU,
                pseudo_derivative_support=pseudo_derivative_support_EGRU,
                thr_mean=thr_mean_EGRU,
                weight_initialization_gain=weight_initialization_gain_EGRU,
                grad_clip=grad_clip_EGRU,
                ))
            if include_linear_layer>0:
                self.linear_layers.append( nn.Linear(self.d_hidden, self.d_output) )
            else:
                self.linear_layers.append( nn.Identity() )

            self.normalize = False
            if norm == "batchnorm":
                self.norms.append( nn.BatchNorm1d(self.d_output, momentum=0.05) )
                self.normalize = True
            elif norm == "layernorm":
                self.norms.append( nn.LayerNorm(self.d_output) )
                self.normalize = True
            
            self.dropouts.append( ssm_models.DropoutNd(self.dropout, tie=self.dropout_tie) )

            if non_linearity == 'relu':
                self.non_linearities.append( nn.ReLU() )
            elif non_linearity == 'gelu':
                self.non_linearities.append( nn.GELU() )
            elif 'binary' in non_linearity:
                self.sg_smoothstep = float(non_linearity.split('_')[-1])
                self.non_linearities.append( SmoothStepModule(self.sg_smoothstep) )
            elif 'sparse_relu' in non_linearity:
                if 'tie' in non_linearity:
                    tie_thr = True                  # sparse_relu_tie_X
                else:
                    tie_thr = False                 # sparse_relu_X
                self.sg_smoothstep = float(non_linearity.split('_')[-1])
                self.non_linearities.append( SparseReLu(self.d_output, tie_thr, self.sg_smoothstep) )
            elif 'sparse_sugar' in non_linearity:
                self.sg_smoothstep = float(non_linearity.split('_')[-1])
                self.non_linearities.append( SparseSugar(self.sg_smoothstep) )
            elif non_linearity == 'none':
                self.non_linearities.append( nn.Identity() )
            else:
                raise NotImplementedError(f"non_linearity {non_linearity} not implemented")
        
        if self.include_decoder:
            # Linear decoder 
            self.decoder = nn.Linear(self.d_output, self.d_output)
        else:
            self.decoder = nn.Identity()


    def forward(self, x):
        
        if not self.ONNX_export:
            # x shape (T,N,C) to (N,T,C)
            x = x.permute(1,0,2) 
        else: 
            print('ONNX export mode, not permuting input')

        if self.mean_pooling>1:
            x = self.pooling(x.permute(0,2,1)).permute(0,2,1)

        x = self.encoder(x)  # (N,T,C) -> (N,T,d_hidden) if encoder is present else (N,T,C)

        # Initial hidden state
        num_directions = 2 if self.bidirectional else 1
        device = x.device
        # x has shape (N,T,C)       # C input channels
        # h0 has shape (*, N, H)    # H hidden and output channels

        if not self.ONNX_export:
            sparsity_after_non_linearity_val = {}
            sparsity_after_non_linearity_size = {}
            sparsity_after_non_linearity_zeros = {}

            sparsity_hidden_EGRU_val = {}
            sparsity_hidden_EGRU_size = {}
            sparsity_hidden_EGRU_zeros = {}

        i_layer = 0

        for layer, lin, non_lin, norm, dropout in zip(self.gru_layers, self.linear_layers, self.non_linearities, self.norms, self.dropouts):
            h0 = torch.zeros( num_directions*self.num_layers, x.size(0), self.d_hidden ).to(device)

            # Forward propagate the GRU
            T_out = x.size(1)//(self.HLfreq_ratio//self.mean_pooling)
            z = torch.zeros(x.size(0), T_out, self.d_hidden).to(device)     # N, T_out, H
            if self.detach_hidden_on_chunks:
                for i_chunk in range(T_out):
                    x_chunk = x[:, i_chunk*self.HLfreq_ratio//self.mean_pooling:(i_chunk+1)*self.HLfreq_ratio//self.mean_pooling, :]

                    if self.gru_model == 'GRU':
                        out_chunk, hn = layer(x_chunk, h0)          #  out_chunk has shape (N,T_chunk,H=d_hidden); hn has shape (num_layers*num_directions, N, H=d_hidden)
                    elif self.gru_model == 'EventBasedGRU':
                        out_chunk, hn = layer(x_chunk, h0)  
                        c_vals, o_vals, tr_vals = hn        # c_vals, o_vals, tr_vals and out_chunk have shape N,T_chunk,H

                        if self.binary_output_EGRU:
                            out_chunk = o_vals

                        if not self.ONNX_export:
                            if i_layer not in sparsity_hidden_EGRU_val:
                                sparsity_hidden_EGRU_val[i_layer] = []
                                sparsity_hidden_EGRU_size[i_layer] = 0
                                sparsity_hidden_EGRU_zeros[i_layer] = 0
                            if self.sparsity_hidden_EGRU>0:
                                sparsity_hidden_EGRU_val[i_layer].append( torch.mean(o_vals) )
                            else:
                                sparsity_hidden_EGRU_val[i_layer].append( torch.mean(o_vals).detach().cpu().item() )
                            sparsity_hidden_EGRU_size[i_layer] += o_vals.numel()
                            sparsity_hidden_EGRU_zeros[i_layer] += ( o_vals.numel() - torch.count_nonzero(o_vals).item())

                        hn_shape = c_vals.shape
                        hn = c_vals[:,-1,:]
                        hn = hn.reshape(1, hn_shape[0], hn_shape[2])
                    
                    h0 = hn.detach()
                    if self.reduction_mode=='last':
                        z[:, i_chunk, :] = out_chunk[:, -1, :] # this should equl to hn [of the last layer]
                    elif self.reduction_mode=='mean':
                        z[:, i_chunk, :] = out_chunk.mean(dim=1)
            else:
                if self.gru_model == 'GRU':
                    z, hn = layer(x, h0)        # z has shape (N,T,H=d_hidden)
                else:
                    z, hn = layer(x, h0)  
                    c_vals, o_vals, tr_vals = hn        # c_vals, o_vals, tr_vals and z have shape N,T,H

                    if self.binary_output_EGRU:
                        z = o_vals

                    if not self.ONNX_export:
                        if i_layer not in sparsity_hidden_EGRU_val:
                            sparsity_hidden_EGRU_val[i_layer] = []
                            sparsity_hidden_EGRU_size[i_layer] = 0
                            sparsity_hidden_EGRU_zeros[i_layer] = 0
                        if self.sparsity_hidden_EGRU>0:
                            sparsity_hidden_EGRU_val[i_layer].append( torch.mean(o_vals) )
                        else:
                            sparsity_hidden_EGRU_val[i_layer].append( torch.mean(o_vals).detach().cpu().item() )
                        sparsity_hidden_EGRU_size[i_layer] += o_vals.numel()
                        sparsity_hidden_EGRU_zeros[i_layer] += ( o_vals.numel() - torch.count_nonzero(o_vals).item())

                if self.HLfreq_ratio>1:
                    z_cut = (z.shape[1]//(self.HLfreq_ratio//self.mean_pooling))*(self.HLfreq_ratio//self.mean_pooling)  # cut the last elements to make it divisible by HLfreq_ratio//mean_pooling
                    z = z[ :, :z_cut ] # cut the last elements to make it divisible by HLfreq_ratio//mean_pooling
                    z = z.reshape(z.shape[0], z.shape[1]//(self.HLfreq_ratio//self.mean_pooling), self.HLfreq_ratio//self.mean_pooling, z.shape[2])
                    if self.reduction_mode=='last':
                        z = z[:,:,-1,:]
                    elif self.reduction_mode=='mean':
                        z = z.mean(dim=2)
            

            # z has shape (N,T_out,H=d_output), dropout requires N,H,T
            z = dropout( z.transpose(-1,-2) ).transpose(-1,-2)
            z = lin(z)        # linear layer if include linear is out has shape (N,T_out,H=d_output)
            if self.include_residual:
                x = z + x
            else:
                x = z

            # x has shape (N,T_out,H)
            if not self.ONNX_export or self.ONNX_export_perform_norm:
                if self.normalize:
                    _x = norm(x.reshape(x.shape[0] * x.shape[1], x.shape[2]))        # normalization, considering batch and time as a single dim: 
                                                                                    # batch norm average across time also, layernorm average across features, separately for each time and batch
                    x = _x.reshape(x.shape[0], x.shape[1], x.shape[2])
            else:
                assert x.shape[0] == 1, "ONNX export mode only works with batch size 1"
                print('not performing BN... can be aggregated with linear layer')

            # non-linearity
            x = non_lin(x)
            if type(x) == tuple:  # if SparseReLu or SparseSugar
                x, o = x
            else:
                o = x
            if not self.ONNX_export:
                if self.sparsity_after_non_linearity>0:
                    sparsity_after_non_linearity_val[i_layer] = torch.mean(o)
                else:
                    sparsity_after_non_linearity_val[i_layer] = torch.mean(o).detach().cpu().item()
                sparsity_after_non_linearity_size[i_layer] = o.numel()
                sparsity_after_non_linearity_zeros[i_layer] = sparsity_after_non_linearity_size[i_layer] - torch.count_nonzero(o).item()
            
            i_layer += 1

        x = self.decoder(x)
        if not self.ONNX_export:
            x = x.permute(1,0,2)    #  out shape: (N,T_out,H=d_output) to (T_out,N,H=d_output)

        if not self.ONNX_export:
            sparsity_dict = {}
            for i_layer in range(self.num_layers):
                sparsity_dict[f'sparsity_GRU_after_non_linearity_{i_layer}'] = [ sparsity_after_non_linearity_val[i_layer], sparsity_after_non_linearity_size[i_layer], sparsity_after_non_linearity_zeros[i_layer] ]
                if self.gru_model == 'EventBasedGRU':
                    if self.sparsity_hidden_EGRU>0:
                        sparsity_dict[f'sparsity_hidden_EGRU_{i_layer}'] = [ torch.mean( torch.stack(sparsity_hidden_EGRU_val[i_layer]) ), sparsity_hidden_EGRU_size[i_layer], sparsity_hidden_EGRU_zeros[i_layer] ]
                    else:
                        sparsity_dict[f'sparsity_hidden_EGRU_{i_layer}'] = [ np.mean( sparsity_hidden_EGRU_val[i_layer] ), sparsity_hidden_EGRU_size[i_layer], sparsity_hidden_EGRU_zeros[i_layer] ]
            return x, sparsity_dict
        else:
            return x




class GRULayerPooled(nn.Module):

    """
        This module is inspired from S4Model and consists of:
            - encoder
            - stack of GRU layers + GLU (optional) with dropout + residual (optional) + pooling (optional) + normalization
            - decoder (optional)
    """

    def __init__(
            self,
            d_input,    # input_size
            d_hidden,   # hidden_size

            norm,
            n_layers,
            bidirectional,
            dropout,
            dropout_tie,
            include_GLU,
            include_residual,

            include_decoder,        # d_model to d_model
    
            end_layer_pools = [],
            end_layer_pool_mode = 'mean',

    ):
        super().__init__()

        # Fixed parameters
        self.num_layers = n_layers
        self.include_GLU = include_GLU

        self.norm = norm
        self.bidirectional = bidirectional

        self.dropout = dropout
        self.dropout_tie = dropout_tie
        self.include_decoder = include_decoder          # dmodel to dmodel
        self.include_residual = include_residual

        self.d_hidden = d_hidden
        self.d_output = d_hidden

        self.HLfreq_ratio = 1
        if end_layer_pool_mode != 'none':
            self.HLfreq_ratio = np.prod(end_layer_pools)
            assert len(end_layer_pools) == self.num_layers, f"end_layer_pools ({end_layer_pools}) must have length equal to num_layers ({self.num_layers})"

        # Linear encoder 
        self.encoder = nn.Linear(d_input, self.d_hidden)
        self.encoder_activation = nn.GELU()

        self.poolings = nn.ModuleList()
        self.gru_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.output_linears = nn.ModuleList()

        for i_layer in range(self.num_layers):
            
            self.gru_layers.append( nn.GRU(
                input_size=self.d_hidden,
                hidden_size=self.d_hidden,
                num_layers=1,
                bidirectional=self.bidirectional,
                batch_first=True  # Assuming input tensor is (batch, seq, feature)
            ))

            if self.include_GLU:
                self.output_linears.append(nn.Sequential(
                    nn.Conv1d(self.d_hidden, 2*self.d_hidden, kernel_size=1),     # it is basically a linear layer identically applied over the sequence lenght (L) ----> B 2H L
                    nn.GLU(dim=-2),                                               # GLU = gated linear unit: introduces a gating mechanism on top of the linear layer  ----> B H L x sigmoid(B H L) = B H L
                ))
            else:
                self.output_linears.append(nn.Identity())
            
            self.dropouts.append( ssm_models.DropoutNd(self.dropout, tie=self.dropout_tie) )

            if self.norm == "batchnorm":
                self.norms.append( nn.BatchNorm1d(self.d_output, momentum=0.05) )
            elif self.norm == "layernorm":
                self.norms.append( nn.LayerNorm(self.d_output) ) 
            else:
                self.norms.append( nn.Identity() )

            if end_layer_pool_mode != 'none' and end_layer_pools[i_layer]>1:
                if end_layer_pool_mode == 'mean':
                    self.poolings.append( nn.AvgPool1d(end_layer_pools[i_layer]) )
                else:
                    raise NotImplementedError(f"end_layer_pool_mode {end_layer_pool_mode} not implemented")
            else:
                self.poolings.append( nn.Identity() )


        if self.include_decoder:
            self.decoder = nn.Linear(self.d_output, self.d_output)
            self.decoder_activation = nn.GELU()
        else:
            self.decoder = nn.Identity()
            self.decoder_activation = nn.Identity()


    def forward(self, x):
        
        # x shape (T,N,C) to (N,T,C)
        x = x.permute(1,0,2) 
       
        x = self.encoder(x)  # (N,T,C) -> (N,T,d_hidden)
        x = self.encoder_activation(x)

        # Initial hidden state
        num_directions = 2 if self.bidirectional else 1
        device = x.device
        # x has shape (N,T,d_hidden)  
        # h0 has shape (*, N, H)    # H hidden and output channels

        i_layer = 0
        for layer, out_lin, norm, dropout, pooling in zip(self.gru_layers, self.output_linears, self.norms, self.dropouts, self.poolings):
            h0 = torch.zeros( num_directions, x.size(0), self.d_hidden ).to(device)

            z = x
            z, hn = layer(z, h0)        # z has shape (N,T,H=d_hidden)

            z = out_lin( z.transpose(1,2) ).transpose(1,2)   # (N,T,H) -> (N,H,T) -> (N,2H,T) -> (N,H,T) -> (N,T,H)

            z = dropout(z)

            if self.include_residual:
                x = x + z
            else:
                x = z

            x = pooling(x.transpose(1,2)).transpose(1,2)   # (N,T,H) -> (N,H,T) -> (N,H,T//pool) -> (N,T//pool,H)

            # normalization
            if self.norm == "batchnorm":
                # batchnorm expects N,C,T # and average over N, T
                x = norm(x.transpose(1,2)).transpose(1,2)
            elif self.norm == "layernorm":
                x = norm(x)
            else:
                pass

            i_layer += 1

        # x has shape N,T_out,H
        x = self.decoder(x)
        x = self.decoder_activation(x)

        x = x.permute(1,0,2)    #  out shape: (N,T_out,H=d_output) to (T_out,N,H=d_output)
        return x


class SmoothStep(torch.autograd.Function):
    """
    Here, we define a surrogate gradient for the Heaviside step function.
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """
    @staticmethod
    def forward(aux, x, sg):
        aux.save_for_backward(x)
        aux.sg = sg
        return (x >= 0).float() # Behavior similar to Heaviside step function

    def backward(aux, grad_output): # Define the behavior for the backward pass
        input, = aux.saved_tensors
        beta   = aux.sg
        # surrogate = 1.0/(beta*torch.abs(input) + 1.0)**2
        surrogate = 1.0/((2*beta*torch.abs(input))**2 + 1.0)        # arctan surrogate
        grad_input = grad_output * surrogate        # modified by AO
        # grad_input = grad_output.clone() * surrogate
        return grad_input, None




class SmoothStepModule(nn.Module):
    def __init__(self, sg_smoothstep):
        super().__init__()
        self.func = SmoothStep()
        self.sg_smoothstep = sg_smoothstep
    
    def forward(self, x):
        return self.func.apply(x, self.sg_smoothstep)



class SJ_SNN(nn.Module):
    def __init__(self,
                 d_input:       int,
                 layer_sizes:   Sequence[int],
                 neuron_type:   str, 
                 norm:          str,    # none or batchnorm --- chunkwise....
            
                 HLfreq_ratio:      int,
                 reduction_mode:    str,   # mean_act, mean_mempot or last_mempot
                 detach_hidden_on_chunks:    bool,  

                 decay_input:   bool,
                 v_reset:       float | str,  # reset value for the membrane potential
                 detach_reset:  bool,
                 v_threshold:   float,
                 
                 alpha_surr_grad:  float,  # alpha for surrogate gradient
                 tau_mem:     float | None,

                 include_neurons_in_last_layer: bool = True, 
                 activation_post_reduction: str = 'none', 

                 sparsity_spiking_activity: float = 0.0,  # sparsity of the spiking activity (mean)  

                 tau_synapse_filter: float=-1.,  
                 include_bias: bool=True, 
                 
            ):  # membrane time constant):
        
        if v_reset == 'none':
            v_reset = None

        super().__init__()

        self.HLfreq_ratio = HLfreq_ratio
        self.detach_hidden_on_chunks = detach_hidden_on_chunks
        self.reduction_mode = reduction_mode
        self.sparsity_spiking_activity = sparsity_spiking_activity
        self.num_layers = len(layer_sizes)
        self.d_output = layer_sizes[-1]
        self.include_bias = include_bias

        assert norm in [ 'none', 'batchnorm' ], f"not implemented norm: {norm}"

        if not include_neurons_in_last_layer:
            assert reduction_mode in [ 'last_mempot', 'mean_mempot' ], f"include_neurons_in_last_layer=False requires reduction_mode in [ 'last_mempot', 'mean_mempot' ]"

        surrog_f = surrogate.ATan(alpha=alpha_surr_grad)
        config_dict = {
                        'tau':                  tau_mem,
                        'decay_input':          decay_input,
                        'v_threshold':          v_threshold,
                        'v_reset':              v_reset,
                        'surrogate_function':   surrog_f,
                        'detach_reset':         detach_reset,
                        'store_v_seq':          True if reduction_mode=='mean_mempot' else False,

        }
        if neuron_type == 'PLIF':
            config_dict['init_tau'] = config_dict['tau']
            config_dict.pop('tau')

        self.layers = nn.ModuleList()
        d_in = d_input
        for i_layer in range(len(layer_sizes)):
            self.layers.append( layer.Linear(d_in, layer_sizes[i_layer], bias=self.include_bias) )
            if norm=='batchnorm':
                self.layers.append( layer.BatchNorm1d(layer_sizes[i_layer], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode='s') )
            
            if include_neurons_in_last_layer or i_layer < len(layer_sizes)-1:
                if i_layer == len(layer_sizes)-1 and reduction_mode in [ 'last_mempot', 'mean_mempot' ]:
                    config_dict['v_reset'] = 0.0
                    config_dict['v_threshold'] = np.inf
                if neuron_type == 'LIF':
                    self.layers.append( neuron.LIFNode(**config_dict) )
                elif neuron_type == 'PLIF':
                    self.layers.append( neuron.ParametricLIFNode(**config_dict) )    

                if i_layer < len(layer_sizes)-1:
                    if tau_synapse_filter>0:
                        self.layers.append( layer.SynapseFilter(tau=tau_synapse_filter, learnable=True, step_mode='s') )
            else:
                # include_neurons_in_last_layer is False and this is the last layer
                pass

            d_in = layer_sizes[i_layer]

        if activation_post_reduction == 'relu':
            self.non_linearity = nn.ReLU()
        elif activation_post_reduction == 'gelu':
            self.non_linearity = nn.GELU()
        elif "sparse_relu" in activation_post_reduction:
            if 'tie' in activation_post_reduction:
                    tie_thr = True                  # sparse_relu_tie_X
            else:
                tie_thr = False                 # sparse_relu_X
            self.sg_smoothstep = float(activation_post_reduction.split('_')[-1])
            self.non_linearity = SparseReLu(self.d_output, tie_thr, self.sg_smoothstep) 
        elif 'sparse_sugar' in activation_post_reduction:
            self.sg_smoothstep = float(activation_post_reduction.split('_')[-1])
            self.non_linearity = SparseSugar(self.sg_smoothstep) 
        elif activation_post_reduction == 'none':
            self.non_linearity = nn.Identity()
        else:
            raise NotImplementedError(f"activation_post_reduction {activation_post_reduction} not implemented")


        functional.set_step_mode(self, 'm')
        try: 
            functional.set_backend(self, 'cupy', (neuron.LIFNode, neuron.ParametricLIFNode, neuron.KLIFNode, neuron.QIFNode))
            print('INFO: cupy available, using cupy backend when possible')
            print(self)
        except ImportError:
            print('INFO: cupy not available, using torch backend')

        self.batch_layers = [ True if type(l)==layer.BatchNorm1d else False for l in self.layers ]
        self.neuron_layers = [ True if type(l) in [neuron.LIFNode, neuron.ParametricLIFNode, neuron.KLIFNode, neuron.QIFNode ] else False for l in self.layers ]
        self.vmem_layers = [ True if type(l) in [neuron.LIFNode, neuron.ParametricLIFNode, neuron.KLIFNode, neuron.QIFNode ] and l.v_threshold==np.inf else False for l in self.layers ]
    
    def setup_torch(self):
        functional.set_step_mode(self, 'm')
        functional.set_backend(self, 'torch', (neuron.LIFNode, neuron.ParametricLIFNode, neuron.KLIFNode, neuron.QIFNode))
        print('INFO: setting to torch backend')
        print(self)
       


    def forward(self, x):
        # x has shape T, N, C

        device = x.device
        z = torch.zeros(x.size(0)//self.HLfreq_ratio, x.size(1), self.d_output).to(device)     # T_out, N, d_output
        # print(f"z shape: {z.shape}, HLfreq_ratio: {self.HLfreq_ratio}, reduction_mode: {self.reduction_mode}")

        sparsity_spiking_val = {}
        sparsity_spiking_size = {}
        sparsity_spiking_zeros = {}

        for i_chunk in range(x.size(0)//self.HLfreq_ratio):
            x_chunk = x[i_chunk*self.HLfreq_ratio:(i_chunk+1)*self.HLfreq_ratio, :, :]

            spiking_layer = 0
            for i_l,l in enumerate(self.layers):
                if self.batch_layers[i_l]:
                    x_chunk = x_chunk.unsqueeze(-1)
                x_chunk = l(x_chunk)
                if self.batch_layers[i_l]:
                    x_chunk = x_chunk.squeeze(-1)
                
                if self.vmem_layers[i_l]:
                    if self.reduction_mode=='mean_mempot':
                        x_chunk = self.layers[i_l].v_seq           # v_seq has shape [T, batch_size, n_neur_in_layer]
                    elif self.reduction_mode=='last_mempot':
                        x_chunk = self.layers[i_l].v.unsqueeze(0)  # v has shape [T=1, batch_size, n_neur_in_layer]
                    else:
                        raise NotImplementedError(f"reduction_mode {self.reduction_mode} not implemented for vmem_layers")
                elif self.neuron_layers[i_l]:
                    if f'sparsity_spiking_activity_{spiking_layer}' not in sparsity_spiking_val.keys():
                        sparsity_spiking_val[f'sparsity_spiking_activity_{spiking_layer}'] = []
                        sparsity_spiking_size[f'sparsity_spiking_activity_{spiking_layer}'] = 0
                        sparsity_spiking_zeros[f'sparsity_spiking_activity_{spiking_layer}'] = 0
                    if self.sparsity_spiking_activity>0:
                        sparsity_spiking_val[f'sparsity_spiking_activity_{spiking_layer}'].append(torch.mean(x_chunk))
                    else:
                        sparsity_spiking_val[f'sparsity_spiking_activity_{spiking_layer}'].append(torch.mean(x_chunk).detach().cpu().item())
                    sparsity_spiking_size[f'sparsity_spiking_activity_{spiking_layer}'] += x_chunk.numel()
                    sparsity_spiking_zeros[f'sparsity_spiking_activity_{spiking_layer}'] += (x_chunk.numel() - torch.count_nonzero(x_chunk).item())
                    spiking_layer += 1

            # print(self.reduction_mode, x_chunk.shape)
            if self.reduction_mode in ['mean_act', 'mean_mempot']:
                z[i_chunk, :, :] = x_chunk.mean(dim=0)
            elif self.reduction_mode == 'last_mempot':
                z[i_chunk, :, :] = x_chunk[-1, :, :]
            else:
                raise NotImplementedError(f"reduction_mode {self.reduction_mode} not implemented")

            if self.detach_hidden_on_chunks:
                for i_l,l in enumerate(self.layers):
                    if self.neuron_layers[i_l]:
                        l.detach()

        z = self.non_linearity(z)  # apply non-linearity after reduction
        if type(z) == tuple:  # if SparseReLu or SparseSugar
            z, o = z
        else:
            o = z
        sparsity_post_non_linearity_val = torch.mean(o).detach().cpu().item()
        sparsity_post_non_linearity_size = o.numel()
        sparsity_post_non_linearity_zeros = sparsity_post_non_linearity_size - torch.count_nonzero(o).item()        

        sparsity_dict = { 'sparsity_SJSNN_post_reduction': [ sparsity_post_non_linearity_val, sparsity_post_non_linearity_size, sparsity_post_non_linearity_zeros ] }
        for k in sparsity_spiking_val.keys():
            # print( k, sparsity_spiking_val[k] )
            if self.sparsity_spiking_activity>0:
                sparsity_dict[k] = [ torch.mean( torch.stack(sparsity_spiking_val[k]) ), sparsity_spiking_size[k], sparsity_spiking_zeros[k] ]
                # print( sparsity_dict[k])
            else:
                sparsity_dict[k] = [ np.mean( sparsity_spiking_val[k] ), sparsity_spiking_size[k], sparsity_spiking_zeros[k] ]
                # print( sparsity_dict[k])
        
        return z, sparsity_dict



class TemporalConvolution(nn.Module):

    def __init__(self, d_input, norm, activation, out_channels, strides, kernel_sizes, padding):
        super().__init__()
        self.ONNX_export = False

        self.d_output = out_channels[-1]
        self.HLfreq_ratio = np.prod(strides)

        self.layers = nn.ModuleList()
        d_in = d_input
        for i_layer in range(len(out_channels)):
            self.layers.append( nn.Conv1d(d_in, out_channels[i_layer], kernel_size=kernel_sizes[i_layer], stride=strides[i_layer], padding=padding) )

            if norm == "batchnorm":
                self.layers.append(nn.BatchNorm1d(out_channels[i_layer]))
            elif norm == "layernorm":
                self.layers.append(nn.LayerNorm(out_channels[i_layer]))
            elif norm == "none":
                pass
            else:
                raise NotImplementedError(f"Unknown normalization for TemporalConvolution {norm}")

            if activation == "relu":
                self.layers.append(nn.ReLU())
            elif activation == "gelu":
                self.layers.append(nn.GELU())
            elif 'sparse_relu' in activation:
                if 'tie' in activation:
                    tie_thr = True                  # sparse_relu_tie_X
                else:
                    tie_thr = False                 # sparse_relu_X
                self.sg_smoothstep = float(activation.split('_')[-1])
                self.layers.append(SparseReLu(self.d_output, tie_thr, self.sg_smoothstep))
            elif 'sparse_sugar' in activation:
                self.sg_smoothstep = float(activation.split('_')[-1])
                self.layers.append(SparseSugar(self.sg_smoothstep))
            elif 'binary' in activation:
                if i_layer == len(out_channels) - 1:
                    # last layer, no need for binary activation
                    self.layers.append(nn.GELU())
                else:
                    self.sg_smoothstep = float(activation.split('_')[-1])
                    self.layers.append(SmoothStepModule(self.sg_smoothstep))

            else:
                raise NotImplementedError(f"Unknown activation function for TemporalConvolution {activation}")
            d_in = out_channels[i_layer]
        

    def forward(self, x):
        # x has shape T, N, C
        if not self.ONNX_export:
            x = x.permute(1, 2, 0)  # (N, C, T) for Conv1d
            for l in self.layers:
                if isinstance(l, nn.LayerNorm):
                    # LayerNorm expects input shape (N, T, C)
                    x = l(x.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    x = l(x)
            x = x.permute(2, 0, 1)  # (T, N, C)
        else:
            print('AAA onnx version')
            # if not platform.platform().startswith('macOS'):
            #     raise RuntimeError('Error: ONNX version of TemporalConvolution: see modules.py')
            #  this is for onnx: input with shape (N,T,C)
            x = x.permute(0, 2, 1)  # (N,T,C) to (N,C,T) for Conv1d
            for l in self.layers:
                if isinstance(l, nn.LayerNorm):
                    # LayerNorm expects input shape (N, T, C)
                    x = l(x.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    x = l(x)
            x = x.permute(0, 2, 1) # (N,C,T) to (N,T,C)

        return x



class transposed_AvgPool1d(nn.Module):

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: T,N,bands,channels_per_band
        x_shape = x.shape       # T, N, bands, channels_per_band
        x = x.permute(1, 2, 3, 0).reshape(-1, x_shape[0])
        x = self.pooling(x)  # (N*bands*channels_per_band, T//stride)
        x_out_shape = x.shape 
        x = x.reshape(x_shape[1], x_shape[2], x_shape[3], x_out_shape[-1])  # (N, bands, channels_per_band, T//stride)
        x = x.permute(3, 0, 1, 2)  # (T//stride, N, bands, channels_per_band)
        return x



''' BASELINE SOLUTION MODULES '''

class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        d_input: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0, "block_channels must be non-empty"
        self.d_input = d_input
        self.d_output = d_input

        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                d_input % channels == 0
            ), f"block_channels ({channels}) must evenly divide num_features ({d_input})"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, d_input // channels, kernel_width),
                    TDSFullyConnectedBlock(d_input),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, d_input)