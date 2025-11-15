import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import matplotlib.pyplot as plt
import copy
from emg2qwerty.modules import SparseReLu, SparseSugar


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()

        # N is the latent space dimension (for each input channel)
        H = d_model                 # input channel size for the S4D Layer (each one independent from the other, but stacked in the computations)
        
        # Generate dt
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        # "special" initialization of the S4D A matrix (i.e. diagonal version of diagonal part of the S4 DiagonalPlusLowRank A matrix)
        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)   # A_imag has shape (H, N//2) and every raw contains increasing multpiples of pi
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)                  
        C = torch.view_as_complex(self.C)                   # (H N//2)                      
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N//2)     

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N/2 L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_state, 
                 activation,
                 dropout=0.0, 
                 dropout_tie=True,
                 transposed=True, 

                 mid_layer_pool_mode='none', 
                 mid_layer_pool=0,
                 
                 sparsity_post_actv_S4D=0.0,

                 **kernel_args):
        super().__init__()

        self.mid_layer_pool_mode = mid_layer_pool_mode
        self.mid_layer_pool = mid_layer_pool

        self.sparsity_post_actv_S4D = sparsity_post_actv_S4D
        if sparsity_post_actv_S4D > 0:
            assert 'sparse_relu' in activation or 'sparse_sugar' in activation, "sparsity_post_actv_S4D > 0 requires sparse_relu or sparse_sugar activation"

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed        # True if input is (B, H, L) [default] else (B, L, H)

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif 'sparse_relu' in activation:
            if 'tie' in activation:
                tie_thr = True                  # sparse_relu_tie_X
            else:
                tie_thr = False                 # sparse_relu_X
            self.sg_smoothstep = float(activation.split('_')[-1])
            self.activation = SparseReLu(self.d_output, tie_thr, self.sg_smoothstep)
        elif 'sparse_sugar' in activation:
            self.sg_smoothstep = float(activation.split('_')[-1])
            self.activation = SparseSugar(self.sg_smoothstep)
        else:
            raise NotImplementedError(f"Unknown activation function for S4D {activation}")
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout, tie=dropout_tie) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features    # position is time 
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),     # it is basically a linear layer identically applied over the sequence lenght (L) ----> B 2H L
            nn.GLU(dim=-2),                                 # GLU = gated linear unit: introduces a gating mechanism on top of the linear layer  ----> B H L x sigmoid(B H L) = B H L
        )


        self.upsample_last = 1  # used only if upsample_causal=True, else it must be 1 (and upsampling is handled in modules.S4Model)
        self.upsample_causal = True


    def step_forward(self, x, Abar, A, C, D, u):
        # u: (B, H)
        # Update hidden state
        x = (Abar.unsqueeze(0) * x) + u.unsqueeze(-1)*(Abar-1.)/A    # (B, H, N/2)
        
        # Compute output
        y = 2*torch.einsum('hn,bhn->bh', C, x).real + u * D   # B, H

        return x, y



    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        if not self.transposed: u = u.transpose(-1, -2)
        """ Input u and output y have shape (B, H, L) """
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)        # note: k is real

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)  # (B H L): multiplies by D, for each batch element and for each timestamp


        if self.upsample_last >1 and self.upsample_causal:
            B, H, L = u.shape
            y_upscaled = torch.zeros((B, H, L, self.upsample_last), dtype=torch.float32, device=u.device) 
            y_upscaled[:,:, :, -1] = y

            log_dt = self.kernel.log_dt         # (H)

            A = -torch.exp(self.kernel.log_A_real) + 1j * self.kernel.A_imag      # (H,N//2)
            C = torch.view_as_complex(self.kernel.C)                              # (H,N//2)
            dt = torch.exp(log_dt)                                                # (H)
            dt_upscaled = dt/self.upsample_last                                   # (H)

            Abar = torch.exp(dt.unsqueeze(-1) * A)   # (H,N//2)
            # Cbar = C*(Abar-1.)/A                     # (H,N//2)

            Abar_upscaled = torch.exp(dt_upscaled.unsqueeze(-1) * A)   # (H,N//2)
            # Cbar_upscaled = C*(Abar_upscaled-1.)/A                     # (H,N//2)

            x_t = torch.zeros((B, H, self.n//2), dtype=torch.cfloat, device=u.device)  # initial state
            # maxs = []
            for t in range(L-1):
                # print(f'{t} of {L}')
                u_t = u[:,:, t] 
                x_t, y_t = self.step_forward(x_t, Abar, A, C, self.D, u_t)
                assert torch.max(torch.abs(y_t.flatten()- y_upscaled[:,:, t, -1].flatten())) < 1e-3, f"y_t and y_upscaled do not match: {torch.max(torch.abs(y_t.flatten()- y_upscaled[:,:, t, -1].flatten()))}"
                
                x_t_temp = x_t.clone()
                # use x_t and u_t to predict the (future: between t and t+1) upsampled outputs
                for minor_t in range(self.upsample_last-1):
                    x_t_temp, y_t = self.step_forward(x_t_temp, Abar_upscaled, A, C, self.D, u_t)
                    y_upscaled[:,:, t+1, minor_t] = y_t.clone()
            
            y = y_upscaled.reshape(B, H, L*self.upsample_last)
                


        if self.mid_layer_pool_mode!='none' and self.mid_layer_pool>0:
            if y.shape[2] % self.mid_layer_pool != 0:
                y = y[:,:,:-(y.shape[2] % self.mid_layer_pool)] 
            if self.mid_layer_pool_mode == 'mean':
                y = y.reshape(y.shape[0], y.shape[1], -1, self.mid_layer_pool).mean(dim=-1)
            elif self.mid_layer_pool_mode == 'max':
                y = y.reshape(y.shape[0], y.shape[1], -1, self.mid_layer_pool).max(dim=-1).values
            elif self.mid_layer_pool_mode == 'last':
                y = y.reshape(y.shape[0], y.shape[1], -1, self.mid_layer_pool)[:,:,:, -1]
        
        y = self.activation(y)
        if type(y) == tuple:  # if SparseReLu or SparseSugar
            y, o = y
        else:
            o = y

        if self.sparsity_post_actv_S4D>0:
            sparsity_post_actv_S4D_val = torch.mean(o)
        else:
            sparsity_post_actv_S4D_val = torch.mean(o).detach().cpu().item()
        sparsity_post_actv_S4D_size = o.numel()
        sparsity_post_actv_S4D_zeros = sparsity_post_actv_S4D_size - torch.count_nonzero(o).item()

        y = self.dropout(y)
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None, { 'sparsity_post_actv_S4D': [sparsity_post_actv_S4D_val,sparsity_post_actv_S4D_size,sparsity_post_actv_S4D_zeros] } # Return a dummy state to satisfy this repo's interface, but this can be modified

