# File: models/taylor_mlp_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.activation_derivatives import ACTIVATION_DERIVATIVES_CALCULATORS

class TaylorMLP_Layer(nn.Module):
    def __init__(self, in_features, out_features, D_h_taylor, N_taylor, activation_name="gelu"):
        super(TaylorMLP_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.D_h_taylor = D_h_taylor
        self.N_taylor = N_taylor
        self.activation_name = activation_name
        # self.act_fn_deriv_calculator = ACTIVATION_DERIVATIVES_CALCULATORS[activation_name] # Not directly used if Theta is learned

        # Parameters corresponding to V, b, z0, Theta in Taylor-Unswift model
        # W1, b1: Weights and bias for the first linear transformation (V, b in paper's Algo 1)
        self.W1 = nn.Parameter(torch.Tensor(D_h_taylor, in_features))
        self.b1 = nn.Parameter(torch.Tensor(D_h_taylor))

        # z0: Expansion point (learnable)
        self.z0 = nn.Parameter(torch.Tensor(D_h_taylor))

        # Theta: Learnable Taylor coefficients 
        # Shape: (N_taylor + 1, out_features, D_h_taylor)
        # Corresponds to Theta_{n,o,d} where n is Taylor term, o is output_feature_idx, d is D_h_taylor_idx
        self.Theta = nn.Parameter(torch.Tensor(N_taylor + 1, out_features, D_h_taylor))
        
        self.init_parameters()

    def init_parameters(self):
        # Initialize W1, b1 (first linear layer)
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b1, -bound, bound)

        # Initialize z0 (e.g., to zeros)
        nn.init.zeros_(self.z0)
        
        # Initialize Theta (Taylor coefficients)
        # For each (out_features, D_h_taylor) matrix within Theta[n,:,:]
        for n in range(self.N_taylor + 1):
             nn.init.kaiming_uniform_(self.Theta[n], a=math.sqrt(5))

    def forward(self, x):
        # x shape: (batch_size, self.in_features)
        
        # 1. Compute z = W1 @ x + b1 (first linear transformation)
        # z = F.linear(x, self.V_layer_weights, self.V_layer_bias)
        z = F.linear(x, self.W1, self.b1) # z shape: (batch_size, D_h_taylor)
        
        # 2. Compute diff_z_z0 = z - self.z0 (element-wise for each item in batch)
        # z0 shape: (D_h_taylor)
        diff_z_z0 = z - self.z0 # Broadcasting z0 to (batch_size, D_h_taylor)

        # 3. Compute Taylor series expansion sum efficiently
        # y_i = sum_{n=0 to N} <Theta_{i,n}, (z-z0)^n>
        # We need (z-z0)^n for n=0 to N. Shape for each: (batch_size, D_h_taylor)
        # Stack them: (N_taylor+1, batch_size, D_h_taylor)
        
        taylor_terms_pow_N = []
        for n_idx in range(self.N_taylor + 1):
            if n_idx == 0:
                taylor_terms_pow_N.append(torch.ones_like(diff_z_z0))
            else:
                taylor_terms_pow_N.append(diff_z_z0 ** n_idx)
        
        # Stack along a new dimension (dim 0 for 'n' in einsum)
        # stacked_taylor_terms shape: (N_taylor+1, batch_size, D_h_taylor)
        stacked_taylor_terms = torch.stack(taylor_terms_pow_N, dim=0)

        # Theta shape: (N_taylor+1, out_features, D_h_taylor)
        # einsum: "nbd,nod->bo"
        # n: N_taylor+1 dimension
        # b: batch_size dimension
        # d: D_h_taylor dimension (summed over)
        # o: out_features dimension
        # Result shape: (batch_size, out_features)
        output = torch.einsum("nbd,nod->bo", stacked_taylor_terms, self.Theta)
            
        return output

    def initialize_theta_from_conceptual_mlp(self, W2_conceptual, c2_conceptual, activation_name, N_taylor_override=None):
        """
        Initializes Theta based on conceptual second MLP layer weights (W2, c2),
        and current self.b1, self.z0.
        This is for advanced initialization if starting from a known MLP structure.
        W2_conceptual: (out_features, D_h_taylor)
        c2_conceptual: (out_features)
        """
        N_to_use = N_taylor_override if N_taylor_override is not None else self.N_taylor
        if self.Theta.shape[0] != N_to_use + 1:
            print(f"Warning: Theta shape mismatch during re-initialization. Expected N={self.Theta.shape[0]-1}, got {N_to_use}")
            # Potentially re-create self.Theta here if N changes, or ensure N_to_use matches
            self.Theta = nn.Parameter(torch.Tensor(N_to_use + 1, self.out_features, self.D_h_taylor).to(self.W1.device))


        act_deriv_calculator = ACTIVATION_DERIVATIVES_CALCULATORS[activation_name]
        
        act_input_for_theta_calc = self.z0.data + self.b1.data # Shape: (D_h_taylor)
        
        # Theta_0: W2 * Act(z0+b1) + c2
        act_val_at_z0_plus_b1 = act_deriv_calculator(act_input_for_theta_calc, 0) # Act(z0+b1)
        # W2_conceptual * act_val (element-wise): (out_f, D_h) * (D_h).unsqueeze(0) -> (out_f, D_h)
        term1_theta0 = W2_conceptual * act_val_at_z0_plus_b1.unsqueeze(0)
        self.Theta.data[0] = term1_theta0 + c2_conceptual.unsqueeze(1) # c2 unsqueezed to (out_f, 1)

        # Theta_n for n > 0: W2 * Act^(n)(z0+b1) / n!
        factorials = [math.factorial(i) for i in range(N_to_use + 1)]
        for n in range(1, N_to_use + 1):
            act_deriv_n_val = act_deriv_calculator(act_input_for_theta_calc, n) # Act^(n)(z0+b1)
            term1_thetan = W2_conceptual * act_deriv_n_val.unsqueeze(0)
            self.Theta.data[n] = term1_thetan / factorials[n]
        print(f"Theta coefficients initialized from conceptual MLP parameters using {activation_name}.")