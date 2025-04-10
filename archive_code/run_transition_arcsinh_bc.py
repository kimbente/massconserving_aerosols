import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score as r2

### GPU ###
# I have majority access to GPU on afternoons on even days, and mornings on odd days

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### DATA ###
# define full path 
path_to_data = "/home/kim/data/aerosols/aerosol_emulation_data/"

# NP for now
x_train = np.load(path_to_data + 'X_train.npy')
y_train = np.load(path_to_data + 'y_train.npy')

x_test = np.load(path_to_data + 'X_test.npy')
y_test = np.load(path_to_data + 'y_test.npy')

x_val = np.load(path_to_data + 'X_val.npy')
y_val = np.load(path_to_data + 'y_val.npy')

# Select the correct 24 columns
# x train, val, test
x_train_24 = x_train[:, 8:]
x_val_24 = x_val[:, 8:]
x_test_24 = x_test[:, 8:]

# y train, val, test
y_train_24 = y_train[:, :24]
y_val_24 = y_val[:, :24]
y_test_24 = y_test[:, :24]

# How much has it changes between x (at t = 0)  and y (at t = 1)
y_delta_train_24 = y_train_24 - x_train_24
y_delta_val_24 = y_val_24 - x_val_24
y_delta_test_24 = y_test_24 - x_test_24

### SPECIES ###
# Define column indices for each of the components (24 column version)
so4_indices = [0, 1, 2, 3, 4]
bc_indices = [5, 6, 7, 8]
oc_indices = [9, 10, 11, 12]
du_indices = [13, 14, 15, 16]

# Define aerosol species and their corresponding indices
species_indices = {
    'so4': so4_indices,
    'bc': bc_indices,
    'oc': oc_indices,
    'du': du_indices
}

### SPLIT ###
data_split = ['train', 'val', 'test']

# What are these indices?!
extra_indices = [17, 18, 19, 20, 21, 22, 23] 

##########################
### Normalise the data ###
##########################

def arcsinh_x_per_species(eps = 1e-5):
    # This transformation can be easily reversed and preserved the zero
    # Iterate over species
    # We arcsinh (universal) and normalise by species variance
    for species, indices in species_indices.items():
        # Iterate over data splits (train, val, test)
        for split in data_split:
            # Fetch the data
            x_split_species = globals()[f'x_{split}_24'][:, indices]
            # Make tensor
            x_split_species = torch.tensor(x_split_species).clone()

            # STEP 1: Clamp negative values as these should not be allowed
            nonneg_split_species = torch.clamp_min(x_split_species, min = 0.0)

            # STEP 2: Arcsinh
            arcsinh_split_species = torch.arcsinh(nonneg_split_species)

            # STEP 3: Scale to unit variance i.e. unit std
            arcsinh_unitvar_split_species = arcsinh_split_species / torch.std(arcsinh_split_species)
            # globals()[f'x_{split}_{species}_arcsinh_unitvar'] = arcsinh_split_species_unitvar

            # SAVE
            globals()[f'x_{split}_{species}_arcsinh'] = arcsinh_split_species
            # option
            globals()[f'x_{split}_{species}_arcsinh_unitvar'] = arcsinh_unitvar_split_species

# Call
arcsinh_x_per_species()

def arcsin_y_delta_per_species(eps = 1e-5):
    # Iterate over species
    for species, indices in species_indices.items():
        # Iterate over data splits (train, val, test)
        for split in data_split:
            # Fetch the data
            y_delta_split_species = globals()[f'y_delta_{split}_24'][:, indices]
            # Make tensor
            y_delta_split_species = torch.tensor(y_delta_split_species).clone()
            # We use this variable for og domain comparison
            globals()[f'y_delta_{split}_{species}'] = y_delta_split_species

            # STEP 1: scale to unit var
            y_delta_split_species_unitvar = y_delta_split_species / torch.std(y_delta_split_species)
            # globals()[f'y_delta_{split}_{species}_unitvar'] = y_delta_split_species_unitvar

            # export std to global namespace for renormalization
            globals()[f'std_y_delta_{split}_{species}'] = torch.std(y_delta_split_species)

            # STEP 2: Arcsin
            arcsinh_y_delta_split_species = torch.asinh(y_delta_split_species)
            arcsinh_unitvar_y_delta_split_species = torch.asinh(y_delta_split_species_unitvar)

            # SAVE both options
            globals()[f'y_delta_{split}_{species}_arcsinh'] = arcsinh_y_delta_split_species
            globals()[f'y_delta_{split}_{species}_arcsinh_unitvar'] = arcsinh_unitvar_y_delta_split_species

# Call
arcsin_y_delta_per_species()

# Helper
def n_random_row_incides(x, n = 5000):
    """Returns n random rows from x"""
    # Default is 5000
    indices = np.random.choice(x.shape[0], n, replace = False)
    return indices

#############
### MODEL ###
#############

class TransitionMatrix(nn.Module):
    def __init__(self, in_features, out_features, width, depth = 2, bias = 5.0):
        super(TransitionMatrix, self).__init__()
        # this is the self transition bias, added to the logits (prior knowledge), high val
        self.bias = bias
        self.out_features = out_features
        # define this once and reuse
        self.transition_identity = torch.eye(out_features).unsqueeze(0)
        self.eps = 1e-8

        # the first layer takes in_features * 2 as input, because we are concatenating the inputthe raltive inputs
        self.fc_in = nn.Linear(in_features = in_features * 2, out_features = width)

        self.hidden_layers = nn.ModuleList()
        for _ in range(depth -1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(in_features = width, out_features = width))
            self.hidden_layers.append(nn.ReLU())

        # for the transition matrix, we want to have a square matrix, 1 for scale factor
        self.fc_out = nn.Linear(in_features = width, out_features = (out_features * out_features) + 1)

        # Set all model parameters to double precision
        self.to(torch.float64)

    def forward(self, x):
        # x is shape(batch_size, in_features)
        # ensure double
        x = x.to(torch.float64)
        
        # Normalise the input to relative values (sum to 1), clamp to avoid div by zero
        x_relative = x / torch.clamp_min(x.sum(axis = -1), self.eps).unsqueeze(-1)

        # Concatenate the input with the relative input and pass into the first layer
        # shape(batch_size, in_features * 2)
        state = self.fc_in(torch.cat((x, x_relative), dim = -1))

        for layer in self.hidden_layers:
            state = layer(state)

        state = self.fc_out(state)

        # Divide the output into logits and scale factor
        logits_raw, scale_factor = state[:, :-1], state[:, -1]

        # Reshape to get (batch_size, out_features, out_features)
        logits = logits_raw.view(-1, self.out_features, self.out_features)

        if torch.isnan(logits).any():
            print("NaN detected at logits!")
        if torch.isinf(logits).any():
            print("Inf detected at logits!")

        # Add the bias to the diagonal (self transitions) with in-place operation
        logits.diagonal(dim1 = -2, dim2 = -1).add_(self.bias)

        # Apply softmax across each row so that columns (last dim) add to 1
        # rows add to 1 (From : To): 100% of the source are redistributed
        # Variance of a softmax is 1/N, variance of a arcsinh unit std is 1
        transition_matrix = F.softmax(logits, dim = -1)

        # Transition matrix without self-transitions
        # Repeat for bacth_size
        transition_matrix_no_diag = transition_matrix - self.transition_identity.repeat(transition_matrix.shape[0], 1, 1).to(device)

        # Multiply the input by the transition matrix without diagonal: bmm or matmul work
        deltas = torch.matmul(transition_matrix_no_diag, x.unsqueeze(-1)).squeeze(-1)

        # reshape scale factor to (batch_size, 1)
        # Scale of the estimated scale factor to account for smaller variance in softmax output (1/N) thus * N
        scaled_deltas = deltas * (scale_factor.unsqueeze(-1) * self.out_features)

        # correct delta numerical precision issue, but comp overhead
        corrected_scaled_deltas = scaled_deltas - scaled_deltas.mean(dim = (-1), keepdim = True)
        # print((deltas.mean(dim = (-1), keepdim = True)).shape)

        # return (batch_size, out_features)
        return corrected_scaled_deltas


###########
### RUN ###
###########

# need features to initialize the model: bc has 4 columns
in_feat = x_train_bc_arcsinh_unitvar.shape[-1]
out_feat = y_delta_train_bc_arcsinh_unitvar.shape[-1]

model = TransitionMatrix(in_features = in_feat, out_features = in_feat, width = 2).double().to(device)

# Define loss and optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = torch.nn.MSELoss()

# Make sure its training data
x_train = x_train_bc_arcsinh_unitvar.double().to(device)
y_train = y_delta_train_bc_arcsinh_unitvar.double().to(device)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True)

# Training loop
epochs = 10

# For some validation
val_row_indices = n_random_row_incides(x_val_bc_arcsinh_unitvar, n = 10000)

for epoch in range(epochs):
    model.train()
    
    # Iterate over batches
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        deltas = model(x_batch)
        
        # Calculate loss
        loss = torch.sqrt(criterion(deltas, y_batch))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        # print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

        if batch_idx % 1000 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Training RMSE Loss (transf.): {loss.item():.1f}')

        if batch_idx % 5000 == 0:
            model.eval()

            PRED_y_delta_val_arcsinh_unitvar = model(x_val_bc_arcsinh_unitvar[val_row_indices].to(device)).detach().cpu()
            # Project back by applying sinh first and then scaling
            PRED_y_delta_val_og = torch.sinh(PRED_y_delta_val_arcsinh_unitvar) * std_y_delta_val_bc

            print("Validation R2 Score on delta sample in transformed space:")
            print(f"{r2(PRED_y_delta_val_arcsinh_unitvar, y_delta_val_bc_arcsinh_unitvar[val_row_indices]).item():.3f}")

            print("Validation RMSE on delta sample in transformed space:")
            print(f"{torch.sqrt(criterion(PRED_y_delta_val_arcsinh_unitvar, y_delta_val_bc[val_row_indices])).item():.4f}")

            print("Validation R2 Score on delta sample in og space:")
            print(f"{r2(PRED_y_delta_val_og, y_delta_val_bc[val_row_indices]).item():.3f}")

            print("Validation RMSE on delta sample in og space:")
            print(f"{torch.sqrt(criterion(PRED_y_delta_val_og, y_delta_val_bc[val_row_indices])).item():.1f}")

            model.train()
        
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, RMSE Loss: {loss.item():.1f}')

    # overwrite after each epoch
    torch.save(model.state_dict(), os.path.join("models", "arcsinh_unitvar_transition_bc_10epochs.pth"))
