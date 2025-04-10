import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score as r2

###########
### GPU ###
###########
# I have majority access to GPU on afternoons on eve
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############
### DATA ###
############

# define full path 
path_to_data = "/home/kim/data/aerosols/aerosol_emulation_data/"

# Load and make torch tensors
# Select the correct 24 columns
x_train = torch.tensor(np.load(path_to_data + 'X_train.npy'))[:, 8:]
y_train = torch.tensor(np.load(path_to_data + 'y_train.npy'))[:, :24]

x_test = torch.tensor(np.load(path_to_data + 'X_test.npy'))[:, 8:]
y_test = torch.tensor(np.load(path_to_data + 'y_test.npy'))[:, :24]

x_val = torch.tensor(np.load(path_to_data + 'X_val.npy'))[:, 8:]
y_val = torch.tensor(np.load(path_to_data + 'y_val.npy'))[:, :24]

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

#####################
### PREPROCESSING ###
#####################

def preprocess_x(eps = 1e-5):
    # STEP 1:
    # Define global normalising constant taken from TRAIN (largest dataset)
    for species, indices in species_indices.items():
            x_train_species_arcsinh_std = torch.arcsinh(x_train[:, indices]).std()
            globals()[f"arcsinh_std_x_{species}"] = x_train_species_arcsinh_std

    for split in data_split:

        # Fetch the data
        x_split = globals()[f'x_{split}']

        # STEP 2: Arcsinh
        x_split_arcsinh = torch.arcsinh(x_split)

        # Copy for norm
        x_split_arcsinh_norm = x_split_arcsinh.clone()

        # STEP 3: normalise by species variance
        for species, indices in species_indices.items():
            # Overwrite: normalise by sepcies variance
            x_split_arcsinh_norm[:, indices] = x_split_arcsinh_norm[:, indices] / globals()[f"arcsinh_std_x_{species}"]

        # SAVE
        globals()[f'x_{split}_arcsinh_norm'] = x_split_arcsinh_norm

    ### COMBINE WITH SPECIES raw data ###
    for split in data_split:
        for species, indices in species_indices.items():
            # Fetch og x
            x_split_species = globals()[f'x_{split}'][:, indices]

            # Fetch arcsinh
            x_split_arcsinh_norm = globals()[f'x_{split}_arcsinh_norm']

            # Add relative rows - NOR for now
            # x_split_species_relative = x_split_species.sum(dim = -1)

            # Get indices of rows where the (OG) sum is zero or lower
            # Remove from x and y 
            # zero_sum_indices = (x_split_species.sum(dim = -1) <= 0).nonzero(as_tuple = True)[0]
            # print(split, species, zero_sum_indices.shape[0])
            # any_under_zero_indices = (x_split_species < 0).any(dim = - 1).nonzero(as_tuple = True)[0]
            # print(split, species, any_under_zero_indices.shape[0])

            # Combine data
            x_split_species_combined_data = torch.concat((x_split_arcsinh_norm, x_split_species), dim = -1)
            # print(x_split_species_combined_data.shape)

            # Write
            globals()[f'x_{split}_{species}_combi'] = x_split_species_combined_data

# Preprocesses x_train, x_val, x_test to output x_train_arcsinh_norm, x_val_arcsinh_norm, x_test_arcsinh_norm
preprocess_x()

# We keep y raw actually
def preprocess_y():
    for split in data_split:

        # Fetch the data
        x_split = globals()[f'x_{split}']
        y_split = globals()[f'y_{split}']

        # STEP 1: DELTA
        y_delta_split = y_split - x_split

        # STEP 2: Split into species
        for species, indices in species_indices.items():

            # Fetch y delta
            y_delta_split_species = y_delta_split[:, indices]

            # SAVE 
            globals()[f'y_delta_{split}_{species}'] = y_delta_split_species

preprocess_y()

#############
### MODEL ###
#############

class Transition_model_species_full_input(nn.Module):
    def __init__(self, out_features, width, depth = 2, bias = 13.0):
        super(Transition_model_species_full_input, self).__init__()
        # this is the self transition bias, added to the logits (prior knowledge), high val
        self.bias = bias
        # define to reshape Transition matrix
        self.out_features = out_features
        # define this once and reuse
        self.transition_identity = torch.eye(out_features).unsqueeze(0)
        self.eps = 1e-8

        # This model always takes 24 inputs 
        self.fc_in = nn.Linear(in_features = 24, out_features = width)

        self.hidden_layers = nn.ModuleList()
        for _ in range(depth -1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(in_features = width, out_features = width))
            self.hidden_layers.append(nn.ReLU())

        # for the transition matrix, we want to have a square matrix, 1 for scale factor
        self.fc_out = nn.Linear(in_features = width, out_features = (out_features * out_features))

        # Set all model parameters to double precision
        self.to(torch.float64)

    def forward(self, x):
        # x is shape(batch_size, 24 + species_features)
        # ensure double
        x = x.to(torch.float64)

        # Split the input into two parts
        x_for_network = x[:, :24]
        x_for_transition = x[:, 24:] # may be 4 or 5

        # Pass input into first layer shape(batch_size, 24)
        state = self.fc_in(x_for_network)

        for layer in self.hidden_layers:
            state = layer(state)

        logits = self.fc_out(state)

        # Reshape to get (batch_size, out_features, out_features)
        logits = logits.view(-1, self.out_features, self.out_features)
        # print(logits[0])

        if torch.isnan(logits).any():
            print("NaN detected at logits!")
        if torch.isinf(logits).any():
            print("Inf detected at logits!")

        # Add the bias to the diagonal (self transitions) with in-place operation
        logits.diagonal(dim1 = -2, dim2 = -1).add_(self.bias)
        # print(logits[0])

        # Apply softmax across each row so that column values of that row (last dim) add to 1
        # rows add to 1 (From : To): 100% of the source are redistributed
        # PREVIOUSLY ISSUE (had dim = -1 before which was wrong)
        transition_matrix = F.softmax(logits, dim = -2)

        # Transition matrix without self-transitions
        # Repeat for batch_size
        transition_matrix_no_diag = transition_matrix - self.transition_identity.repeat(transition_matrix.shape[0], 1, 1).to(device)

        # Multiply the input by the transition matrix without diagonal: bmm or matmul work
        deltas = torch.matmul(transition_matrix_no_diag, x_for_transition.unsqueeze(-1)).squeeze(-1)

        # return (batch_size, out_features)
        return deltas

model = Transition_model_species_full_input(out_features = 5, width = 64, depth = 2).to(device)

# Define loss and optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = torch.nn.MSELoss()

# Make sure its training data
x_train = x_train_so4_combi.double().to(device)
y_train = y_delta_train_so4.double().to(device)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True)

# Training loop
epochs = 10

### Validation data ###

# Helper
def n_random_row_incides(x, n = 5000):
    """Returns n random rows from x"""
    # Default is 5000
    indices = np.random.choice(x.shape[0], n, replace = False)
    return indices

# Fix indices for some validation
val_row_indices = n_random_row_incides(x_val, n = 10000)

x_val = x_val_so4_combi.double().to(device)
y_val = y_delta_val_so4.double().to(device)

x_val_subset = x_val[val_row_indices]
# y only ever needed on cpu
y_val_subset = y_val[val_row_indices].cpu()

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
            print(f'Epoch {epoch}, Batch {batch_idx}, Training RMSE Loss (og units): {loss.item():.4f}')

        if batch_idx % 5000 == 0:
            model.eval()

            PRED_y_delta_val = model(x_val_subset.to(device)).detach().cpu()

            print("Validation R2 Score on delta sample (og units):")
            print(f"{r2(PRED_y_delta_val, y_val_subset).item():.4f}")

            print("Validation RMSE on delta sample (og units):")
            print(f"{torch.sqrt(criterion(PRED_y_delta_val, y_val_subset)).item():.4f}")

            model.train()
        
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, RMSE Loss: {loss.item():.4f}')

    # overwrite after a few epoch
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join("models", "full_transition_model_so4_10_epochs.pth"))