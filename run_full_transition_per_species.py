import numpy as np
import pandas as pd
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score as r2

# Notes
# Model takes full 24 input features + 4/5 raw ones
# No relative term
# Output is not transformed
# Save wall
# Save full val results (RMSE and R2) and train results

###########
### GPU ###
###########
# I have majority access to GPU on afternoons on eve
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

######################
### Hyperparameter ###
######################
epochs = 100

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

####################
####################
### SPECIES LOOP ###
####################
####################

for species, indices in species_indices.items():
    species_features = len(indices)
    print(f"Species: {species}, features: {species_features}")

    model = Transition_model_species_full_input(out_features = species_features, width = 64, depth = 2).to(device)

    # Define loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = torch.nn.MSELoss()

    # Dynamically adjust training data based on species
    x_train = globals()[f"x_train_{species}_combi"].double().to(device)
    y_train = globals()[f"y_delta_train_{species}"].double().to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True)

    ### Validation data ###
    x_val = globals()[f"x_val_{species}_combi"].double().to(device)
    y_val = globals()[f"y_delta_val_{species}"].double().to(device)

    val_dataset = TensorDataset(x_val, y_val)
    # no shuffle needed
    val_loader = DataLoader(val_dataset, batch_size = 256, shuffle = False)

    # Initialise lists to track convergence 
    loss_per_epoch = []
    r2_per_epoch = []

    val_loss_per_epoch = []
    val_r2_per_epoch = []

    start_time = time.time()

    for epoch in range(epochs):
        print("Epoch: ", epoch, " / ", epochs)
        model.train()

        # Initlaise lists within this epoch
        loss_per_batch = []
        r2_per_batch = []
        
        # Iterate over batches
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            deltas_batch = model(x_batch)
            
            # Calculate loss for one batch
            loss_batch = torch.sqrt(criterion(deltas_batch, y_batch))
            with torch.no_grad():
                r2_batch = r2(deltas_batch, y_batch)
            
            # Backward pass
            loss_batch.backward()
            optimizer.step()

            loss_per_batch.append(loss_batch.item())
            r2_per_batch.append(r2_batch.item())
        
        loss_per_epoch.append(np.mean(loss_per_batch))
        r2_per_epoch.append(np.mean(r2_per_batch))
        
        # after each epoch
        model.eval()
        with torch.no_grad():

            # Initlaise lists within this epoch
            val_loss_per_batch = []
            val_r2_per_batch = []
            
            # Iterate over batches
            for batch_idx, (val_x_batch, val_y_batch) in enumerate(val_loader):
                
                # Forward pass
                val_deltas_batch = model(val_x_batch)
                
                # Calculate loss
                val_loss_batch = torch.sqrt(criterion(val_deltas_batch, val_y_batch))
                val_r2_batch = r2(val_deltas_batch, val_y_batch)

                val_loss_per_batch.append(val_loss_batch.item())
                val_r2_per_batch.append(val_r2_batch.item())

            val_loss_per_epoch.append(np.mean(val_loss_per_batch))
            val_r2_per_epoch.append(np.mean(val_r2_per_batch))

        # Back to training
        model.train()

        # overwrite after a few epoch
        if epoch % 10 == 0:
            model_path = os.path.join("models", f"full_transition_model_{species}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Epoch {epoch+1}/{epochs} commenced ({(epoch)/epochs*100:.2f}% completed)")

    # Create a DataFrame for losses
    results_df = pd.DataFrame({
        'epoch': list(range(epochs)),
        'train_loss': loss_per_epoch,
        'train_r2': r2_per_epoch,
        'val_loss': val_loss_per_epoch,
        'val_r2': val_r2_per_epoch
    })

    # Export to CSV
    results_df.to_csv(f"convergence/training_metrics_{species}.csv", index = False, float_format = "%.4f")

    # Append total wall clock time to the end of the file
    with open(f"convergence/training_metrics_{species}.csv", "a") as f:
        f.write(f"\nTotal training time (s), {elapsed_time:.2f}\n")
