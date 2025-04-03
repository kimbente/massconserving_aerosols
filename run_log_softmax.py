"""
Description : Train neural network model to predict one time step of M7
Options:

  --signs=<need_extra_signs_for_log_mass>
  --classification=<train_classification_net>
  --scale=<scaler>
  --model=<model_version>
"""

import numpy as np
from utils import standard_transform_x, standard_transform_y, get_model, train_model, create_report, calculate_stats, log_full_norm_transform_x, log_tend_norm_transform_y, create_dataloader, create_test_dataloader
# from models import Softmax_model
from utils import add_nn_arguments_jupyter
import torch.nn as nn 
import torch
import torch.optim as optim

from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F

# define full path 
path_to_data = "/home/kim/data/aerosols/aerosol_emulation_data/"

X_test = np.load(path_to_data + 'X_test.npy')
y_test = np.load(path_to_data + 'y_test.npy')

X_train = np.load(path_to_data + 'X_train.npy')
y_train = np.load(path_to_data + 'y_train.npy')

X_valid = np.load(path_to_data + 'X_val.npy')
y_valid = np.load(path_to_data + 'y_val.npy')

# Select the correct 24 columns
X_test_24 = X_test[:, 8:]
X_train_24 = X_train[:, 8:] 

y_test_24 = y_test[:, :24]
y_train_24 = y_train[:, :24]

y_valid_24 = y_valid[:, :24]
X_valid_24 = X_valid[:, 8:]

# How much has it changes between x (at t = 0)  and y (at t = 1)
y_delta_train_24 = y_train_24 - X_train_24
y_delta_test_24 = y_test_24 - X_test_24
y_delta_valid_24 = y_valid_24 - X_valid_24

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

# What are these indices?!
extra_indices = [17, 18, 19, 20, 21, 22, 23] 

# Define aerosol species and their corresponding indices

### ARGS ###
args = add_nn_arguments_jupyter()
# Overwrite the model name, keep everything else the same
# Have one model for now as each input dim can be different
args.model = 'log_softmax'
# args.model_id = 'transition_' + species # save different models
# Run for only 3 epochs for proof of concept
# Took around 2 mins per epoch
args.epochs = 3 
### DIFFERENT DIMS
# Takes a minute
# stats = calculate_stats(X_train, (y_train - X_train), X_test, (y_test - X_test), args)
# y's can be delata and 24, X is raw
stats = calculate_stats(X_train, y_delta_train_24, X_test, y_delta_test_24, args)

# Look at stats
np.set_printoptions(precision = 4, suppress = True, formatter = {'all': lambda x: f'{x:.4f}'})
# stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for species, indices in species_indices.items():
    print(f"Species: {species} - TRAINING COMMENCED")

    ### ARGS ###
    args = add_nn_arguments_jupyter()
    # Overwrite the model name, keep everything else the same
    # Have one model for now as each input dim can be different
    args.model = 'log_softmax' # Softmax model
    args.model_id = 'logsoftmax_' + species # save different models
    # Run for only 3 epochs for proof of concept
    # Took around 2 mins per epoch
    # args.epochs = 1
    # consider learning rate

    args.epochs = 3

    # args.save_val_scores = True # For oversight
    args.loss = 'rmse'


    ### TRAIN ###
    # Fetch the arrays using globals()
    x_train_species = X_train_24[:, indices]
    y_train_species = y_delta_train_24[:, indices]

    x_valid_species = X_valid_24[:, indices]
    y_valid_species = y_delta_valid_24[:, indices]

    print(x_train_species.shape, y_train_species.shape)

    input_dim = x_train_species.shape[1]
    output_dim = y_train_species.shape[1]

    # Create dataloaders: x, y
    train_data_species = create_dataloader(x_train_species, y_train_species, args)
    valid_data_species = create_test_dataloader(x_valid_species, y_valid_species, args)

    # Initalize model
    model = get_model(
        in_features = input_dim, 
        out_features = output_dim, 
        args = args, 
        constraints_active = False)

    if args.mode == 'train':
            
        optimizer = optim.Adam(
                model.parameters(), 
                lr = args.lr, 
                weight_decay = args.weight_decay)

        train_model(
                model = model, 
                train_data = train_data_species, # data loader
                test_data = valid_data_species, # validation
                optimizer = optimizer, 
                input_dim = input_dim, 
                output_dim = output_dim, 
                stats = stats, # !!! Stats are used for the transforms
                X_test = x_valid_species, #??
                y_test = y_valid_species, #??
                args = args)
        # Saves the model automatically
    
    ### LOAD trained model ###
    model = get_model(
        in_features = input_dim, out_features = output_dim, args = args, constraints_active = True
        # KB: constraints_active = True This is not used for the softmax model
    ) 
    model.load_state_dict(torch.load('./models/' + args.model_id + '.pth') ['state_dict'])
    model.to(device)
    # Evaluate
    model.eval()

    # Fetch the test arrays using globals()
    x_test_species = X_test_24[:, indices]
    y_test_species = y_delta_test_24[:, indices]
    y_test_species_absolute = y_test_24[:, indices]

    # Model output is the tendency: rows of tendencies sum to zero as we just "redistribute" mass
    y_test_species_tend_PRED = model(torch.tensor(x_test_species).to(device).float())
    # Absolue Prediction. Project back using sums from x_test (not y_test itself)
    y_test_species_absolute_PRED = y_test_species_tend_PRED + torch.tensor(x_test_species).to(device).float()

    # sklearn function, same as np.square(relative_error).mean()
    # relative is implicit in naming
    MSE_tend = mean_squared_error(y_test_species, y_test_species_tend_PRED.detach().cpu().numpy())
    R2_tend = r2_score(y_test_species, y_test_species_tend_PRED.detach().cpu().numpy())
    print(f'Species: {species} | MSE tendency:', MSE_tend)
    print(f'Species: {species} | R2 tendency:', R2_tend)

    # true, pred
    MSE_abs = mean_squared_error(y_test_species_absolute, y_test_species_absolute_PRED.detach().cpu().numpy())
    R2_abs = r2_score(y_test_species_absolute, y_test_species_absolute_PRED.detach().cpu().numpy())
    print(f'Species: {species} | MSE absolute:', MSE_abs)
    print(f'Species: {species} | R2 absolute:', R2_abs)

    break

# Call it here
if __name__ == "__main__":
    args = add_nn_arguments()
    main(args)