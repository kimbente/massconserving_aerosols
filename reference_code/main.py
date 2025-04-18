"""
Description : Train neural network model to predict one time step of M7
Options:

  --signs=<need_extra_signs_for_log_mass>
  --classification=<train_classification_net>
  --scale=<scaler>
  --model=<model_version>
"""
import numpy as np
from utils import standard_transform_x, standard_transform_y, get_model, train_model, create_report, calculate_stats, add_nn_arguments, log_full_norm_transform_x, log_tend_norm_transform_y, create_dataloader, create_test_dataloader
import torch.nn as nn 
import torch
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    #loading data
    print('loading data...')    
    X_train = np.load('/home/kim/data/aerosols/aerosol_emulation_data/X_train.npy')
    y_train = np.load('/home/kim/data/aerosols/aerosol_emulation_data/y_train.npy')  
    
    if args.mode == 'train':
        X_test = np.load('/home/kim/data/aerosols/aerosol_emulation_data/X_val.npy')
        y_test = np.load('/home/kim/data/aerosols/aerosol_emulation_data/y_val.npy')
    else:
        X_test = np.load('/home/kim/data/aerosols/aerosol_emulation_data/X_test.npy')
        y_test = np.load('/home/kim/data/aerosols/aerosol_emulation_data/y_test.npy')
    
    if args.old_data:
        X_train = np.delete(X_train, [21,22],axis=1) 
        X_test = np.delete(X_test, [21,22],axis=1)
    #calculate tendencies
    y_train[:,:24] -= X_train[:,8:]
    y_test[:,:24] -= X_test[:,8:]
    stats = calculate_stats(X_train, y_train, X_test, y_test, args)
    
    if args.signs:
        y_train_sign = np.sign(y_train)
        y_test_sign = np.sign(y_test)
        
    if args.model == 'classification':
        inds = [1,2,3,5,6,9,10,13,17,18,19]
        y_train_signs = np.ones_like(y_train)
        y_test_signs = np.ones_like(y_test)
        y_train_signs[y_train<0] = 0
        y_test_signs[y_test<0] = 0
        y_train = y_train_signs[:,inds]
        y_test = y_test_signs[:,inds]

    #transform
    if args.scale == 'z':
        X_train = standard_transform_x(stats, X_train)
        X_test = standard_transform_x(stats, X_test)
        if not args.model == 'classification':
            y_test= standard_transform_y(stats, y_test)
            y_train = standard_transform_y(stats, y_train)
    elif args.scale == 'log':
        X_train = log_full_norm_transform_x(stats, X_train)
        X_test = log_full_norm_transform_x(stats, X_test)    
        if not args.model == 'classification':
            y_train = log_tend_norm_transform_y(stats, y_train)
            y_test= log_tend_norm_transform_y(stats, y_test)
    if args.signs:
        X_train = np.concatenate((X_train, y_train_signs), axis=1)
        X_test = np.concatenate((X_test, y_test_signs), axis=1)
        
        
    print(y_train.shape, y_test.shape)

    ### Insert more code here

if __name__ == "__main__":
    args = add_nn_arguments()
    main(args)