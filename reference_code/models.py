import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


###### models

class Base(nn.Module):
    def __init__(self, in_features, out_features, width, depth = 2):
        super(Base, self).__init__()        
        self.fc_in = nn.Linear(in_features = in_features, out_features = width)
        self.hidden_layers = nn.ModuleList()
        for i in range(depth -1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(in_features = width, out_features = width))
            self.hidden_layers.append(nn.ReLU())
        self.fc_out = nn.Linear(in_features = width, out_features = out_features)
    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_out(x)
        return x

### KB add ###
class LogSoftmax_model(nn.Module):
    def __init__(self, in_features, out_features, width, depth = 2):
        super(LogSoftmax_model, self).__init__()
        self.fc_in = nn.Linear(in_features = in_features, out_features = width)
        # Create the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(depth - 1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(in_features = width, out_features = width))
            self.hidden_layers.append(nn.ReLU())
        # Output layer (fc: fully connected)
        self.fc_out = nn.Linear(in_features = width, out_features = out_features + 1)
        # ADD softmax layer: same as probabilities per class (classification)
        self.softmax = nn.Softmax(dim = -1)  # Apply softmax along the output dimension

    def forward(self, x):
        # x_relative = 
        x_log = torch.log(torch.clamp(x.clone(), min = 1e-8).requires_grad_())  # Apply log transformation to the input
        # some values are small zero so we need to clamp them

        if torch.isnan(x_log).any():
            print("NaNs detected in log!")

        # Pass through the input layer (fully connected)
        out = self.fc_in(x_log)
        # Pass through hidden layers
        for layer in self.hidden_layers:
            out = layer(out)
        # Final output layer
        out = self.fc_out(out)

        # Split it up
        softmax_input = out[:, :-1]
        scalar = out[:, -1]

        # Apply softmax to the final output for classification
        # calculate in double precision
        softmax_out = self.softmax(softmax_input) #.double()

        if torch.isnan(softmax_out).any():
            print("NaNs detected in softmax_out!")

        # Make softmax zero_sum
        zero_sum_output = softmax_out - softmax_out.mean(dim = -1, keepdim = True)

        scaled_zero_sum_output = zero_sum_output * scalar.unsqueeze(1)

        if torch.isnan(scaled_zero_sum_output).any():
            print("NaNs detected in scaled zero sum!")

        # Avoid division by zero by ensuring denominator is never zero
        denominator = scaled_zero_sum_output.clone()
        denominator = torch.where(denominator == 0, torch.tensor(1e-10).to(denominator.device), denominator)

        ### Ensure non-negativivity constraint ###
        safe_beta = torch.where((
            (scaled_zero_sum_output + x) < 0.0), # In the case of negative values
            (- x / denominator), # scalar candidates, maybe add noise?! works per row
            torch.tensor(float('inf')) # infinity if no violation (so it doesn't get selected)
            ).min(dim = 1)[0].unsqueeze(-1) # select the minimum value over columns (for each row)
            # minimum by which we have to scale it backk. 0 in worst case
        # unsqueeze to make torch.Size([n_batch, 1])

        # In case of no violation (safe_beta == inf), set beta to 1 - no change occurs
        # In case of violation, set beta to the minimum value (zero in worst case)
        # row-wise i.e. batch-wise min selection
        beta = torch.min(torch.ones(safe_beta.shape).to(safe_beta.device), safe_beta)

        safe_out = beta * zero_sum_output

        if torch.isnan(safe_out).any():
            print("NaNs detected in safe out!")

        return safe_out

class TransitionMM(nn.Module):
    def __init__(self, in_features, out_features, width, depth = 2, alpha = 0.999):
        super(TransitionMM, self).__init__()
        self.alpha = alpha
        self.out_features = out_features
        self.fc_in = nn.Linear(in_features = in_features, out_features = width)
        self.hidden_layers = nn.ModuleList()
        for i in range(depth -1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(in_features = width, out_features = width))
            self.hidden_layers.append(nn.ReLU())
        # square output
        self.fc_out = nn.Linear(in_features = width, out_features = out_features * out_features) # square output
    def forward(self, x):
        # Pass in relative values into the model as some normalisation
        # consider concating both
        x_norm = x / x.sum(axis = 1).unsqueeze(-1)
        state = self.fc_in(x_norm)
        for layer in self.hidden_layers:
            state = layer(state)
        state = self.fc_out(state)
        state = state.view(-1, self.out_features, self.out_features)
        tm = F.softmax(state, dim = 1)  # Apply softmax across each columns to that columns (last dim) add to 1
        # tm is shape(batch_size, out_features, out_features)
        # Stabilise the transition matrix (nudge towards identity)
        stabiliser = torch.eye(tm.shape[-1]).repeat(tm.shape[0], 1, 1).to(tm.device)
        # Convex Combination of Transition Matrices is still a Transition Matrix
        stable_tm = self.alpha * stabiliser + (1 - self.alpha) * tm
        # x is shape(batch_size, in_features)
        out = torch.bmm(stable_tm, x.unsqueeze(-1)).squeeze(-1)
        # predict the delta (tendencies)
        y_delta = out - x
        return torch.log(y_delta)

class Softmax_model(nn.Module):
    def __init__(self, in_features, out_features, width, depth = 2):
        super(Softmax_model, self).__init__()
        self.fc_in = nn.Linear(in_features = in_features, out_features = width)
        # Create the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(depth - 1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(in_features = width, out_features = width))
            self.hidden_layers.append(nn.ReLU())
        # Output layer (fc: fully connected)
        self.fc_out = nn.Linear(in_features = width, out_features = out_features)
        # ADD softmax layer: same as probabilities per class (classification)
        self.softmax = nn.Softmax(dim = -1)  # Apply softmax along the output dimension
    def forward(self, x):
        # Pass through the input layer (fully connected)
        x = self.fc_in(x)
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        # Final output layer
        x = self.fc_out(x)
        # Apply softmax to the final output for classification
        x = self.softmax(x)
        return x
    
# base network concatenating the signs for log case
class SignExtBase(nn.Module):
    def __init__(self, in_features, out_features, width):
        super(SignExtBase, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
    def forward(self, x_in):
        x = self.fc1(x_in)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = torch.cat((x, x_in[:,32:]), dim=1) # concatenate the signs
        return x

# What is this used for?
class ClassificationNN(nn.Module):
    def __init__(self, in_features, out_features, width):
        super(ClassificationNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
        self.act3 = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        return x
    
# outputs inputs as well for positivity enforcement
class PositivityNN(nn.Module):
    def __init__(self, in_features, out_features, width):
        super(PositivityNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
    def forward(self, x_in):
        x = self.fc1(x_in)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x_out = torch.cat((x, x_in[:,8:]), dim=1) 
        return x_out
    
class CompletionLayer(nn.Module):
    def __init__(self, mu_y, si_y):
        super(CompletionLayer, self).__init__()
        self.mu_y = mu_y
        self.si_y = si_y
        
    def forward(self,x):
        x_out = torch.clone(x)
        # fixed which one is completed
        x_out[:,4] =(- torch.sum(x[:,:4]*self.si_y[:4]+self.mu_y[:4], dim=1)-self.mu_y[4])/self.si_y[4]
        inds7 = [5,6,8]
        x_out[:,7] = (-torch.sum(x[:,inds7]*self.si_y[inds7]+self.mu_y[inds7], dim=1)-self.mu_y[7])/self.si_y[7]
        inds11 = [9,10,12]
        x_out[:,11] = (-torch.sum(x[:,inds11]*self.si_y[inds11]+self.mu_y[inds11], dim=1)-self.mu_y[11])/self.si_y[11]
        x_out[:,13] = (-torch.sum(x[:,14:17]*self.si_y[14:17]+self.mu_y[14:17], dim=1)-self.mu_y[13])/self.si_y[13] 
        return x_out

    
class CorrectionLayer(nn.Module):
    def __init__(self, mu_y, si_y, mu_x, si_x):
        super(CorrectionLayer, self).__init__()
        self.mu_y = mu_y
        self.si_y = si_y
        self.mu_x = mu_x
        self.si_x = si_x
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
    def forward(self,x):
        y_orig = x[:,:28]*self.si_y[:28]+self.mu_y[:28] #output in original scal
        x_orig = x[:,28:]*self.si_x[8:]+self.mu_x[8:] #input in orginal scale
        pos = self.relu1(y_orig[:,:24]+x_orig)
        x[:,:24] = pos - x_orig 
        x[:,:24] = (x[:,:24]-self.mu_y[:24])/self.si_y[:24]      
        x[:,24:28] = self.relu2(y_orig[:,24:28])
        x[:,24:28] = (self.relu2(y_orig[:,24:28])-self.mu_y[24:28])/self.si_y[24:28]
        return x[:,:28]
    
    
class CompletionNN(nn.Module):
    def __init__(self, in_features, out_features, width, mu_y, si_y, activate_completion):
        super(CompletionNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
        self.completion = CompletionLayer(mu_y, si_y)
        self.completion_active = activate_completion
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        if self.completion_active:
            x = self.completion(x)
        return x
    
class CorrectionNN(nn.Module):
    def __init__(self, in_features, out_features, width, mu_y, si_y, mu_x, si_x, activate_correction):
        super(CorrectionNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
        self.correction = CorrectionLayer(mu_y, si_y, mu_x, si_x)
        self.correction_active = False
    def forward(self, x_in):
        x = self.fc1(x_in)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        if self.correction_active:
            x = self.correction(torch.cat((x, x_in[:,8:]), dim=1) )
        return x