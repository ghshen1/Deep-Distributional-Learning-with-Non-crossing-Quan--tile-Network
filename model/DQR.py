import numpy as np
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# class DQR_arch(nn.Module):
#     def __init__(self, X_means, X_stds, y_mean, y_std, n_out, width_vec=None):
#         super(DQR_arch, self).__init__()
#         self.X_means = X_means
#         self.X_stds = X_stds
#         self.y_mean = y_mean
#         self.y_std = y_std
#         self.n_in = X_means.shape[1]
#         self.n_out = n_out
#         self.y_dim = 1 if len(y_mean.shape) == 1 else y_mean.shape[1]
#         self.width_vec = width_vec if width_vec is not None else [self.n_in, 200, 200, self.out]

        
#         layers = []
#         for i in range(len(self.width_vec) - 2):
#             layers.append(nn.Linear(self.width_vec[i], self.width_vec[i+1]))
#             #layers.append(nn.Dropout(0.1))
#             layers.append(nn.ReLU())
#             #layers.append(nn.BatchNorm1d(self.width_vec[i+1]))
#         layers.append(nn.Linear(self.width_vec[-2], self.width_vec[-1]))
        
#         self.fc_in = nn.Sequential(*layers)
#         self.softplus = nn.Softplus()
      
#     def forward(self, x):
#         fout = self.fc_in(x)
#         out = torch.cat((fout[:,0].unsqueeze(1),fout[:,0].unsqueeze(1) + torch.cumsum(self.softplus(fout[:,1:]),dim=1)),dim=1)
#         return out

#     def predict(self, X):
#         self.eval()
#         self.zero_grad()
#         tX = autograd.Variable(torch.FloatTensor((X - self.X_means) / self.X_stds), requires_grad=False)
#         with torch.no_grad():
#             pred = self.forward(tX) * self.y_std + self.y_mean
#         return pred



class DQR_arch(nn.Module):
    def __init__(self, X_means, X_stds, y_mean, y_std, n_out, width_vec=None, non_crossing=True):
        super(DQR_arch, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_in = X_means.shape[1]
        self.n_out = n_out
        self.y_dim = 1 if len(y_mean.shape) == 1 else y_mean.shape[1]
        self.non_crossing = non_crossing
        self.width_vec = width_vec if width_vec is not None else [self.n_in, 200, 200, self.n_out]

        layers = []
        for i in range(len(self.width_vec) - 2):
            layers.append(nn.Linear(self.width_vec[i], self.width_vec[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.width_vec[-2], self.width_vec[-1]))
        self.fc_in = nn.Sequential(*layers)

        if self.non_crossing:
            self.softplus = nn.Softplus()

    def forward(self, x):
        fout = self.fc_in(x)
        if self.non_crossing:
            # Non-crossing output: first quantile + cumulative softplus increments
            out = torch.cat(
                (
                    fout[:, 0].unsqueeze(1),
                    fout[:, 0].unsqueeze(1) + torch.cumsum(self.softplus(fout[:, 1:]), dim=1)
                ),
                dim=1
            )
        else:
            # Standard MLP output (no special activation)
            out = fout
        return out

    def predict(self, X):
        self.eval()
        self.zero_grad()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means) / self.X_stds), requires_grad=False)
        with torch.no_grad():
            pred = self.forward(tX) * self.y_std + self.y_mean
        return pred




class DQR:
    def __init__(self, quantiles, non_crossing=True):
        self.quantiles = quantiles
        self.label = 'DQR'
        self.filename = 'DQR'
        self.non_crossing=non_crossing
        
    def fit(self, X, y, width_vec=[256,256,256], lr =0.1, epochs=100):
        if width_vec==None:
            width_vec =[X.shape[1],200,200,len(self.quantiles)]
  
        self.model = fit_quantiles(X, y, quantiles= self.quantiles, non_crossing = self.non_crossing, width_vec = width_vec, 
                                       lr =lr, nepochs = epochs)

    def predict(self, X):
        return self.model.predict(X)

def fit_quantiles(X, y, quantiles=0.5, width_vec=[256,256,256], non_crossing=True,
                    nepochs=100, val_pct=0.25,
                    batch_size=None, target_batch_pct=0.01,
                    min_batch_size=20, max_batch_size=100,
                    verbose=False, lr=1e-1, weight_decay=0.0, patience=5,
                    init_model=None, splits=None, 
                    clip_gradients=False, clip_value=5,**kwargs):
   
    if batch_size is None:
        batch_size = min(X.shape[0], max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct)))))
        if verbose:
            print('Auto batch size chosen to be {}'.format(batch_size))

    # Standardize the features and response (helps with gradient propagation)
    Xmean = X.mean(axis=0, keepdims=True)
    Xstd = X.std(axis=0, keepdims=True)
    Xstd[Xstd == 0] = 1 # Handle constant features
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    tX = autograd.Variable(torch.FloatTensor((X - Xmean) / Xstd), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - ymean) / ystd), requires_grad=False)

    # Create train/validate splits
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits
        
    train_dataset = TensorDataset(tX[train_indices], tY[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_valid = tX[validate_indices]
    Y_valid = tY[validate_indices]

    if np.isscalar(quantiles):
        quantiles = np.array([quantiles])
 
    tquantiles = autograd.Variable(torch.FloatTensor(quantiles), requires_grad=False)

    model = DQR_arch(Xmean, Xstd, ymean, ystd, quantiles.shape[0],width_vec=width_vec, non_crossing=non_crossing) if init_model is None else init_model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=True, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())

    # Univariate quantile loss
  
    def quantile_loss(yhat, y_true):
        z = y_true - yhat
        return torch.max(tquantiles.view(1, -1) * z, (tquantiles.view(1, -1) - 1) * z)
    

    # Create the quantile loss function
    lossfn = quantile_loss
            

    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()
        train_loss = torch.Tensor([0])
        for batch_X, batch_Y in train_loader:
            model.train()
            model.zero_grad()
            yhat = model(batch_X)
            loss = lossfn(yhat, batch_Y)
            loss = loss.mean()
            loss.backward()
            
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            
            optimizer.step()
            train_loss += loss.data
            
        
            if np.isnan(loss.data.numpy()):
                import warnings
                warnings.warn('NaNs encountered in training model.')
                break

        validate_loss = torch.Tensor([0])

        with torch.no_grad():
            model.eval()
            model.zero_grad()
            yhat = model(X_valid)
            loss = lossfn(yhat, Y_valid)
            validate_loss = loss.sum().item()

        train_losses[epoch] = train_loss / float(len(train_indices))
        val_losses[epoch] = validate_loss / float(len(validate_indices))


        # Adjust the learning rate down if the validation performance is bad
        if num_bad_epochs > patience:
            if verbose:
                print('Decreasing learning rate to {}'.format(lr*0.5))
            scheduler.step()
            lr *= 0.5
            num_bad_epochs = 0


        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            
        else:
            num_bad_epochs += 1
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss))


    print('DQR training finished.')
    # Return the conditional density model that marginalizes out the grid
    return model