import numpy as np
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ReLU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return nn.functional.relu(x).pow(2)
    
class DQRP_arch(nn.Module):
    def __init__(self, X_means, X_stds, y_mean, y_std, width_vec=None, activation='ReQU'):
        super(DQRP_arch, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_in = X_means.shape[1]+1
        self.y_dim = 1 if len(y_mean.shape) == 1 else y_mean.shape[1]
        self.width_vec = width_vec if width_vec is not None else [self.n_in, 200, 200, 1]
        self.activation = activation

        modules = []
        if self.activation.lower() =='requ':
            for i in range(len(self.width_vec) - 2):
                modules.append(
                    nn.Sequential(
                        nn.Linear(self.width_vec[i],self.width_vec[i+1]),
                        ReLU2()))
                
        if self.activation.lower() =='relu':
            for i in range(len(self.width_vec) - 2):
                modules.append(
                    nn.Sequential(
                        nn.Linear(self.width_vec[i],self.width_vec[i+1]),
                        nn.ReLU()))

        self.net = nn.Sequential(*modules,
                                 nn.Linear(self.width_vec[-2], self.width_vec[-1]))

    def forward(self, x,u):
        z =torch.cat((x,u),dim=1)
        output = self.net(z)
        return  output
    
    

    def predict(self, X, quantiles):
        self.eval()
        self.zero_grad()
        self.quantiles = quantiles
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means) / self.X_stds), requires_grad=False)
        size = X.shape[0]
        preds = torch.zeros([size, len(self.quantiles)])
        for t in range(len(self.quantiles)):
            z = torch.cat((tX, self.quantiles[t].repeat(size, 1).float()),dim=1)
            preds[:, t] = self.net(z).detach().squeeze()*self.y_std + self.y_mean
        return preds





class DQRP:
    def __init__(self, quantiles):
        self.quantiles = quantiles
       
    def fit(self, X, y, width_vec= None, activation='ReQU',lr =0.1, epochs=100):
        
        if width_vec==None:
            width_vec =[X.shape[1]+1, 200,200,1]

        self.model = fit_quantiles(X, y, width_vec = width_vec, 
                                activation=activation ,lr =lr, nepochs = epochs)

    def predict(self, X, quantiles):
         return self.model.predict(X, quantiles)
     
        

def fit_quantiles(X, y, width_vec=[256,256,256], activation='ReQU', penalty=None,
                    nepochs=100, val_pct=0.25,
                    batch_size=None, target_batch_pct=0.01,
                    min_batch_size=20, max_batch_size=100,
                    verbose=False, lr=1e-1, weight_decay=0.0, patience=5,
                    init_model=None, splits=None, 
                    clip_gradients=False, clip_value=5,**kwargs):
    if penalty is None:
        penalty = np.log(X.shape[0])
    
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
    u  = autograd.Variable(torch.rand(tY.shape), requires_grad=True)
    
    # Create train/validate splits
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits
        
    train_dataset = TensorDataset(tX[train_indices], tY[train_indices], u[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_valid = tX[validate_indices]
    Y_valid = tY[validate_indices]
    u_valid = u[validate_indices]
    
 
    #tquantiles = autograd.Variable(torch.FloatTensor(u.detach()), requires_grad=False)

    model = DQRP_arch(Xmean, Xstd, ymean, ystd, width_vec=width_vec, activation=activation) if init_model is None else init_model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())

    # Univariate quantile loss
  
    def quantile_loss(yhat, y_true, u):
        z = y_true - yhat
        return torch.max(u.view(1, -1) * z, (u.view(1, -1) - 1) * z)
    

    # Create the quantile loss function
    lossfn = quantile_loss
            

    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()
        train_loss = torch.Tensor([0])
        model.train()
        for batch_X, batch_Y, batch_u in train_loader:
            batch_Y = batch_Y.float()
            batch_X = autograd.Variable(batch_X.float(), requires_grad=True)
            #batch_u = autograd.Variable(torch.rand(batch_Y.shape), requires_grad=True)
            batch_u = autograd.Variable(batch_u.float(), requires_grad=True)
            
           
            batch_yhat = model(batch_X, batch_u)
            loss = lossfn(batch_yhat, batch_Y, batch_u).mean()
            
            grads = autograd.grad(outputs = batch_yhat, inputs= batch_u, grad_outputs=torch.ones_like(batch_yhat),
                         retain_graph=True, 
                         create_graph=True, 
                         only_inputs=True)[0]
            grads = grads.view(grads.size(0), -1)
            loss_grad = penalty * torch.max(-grads, torch.zeros_like(grads)).mean()
        
            total_loss = loss + loss_grad
            optimizer.zero_grad()
            
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.data
            
            if np.isnan(total_loss.data.numpy()):
                import warnings
                warnings.warn('NaNs encountered in training model.')
                break
            
            #del batch_X, batch_Y, batch_u, loss, loss_grad, grads, total_loss
            
        validate_loss = torch.Tensor([0])

        with torch.no_grad():
            model.eval()
            model.zero_grad()
            yhat = model(X_valid, u_valid)
            loss = lossfn(yhat, Y_valid, u_valid)
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


    print('DQRP training finished.')
    # Return the conditional density model that marginalizes out the grid
    return model