from sklearn.linear_model import QuantileRegressor
import numpy as np

class LinearQR:
    def __init__(self, quantiles=0.5):
        self.quantiles = quantiles
        self.models = []
        self.label = 'Linear QR'
        self.filename = 'LinearQR'


    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.detach().cpu().numpy() if hasattr(X, 'detach') else np.array(X)
        X = X.astype(float)
        if not isinstance(y, np.ndarray):
            y = y.detach().cpu().numpy() if hasattr(y, 'detach') else np.array(y)
        y = y.ravel().astype(float)
        
        for t in range(len(self.quantiles)):
            reg = QuantileRegressor(quantile=self.quantiles[t].numpy().item(), alpha=0)
            reg.fit(X, y)
            self.models.append(reg)
        print('Linear QR training finished.')

    def predict(self, X):
        n = X.shape[0];
        preds = np.zeros([n,len(self.quantiles)])
        for t, model in enumerate(self.models):
            preds[:,t] = model.predict(X)

        return preds



