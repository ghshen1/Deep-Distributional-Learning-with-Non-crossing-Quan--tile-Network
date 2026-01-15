import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures,SplineTransformer
import numpy as np


class SplineQR:
    def __init__(self, quantiles=0.5,poly_base=True):
        self.quantiles = quantiles
        self.models = []
        self.label = 'Spline QR'
        self.filename = 'SplineQR'
        self.poly_base = poly_base
    
    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.detach().cpu().numpy() if hasattr(X, 'detach') else np.array(X)
        X = X.astype(float)
        if not isinstance(y, np.ndarray):
            y = y.detach().cpu().numpy() if hasattr(y, 'detach') else np.array(y)
        y = y.ravel().astype(float)
        
        spl = SplineTransformer(degree=3, n_knots=5, include_bias=True)
        x_spl= spl.fit_transform(X)
        self.spl = spl
        if self.poly_base:
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
            self.poly = poly
            x_spl= poly.fit_transform(x_spl)
        mod = sm.QuantReg(y, x_spl)
        for t in range(len(self.quantiles)):
            res = mod.fit(q=self.quantiles[t])
            self.models.append(res)
        print('Spline QR training finished.')

    def predict(self, X):
        n = X.shape[0];
        x_test_spl = self.spl.transform(X)
        if self.poly_base:
            x_test_spl = self.poly.fit_transform(x_test_spl)
        preds = np.zeros([n,len(self.quantiles)])
        for t, model in enumerate(self.models):
            preds[:,t] = model.predict(x_test_spl)
        return preds



