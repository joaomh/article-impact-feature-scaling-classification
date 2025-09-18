from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from typing import List
import pandas as pd
import numpy as np
import math

class MeanCentered(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.means_ = {col: np.mean(X[col]) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = (X[col] - self.means_[col])
        return X.to_numpy()
    
class VariableStabilityScaling(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.means_ = {col: np.mean(X[col]) for col in self.cols}
        self.stds_ = {col: np.std(X[col]) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = ((X[col] - self.means_[col]) / self.stds_[col])*(self.stds_[col]/self.means_[col])
        return X.to_numpy()
    
class ParetoScaling(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.means_ = {col: np.mean(X[col]) for col in self.cols}
        self.stds_ = {col: np.std(X[col]) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = (X[col] - self.means_[col]) / np.sqrt(self.stds_[col])
        return X.to_numpy()
    
class DecimalScaling(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.scale_ = {}
        for col in self.cols:
            max_abs = np.max(np.abs(X[col]))
            if max_abs == 0:
                j = 0  
            else:
                j = math.ceil(np.log10(max_abs + 1e-17))  
            self.scale_[col] = j
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            j = self.scale_[col]
            X[col] = X[col] / (10 ** j)
        return X.to_numpy()

class TanhTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.means_ = {col: np.mean(X[col]) for col in self.cols}
        self.stds_ = {col: np.std(X[col]) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = 0.5*(np.tanh(0.01*((X[col] - self.means_[col])/ self.stds_[col])) + 1)
        return X.to_numpy()

class LogisticSigmoidTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.means_ = {col: np.mean(X[col]) for col in self.cols}
        self.stds_ = {col: np.std(X[col]) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = 1 /(1 + np.exp(-(X[col] - self.means_[col]) / self.stds_[col]))
        return X.to_numpy()
    
class HyperbolicTangentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.means_ = {col: np.mean(X[col]) for col in self.cols}
        self.stds_ = {col: np.std(X[col]) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = (1 - np.exp(-(X[col] - self.means_[col]) / self.stds_[col])) / (1 + np.exp(-(X[col] - self.means_[col]) / self.stds_[col]))
        return X.to_numpy()

scaling_list = {
    'MinMaxScaler':MinMaxScaler(),
    'MaxAbsScaler':MaxAbsScaler(),
    'StandardScaler':StandardScaler(),
    'ParetoScaling':ParetoScaling,
    'VariableStabilityScaling':VariableStabilityScaling,
    'MeanCentered':MeanCentered,
    'None': 'None',
    'RobustScaler':RobustScaler(),
    'QuantileTransformer':QuantileTransformer(),
    'DecimalScaling':DecimalScaling,
    'TanhTransformer':TanhTransformer,
    'LogisticSigmoidTransformer':LogisticSigmoidTransformer,
    'HyperbolicTangentTransformer':HyperbolicTangentTransformer
}
common_scalers = ['MinMaxScaler','MaxAbsScaler','StandardScaler','RobustScaler','QuantileTransformer']
