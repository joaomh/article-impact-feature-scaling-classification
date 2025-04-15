from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from typing import List
import pandas as pd
import numpy as np

class MeanCentered(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.means_ = {col: np.mean(X[col]) for col in self.cols}
        self.stds_ = {col: np.std(X[col]) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = (X[col] - self.means_[col])
        return X
    
class MedianAbsScaler(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.median_ = {col: np.median(X[col]) for col in self.cols}
        self.mad_ = {col: np.median(np.absolute(X[col]-np.median(X[col]))) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = (X[col] - self.median_[col]) / self.mad_[col]
        return X
    
class StandardStabilityScaling(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.means_ = {col: np.mean(X[col]) for col in self.cols}
        self.stds_ = {col: np.std(X[col]) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = ((X[col] - self.means_[col]) / self.stds_[col])*(self.means_[col]/self.stds_[col])
        return X
    
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
        return X
    
class DecimalScaling(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.len_ = {col: len(str(abs(np.max((X[col]))))) for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col] / 10**self.len_[col]
        return X

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
        return X

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
        return X
    
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
            X[col] = ((1 - np.exp(-(X[col] - self.means_[col]) / self.stds_[col]))) / (1 + np.exp(-(X[col] - self.means_[col]) / self.stds_[col]))
        return X

scaling_list = {
    'MinMaxScaler':MinMaxScaler(),
    'MaxAbsScaler':MaxAbsScaler(),
    'StandardScaler':StandardScaler(),
    'ParetoScaling':ParetoScaling,
    'StandardStabilityScaling':StandardStabilityScaling,
    'MeanCentered':MeanCentered,
    'None': 'None',
    'RobustScaler':RobustScaler(),
    'QuantileTransformer':QuantileTransformer(),
    'DecimalScaling':DecimalScaling,
    'TanhTransformer':TanhTransformer,
    'LogisticSigmoidTransformer':LogisticSigmoidTransformer,
    'HyperbolicTangentTransformer':HyperbolicTangentTransformer
}
common_scalers = ['MinMaxScaler','MaxAbsScaler','StandardScaler','Normalizer','RobustScaler','QuantileTransformer']