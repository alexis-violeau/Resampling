from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import models
import resampling
import metrics


class SearchCV:
    
    def __init__(self,resampler_list = [None,'RandomOverSampler','RandomUnderSampler'], class_ratio_list = [0.1,0.5], metric = 'f1', cv = 5):
        self.resampler_list = resampler_list
        self.class_ratio_list = class_ratio_list
        self.metric = metric
        self.cv = cv
        self.fitted = False
        self.best_params = {'method' : None, 'class_ratio' : None}
        
    def fit(self,X,y):
        result = []

        for resampler_name in self.resampler_list:
            for class_ratio in self.class_ratio_list:
                
                score = []
                
                for train_index, test_index in StratifiedKFold(n_splits= self.cv).split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    resampler = resampling.get_resampler(method = resampler_name, class_ratio = class_ratio)
                    model = models.get_model()
                    
                    X_train_resample, y_train_resample = resampler.fit_resample(X_train,y_train)
                    model.fit(X_train_resample,y_train_resample)
                    
                    y_pred = model.predict(X_test)
                    
                    score.append(metrics.score(y_test,y_pred, metric = self.metric))
                    
                
                result.append(
                    {'method' : resampler_name,
                    'class_ratio' : class_ratio,
                    'metric' : self.metric,
                    'cv' : self.cv,
                    'mean_score' : np.mean(np.array(score)),
                    'std_score' : np.std(np.array(score))
                })
                
        result = sorted(result, key=lambda d: d['mean_score'])
        
        self.best_params['method'] = result[0]['method']
        self.best_params['class_ratio'] = result[0]['class_ratio']
                
        return pd.DataFrame(result)