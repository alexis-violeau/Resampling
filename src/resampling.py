from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

resampler = {'RandomOverSampler' : RandomOverSampler,
                     'RandomUnderSampler' : RandomUnderSampler,
                     'SMOTE' : SMOTE,
                     'TOMEK' : TomekLinks,
                     'SMOTE TOMEK' : SMOTETomek
                     }


def get_resampler(method = 'RandomUnderSampler',class_ratio = 0.5):
    if method is not None:
        return resampler[method](sampling_strategy = class_ratio)
    else:
        return DummyResampler()


class DummyResampler:
    def __init__(self):
        pass
            
    def fit_resample(self,X,y):
        return X,y
    
    def fit(self,X,y):
        return self
    
    def resample(self,X,y):
        return X,y