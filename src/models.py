from sklearn.ensemble import RandomForestClassifier

def get_model():
    return RandomForestClassifier(n_estimators = 100,criterion = 'gini',max_features = 'sqrt')