from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

metric_dict = {
            'accuracy' : accuracy_score,
            'precision' : precision_score,
            'recall' : accuracy_score,
            'f1' : f1_score,
            'auc' : roc_auc_score
            }

def score(y_true,y_pred,metric = 'f1'):
    return metric_dict[metric](y_true,y_pred)