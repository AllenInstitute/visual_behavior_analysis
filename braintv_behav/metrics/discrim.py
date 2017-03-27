"""
the metrics contained in this module conform to scikit-learn styled `metric(y_true,y_pred,**args,**kwargs)`

"""

def hit_rate(y_true,y_pred):

    return y_pred[y_true].mean()

def false_alarm_rate(y_true,y_pred):
    
    return y_pred[~y_true].mean()


def d_prime(y_true,y_pred,eps=None):

    C = metrics.confusion_matrix(y_true,y_pred).astype(float) 
    
    try:
        tn, fp, fn, tp = C.ravel()

        HR = tp / (tp + fn)
        FAR = fp / (tn + fp)

        if eps is None:
            # a reasonable minimum value is 1 / sum of trials
            eps = 1.0/C.sum()

        HR = max(HR,eps)
        HR = min(HR,1.0-eps)

        FAR = max(FAR,eps)
        FAR = min(FAR,1.0-eps)

        return stats.norm.ppf(HR) - stats.norm.ppf(FAR)
    except ValueError:
        return np.nan

def discrim_p(y_true,y_pred):

    C = metrics.confusion_matrix(y_true,y_pred)
    
    try:
        return stats.chi2_contingency(C)[1]
    except ValueError:
        return 1.0
    