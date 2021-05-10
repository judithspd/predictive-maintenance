# AUTOMATIC CLASSIFICATION OF VARIABLE SPEED SIGNALS OBTAINED FROM 
# https://www.sciencedirect.com/science/article/pii/S2352340918314124

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
from scipy.stats import kurtosis, skew
from scipy import signal
import joblib 
from collections import Counter

def highlowfilter(k_filter, input_signal):
    '''
    Given a signal, it applies the high pass or low pass band filter to it, depending on the input choice.
    INPUT:
        - k_filter: kind of filter applied: 
            * 'hp' high pass
            * 'low': low pass
        - input_signal: signal to which to apply the filter 
    '''
    b, a = signal.butter(3, 0.05, k_filter) 

    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, input_signal, zi = zi * input_signal[0])

    z2, _ = signal.lfilter(b, a, z, zi = zi * z[0])

    y = signal.filtfilt(b, a, input_signal)
    
    return z, z2, y

def parameters_hamming(xsignal):
    '''
    Given the signal 'xsignal', it applies the Hamming window function, a low pass filter, 
    and calculates certain statistics and parameters. 
    '''
    xsignal = xsignal * signal.hamming(len(xsignal))
    _, _, x = highlowfilter('low', xsignal)
    N = len(x)
    n_inf = max(abs(x))
    kurt = kurtosis(x)
    impulse_factor = N * n_inf / sum(abs(x))
    RMS = np.sqrt(sum(x**2))
    margin_factor = n_inf / RMS**2
    skewness = skew(x)
    shape_factor = N * RMS / sum(abs(x))
    peak_to_peak = max(x) - min(x)
    crest_factor = n_inf / RMS
    
    return kurt, impulse_factor, RMS, margin_factor, skewness, shape_factor, peak_to_peak, crest_factor


def clasification(signal_1s):
    '''
    Given a signal of duration one second, the parameters_hamming function is applied to preprocess it and find the statistics to 
    be used as input for the machine learning models. We will use 4 models that have been stored and trained and will be loaded using the
    will be used and loaded using the joblib library:
    * kNN. Trained with k =3. In addition to saving the trained model, we also use the corresponding scaler, 
           saved as scaler.fitted.
    * Random forest. Trained with 50 trees.
    * Decision tree. With tree depth 7.
    * SVM. With the optimal configuration found using cross validation: 'C': 5000.0, 'gamma': 0.001.
    We predict using the previous 4 models and return the majority prediction. In case of a 2-2 tie, we keep the prediction given by 
    the random forest.
    '''
    kurt, impulse_factor, RMS, margin_factor, skewness, shape_factor, peak_to_peak, crest_factor = parameters_hamming(signal_1s)
    xsignal = np.array([[kurt, impulse_factor, RMS, margin_factor, skewness, shape_factor, peak_to_peak, crest_factor]])
    
    scaler = joblib.load("scaler_traintest.pkl") 
    xsignal_sca = scaler.transform(xsignal)
    
    randomForest = joblib.load('randomForest_traintest_trained.pkl') 
    svmModel = joblib.load('svm_traintest_trained.pkl') 
    treeModel = joblib.load('tree_traintest_trained.pkl') 
    knnModel = joblib.load('knn_traintest_trained.pkl')
    
    pred_rf = randomForest.predict(xsignal)[0]
    pred_svm = svmModel.predict(xsignal)[0]
    pred_tree = treeModel.predict(xsignal)[0]
    pred_knn = knnModel.predict(xsignal_sca)[0]
    
    rep = Counter([pred_rf, pred_svm, pred_tree, pred_knn])
    tipos = ['Inner', 'Sano', 'Outer']
    numRep = [rep['Inner'], rep['Sano'], rep['Outer']]
    print('Pred rf: ' + pred_rf + ' Pred SVM: ' + pred_svm + ' Pred tree: ' + pred_tree + ' Pred kNN: ' + pred_knn)
    if max(numRep) > 2: # 3 or 4 matches -> majority
        return tipos[np.argmax(numRep)]
    elif max(numRep) == 2 and min(numRep) == 0: # If there is a tie, we select random forest prediction
        return pred_rf
    elif max(numRep) == 2 and min(numRep) == 1: # There are 2 equal predictions (majority) and one of each other of the two types
        return tipos[np.argmax(numRep)]
 