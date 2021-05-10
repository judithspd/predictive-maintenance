import scipy.stats 
from scipy.stats import kurtosis, skew
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

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