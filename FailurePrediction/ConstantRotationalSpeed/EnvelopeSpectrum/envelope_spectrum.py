import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import statistics as stats
from scipy.fft import fft, fftfreq, fftshift
import pandas as pd
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks

def envelope_spectrum(signal, fs, BPFI, BPFO, title, imp = 0):
    '''
    ENVELOPE SPECTRUM

    signal: array con la señal
    fs: entero con la frecuencia
    BPFI: entero con el valor de BPFI 
    BPFO: entero con el valor de BPFO
    imp: nivel de impresión de resultados. 
        iff imp = 1 muestra los picos tenidos en cuenta para la clasificación
    '''
    analytic_signal = hilbert(signal - np.mean(signal))
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
    
    fSpec = []
    for i in range(0, len(amplitude_envelope)):
        fSpec.append(i/len(amplitude_envelope)*fs)
    xSpec = 1/len(amplitude_envelope)*abs(fft(amplitude_envelope));
    
    # Compute one-sided spectrum. Compensate the amplitude for a two-sided
    # spectrum. Double all points except DC and nyquist.
    if len(xSpec) % 2 == 0: 
        # Even length two-sided spectrum
        fSpec = fSpec[0:int(len(fSpec)/2+1)]
        xSpec = xSpec[0:int(len(xSpec)/2+1)]
        xSpec[1:len(xSpec)-1] = 2*xSpec[1:len(xSpec)-1]

    else:
        # Odd length two-sided spectrum
        fSpec = fSpec[0:int((len(fSpec)+1)/2)]
        xSpec = xSpec[0:int((len(xSpec)+1)/2)]
        xSpec[1:len(xSpec)] = 2*xSpec[1:len(xSpec)]
        
    fig = plt.figure(figsize=(20,10))
    
    peaks_c, _ = find_peaks(xSpec, np.percentile(xSpec,99))
    peaks = xSpec[peaks_c]
    cota = np.percentile(peaks, 90)
    fSpecG = []
    for i in range(0,len(xSpec)):
        if xSpec[i] >= cota:
            fSpecG.append(fSpec[i])

    plt.plot(fSpec,xSpec, label='Envelope spectrum')
    BPFI_coords = np.arange(0, fSpecG[len(fSpecG) - 1] + BPFI, BPFI)
    for xc in BPFI_coords:
        if xc == BPFI_coords[0]:
            plt.axvline(x=xc, color = 'r', linestyle = '--', lw=1.5, alpha = 0.5,label='BPFI')
        else:
            plt.axvline(x=xc, color = 'r', linestyle = '--', lw=1.5, alpha = 0.5)
    BPFO_coords = np.arange(0, fSpecG[len(fSpecG) - 1] + BPFO, BPFO)
    for xc2 in BPFO_coords:
        if xc2 == BPFO_coords[0]:
            plt.axvline(x=xc2, color = 'g', linestyle = '--', lw=1.5, alpha = 0.5, label='BPFO')
        else:
            plt.axvline(x=xc2, color = 'g', linestyle = '--', lw=1.5, alpha = 0.5)
    
    if imp == 1:
        peaks_r_coord, _ = find_peaks(xSpec, cota)
        peaks_r = xSpec[peaks_r_coord]
        fSpec = np.array(fSpec)
        plt.plot(fSpec[peaks_r_coord], peaks_r, "o")
        
    plt.title(title, fontsize=15)
    plt.legend(fontsize=12)
    
    return fSpec, xSpec, fSpecG, BPFI_coords, BPFO_coords


def envelope_spectrum2(signal, Fs):
    analytic_signal = hilbert(signal - np.mean(signal))
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)

    fSpec = []
    for i in range(0, len(amplitude_envelope)):
        fSpec.append(i/len(amplitude_envelope)*Fs)
    xSpec = 1/len(amplitude_envelope)*abs(fft(amplitude_envelope));

    # Compute one-sided spectrum. Compensate the amplitude for a two-sided
    # spectrum. Double all points except DC and nyquist.
    if len(xSpec) % 2 == 0: 
        # Even length two-sided spectrum
        fSpec = fSpec[0:int(len(fSpec)/2+1)]
        xSpec = xSpec[0:int(len(xSpec)/2+1)]
        xSpec[1:len(xSpec)-1] = 2*xSpec[1:len(xSpec)-1]

    else:
        # Odd length two-sided spectrum
        fSpec = fSpec[0:int((len(fSpec)+1)/2)]
        xSpec = xSpec[0:int((len(xSpec)+1)/2)]
        xSpec[1:len(xSpec)] = 2*xSpec[1:len(xSpec)]

        
    return fSpec, xSpec 

def clasificacion_envelope(fSpec, xSpec, fr, BPFO, BPFI):
    peaks_c, _ = find_peaks(xSpec, np.percentile(xSpec,99))
    peaks = xSpec[peaks_c]
    cota = np.percentile(peaks, 95)
    fSpecG = []
    xSpecG = []
    for i in range(len(xSpec)):
        if (xSpec[i] >= cota and fSpec[i] <= 2 * max([fr, BPFO, BPFI])):
            fSpecG.append(fSpec[i])
            xSpecG.append(xSpec[i])

    fMax = fSpecG[np.argmax(xSpecG)]
    esado = ''
    if (fMax <= BPFO + fr/10 and fMax >= BPFO - fr/10):
        estado = 'Fallo Outer Race'
    elif (fMax <= BPFI + fr/10 and fMax >= BPFI - fr/10):
        estado = 'Fallo Inner Race'
    elif (fMax <= fr + fr/10 and fMax >= fr - fr/10):
        estado = 'Sano'
    else:
        estado = 'No concluyente'
    
    if (estado =='No concluyente'):
        i = 2
        fr_mult = i * fr
        while (fr_mult <= 2 * max([fr, BPFO, BPFI])):
            if (fMax <= fr_mult + fr/10 and fMax >= fr_mult - fr/10):
                estado = 'Probablemente Sano'
            i += 1
            fr_mult = i * fr
 
    return estado
