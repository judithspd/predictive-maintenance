import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import statistics as stats
from scipy.fft import fft, fftfreq, fftshift
import pandas as pd
from scipy.signal import hilbert, chirp, detrend
from scipy.signal import find_peaks
from scipy.stats import kurtosis
import math

def lee_dataset(dataset_mat):
    '''
    Retorna la información de los datsets de https://csegroups.case.edu/bearingdatacenter/pages/download-data-file
    '''
    ds = sio.loadmat(dataset_mat)
    dataname = 'X'
    cont = 0
    for i in dataset_mat:
        if i != '.' and i != 'm' and i != 'a' and i != 't':
            cont = cont + 1
            dataname = dataname + i
    if cont == 2:
        dataname = 'X0' + dataname[len(dataname)-2:len(dataname)]
        
    DS_DE_1 = ds[dataname + '_DE_time']
    DS_FE_1 = ds[dataname + '_FE_time']
    if dataname + 'RPM' in ds:
        RPM = int(ds[dataname + 'RPM'])
    else:
        RPM = np.NaN
    
    n_DE = len(DS_DE_1)
    DS_DE = np.zeros(n_DE)
    for i in range(0, n_DE):
        DS_DE[i] = DS_DE_1[i][0]
        
    n_FE = len(DS_FE_1)
    DS_FE = np.zeros(n_FE)
    for i in range(0, n_FE):
        DS_FE[i] = DS_FE_1[i][0]
        
    samples_s_DE = 12000
    samples_s_FE = 12000
    
    dt_DE = 1/samples_s_DE
    t_DE = []
    for i in range(0, n_DE):
        t_DE.append(i * dt_DE)
    t_DE = np.array(t_DE)
        
    dt_FE = 1/samples_s_FE
    t_FE = []
    for i in range(0, n_FE):
        t_FE.append(i * dt_FE)
    t_FE = np.array(t_FE)
    
    return DS_DE, DS_FE, t_DE, t_FE, RPM, samples_s_DE, samples_s_FE

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

def busca_maximos_locales(frec_pos, fft_pos, BPFI, BPFO, fr, fs):
    intervalo = min(BPFI, BPFO) - fr
    frec = []
    picos = []
    for i in range(0, len(frec_pos)):
        if frec_pos[i] <= fs/8:
            frec.append(frec_pos[i])   
            picos.append(fft_pos[i])
    
    frec = np.array(frec)
    picos = np.array(picos)
    cont = 1
    maximos = []
    cont1 = (cont - 1) 
    while cont * intervalo < frec[len(frec) - 1]:
        cont2 = len(frec[frec < cont * intervalo]) - 1
        max_int = max(picos[int(cont1):int(cont2)])
        #print('MAX INT')
        #print(max_int)
        #print('3 POR LA MEDIA')
        #print(3 * np.mean(picos[int(cont1):int(cont2)]))
        if max_int > 2.5 * np.mean(picos[int(cont1):int(cont2)]):
            maximos.append(max_int)
        cont1 = cont2
        cont = cont + 1
    
    frec_max = []
    for m in maximos:
        for i in range(0, len(frec_pos)):
            if fft_pos[i] == m:
                frec_max.append(frec_pos[i])
    return maximos, frec_max

def por_comunes_fft(fft_picos, BPFI_coords, BPFO_coords, fr):
    '''
    Porcentaje de coincidencias entre los picos de la FFT y 
    las rectas correspondientes con BPFI y BPFO
    '''
    BPFI_array_v = BPFI_coords
    comunes_BPFI = 0
    for f in fft_picos:
        for i in range(0,len(BPFI_array_v)-1):
            if f < BPFI_array_v[i] + fr and f > BPFI_array_v[i] - fr:
                comunes_BPFI = comunes_BPFI + 1
                BPFI_array_v = np.delete(BPFI_array_v, i)
                break

    por_comunes_BPFI = comunes_BPFI * 100 / min(len(BPFI_coords),len(fft_picos))
    
    BPFO_array_v = BPFO_coords
    comunes_BPFO = 0
    for f in fft_picos:
        for i in range(0,len(BPFO_array_v)-1):
            if f < BPFO_array_v[i] + fr and f > BPFO_array_v[i] - fr:
                comunes_BPFO = comunes_BPFO + 1
                BPFO_array_v = np.delete(BPFO_array_v, i)
                break

    por_comunes_BPFO = comunes_BPFO * 100 / min(len(BPFO_coords),len(fft_picos))
    
    return por_comunes_BPFI, por_comunes_BPFO

def complex_cepstrum(x):
    """Compute the complex cepstrum of a real sequence.

    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.
    Returns
    -------
    ceps : ndarray
        The complex cepstrum of the real data sequence `x` computed using the
        Fourier transform.
    """

    spectrum = np.fft.fft(x)/len(x)
    spectrum = np.fft.fftshift(spectrum)
    unwrapped_phase = np.unwrap(np.angle(spectrum))
    log_amplitud = np.log(np.abs(spectrum))
    log_spectrum = log_amplitud + 1j * unwrapped_phase
    ceps = np.fft.ifftshift(log_spectrum).real

    return ceps


def algoritmo_clasifica_cpw(dataset, conjunto, BPFI_cte, BPFO_cte, fs):
    DE, FE, t_DE, t_FE, RPM, samples_s_DE, samples_s_FE = lee_dataset(dataset)
    fr = RPM/60
    if math.isnan(fr):
        fr = 29.95
    if conjunto == 'DE':
        datos = DE * np.hamming(len(DE))
        dt = t_DE[1] - t_DE[0]
        kurt = kurtosis(datos)
    elif conjunto == 'FE':
        datos = FE * np.hamming(len(FE))
        dt = t_FE[1] - t_FE[0]
        kurt = kurtosis(datos)
    else:
        print('Tipo de conjunto erroneo')

    ceps = complex_cepstrum(datos)
    BPFI = BPFI_cte * fr
    BPFO = BPFO_cte * fr
    title = 'Envelope spectrum'
    fSpecCeps, xSpecCeps, fSpecGCeps, BPFI_coordsCeps, BPFO_coordsCeps = envelope_spectrum(ceps, fs, BPFI, BPFO, title, 1)
    
    fft_envspec = fftshift(fft(detrend(xSpecCeps)))
    fft_envspec_norm = abs(fft_envspec)/np.max(abs(fft_envspec))
    frec_ceps = fftshift(fftfreq(len(xSpecCeps), dt))
    
    fig = plt.figure(figsize=(20,10))
    plt.plot(frec_ceps, fft_envspec_norm)
    BPFI_coords = np.arange(BPFI, fs/8, BPFI)
    for xc in BPFI_coords:
        if xc == BPFI_coords[0]:
            plt.axvline(x=xc, color = 'r', linestyle = '--', lw=1.5, alpha = 0.5,label='BPFI')
        else:
            plt.axvline(x=xc, color = 'r', linestyle = '--', lw=1.5, alpha = 0.5)
    BPFO_coords = np.arange(BPFO, fs/8, BPFO)
    for xc2 in BPFO_coords:
        if xc2 == BPFO_coords[0]:
            plt.axvline(x=xc2, color = 'g', linestyle = '--', lw=1.5, alpha = 0.5, label='BPFO')
        else:
            plt.axvline(x=xc2, color = 'g', linestyle = '--', lw=1.5, alpha = 0.5)

    plt.xlim(0,fs/2)
    plt.legend(fontsize=12)
    
    fft_envspec_norm_pos = fft_envspec_norm[len(fft_envspec_norm)//2:len(fft_envspec_norm)]
    frec_pos = frec_ceps[len(frec_ceps)//2:len(frec_ceps)]
    maximos, frec_max = busca_maximos_locales(frec_pos, fft_envspec_norm_pos, BPFI, BPFO, fr, fs)
    por_comunes_BPFI, por_comunes_BPFO = por_comunes_fft(frec_max, BPFI_coords, BPFO_coords, fr)
    print(kurt)
    print(por_comunes_BPFI, por_comunes_BPFO)
    # CLASIFICACIÓN
    clasificacion = ''
    if por_comunes_BPFI > 70 and por_comunes_BPFI > 10 + por_comunes_BPFO and kurt > 20:
        clasificacion = 'MUY probable: fallo en inner race'
    elif por_comunes_BPFO > 70 and por_comunes_BPFO > 10 + por_comunes_BPFI and 15 > kurt > 2:
        clasificacion = 'MUY probable: fallo en outer race'
    elif abs(por_comunes_BPFI - por_comunes_BPFO) < 10 and kurt < 3:
        clasificacion = 'MUY probable: sano'
    elif por_comunes_BPFI < 50 and por_comunes_BPFO < 50 and abs(por_comunes_BPFO - por_comunes_BPFO) > 10 and kurt < 3:
        clasificacion = 'Probablemente sano'
    elif por_comunes_BPFO > 50 and por_comunes_BPFO > por_comunes_BPFI and 15 > kurt > 2:
        clasificacion = 'Probablemente fallo en outer race'
    elif por_comunes_BPFI > 50 and por_comunes_BPFI > por_comunes_BPFO and kurt > 20:
        clasificacion = 'Probablemente fallo en inner race'
    elif kurt > 20:
        clasificacion = 'No concluyente, posible fallo en inner race'
    elif 15 > kurt > 2:
        clasificacion = 'No concluyente, posible fallo en outer race'
    else:
        clasificacion = 'No concluyente'
    
    return clasificacion
