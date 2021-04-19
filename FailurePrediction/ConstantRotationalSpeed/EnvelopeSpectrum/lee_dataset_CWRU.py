import scipy.io as sio
import numpy as np
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