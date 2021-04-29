import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

def pca_reconstruccion_error_reconstruccion_primerCuarto(X, n_comp, imp = 0):
    '''
    Se halla la recoinstrucción del conjunto de datos X usando PCAs con n_comp componentes principales.
    Se calcula el error en la reconstrucción.
    
    INPUT:
        X: conjunto de datos que se descompondrán usando PCAs.
        n_comp: número de componentes principales.
        imp: indicador del niel de impresión. Si es 1 se muestra la gráfica con la varianza explicada acumulada.
    
    OUTPUT:
        reconstruccion: conjunto de datos con la reconstrucción de los datos originales.
        error_reconstruccion: array con el RMSE entre el dataset original y el reconstruido.
    
    '''
    
    # Entrenamiento modelo PCA con escalado de los datos
    X_primerCuarto = X[:int(len(X)/4)]
    pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components = n_comp))
    pca_pipeline.fit(X_primerCuarto)
    
    # Proyectar los datos
    proyecciones = pca_pipeline.transform(X)
    
    # Reconstrucción
    reconstruccion = pca_pipeline.inverse_transform(X = proyecciones)
    reconstruccion = pd.DataFrame(reconstruccion, columns = X.columns, index = X.index)
    
    # RMSE: 
    error_reconstruccion = np.sqrt(((reconstruccion - X) ** 2).mean(axis=1))
    
    # Si imp es 1, mostramos el plot de la varianza explicada acumulada
    if imp == 1:
        modelo_pca = pca_pipeline.named_steps['pca']
        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
        ax.plot(np.arange(modelo_pca.n_components_) + 1, prop_varianza_acum, marker = 'o')

        for x, y in zip(np.arange(modelo_pca.n_components_) + 1, prop_varianza_acum):
            label = round(y, 2)
            ax.annotate( label, (x,y), textcoords = "offset points", xytext = (0,10), ha = 'center')

        ax.set_ylim(0, 1.2)
        ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
        ax.set_title('Varianza explicada acumulada')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Varianza explicada');
    
    return reconstruccion, error_reconstruccion