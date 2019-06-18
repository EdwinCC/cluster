import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

#matplotlib inline

from mpl_toolkits.mplot3d import Axes3D

dataframe = pd.read_csv(r"/home/edwin/Escritorio/databeaconsoriginal/Tx_0x04.csv")

print("imprimiendo dataframe (datos recien importados)")
print(dataframe)
print("termino de imprimir dataframe")

X=dataframe.values
X=np.array(X)

#print(X)

puntos=X[:,0:5:4]

print(puntos)

#plt.scatter(puntos[:,0], puntos[:,1], c=['lightblue'], label='points in (x,y)')
plt.scatter(puntos[:,0], puntos[:,1], c=['lightblue'])


plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('grafica')
plt.show()

