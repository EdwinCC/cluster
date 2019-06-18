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
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')

numsectores=15 #fijo para los datos que tengo, cambiar si es diferente

normalizar=0 #1 si/ 0 no
dimensiones=5
numclusters=15

dataframe = pd.read_csv(r"/home/edwin/Escritorio/databeaconsoriginal/Tx_0x04.csv")

print("imprimiendo dataframe (datos recien importados)")
print(dataframe)
print("termino de imprimir dataframe")

#calculamos la correlacion

columnas=["B01","B02","B03","B04","B05"]
pearson=np.zeros((6,6))

print(pearson)

for i in range(1,6):
	for j in range(1,6):
		print(i,j)
		pearson[i][j]=dataframe[columnas[i-1]].corr(dataframe[columnas[j-1]])

print(pearson)

#time.sleep(10)

X=dataframe.values #saca los valores del csv como un array

print("X:")
print(X)

"""
print("imprimiendo dataframe.values=X")
print(X)
print("terminando de imprimir dataframe.values=X")

print("prueba")
print(X[:,0:5]) 
print("fin prueba")
"""

#guardo el array de datos original (datos de intensidad de señal junto con sectores)
Xtemp=X

#separo los datos de intensidad de señal de los sectores


#cambio X por data para distinguirlo mejor y guardar el array original sin tocarlo
data=X[:,0:5] #aqui se guardan los datos de intensidades
sectores=X[:,5] #aqui se guarda el campo de sectores


print("data:")
print(data)


#se aplica normalizacion a los datos

###############################################################################
if normalizar==1 :
	data=normalize(data)


"""

print("imprimo sectores")
print(sectores)
print("fin imp sectores")

"""




#Scaling the values
#X = scale(X) #normaliza los valores (necesito ver los valores como son)

pca = PCA(n_components=5)


print("pca")
print(pca)
print("fin pca")


pca.fit(data) #ajusta el modelo / data -> linea 43


print("pca.fit")
print(pca)
print("fin pca.fit")

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print("varianza acumulativa")
print(var1)

#plt.plot(var1)
#plt.show()



#escoge el numero de dimensiones--------------------------------------

#Looking at above plot I'm taking 30 variables
pca = PCA(n_components=dimensiones) #dim_inicial=5 -> dim_final=5 (coord de componentes principales)
pca.fit(data) #data -> linea 43

print(pca)




X1=pca.fit_transform(data) #ajusta el modelo y aplica reduccion de dimensiones

print("matriz final")

print(X1) #X1 : matriz a la que se aplico el pca
print(X1.shape)

#X1=X ###########si comenta usa pca/si no, no usa pca
#X contiene datos como se leyo del csv
#X1 resultado pca

if dimensiones==2:

	#muestra datos de pca en 2d

	#plt.plot(X1[2:,0],X1[2:,1], markersize=7, color='blue')
	#plt.plot(X1[2:123,0],X1[2:123,1], markersize=7, color='blue', alpha=0.5, label='sector1')
	#plt.plot(X1[123:213,0], X1[123:213,1], markersize=7, color='red', alpha=0.5, label='sector2')
	#plt.plot(X1[213:221,0], X1[213:221,1], markersize=7, color='orange', alpha=0.5, label='sector3')

	plt.scatter(X1[:,0], X1[:,1], c=['blue'], label='points in (x,y)')
	
	plt.xlabel('x_values')
	plt.ylabel('y_values')
	
	plt.legend()
	plt.title('Visualización de datos con PCA 2-dimensiones')
	
	#plt.show()





if dimensiones ==3:
	#muestra datos de pca en 3d

	fig = plt.figure()
	ax = Axes3D(fig)

	print("control")


	ax.scatter(X1[0:322, 0], X1[0:322, 1], X1[0:322, 2], color='blue',s=60)
	#ax.scatter(X1[123:213, 0], X1[123:213, 1], X1[123:213, 2], color='red',s=60)
	#ax.scatter(X1[213:221, 0], X1[213:221, 1], X1[213:221, 2], color='black',s=60)



	#ax.scatter(X1[221:365, 0], X1[221:365, 1], X1[221:365, 2], color='black',s=60)

	#ax.scatter(C[0:5, 0], C[0:5, 1], C[0:5, 2], marker='+', c=colores[0:5], s=1000)

	#ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=asignar,s=60)
	#ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='+', c=colores, s=1000)

	#ax.scatter(X1[:, 0], X1[:, 1], c=asignar,s=60)
	#ax.scatter(C[:, 0], C[:, 1], marker='+', c=colores, s=1000)

	#ax.scatter(X1[:, 0], c=asignar, s=60)
	#ax.scatter(C[:, 0], marker='+', c=colores, s=1000)

	plt.xlabel('x')
	plt.ylabel('y')

	#plt.show()










"""
print(X1.shape)

x_,y_=X1.shape

Xfinal=np.zeros((x_,y_+1))
for i in range(x_):
	for j in range(y_):
		Xfinal[i][j]=X1[i][j]
	Xfinal[i][j+1]=Xtemp[i][5]

print(Xfinal)
"""






#X1 es la matriz que resulto al aplicar el pca

###############################################################################

#normalizacion antes de k means

#X1=normalize(X1)


#calculo de k
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
#kmeans
score = [kmeans[i].fit(X1).score(X1) for i in range(len(kmeans))]
#score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
#plt.show()

#aplicamos kmeans con el valor de k elegido de la grafica

kmeans = KMeans(n_clusters=numclusters).fit(X1) #---------N° de clusters
centroids = kmeans.cluster_centers_
print("centroides")
print(centroids)


#grafica 3d de los clusters

# Predicting the clusters
labels = kmeans.predict(X1)

print("labels:")

print(labels)
labx=len(labels)
print("tamaño labels= ",labx)


#valores de rango de los sectores de los labels (para Tx_0x04)


#puntos muy juntos (ningun sector logra la precision del sector 7 de Tx_0x04)
#Tx_0x01
maxs1=359
maxs2=699
maxs3=1020
maxs4=1359
maxs5=1717
maxs6=2038
maxs7=2381
maxs8=2727
maxs9=3106
maxs10=3446
maxs11=3796
maxs12=4077
maxs13=4335
maxs14=4655
maxs15=5003


"""
#Tx_0x04 sector 7 100% otros ~50%
maxs1=321
maxs2=635
maxs3=945
maxs4=1289
maxs5=1637
maxs6=1973
maxs7=2323
maxs8=2656
maxs9=2967
maxs10=3432
maxs11=3764
maxs12=4066
maxs13=4398
maxs14=4729
maxs15=5134
"""


"""
#Tx_0x06 
maxs1=322
maxs2=571
maxs3=848
maxs4=1181
maxs5=1509
maxs6=1748
maxs7=2040
maxs8=2314
maxs9=2530
maxs10=2831
maxs11=3110
maxs12=3318
maxs13=3612
maxs14=3879
maxs15=4198
"""




#prueba: imprimo los valores del primer cluster para asegurarme

print("data sector15")
for i in range (maxs14,maxs15): #para sect1 0:maxs1, sectn maxs(n-1):maxsn
	print(data[i])
print("fin data sector15")


#evaluo la distribucion de los cluster en donde deberian estar cada cluster

#sect1, sect2, etc guardan los labels predichos para cada punto en cada sector
#ahora las guardo en una sola matriz para no tener que concatenar al final
#*********************************************************

#MATRIZ

sectores=np.zeros((numsectores+1,numclusters+1)) #necesito +1 sector para ir de 1 a 15
#en el cero voy a guardar los indices de clusters
#********************************************************
#inicializo con los indices de sectores y clusters
for i in range(1,sectores.shape[0]):
	for j in range(1,sectores.shape[1]):
		sectores[0][j]=j
		sectores[i][0]=i


#sector1

minncluster=1
maxncluster=numclusters+1
#maxncluster=16



#sect1=np.zeros(maxncluster) #creo matriz numpy de 0 a 15
for i in range(0,maxs1): #itero de 0 al limite del sector 1 (todos datos del sector1)
	for j in range(minncluster,maxncluster): #itero de 1 a 15
		if labels[i]+1==j: #si el label predicho es igual al sector real
			sectores[1][j]=sectores[1][j]+1 #aumenta el contador de aciertos en 1
	
print("distribucion sector 1")


for i in range(1,len(sectores[1])):
	print(i,sectores[1][i])

#sector2

#sect2=np.zeros(maxncluster)
for i in range(maxs1,maxs2):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[2][j]=sectores[2][j]+1
	
print("distribucion sector 2")


for i in range(1,len(sectores[2])):
	print(i,sectores[2][i])

#sector3

#sect3=np.zeros(maxncluster)
for i in range(maxs2,maxs3):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[3][j]=sectores[3][j]+1
	
print("distribucion sector 3")


for i in range(1,len(sectores[3])):
	print(i,sectores[3][i])

#sector4

#sect4=np.zeros(maxncluster)
for i in range(maxs3,maxs4):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[4][j]=sectores[4][j]+1
	
print("distribucion sector 4")


for i in range(1,len(sectores[4])):
	print(i,sectores[4][i])

#sector5

#sect5=np.zeros(maxncluster)
for i in range(maxs4,maxs5):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[5][j]=sectores[5][j]+1
	
print("distribucion sector 5")


for i in range(1,len(sectores[5])):
	print(i,sectores[5][i])

#sector6

#sect6=np.zeros(maxncluster)
for i in range(maxs5,maxs6):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j: #labels son de 0 a 14, mis sectores son de 1 a 15
			sectores[6][j]=sectores[6][j]+1
	
print("distribucion sector 6")


for i in range(1,len(sectores[6])):
	print(i,sectores[6][i])


#sector7

#sect7=np.zeros(maxncluster)
for i in range(maxs6,maxs7):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[7][j]=sectores[7][j]+1
	
print("distribucion sector 7")


for i in range(1,len(sectores[7])):
	print(i,sectores[7][i])

#sector8

#sect8=np.zeros(maxncluster)
for i in range(maxs7,maxs8):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[8][j]=sectores[8][j]+1
	
print("distribucion sector 8")


for i in range(1,len(sectores[8])):
	print(i,sectores[8][i]) #es un indice menos que el sector porque el array sectores comienza desde cero

#sector9

#sect9=np.zeros(maxncluster)
for i in range(maxs8,maxs9):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[9][j]=sectores[9][j]+1
	
print("distribucion sector 9")


for i in range(1,len(sectores[9])):
	print(i,sectores[9][i])

#sector10

#sect10=np.zeros(maxncluster)
for i in range(maxs9,maxs10):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[10][j]=sectores[10][j]+1
	
print("distribucion sector 10")


for i in range(1,len(sectores[10])):
	print(i,sectores[10][i])

#sector11

#sect11=np.zeros(maxncluster)
for i in range(maxs10,maxs11):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[11][j]=sectores[11][j]+1
	
print("distribucion sector 11")


for i in range(1,len(sectores[11])): #imprimo la distribucion de label 1 a label 15 
	print(i,sectores[11][i])

#sector12

#sect12=np.zeros(maxncluster)
for i in range(maxs11,maxs12):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[12][j]=sectores[12][j]+1
	
print("distribucion sector 12")


for i in range(1,len(sectores[12])):
	print(i,sectores[12][i])

#sector13

#sect13=np.zeros(maxncluster)
for i in range(maxs12,maxs13):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[13][j]=sectores[13][j]+1
	
print("distribucion sector 13")


for i in range(1,len(sectores[13])):
	print(i,sectores[13][i])

#sector14

#sect14=np.zeros(maxncluster)
for i in range(maxs13,maxs14):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[14][j]=sectores[14][j]+1
	
print("distribucion sector 14")


for i in range(1,len(sectores[14])):
	print(i,sectores[14][i])


#sector15

#sect15=np.zeros(maxncluster)
for i in range(maxs14,maxs15-1):
	for j in range(minncluster,maxncluster):
		if labels[i]+1==j:
			sectores[15][j]=sectores[15][j]+1
	
print("distribucion sector 15")


for i in range(1,len(sectores[15])):
	print(i,sectores[15][i])




print("sector 1------------------------------------------------")
for i in range(labx):
	
	if i==maxs1+1:
		print("sector 2----------------------------------------------")
	if i==maxs2+1:
		print("sector 3----------------------------------------------")
	if i==maxs3+1:
		print("sector 4----------------------------------------------")
	if i==maxs4+1:
		print("sector 5----------------------------------------------")
	if i==maxs5+1:
		print("sector 6----------------------------------------------")
	if i==maxs6+1:
		print("sector 7----------------------------------------------")
	if i==maxs7+1:
		print("sector 8----------------------------------------------")
	if i==maxs8+1:
		print("sector 9----------------------------------------------")
	if i==maxs9+1:
		print("sector 10----------------------------------------------")
	if i==maxs10+1:
		print("sector 11----------------------------------------------")
	if i==maxs11+1:
		print("sector 12----------------------------------------------")
	if i==maxs12+1:
		print("sector 13----------------------------------------------")
	if i==maxs13+1:
		print("sector 14----------------------------------------------")
	if i==maxs14+1:
		print("sector 15----------------------------------------------")
	


	print(labels[i])

#vemos las frecuencias de predicciones para los puntos y ordenamos para saber los
#representantes

#***************************cambiar lo de arriba por esto

#print("sector 1 --------------------------------------------------------------")
#sector1=labels[0:322]
#for i in range(0,sector):
	



print("fin labels")

# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['blue','red','gray','green','cyan','magenta','yellow','coral','orange','purple','black','brown','darkcyan','lime','peru']
asignar=[]
for row in labels:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)

print("control")


#solo sector 1

##ax.scatter(X1[0:322, 0], X1[0:322, 1], X1[0:322, 2], c=asignar[0:322],s=60)
##ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='+', c=colores, s=1000)


#ax.scatter(X1[0:123, 0], X1[0:123, 1], c=asignar[0:123],s=60)


#ax.scatter(C[0:4, 0], C[0:4, 1], C[0:4, 2], marker='+', c=colores[0:4], s=1000)


#hasta sector 3
#ax.scatter(X1[0:221, 0], X1[0:221, 1], X1[0:221, 2], c=asignar[0:221],s=60)
#ax.scatter(C[0:3, 0], C[0:3, 1], C[0:3, 2], marker='+', c=colores[0:3], s=1000)

if dimensiones==3:
	ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=asignar,s=60)
	ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores[:numclusters], s=1000)

if dimensiones==2:
	ax.scatter(X1[:, 0], X1[:, 1], c=asignar,s=60)
	ax.scatter(C[:, 0], C[:, 1], marker='*', c=colores[:numclusters], s=1000)

#ax.scatter(X1[:, 0], c=asignar, s=60)
#ax.scatter(C[:, 0], marker='+', c=colores, s=1000)

plt.xlabel('x')
plt.ylabel('y')
#plt.show()

#******************************************************************************
#cambiar el numero para obtener un nuevo csv

"""
with open('dim5cluster5connorm12.csv','a') as f:
	writer=csv.writer(f)
	#for i in range(maxncluster): #maxncluster: 16 (15 clusters) linea 304
	#	writer.writerow(str('sector'+i))
	writer.writerow(sect1)
	writer.writerow(sect2)
	writer.writerow(sect3)
	writer.writerow(sect4)
	writer.writerow(sect5)
	writer.writerow(sect6)
	writer.writerow(sect7)
	writer.writerow(sect8)
	writer.writerow(sect9)
	writer.writerow(sect10)
	writer.writerow(sect11)
	writer.writerow(sect12)
	writer.writerow(sect13)
	writer.writerow(sect14)
	writer.writerow(sect15)
"""


"""
sectores=np.concatenate((sect1,sect2,sect3,sect3,sect4))
maximosect=np.zeros(numcluster)

for i in range(numcluster):
	maximosect[i]=sect

print("maxsect1"+maximosect1)
print("maxsect2"+maximosect2)
print("maxsect3"+maximosect3)
print("maxsect4"+maximosect4)
print("maxsect5"+maximosect5)
print("maxsect6"+maximosect6)
print("maxsect7"+maximosect7)
print("maxsect8"+maximosect8)
print("maxsect9"+maximosect9)
print("maxsect10"+maximosect10)
print("maxsect11"+maximosect11)
print("maxsect12"+maximosect12)
print("maxsect13"+maximosect13)
print("maxsect14"+maximosect14)
print("maxsect15"+maximosect15)
"""
#NOTA: el primer elemento no cuenta porque tomaba de 1 a 15 los indices para no equivocarme
"""
print("sect1 ",sect1.shape,"sect2 ",sect2.shape)
print(sect1)
sect1y2=np.append([sect1],[sect2],axis=0)
print(sect1y2)
"""

print(sectores)


######################################
#array de totales de puntos de sectores
totalhorizontal=np.zeros(numsectores+1) #de 1 a 15
totalvertical=np.zeros(numclusters+1)
frecvertical=np.zeros(numsectores+1)
frechorizontal=np.zeros(numsectores+1)
rephorizontal=np.zeros(numsectores+1)
###############################
#muy importante contiene los clusters agrupados
freclusters=np.full((numclusters+1,15),np.inf) #1 a cantidad de clusters
##############################
counttemp=np.zeros(numclusters+1) #para guardar en que indice de frecluster va al guardar

for i in range(1,sectores.shape[0]):
	for j in range(1,sectores.shape[1]):
		totalhorizontal[i]=totalhorizontal[i]+sectores[i][j]
		totalvertical[j]=totalvertical[j]+sectores[i][j]
	rephorizontal[i]=sectores[i].max()
	indicemax=np.argmax(sectores[i])
	print("indicemax= ",indicemax)
	#np.append(freclusters[indicemax],i)
	#freclusters[indicemax].extend(i)
	indexcount=counttemp[indicemax]
	freclusters[int(indicemax)][int(indexcount)]=i
	counttemp[int(indicemax)]=counttemp[int(indicemax)]+1

print("total horizontal")
print(totalhorizontal)
print("total vertical")
print(totalvertical)
print("representante horizontal")
print(rephorizontal)
print("matriz de frec clusters")
print(freclusters)

with open('./clusters/dim5cluster15connnorm.csv','a') as f:
	writer=csv.writer(f)
	#for i in range(maxncluster): #maxncluster: 16 (15 clusters) linea 304
	#	writer.writerow(str('sector'+i))
	for i in range(1,numclusters+1):
		writer.writerow(freclusters[i])
	writer.writerow(np.zeros(numclusters+1))

