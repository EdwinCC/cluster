from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import csv

dataframe = pd.read_csv(r"/home/edwin/Escritorio/databeaconsoriginal/Tx_0x04.csv")
print("imprimiendo dataframe (datos recien importados)")
print(dataframe)
print("termino de imprimir dataframe")

X=dataframe.values #saca los valores del csv como un array

print("X:")
print(X)

#cambio X por data para distinguirlo mejor y guardar el array original sin tocarlo
data=X[:,0:5] #aqui se guardan los datos de intensidades
sectores=X[:,5] #aqui se guarda el campo de sectores

#data=np.asarray(data)

print("data:")
print(data)

Z=linkage(data,'ward')
#Z=linkage(data,'single')
#Z=linkage(data,'complete')
#Z=linkage(data,'average')

print("linkage")
print(Z)

c, coph_dists=cophenet(Z,pdist(data))
#c, coph_dists=cophenet(Z,pdist(data,'cityblock'))
#c, coph_dists=cophenet(Z,pdist(data,'hamming'))

print("cophenet correlation distance")
print(c)

print("cophenetic distance matrix")
print(coph_dists)

plt.figure(figsize=(25,10))
plt.title('Hierarchical Clustering Dendogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
	Z,
	truncate_mode='lastp',
	p=15, #hasta cuantas divisiones mostrar (yo solo quiero q muestre 15 clusters)
	#show_leaf_counts=False,
	leaf_rotation=90., #rotates the x axis labels
	leaf_font_size=8., #font size for the x axis labels
	show_contracted=True,
)
plt.show()

k=15 #---------------------------------------------- N° clusters
clusters=fcluster(Z,k,criterion='maxclust')

#imprimo las predicciones
print("labels:")
tam=len(clusters)
print("tamaño clusters = {0}".format(tam))
for i in range(len(clusters)):
	print(clusters[i])


#Tx_0x04
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

#prueba: imprimo los valores del primer cluster para asegurarme

print("data sector14")
for i in range (maxs13,maxs14): #para sect1 0:maxs1, sectn maxs(n-1):maxsn
        print(data[i])
print("fin data sector14")

#evaluo la distribucion de los cluster en donde deberian estar cada cluster

#sect1, sect2, etc guardan los labels predichos para cada punto en cada sector

#sector1

minncluster=1
maxncluster=16

sect1=np.zeros(16) #creo matriz numpy de 0 a 15
for i in range(0,maxs1): #itero de 0 al limite del sector 1 (todos datos del sector1)
        for j in range(minncluster,maxncluster): #itero de 1 a 15
                if clusters[i]==j: #si el label predicho es igual al sector real
                        sect1[j]=sect1[j]+1 #aumenta el contador de aciertos en 1

print("distribucion sector 1")


for i in range(1,len(sect1)):
        print(i,sect1[i])

#sector2

sect2=np.zeros(16)
for i in range(maxs1,maxs2):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect2[j]=sect2[j]+1

print("distribucion sector 2")


for i in range(1,len(sect2)):
        print(i,sect2[i])

#sector3

sect3=np.zeros(16)
for i in range(maxs2,maxs3):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect3[j]=sect3[j]+1

print("distribucion sector 3")


for i in range(1,len(sect3)):
        print(i,sect3[i])

#sector4

sect4=np.zeros(16)
for i in range(maxs3,maxs4):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect4[j]=sect4[j]+1

print("distribucion sector 4")


for i in range(1,len(sect4)):
        print(i,sect4[i])

#sector5

sect5=np.zeros(16)
for i in range(maxs4,maxs5):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect5[j]=sect5[j]+1

print("distribucion sector 5")


for i in range(1,len(sect5)):
        print(i,sect5[i])

#sector6

sect6=np.zeros(16)
for i in range(maxs5,maxs6):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j: #labels son de 0 a 14, mis sectores son de 1 a 15
                        sect6[j]=sect6[j]+1

print("distribucion sector 6")


for i in range(1,len(sect6)):
        print(i,sect6[i])


#sector7

sect7=np.zeros(16)
for i in range(maxs6,maxs7):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect7[j]=sect7[j]+1

print("distribucion sector 7")

for i in range(1,len(sect7)):
        print(i,sect7[i])

#sector8

sect8=np.zeros(16)
for i in range(maxs7,maxs8):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect8[j]=sect8[j]+1

print("distribucion sector 8")


for i in range(1,len(sect8)):
        print(i,sect8[i])

#sector9

sect9=np.zeros(16)
for i in range(maxs8,maxs9):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect9[j]=sect9[j]+1

print("distribucion sector 9")


for i in range(1,len(sect9)):
        print(i,sect9[i])

#sector10

sect10=np.zeros(16)
for i in range(maxs9,maxs10):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect10[j]=sect10[j]+1

print("distribucion sector 10")

for i in range(1,len(sect10)):
        print(i,sect10[i])

#sector11

sect11=np.zeros(16)
for i in range(maxs10,maxs11):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect11[j]=sect11[j]+1

print("distribucion sector 11")


for i in range(1,len(sect11)): #imprimo la distribucion de label 1 a label 15 
        print(i,sect11[i])

#sector12

sect12=np.zeros(16)
for i in range(maxs11,maxs12):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect12[j]=sect12[j]+1

print("distribucion sector 12")


for i in range(1,len(sect12)):
        print(i,sect12[i])

#sector13

sect13=np.zeros(16)
for i in range(maxs12,maxs13):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect13[j]=sect13[j]+1

print("distribucion sector 13")

for i in range(1,len(sect13)):
        print(i,sect13[i])

#sector14

sect14=np.zeros(16)
for i in range(maxs13,maxs14):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect14[j]=sect14[j]+1

print("distribucion sector 14")


for i in range(1,len(sect14)):
        print(i,sect14[i])


#sector15

sect15=np.zeros(16)
for i in range(maxs14,maxs15-1):
        for j in range(minncluster,maxncluster):
                if clusters[i]==j:
                        sect15[j]=sect15[j]+1

print("distribucion sector 15")


for i in range(1,len(sect15)):
        print(i,sect15[i])


print("sector 1------------------------------------------------")
for i in range(len(clusters)):
	if i==322:
		print("sector 2----------------------------------------------")
	if i==636:
		print("sector 3----------------------------------------------")
	if i==946:
		print("sector 4----------------------------------------------")
	if i==1290:
		print("sector 5----------------------------------------------")
	if i==1638:
		print("sector 6----------------------------------------------")
	if i==1974:
		print("sector 7----------------------------------------------")
	if i==2324:
		print("sector 8----------------------------------------------")
	if i==2657:
		print("sector 9----------------------------------------------")
	if i==2968:
		print("sector 10----------------------------------------------")
	if i==3433:
		print("sector 11----------------------------------------------")
	if i==3765:
		print("sector 12----------------------------------------------")
	if i==4067:
		print("sector 13----------------------------------------------")
	if i==4399:
		print("sector 14----------------------------------------------")
	if i==4730:
		print("sector 15----------------------------------------------")
	print(clusters[i])


with open('./hierarchical/salida15clusters.csv','a') as f:
        writer=csv.writer(f)
        #for i in range(maxncluster): #maxncluster: 16 (15 clusters) linea 304
        #       writer.writerow(str('sector'+i))
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

