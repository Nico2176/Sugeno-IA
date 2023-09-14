"""Subtractive Clustering Algorithm
"""
__author__ = 'Daniel Albornoz'


import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class fisRule:
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids


    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
      #  plt.figure()                                                        //grafica las gaussianas !!!
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
           # plt.plot(x,y)

class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []



    def genfis(self, data, radii):

        start_time = time.time()
        labels, cluster_center = subclust2(data, radii)

        print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        print("nivel acti")
       # print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        print("sumMu")
      #  print(sumMu)
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]


        A = acti*inp/sumMu

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
       # print(solutions)
        return 0

    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        for input in self.inputs:
            input.view()


def subclust2(data, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
    if Rb==0:
        Rb = Ra*1.15

    scaler = MinMaxScaler()
    scaler.fit(data)
    ndata = scaler.transform(data)

    # 14/05/2020 cambio list comprehensions por distance matrix
    #P = np.array([np.sum([np.exp(-(np.linalg.norm(u-v)**2)/(Ra/2)**2) for v in ndata]) for u in ndata])
    #print(P)
    P = distance_matrix(ndata,ndata)
    alpha=(Ra/2)**2
    P = np.sum(np.exp(-P**2/alpha),axis=0)

    centers = []
    i=np.argmax(P)
    C = ndata[i]
    p=P[i]
    centers = [C]

    continuar=True
    restarP = True
    while continuar:
        pAnt = p
        if restarP:
            P=P-p*np.array([np.exp(-np.linalg.norm(v-C)**2/(Rb/2)**2) for v in ndata])
        restarP = True
        i=np.argmax(P)
        C = ndata[i]
        p=P[i]
        if p>AcceptRatio*pAnt:
            centers = np.vstack((centers,C))
        elif p==1:
                centers = np.vstack((centers,C))
        else:
                P[i]=0
                restarP = False
        if not any(v>0 for v in P):
            continuar = False
    distancias = [[np.linalg.norm(p-c) for p in ndata] for c in centers]
    labels = np.argmin(distancias, axis=0)
    centers = scaler.inverse_transform(centers)
    return labels, centers



#test commit con comandos che




# c1 = np.random.rand(15,2)+[1,1]
# c2 = np.random.rand(10,2)+[10,1.5]
# c3 = np.random.rand(5,2)+[4.9,5.8]
# m = np.append(c1,c2, axis=0)
# m = np.append(m,c3, axis=0)

# r,c = subclust2(m,2)

# plt.figure()
# plt.scatter(m[:,0],m[:,1])
# plt.scatter(c[:,0],c[:,1], marker='X')
# print(c)
     

"""c1 = np.random.rand(150,2)+[1,1]
c2 = np.random.rand(100,2)+[10,1.5]
c3 = np.random.rand(50,2)+[4.9,5.8]
m = np.append(c1,c2, axis=0)
m = np.append(m,c3, axis=0)"""




erroresPorRadio = []
datos = np.loadtxt(r"D:\Carpetas de usuario\Documentos\Repos\IA2023\samplesVDA3.txt")
#print(datos)
datosbi = np.zeros((len(datos),2))
#print(datosbi)
for i,valor in enumerate(datos):
    # datosbi[i,0]=i
     datosbi[i,0]=i*2.5   #2.5 para que las unidades queden en ms, ver luego si es conveniente
     datosbi[i,1]=datos[i]
#print(datosbi)

arreglo_cantClusters = []
for numero in range(3, 20, 1):
    valor_decimal = numero / 10.0
    print(valor_decimal)
    arreglo_valores = []
    
    r,c = subclust2(datosbi,valor_decimal)
    arreglo_valores.append(valor_decimal)
    arreglo_cantClusters.append(c.shape[0])
    def my_exponential(A, B, C, x):
        return A*np.exp(-B*x)+C

    #data_x = np.arange(-10,10,0.1)
    data_x = datosbi[:,0]
    #data_y = -0.5*data_x**3-0.6*data_x**2+10*data_x+1 #my_exponential(9, 0.5,1, data_x)
    data_y = datosbi[:,1]
   # plt.plot(data_x, data_y)
    # plt.ylim(-20,20)
   # plt.xlim(-7,7)

    data = np.vstack((data_x, data_y)).T

    fis2 = fis()
    fis2.genfis(data, valor_decimal)
    fis2.viewInputs()
    r = fis2.evalfis(np.vstack(data_x))  #valores en y de la funcion sugeno

    error=0
    aux=0
    valoresMSE = []
    
    for i,valor in enumerate(data_y):
        aux=(data_y[i]-r[i])**2
        valoresMSE.append(aux)
      #  print ("Datay[",i,"]: ", data_y[i], " r[", i, "]: ", r[i], "Error relativo en el punto: ", aux)
        error=error+aux

    
    valoresMSE = np.array(valoresMSE)

    print("Cantidad elementos data y: ", len(data_y))
    print("Cantidad elementos r: ", len(r))

    print("error cuadratico medio: ", error/len(data_y))
    erroresPorRadio.append(error/len(data_y))

    """"
    plt.figure()
    titulo = "Cantidad de clusters: {}".format(c.shape[0])
    plt.title(titulo)
    plt.xlabel("Tiempo [ms]")
    plt.ylabel("MSE")
    plt.plot(data_x,data_y)
    plt.plot(data_x,r,linestyle='--')
 #   plt.show()
    """
    #plt.plot(np.arange(0,len(valoresMSE),1),valoresMSE)
    #plt.plot(valoresMSE,np.arange(0,len(valoresMSE),1))
   # plt.show()

    fis2.solutions



print("Valores: ", arreglo_valores)
print("Cant Clusters: ", arreglo_cantClusters)

valoresNP = np.array(arreglo_valores)
cantClustersNP = np.array(arreglo_cantClusters)

""" plt.figure()
plt.title("Cant clusters vs radio intercluster")
plt.grid(True)
plt.xlabel("Radio intercluster")
plt.ylabel("Clusters")
plt.plot(arreglo_valores,arreglo_cantClusters,color='b')
plt.show() """

print("Errores: ", erroresPorRadio)
print("Cant Clusters: ", arreglo_cantClusters)
erroresPorRadio = np.array(erroresPorRadio)

plt.figure()
plt.title("R vs MSE")
plt.grid(True)
plt.xlabel("R")
plt.ylabel("MSE")
plt.plot(arreglo_cantClusters,erroresPorRadio,color='b')
plt.show()





# print("r:",r) R es la matriz de pertenencia de cada dato
# print("c:",c) c es la matriz de clusters con las coordenadas de los centros de cluster. su cantidad de filas nos dará la cantidad de clusters

"""plt.figure()
plt.xlabel("Tiempo [ms]")  #plt.xlabel("Tiempo [2.5ms.]")
plt.ylabel("VDA")
plt.grid(False)   #cuadriculado
plt.scatter(datosbi[:,0],datosbi[:,1], c=r)  
plt.scatter(c[:,0],c[:,1], marker='X',color='m')
plt.show()"""

exit()

def my_exponential(A, B, C, x):
    return A*np.exp(-B*x)+C

#data_x = np.arange(-10,10,0.1)
data_x = datosbi[:,0]
#data_y = -0.5*data_x**3-0.6*data_x**2+10*data_x+1 #my_exponential(9, 0.5,1, data_x)
data_y = datosbi[:,1]
plt.plot(data_x, data_y)
# plt.ylim(-20,20)
plt.xlim(-7,7)

data = np.vstack((data_x, data_y)).T

fis2 = fis()
fis2.genfis(data, 1)
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))  #valores en y de la funcion sugeno

error=0
aux=0
valoresMSE = []
for i,valor in enumerate(data_y):
    aux=(data_y[i]-r[i])**2
    valoresMSE.append(aux)
    print ("Datay[",i,"]: ", data_y[i], " r[", i, "]: ", r[i], "Error relativo en el punto: ", aux)
    error=error+aux

valoresMSE = np.array(valoresMSE)

print("Cantidad elementos data y: ", len(data_y))
print("Cantidad elementos r: ", len(r))

print("error cuadratico medio: ", error/len(data_y))

plt.figure()
plt.plot(data_x,data_y)
plt.plot(data_x,r,linestyle='--')
plt.show()

#plt.plot(np.arange(0,len(valoresMSE),1),valoresMSE)
#plt.plot(valoresMSE,np.arange(0,len(valoresMSE),1))
plt.show()

fis2.solutions