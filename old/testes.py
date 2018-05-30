import timeit
import numpy as np

def foo(indice,pesos,prots):
    distancias[:, indice] = np.sum(pesos[indice]*(prots[indice] - dados)**2,1)

if __name__ == '__main__':
    #global dados
    #global distancias
    #distancias = np.zeros((292,32))
    #dados = np.arange(292*32).reshape(292,32)
    #prots = np.arange(32*32).reshape(32,32)
    #pesos = np.ones((32,32))*0.5
    
    #fv = np.vectorize(foo, excluded=[1,2])
    
    #fv(np.arange(32),pesos,prots)
    
    #print distancias
    
    setup = '''
from numpy import arange
from numpy import sum
from numpy import reshape
from numpy import zeros
from numpy import ones
from numpy import vectorize
from numpy import sqrt
from scipy.spatial.distance import cdist

def foo(indice,pesos,prots):
    distancias[:, indice] = cdist([sqrt(pesos[indice])*prots[indice]],sqrt(pesos[indice])*dados)[0]
    
global dados
global distancias
distancias = zeros((10000,32))
dados = arange(10000*32).reshape(10000,32)
prots = arange(32*32).reshape(32,32)
pesos = ones((32,32))*0.5

fv = vectorize(foo, excluded=[1,2])

'''

    setup3 = '''
from numpy import arange
from numpy import sum
from numpy import reshape
from numpy import zeros
from numpy import ones
from numpy import shape
from numpy import sqrt

def foo(pesos,prots,dados):
    n = shape(dados)[0]
    k = shape(prots)[0]
    distancias = zeros((n,k))
    for indice in range(k):
        distancias[:, indice] = sqrt(sum(pesos[indice]*(prots[indice] - dados)**2,1))
    return distancias

dados = arange(292*32).reshape(292,32)
prots = arange(32*32).reshape(32,32)
pesos = ones((32,32))*0.5


'''
    
    setup4 = '''
from numpy import arange
from numpy import sum
from numpy import reshape
from numpy import ones
from numpy import repeat

dados = arange(100000*32).reshape(100000,32)
prots = arange(32*32).reshape(32,32)
pesos = ones((32,32))*0.5
dados_prontos = dados.repeat(32,axis=0).reshape(32,100000,32)

'''    
    
    setup2 = '''
from numpy import zeros
from numpy import random
from numpy import ones
from numpy import sum
from numpy import around
from numpy import shape

def calcularDistancias(prototipos, dados, pesos, variaveisConsideradas, n, k, intervalar=False):
    consideradas = around(variaveisConsideradas)
    distancias = zeros((n,k))
    for prot in range(k):
        if intervalar:                 
            mins = consideradas * pesos[prot,:] * ((prototipos[prot,::2] - dados[:,::2])**2)
            maxs = consideradas * pesos[prot,:] * ((prototipos[prot,1::2] - dados[:,1::2])**2)
            distancias[:,prot] = sum(mins + maxs,1) + 0.0000000001
        else:
            distancias[:,prot] = sum((consideradas * pesos[prot,:] * (prototipos[prot,:] - dados)**2),1) + 0.0000000001
    return distancias
    
prots = random.rand(32,32)
dados = random.rand(292,32)
pesos = ones((shape(prots)[0],16))
variaveisCons = ones(16)
'''

    setup4 = '''
from numpy import arange
from numpy import sum
from numpy import reshape
from numpy import ones
from numpy import repeat

dados = arange(100000*32).reshape(100000,32)
prots = arange(32*32).reshape(32,32)
pesos = ones((32,32))*0.5
dados_prontos = dados.repeat(32,axis=0).reshape(32,100000,32)

'''    
    
    setup5 = '''
from numpy import random
from numpy import copy

    
a = random.rand(10000,32)
'''
    
    print min(timeit.repeat("c = a + 0", setup=setup5,number=7,repeat=1000))
