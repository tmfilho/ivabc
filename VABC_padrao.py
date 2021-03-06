from numpy import *

import old.fastslvq as slvq
import old.fcm as fcm
from scipy.spatial.distance import cdist
import sys
from old.util import sendMail
from util.functions import print_confusion_matrix


def distanciaVetorizavelClassica(indice,prots):
    distancias[:, indice] = cdist([prots[indice]/(maxs-mins)],dados/(maxs-mins),'sqeuclidean')[0]/pa

def distanciaVetorizavelIntervalar(indice,prots):
    pMins = cdist([(prots[indice,::2]/(maxs-mins))],(dados[:,::2]/(maxs-mins)),'sqeuclidean')[0]
    pMaxs = cdist([(prots[indice,1::2]/(maxs-mins))],(dados[:,1::2]/(maxs-mins)),'sqeuclidean')[0]
    distancias[:, indice] = (pMins + pMaxs)/(2*pa)

def calcularDistancias(prototipos, dadosp, n, k):
    global distancias   
    global dados
    
    dados = dadosp    
    distancias = zeros((n,k))
    
    if intervalar:
        distanciaIntervalarVetorizada(arange(k),prototipos)
    else:
        distanciaClassicaVetorizada(arange(k),prototipos)
    return distancias

def gerarSolucao(nProts):  
    [prototipos,classes] = slvq.iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nProts)
          
    return [prototipos, classes]

def iniciarVelocidades(np):
    global velocidades
    velocidades = random.uniform(-1,1,(np,k,(intervalar+1)*pa))
    if intervalar:
        velocidades[:,:,::2] = velocidades[:,:,::2]*(maxs-mins)
        velocidades[:,:,1::2] = velocidades[:,:,1::2]*(maxs-mins)
    else:
        velocidades = velocidades*(maxs-mins)

def iniciarVelocidade():
    velocidade = random.uniform(-1,1,(k,(intervalar+1)*pa))
    if intervalar:
        velocidade[:,::2] = velocidade[:,::2]*(maxs-mins)
        velocidade[:,1::2] = velocidade[:,1::2]*(maxs-mins)
    else:
        velocidade = velocidade*(maxs-mins)
    return velocidade

def ajustarVelocidade(velocidade):  
    if intervalar:   
        minimos = velocidade[:,::2]
        maximos = velocidade[:,1::2]
        indMins = where(minimos < -(maxs-mins))
        minimos[indMins] = -(maxs-mins)[indMins[1]]
        indMaxs = where(minimos > (maxs-mins))
        minimos[indMaxs] = (maxs-mins)[indMaxs[1]]
        
        indMins = where(maximos < -(maxs-mins))
        maximos[indMins] = -(maxs-mins)[indMins[1]]
        indMaxs = where(maximos > (maxs-mins))
        maximos[indMaxs] = (maxs-mins)[indMaxs[1]]
        velocidade[:,::2] = minimos
        velocidade[:,1::2] = maximos
    else:
        indMins = where(velocidade < -(maxs-mins))
        velocidade[indMins] = -(maxs-mins)[indMins[1]]
        indMaxs = where(velocidade > (maxs-mins))
        velocidade[indMaxs] = (maxs-mins)[indMaxs[1]]
    return velocidade

def inicializar(nProts, np):
    global particulas
    global classesParticulas
    global PBEST
    global criteriosPBEST
    global indiceGBEST
    global criterios
    global limites
    
    limites = zeros(np)
    
    
    particulas = zeros((np,sum(nProts),shape(treinamento)[1]))
    iniciarVelocidades(np)
    classesParticulas = zeros((np,sum(nProts)))
    criterios = zeros(np)
    for particula in range(np):
        [particulas[particula], classesParticulas[particula]] = gerarSolucao(nProts)
        criterios[particula] = calcularCriterioJ(particulas[particula], classesParticulas[particula])
    PBEST = copy(particulas)
    criteriosPBEST = copy(criterios)
    indiceGBEST = argmin(criterios) 
 
def calcularCriterioJ(particula, classesParticula):
    [erro,criterio] = testar(treinamento, classesTreinamento, particula, classesParticula)
    return alfa * (erro/100) + beta * criterio            

def ajustarMinsMaxs(dados):
    if intervalar:
        minimos = dados[:,::2]
        maximos = dados[:,1::2]
        ind = minimos > maximos
        if any(ind == True):
            temp = minimos[ind]
            minimos[ind] = maximos[ind]
            maximos[ind] = temp
            
            dados[:,::2] = minimos
            dados[:,1::2] = maximos   
        indMins = where(minimos < mins)
        minimos[indMins] = mins[indMins[1]]
        indMaxs = where(minimos > maxs)
        minimos[indMaxs] = maxs[indMaxs[1]]
        indMins = where(maximos < mins)
        maximos[indMins] = mins[indMins[1]]
        indMaxs = where(maximos > maxs)
        maximos[indMaxs] = maxs[indMaxs[1]]
    else:
        indMins = where(dados < mins)
        dados[indMins] = mins[indMins[1]]
        indMaxs = where(dados > maxs)
        dados[indMaxs] = maxs[indMaxs[1]]
    return dados

def VABC(np, nProts):    
    MAX_ITERACAO = 200 
    
    r = 1  
    repeticoes = 0
    f_antes = criteriosPBEST[indiceGBEST]
    while r < MAX_ITERACAO and repeticoes < 50:
        r = r + 1
        # atualizar PBESTs e GBEST
        enviarTrabalhadoras(np)
        enviarObservadoras(np,r,MAX_ITERACAO)
        enviarEscoteiras(np, nProts)
        
        f_depois = criteriosPBEST[indiceGBEST]
        if f_depois == f_antes:
            repeticoes = repeticoes + 1
        else:
            repeticoes = 0      
        f_antes = f_depois 
    
def setarVariavelFonte(fonte,fonteK,prot,j):
    fi = 2 * random.random_sample() -1
    xTemp = copy(particulas[fonte])
    if intervalar:
        mi = particulas[fonte,prot,2*j] + fi * (particulas[fonte,prot,2*j] - particulas[fonteK,prot,2*j])
        ma = particulas[fonte,prot,2*j+1] + fi * (particulas[fonte,prot,2*j+1] - particulas[fonteK,prot,2*j+1])
        if mi > ma:
            temp = ma
            ma = mi
            mi = temp
        if ma > maxs[j]:
            ma = maxs[j]
        elif ma < mins[j]:
            ma = mins[j]
        if mi > maxs[j]:
            mi = maxs[j]
        elif mi < mins[j]:
            mi = mins[j]
        xTemp[prot,2*j] = mi
        xTemp[prot,2*j+1] = ma
    else:
        var = particulas[fonte,prot,j] + fi * (particulas[fonte,prot,j] - particulas[fonteK,prot,j])
        if var > maxs[j]:
            var = maxs[j]
        elif var < mins[j]:
            var = mins[j]
        xTemp[prot,j] = var
    return xTemp

def enviarTrabalhadoras(np):
    global limites
    global particulas
    global PBEST
    global criteriosPBEST
    global indiceGBEST
    global criterios
    
    for fonte in range(np):
        j = random.choice(pa) #selecionar uma dimensao aleatoria do problema
        prot = random.choice(k) #selecionar um prototipo aleatorio do problema
        #if prot > 0:
        #    j = random.choice(delete(arange(pa),where(around(particulas[fonte,0,:pa]) ==0.0)))
        fonteK = random.choice(delete(arange(np),fonte)) #selecionar uma fonte aleatoria
        
        xTemp = setarVariavelFonte(fonte,fonteK,prot,j)
        f = calcularCriterioJ(xTemp, classesParticulas[fonte])
        if f < criterios[fonte]:
            criterios[fonte] = f
            particulas[fonte] = xTemp
            limites[fonte] = 0
        else:
            limites[fonte] = limites[fonte] + 1
        if f < criteriosPBEST[fonte]:
            criteriosPBEST[fonte] = f
            PBEST[fonte] = xTemp
        
    indiceGBEST = argmin(criteriosPBEST)

def enviarObservadoras(np, t, tmax):
    global limites
    global particulas
    global velocidades
    global PBEST
    global criteriosPBEST
    global indiceGBEST
    global criterios
    
    fitness = 1.0 / (criterios + 1.0)
    probs = fitness / sum(fitness)
    
    wmin = 0.4
    wmax = 0.9
    
    w = wmin+(wmax-wmin)*(tmax-t)/tmax
    c1 = 0.5+ random.rand()/2.0
    c2 = 0.5+ random.rand()/2.0

    for index in range(np):    
        fonte = random.choice(int(np),p=probs)
        r1 = random.rand()
        r2 = random.rand()
        
        velocidades[fonte] = ajustarVelocidade(w*velocidades[fonte] + c1*r1*(PBEST[fonte] - particulas[fonte]) + c2*r2*(PBEST[indiceGBEST] - particulas[fonte]))
          
        xTemp = ajustarMinsMaxs(particulas[fonte] + velocidades[fonte]) # calculo da nova posicao
        # ajuste da posicao para que os valores fiquem no intervalo [1;c], que sao os possiveis clusters
        f = calcularCriterioJ(xTemp, classesParticulas[fonte])
        if f < criterios[fonte]:
            criterios[fonte] = f
            particulas[fonte] = xTemp
            limites[fonte] = 0
        else:
            limites[fonte] = limites[fonte] + 1
        if f < criteriosPBEST[fonte]:
            criteriosPBEST[fonte] = f
            PBEST[fonte] = xTemp   
        if f < criteriosPBEST[indiceGBEST]:
            indiceGBEST = fonte

def enviarEscoteiras(np, nProts):
    global limites
    global particulas
    global velocidades
    global PBEST
    global criteriosPBEST
    global indiceGBEST
    global criterios
    
    for fonte in arange(np)[limites >= 10]:
        [particulas[fonte], classesParticulas[fonte]] = gerarSolucao(nProts)
        criterios[fonte] = calcularCriterioJ(particulas[fonte], classesParticulas[fonte])
        
        velocidades[fonte] = iniciarVelocidade()
            
        if criterios[fonte] < criteriosPBEST[fonte]:
            criteriosPBEST[fonte] = criterios[fonte]
            PBEST[fonte] = copy(particulas[fonte])
        limites[fonte] = 0          
            
    indiceGBEST = argmin(criteriosPBEST)

def testar(teste, classesTeste, particula, classesParticula):
    ne = shape(teste)[0]
    distancias = calcularDistancias(particula, teste, ne, k)
    
    particao = argmin(distancias,1)       
    classesResultantes = classesParticula[particao]
    if len(classesTeste) < 200:
        return classesResultantes
    numeroErros = float(sum(classesTeste != classesResultantes))
    sumDistanciasProtsMembros = sum(distancias.min(1) * (classesTeste == classesParticula[particao]))
    criterio = sumDistanciasProtsMembros / ne
    
    return [(numeroErros / ne)*100.0,criterio]

def rodarValidacaoCruzada(dados, classes, nProts, montecarlo, nFolds, np): 
    global pa
    global k
    global n    
    global treinamento
    global classesTreinamento
    
    if len(nProts) == 1:
        nProts = ones(max(classes)+1) * nProts[0] 
    
    k = sum(nProts)
    
    erros = zeros(montecarlo*nFolds)
    pa = shape(dados)[1]/2
    if not intervalar:
        pa = shape(dados)[1]
    nDados = size(classes)
    for i in range(montecarlo):
        indices = arange(nDados)
        random.shuffle(indices)
        dadosEmbaralhados = dados[indices,:]
        classesEmbaralhadas = classes[indices]
        folds = slvq.separarFolds(dadosEmbaralhados, classesEmbaralhadas, nFolds)
        for fold in range(nFolds):
            print i*nFolds + fold
            [treinamento, classesTreinamento, teste, classesTeste] = slvq.separarConjuntos(folds, dadosEmbaralhados, classesEmbaralhadas, fold)      
            n = shape(treinamento)[0]
            preds = zeros((30, len(classesTeste)))
            from tqdm import tqdm
            for l in tqdm(arange(30)):
                inicializar(nProts, np)
                VABC(np, nProts)
                GBEST = PBEST[indiceGBEST]
                #print sum(removidosPBEST[indiceGBEST])
                # [erros[i*nFolds + fold], d] = testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST])
                preds[l] = testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST])
            predictions = around(mean(preds, axis=0))
            print_confusion_matrix(classesTeste, predictions)
            exit()
            # print erros[i*nFolds + fold]
    print erros
    print mean(erros)
    print std(erros)       
    return erros , "\n erro medio:" , mean(erros) , "\n desvio:" , std(erros) 

def rodarLOO(dados, classes, nProts, montecarlo, np): 
    global pa
    global k
    global n    
    global treinamento
    global classesTreinamento
    
    if len(nProts) == 1:
        nProts = ones(max(classes)+1) * nProts[0] 
    
    k = sum(nProts)
    rem = 0.0
    pa = shape(dados)[1]/2
    if not intervalar:
        pa = shape(dados)[1]
    consideradas = zeros(pa)
    nDados = size(classes)
    erros = zeros(montecarlo*nDados)
    for i in range(montecarlo):
        indices = arange(nDados)
        random.shuffle(indices)
        dadosEmbaralhados = dados[indices,:]
        classesEmbaralhadas = classes[indices]
        for dado in range(nDados):
            print i*nDados + dado
            teste = array([dadosEmbaralhados[dado,:]])
            classesTeste = array([classesEmbaralhadas[dado]])
            indTreinamento = delete(arange(nDados),dado)
            treinamento = dadosEmbaralhados[indTreinamento,:]
            classesTreinamento = classesEmbaralhadas[indTreinamento]
            n = shape(treinamento)[0]
            inicializar(nProts, np)
            VABC(np, nProts)
            GBEST = PBEST[indiceGBEST]
            consideradas = consideradas + GBEST[0,:pa]
            print sum(removidosPBEST[indiceGBEST])    
            rem = rem + float(sum(removidosPBEST[indiceGBEST]))        
            [erros[i*nDados + dado], d] =  testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST], pesosPBEST[indiceGBEST], removidosPBEST[indiceGBEST])
            print erros[i*nDados + dado]
    print erros
    print mean(erros)
    print std(erros)        
    print consideradas / (montecarlo * nDados)
    print rem / (montecarlo * nDados)

def geraNormaisMultiVariadas(parametros,intervalo):
    quantidades = parametros[:,-2]
    nDados = sum(quantidades)
    dados = zeros((nDados,2*pa))
    classes = zeros(nDados)
    nRegioes = shape(parametros)[0]
    inicio = 0
    fim = 0
    par = parametros[:,0:pa*2]
    for regiao in range(nRegioes):
        inicio = fim
        fim = fim + quantidades[regiao]
        medias = par[regiao,0::2]#array([parametros[regiao,0], parametros[regiao,2]])
        covs = diag(par[regiao,1::2])#array([parametros[regiao,1], parametros[regiao,3]]))
        d = random.multivariate_normal(medias,covs,quantidades[regiao])
        delta = random.randint(1,intervalo,pa)
        delta = delta.astype(float)
        for var in range(pa):
            dados[inicio:fim,2*var] = d[:,var] - (delta[var]/2)
            dados[inicio:fim,2*var+1] = d[:,var] + (delta[var]/2)
        classes[inicio:fim] = parametros[regiao,-1]
    return [dados,classes.astype(int)]
    
def rodarSimulados(parametros, intervalo, nProts, montecarlo, np): 
    global pa
    global k
    global n    
    global treinamento
    global classesTreinamento
    global mins
    global maxs
        
    k = sum(nProts)
    
    erros = zeros(montecarlo)
    pa = (shape(parametros)[1] - 2)/2
    consideradas = zeros(pa)
    for i in range(montecarlo):
        print i
        [dados,classes] = geraNormaisMultiVariadas(parametros,intervalo)
        if not intervalar:
            dados = (dados[:,::2] + dados[:,1::2])/2.0
        if intervalar:
            mins = dados[:,::2].min(0)
            maxs = dados[:,1::2].max(0)
        else: 
            mins = dados.min(0)
            maxs = dados.max(0)
        #qtd = shape(dados)[0]
        #for instancia in range(qtd):
        #    print dados[instancia,], classes[instancia]
        #exit()
        #for var in arange(pa):
        #    dados[:,[2*var,2*var+1]] = (dados[:,[2*var,2*var+1]] - dados[:,[2*var,2*var+1]].min())/(dados[:,[2*var,2*var+1]].max() - dados[:,[2*var,2*var+1]].min())
        
        [treinamento, classesTreinamento, teste, classesTeste] = slvq.separarHoldOut(dados, classes, 0.5)
        n = shape(treinamento)[0]
        inicializar(nProts, np)
        VABC(np, nProts)
        GBEST = PBEST[indiceGBEST]
        #print GBEST[0,:pa]
        #print sum(removidosPBEST[indiceGBEST])          
        [erros[i], d] =  testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST])
        print erros[i]
    print erros
    print mean(erros)
    print std(erros)   
    return erros , "\n erro medio:" , mean(erros) , "\n desvio:" , std(erros) 
    
def lerDados(nome):
    if intervalar:
        pasta = "dados/"
    else:
        pasta = "dadosC/"
    with open(pasta + nome, 'r') as f:
        dados = array([line.split() for line in f])
        dados = dados.astype(float)
        dados = dados[dados[:,shape(dados)[1]-1].argsort()]
        classes = dados[:,shape(dados)[1]-1].astype(int)
        dados = dados[:,0:shape(dados)[1]-1]
    return [dados, classes]

if __name__ == "__main__":
    random.seed(1)
    
    global intervalar
    global alfa
    global beta
    global distanciaClassicaVetorizada
    global distanciaIntervalarVetorizada
    global mins
    global maxs
    
    distanciaClassicaVetorizada = vectorize(distanciaVetorizavelClassica, excluded=[1])
    distanciaIntervalarVetorizada = vectorize(distanciaVetorizavelIntervalar, excluded=[1])
    
    intervalar = True
    alfa = 0.5
    beta = 1.0 - alfa
    
    prots = [34, 28, 12, 20, 12, 42, 38, 34, 6, 40]
    nome = "base_meses_corrigida_nao_normalizados.txt"
    
    # titulo = "vabc", prots, "2,2 novo"
    # print titulo
    # print nome
    # titulo = titulo , nome
    [dados, classes] = lerDados(nome)
    if intervalar:
        mins = dados[:,::2].min(0)
        maxs = dados[:,1::2].max(0)
    else:
        mins = dados.min(0)
        maxs = dados.max(0)
    texto = rodarValidacaoCruzada(dados, classes, prots, 10, 10, 25)
    #rodarLOO(dados, classes, prots, 10, 25) 
    # intervalo = 10
    # texto = rodarSimulados(array([[99,9,99,169,200,0],[104,16,118,16,150,0],[104,16,80,16,150,1],[100,9,99,169,200,1]] ), intervalo, prots, 100, 25)
    # sendMail(titulo,texto)