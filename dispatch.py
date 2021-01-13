import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def resolve_de(Pd,beta,gamma):
    n = len(beta)
    b = [-beta[i] for i in range(n)]
    b.append(Pd)
    c = np.zeros((n+1,n+1))
    for i in range(n): c[i][i] = gamma[i]*2
    c[ :, -1] = -1
    c[-1,  :] =  1
    c[-1, -1] =  0
    de = np.linalg.solve(c,b)
    return de

def sem_perdas(Pd,beta,gamma,pmin,pmax):
    n = len(beta)
    pg = resolve_de(Pd,beta,gamma)[:-1]
    bet = []
    gam = []
    state = []
    carga = Pd
    #print(f'[0] Carga inicial: {carga} (MW)'.format())
    for i in range(n):  
        if(pmin[i] <= pg[i] <= pmax[i]):
            #print(f'G{i+1}] gerando {pg[i]} (MW) está dentro dos limites operacionais!'.format())
            state.append(0)
            bet.append(beta[i])
            gam.append(gamma[i])
        else:
            if(pmin[i]> pg[i]):
                #print(f'G{i+1}] gerando {pg[i]} (MW) é menor que {pmin[i]} (MW)!'.format())
                #print('Ultrapassando a capacidade minima de geração da unidade!')
                pg[i] = pmin[i]
                state.append(-1)
                bet.append(beta[i])
                gam.append(gamma[i])
            else:
                #print(f'G{i+1}] gerando {pg[i]} (MW) é maior que {pmax[i]} (MW)!'.format())
                #print('Ultrapassando a capacidade máxima de geração da unidade!')
                pg[i] = pmax[i]
                state.append(1)
                carga = carga - pg[i]
        #print(f'[{i+1}]Carga atual: {carga} (MW)'.format())
    b = np.array(bet)
    c = np.array(gam)
    #print('STATE: ', state)
    #print('B:',b, 'C:',c)
    de = resolve_de(carga, b, c)
    #print('DE: ',de)
    k=0
    for i in range(n):
        if(state[i]>0):
            k=k+1
        else:
            pg[i] = de[i-k]
        pg = np.array(pg)
        lbd = np.float(de[-1])
    #print('PG:',pg,'\n')
    #print('lbd:',de[-1],'\n')
    return ([pg, lbd])

def despacho(path,Pd):
    # ENTRADA DE DADOS:
    data = pd.read_csv(path)
    data.head()
    print(f'Problema:\n{data}\nDemanda: {Pd} (MW)'.format())
    u = data['Unid.'] 
    alpha = data['A']
    beta  = data['B']
    gamma = data['C']
    custo = data['Custo']
    pmin  = data['Min']
    pmax  = data['Max']
    n = len(u)


    a = [alpha[i]*custo[i] for i in range(n)]
    b = [ beta[i]*custo[i] for i in range(n)]
    c = [gamma[i]*custo[i] for i in range(n)]
    de = sem_perdas(Pd, b,c,pmin,pmax)
    print('PG:',  de[0])
    print('Cin:', de[1])
    return de

def simu_de():
    print('Iniciar simulação...\n')


def visualize():
    import matplotlib
    matplotlib.axes.Axes.legend
    matplotlib.pyplot.legend
    matplotlib.legend.Legend
    matplotlib.legend.Legend.get_frame
    nt = 24 # horas (1 dia)
    ng = 3 # numero de geradoras 
    
    p = np.zeros((nt,ng)) # matriz 24x3

    for i in range(nt):
        Pd = random.randint(450,1000)
        de = despacho('input2.csv', Pd)
        p[i,:] = de[0]
    

    print(p)

    tempo = [i for i in range(len(p))]

    ax = plt.subplot(111)

    for i in range(3):
        ger = f'G{i+1}'.format()
        plt.plot(tempo, p[:, i], label = ger)

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.8)
    plt.xlabel('Tempo (h)')
    plt.ylabel('Potência (MW)')
    plt.title('Demanda/Unid. Geradora')

    plt.show()









if __name__ == '__main__':
    import random
    path = 'input2.csv'
    visualize()

