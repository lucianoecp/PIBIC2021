import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def resolve_de(Pd,beta,gamma):
    print('\nDespacho econômico...\n')
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

def com_limites(Pd, beta, gama, pmin, pmax):
    print('\nConsiderando os limites operacionais...\n')
    n   = len(beta)
    bet = np.array(beta)
    gam = np.array(gama)
    carga = Pd

    pg = []

    for i in range(n-1):
        de = resolve_de(carga, bet, gam)
        pg[i:] = de[:-1]

        if(pg[i]>=pmin[i] and pg[i]<=pmax[i]):
            print(f'G{1+1}: Ok!\n'.format())
        
        else:
            if(pg[i]<pmin[i]):
                print(f'G{i+1}: min.'.format())
                pg[i] = pmin[i]
            else:
                print(f'G{i+1}: max'.format())
                pg[i] = pmax[i]

            carga = carga - pg[i]

            bet = np.delete(bet, i)
            gam = np.delete(gam, i)   

        print('DE: ',de)
        print('PG: ',pg)

    return [pg, de[-1]]

def despacho(path,Pd):

    data = pd.read_csv(path)
    data.head()

    print(f'Problema:\n{data}\nDemanda: {Pd} (MW)')

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

    print(com_limites(Pd, b, c, pmin, pmax))
        
if __name__ == '__main__':

    Pd = 850
    despacho('input2.csv',Pd)