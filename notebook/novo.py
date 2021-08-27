from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

import pyomo.environ as pyo, numpy as np, pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory

# Entrada de dados:
#
## Geradores
arquivo = 'input4.xlsx'
ute = pd.read_excel(arquivo, sheet_name='UTE')
uhe = pd.read_excel(arquivo, sheet_name='UHE')
uel = pd.read_excel(arquivo, sheet_name='UEL')
aero = pd.read_excel(arquivo, sheet_name='aero')
demanda   = pd.read_excel(arquivo, sheet_name='demanda')

### termoelétricas:
cos = ute.Custo.values
a = ute.A.values*cos
b = ute.B.values*cos
c = ute.C.values*cos
pmin = ute.Pmin.values
pmax = ute.Pmax.values
ur = ute.UR.values
dr = ute.DR.values

### hidroelétricas: 
ah = uhe.A.values
bh = uhe.B.values
ch = uhe.C.values
phmin = uhe.Pmin
phmax = uhe.Pmax
vmax = uhe.Vmax

### eólicas
Nt = uel.Nt.values
Pt = uel.Pt.values
Wmax = uel.Wmax.values
disp = uel.d.values


## Demanda
#Pd = demanda.dem.values
#t = demanda.t.to_list()
Pd = [3000,3100]
t = [0,1]

Nd = len(Pd)
Ng = len(ute)
Nh = len(uhe)
Nw = len(uel)

model = pyo.ConcreteModel()

model.Pg = pyo.Var(range(Ng), range(Nd), bounds=(0,None))
Pg = model.Pg
model.Ph = pyo.Var(range(Nh), range(Nd), bounds=(0,None))
Ph = model.Ph
model.Pw = pyo.Var(range(Nw), range(Nd), bounds=(0,None))
Pw = model.Pw

model.deficit = pyo.Var(range(Nd), bounds=(0,None))
deficit = model.deficit


def gPdw(Wmax):
	L = np.arange(0, Wmax, 0.0001)
	tam = len(L)
	n = random.randint(0,tam)
	np.random.shuffle(L)
	return L[n]

def fg(Pg,a,b,c,t):
    return sum([a[g] + b[g]*Pg[g,t] + c[g]*Pg[g,t]*Pg[g,t] for g in range(Ng)])

def fh(Ph,ah,bh,ch,t):
    return sum([ah[h] + bh[h]*Ph[h,t] + ch[h]*Ph[h,t]*Ph[h,t] for h in range(Nh)])

def Li(Vr,Vi):
    return ((Vr-Vi)/Vi)

'''def Weibull(Pw,Wmax,Vr,Vi,c,k):
    rho = Pw/Wmax
    l = Li(Vr,Vi)
    fW = lambda Pw : ((k*l*Vi)/c)*((((1+rho*l)*Vi)/c)**(k-1))*exp(-(((1+rho*l)*Vi)/c)**k)
    return fW(Pw)'''

def Weibull(Pw,Wmax,Vr,Vi,c,k):
    rho = Pw/Wmax
    l = Li(Vr,Vi)
    return ((k*l*Vi)/c)*((((1+rho*l)*Vi)/c)**(k-1))*exp(-(((1+rho*l)*Vi)/c)**k)

def fw(Pw,disp,Wmax,t):
    Vi = 5
    Vr = 15
    Kp = 0.95
    Kr = 0.05
    c = 8.47
    k = 2.26

    Pdw = [gPdw(Wmax[w]) for w in range(Nw)]
    sum_Cl = sum([disp[w]*Pw[w,t] for w in range(Nw)])
    sum_Cp = sum([Kp*(Pw[w,t]-Pdw[w])*Weibull(Pw[w,t],Wmax[w],Vr,Vi,c,k) for w in range(Nw)])
    sum_Cr = sum([Kr*(Pw[w,t]-Pdw[w])*Weibull(Pw[w,t],Wmax[w],Vr,Vi,c,k) for w in range(Nw)])
    
    return sum_Cl+sum_Cp+sum_Cr


def fun_custo_total(Pg,a,b,c,Ph,ah,bh,ch,Pw,disp,Wmax,t):
    Fg = fg(Pg,a,b,c,t)
    Fh = fh(Ph,ah,bh,ch,t)
    Fw = fw(Pw,disp,Wmax,t)
    return Fg+Fh+Fw

# Objetivo
model.obj = pyo.Objective(expr= sum(fun_custo_total(Pg,a,b,c,Ph,ah,bh,ch,Pw,disp,Wmax,t) + deficit[t]*99999 for t in range(Nd)), sense=minimize)

# balanco de potencia:
model.balanco = ConstraintList()
for t in range(Nd):
    sum_Pg = sum(Pg[g,t] for g in range(Ng))
    sum_Ph = sum(Ph[h,t] for h in range(Nh))
    sum_Pw = sum(Pw[w,t] for w in range(Nw))
    model.balanco.add(expr= (sum_Pg + sum_Ph + sum_Pw + deficit[t]) == Pd[t])

# limites geração termo:
model.limger = pyo.ConstraintList()
for g in range(Ng):
    for t in range(Nd):
        model.limger.add(inequality(pmin[g], Pg[g,t], pmax[g]))

# limites geração hidreletricas:
model.limgerh = pyo.ConstraintList()
for h in range(Nh):
    for t in range(Nd):
        model.limgerh.add(inequality(phmin[h], Ph[h,t], phmax[h]))

# limite de geração eólica:
model.limgerw = pyo.ConstraintList()
for w in range(Nw):
    for t in range(Nd):
        model.limgerw.add(inequality(0, Pw[w,t], Wmax[w]))

# limites de rampas
model.ramp_up = pyo.ConstraintList()
for g in range(Ng):
    for t in range(Nd):
        if(t==0):
            Constraint.Skip
        else:
            model.ramp_up.add(expr= Pg[g,t] - Pg[g,t-1] <= ur[g])

model.ramp_down = pyo.ConstraintList()
for g in range(Ng):
    for t in range(Nd):
        if(t==0):
            Constraint.Skip
        else:
            model.ramp_down.add(expr= Pg[g,t-1] - Pg[g,t] <= dr[g])

# limites de armazenamento hidro:
model.limarm = pyo.ConstraintList()
for h in range(Nh):
    #Pht = sum([Ph[h,t] for t in range(Nd)])
    Pht = sum([ah[h] + bh[h]*Ph[h,t] + ch[h]*Ph[h,t]*Ph[h,t] for t in range(Nd)])
    model.limarm.add(expr= Pht <= vmax[h])

opt = SolverFactory('couenne', executable='C:\\couenne\\bin\\couenne.exe')
opt.solve(model)

model.pprint()

'''for t in range(Nd):
    print(f'Pd[{t}] = {pyo.value(Pd[t])} MW')
    for g in range(Ng):
        print(f' Pg[{g}] = {pyo.value(Pg[g,t])}')
    for h in range(Nh):
        print(f' Ph[{h}] = {pyo.value(Ph[h,t])}')
    for w in range(Nw):
        print(f' Pw[{w}] = {pyo.value(Pw[w,t])}')
    
    print(f'Pt[{t}] = {sum(pyo.value(Pg[:,t])) + sum(pyo.value(Ph[:,t])) + sum(pyo.value(Pw[:,t]))}')
    print(f'corte[{t}] = {pyo.value(deficit[t])}\n')'''

# 
f = open('novo.csv', 'w')

f.write("t,Pd,")
for g in range(Ng):
    f.write(f"G[{g+1}],")
for h in range(Nh):
    f.write(f"H[{h+1}],")
for w in range(Nw):
    f.write(f"W[{w+1}],")
f.write("Gerado,Corte"+"\n")


for t in range(Nd):
    f.write(str(t)+","+str(Pd[t])+",")
    for g in range(Ng):
        f.write(str(pyo.value(Pg[g,t]))+",")
    for h in range(Nh):
        f.write(str(pyo.value(Ph[h,t]))+",")
    for w in range(Nw):
        f.write(str(pyo.value(Pw[w,t]))+",")
    f.write(str(sum(pyo.value(Pg[:,t]))+sum(pyo.value(Ph[:,t]))+ sum(pyo.value(Pw[:,t])))+","+str(int(pyo.value(deficit[t])))+"\n")