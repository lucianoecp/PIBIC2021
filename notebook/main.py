from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyomo.environ as pyo, numpy as np, pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory


# Entrada de dados:
#
## Geradores
geradores = pd.read_excel('input2.xlsx', sheet_name='geradores')
demanda   = pd.read_excel('input2.xlsx', sheet_name='demanda')

cos = geradores.Custo.values
a = geradores.A.values*cos
b = geradores.B.values*cos
c = geradores.C.values*cos
pmin = geradores.Pmin.values
pmax = geradores.Pmax.values
ur = geradores.UR.values
dr = geradores.DR.values

## Demanda
Pd = demanda.dem.values
#Pd = [5550]
t = demanda.t.to_list()

Ng = len(geradores)
Nd = len(Pd)

model = pyo.ConcreteModel()

model.Pg = pyo.Var(range(Ng), range(Nd), bounds=(0,None))
Pg = model.Pg
model.deficit = pyo.Var(range(Nd), bounds=(0,None))
deficit = model.deficit

def CustoUTEs(Pg,a,b,c,t):
    return sum([a[g] + b[g]*Pg[g,t] + c[g]*Pg[g,t]*Pg[g,t] for g in range(Ng)])

# Objetivo
model.obj = pyo.Objective(expr= sum(CustoUTEs(Pg,a,b,c,t) + deficit[t]*99999 for t in range(Nd)), sense=minimize)

# balanco:
model.balanco = ConstraintList()
for t in range(Nd):
    sum_Pg = sum(Pg[g,t] for g in range(Ng))
    model.balanco.add(expr= (sum_Pg + deficit[t]) == Pd[t])

# limites geração:
model.limger = pyo.ConstraintList()
for g in range(Ng):
    for t in range(Nd):
        model.limger.add(inequality(pmin[g], Pg[g,t], pmax[g]))

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

opt = SolverFactory('couenne', executable='C:\\couenne\\bin\\couenne.exe')
opt.solve(model)

model.pprint()

for t in range(Nd):
    print(f'Pd[{t}] = {pyo.value(Pd[t])} MW')
    for g in range(Ng):
        print(f' Pg[{g}] = {pyo.value(Pg[g,t])}')
    
    print(f'Pt[{t}] = {sum(pyo.value(Pg[:,t]))}')
    print(f'corte[{t}] = {pyo.value(deficit[t])}\n')

# 
f = open('gen_result.csv', 'w')

f.write("t,Pd,")
for g in range(Ng):
    f.write(f"G[{g+1}],")
f.write("Gerado,Corte"+"\n")

for t in range(Nd):
    f.write(str(t)+","+str(Pd[t])+",")
    for g in range(Ng):
        f.write(str(pyo.value(Pg[g,t]))+",")
    f.write(str(sum(pyo.value(Pg[:,t])))+","+str(pyo.value(deficit[t]))+"\n")