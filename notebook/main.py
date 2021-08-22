from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyomo.environ as pyo, numpy as np, pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory


geradores = pd.read_excel('input.xlsx', sheet_name='geradores')
demanda   = pd.read_excel('input.xlsx', sheet_name='demanda')

# Entrada de dados:
#
# Geradores
cos = geradores.Custo.values
a = geradores.A.values*cos
b = geradores.B.values*cos
c = geradores.C.values*cos
pmin = geradores.Pmin.values
pmax = geradores.Pmax.values
ur = geradores.UR.values
dr = geradores.DR.values
#cos = geradores.Custo.to_list()
#pmin = geradores.Pmin.to_list()
#pmax = geradores.Pmax.to_list()
#ur = geradores.UR.to_list()
#dr = geradores.DR.to_list()

# Demanda
Pd = demanda.dem.values
t = demanda.t.to_list()

Ng = len(geradores)
Nd = len(Pd)

# Modelagem
model = pyo.ConcreteModel()

model.Pg = pyo.Var(range(Ng), range(Nd), bounds=(0,None))
Pg = model.Pg

# Objetivo
model.obj = pyo.Objective(expr= sum([a[g] + b[g]*Pg[g,t] + c[g]*Pg[g,t]*Pg[g,t] for g in range(Ng) for t in range(Nd)]), sense=minimize)

# Balanco de potência:
model.balanco = ConstraintList()
for t in range(Nd):
    sum_Pg = sum(Pg[g,t] for g in range(Ng))
    model.balanco.add(expr= sum_Pg == Pd[t])

# Limites geração:
model.limger = pyo.ConstraintList()
for g in range(Ng):
    for t in range(Nd):
        model.limger.add(inequality(pmin[g], Pg[g,t], pmax[g]))

opt = SolverFactory('couenne', executable='C:\\couenne\\bin\\couenne.exe')
#opt = SolverFactory('glpk')
opt.solve(model)

model.pprint()

for g in range(Ng):
    for t in range(Nd):
        print(f"Pg[{g},{t}] = {pyo.value(Pg[g,t])} / Pd[{t} == {pyo.value(Pd[t])}]")