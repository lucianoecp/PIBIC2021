import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('novo.csv')

# Arredondando para 2 casas decimais
data.describe().round(4)
print(data)

data.plot(x="t", y=["G[1]","G[2]","G[3]","G[4]","G[5]","G[6]","G[7]","G[8]","G[9]","G[10]","H[1]","H[2]","W[1]","W[2]","W[3]","W[4]","W[5]"], kind="line")
plt.show()
data.plot(x="t", y=["W[1]","W[2]","W[3]","W[4]","W[5]"], kind="line")
plt.show()

'''arquivo = 'input4.xlsx'
ute = pd.read_excel(arquivo, sheet_name='UTE')
print(sum(ute.Pmin.values))'''