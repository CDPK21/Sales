import pandas as pd
import matplotlib.pyplot as plt
import re
import datetime as dt
import math


# Cargar la base de datos
base = pd.read_csv("sales_data_sample.csv", encoding = "ISO-8859-1", engine='python')

# Primeras filas de la base de datos
print(base.head(1))

# Nombres de las columnas
nombres_columns = base.columns.tolist()
print(nombres_columns)

# Número de columnas
n_columns = len(nombres_columns)
print(n_columns)


# Estructura básica de la base de datos (Estadísticas básicas)
estructura = base.describe()
print(estructura)

# Imprimir toda una columna
print(estructura['SALES'])
# Imprimir toda una columna usando .loc
print(base.loc[:, 'SALES'])


## Ejemplo de como usar el ciclo For
#[0,1,2,3,4,5,6,7,8,9]
#for i in range(10):
#  print(i)


## Existen datos faltantes?
for i in range(n_columns):
    if(base.iloc[:, i].isnull().values.any()):
        print("Columna: ", nombres_columns[i], " ...contiene datos faltantes")


##  Ver boxplot de la variable SALES
plt.boxplot(base.loc[:, "SALES"])
plt.show()
#
##  Ver el hisograma de la variable SALES
plt.hist(base.loc[:, 'SALES'], 50)
plt.show()
#
##  Ver la densidad de la variable SALES
base.loc[:, 'SALES'].plot(kind="kde")
plt.show()

# Estructura básica por boxplot (Variable QUANTITY) por estratos (STATE)
# Eliminar los datos faltantes de la variable STATE
print("No filas con datos faltantes: ", len(base))
base2 = base[~base['STATE'].isnull()]
print("No filas sin datos faltantes: ", len(base2))
#
base_state = base2.loc[:, ['QUANTITYORDERED', 'STATE']]
var_state = base_state['STATE'].unique()
#
print(base_state.loc[base_state.loc[:, 'STATE'] == 'NY'])
print(var_state)
#
base_state.groupby('STATE').boxplot(column='QUANTITYORDERED')
plt.show()

# Ventas superiores a $12,000
print(base.loc[base.loc[:, 'SALES'] > 12001.0, ['SALES', 'PRICEEACH']])


# Quitar los 00:00 de la columna ORDERDATE
base['ORDERDATE'] = base['ORDERDATE'].replace("0:00", "", regex=True)
base['ORDERDATE'] = base['ORDERDATE'].replace("/", "-", regex=True)
base['ORDERDATE'] = base['ORDERDATE'].replace(" ", "", regex=True)

# Serie de tiempo de las ventas:
base_sales = base.loc[:, ['ORDERDATE', 'SALES']]


# Arreglar mes en la variable ORDERDATE
# Miniejemplo
temp = base_sales.iloc[0:1, 0:1].values[0][0]
print("Data", temp)
print("Tipo: ", type(temp))
temp_date = dt.datetime.strptime(temp, '%m-%d-%Y').date()
print("FECHA", temp_date)
print("Tipo: ", type(temp_date))
print("--------------------------------")

# Manera en que se arregla ORDERDATE
def date_var(variable):
    temp = dt.datetime.strptime(variable, '%m-%d-%Y').date()
    return temp
base_sales['ORDERDATE'] = base_sales['ORDERDATE'].apply(date_var)
# Base original
base['ORDERDATE'] = base['ORDERDATE'].apply(date_var)


### Comprobación
###temp_d = base_sales.iloc[0:1, 0:1].values[0][0]
###print("Data", temp_d)
###print("Tipo: ", type(temp_d))#


# Agruparlos por fecha
base_sales = base_sales.groupby(['ORDERDATE']).sum()
print("Después de agrupar, No filas: ", len(base_sales))
print(base_sales.head())
base_sales = base_sales.reset_index()
print("Después de reset_index, No filas: ", len(base_sales))
print(base_sales.head())

##
plt.plot(base_sales['ORDERDATE'], base_sales['SALES'])
plt.show()
##
q = 30
base_sales['SALES_MA'] = base_sales['SALES']
base_sales['SALES_MA'] = base_sales['SALES_MA'].rolling(window=q).mean()
print(base_sales.head(12))
base_sales = base_sales.iloc[(q-1):]


plt.plot(base_sales['ORDERDATE'], base_sales['SALES_MA'])
plt.show()






