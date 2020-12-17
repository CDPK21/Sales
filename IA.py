import pandas as pd
import datetime as dt

from scipy.stats import zscore

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense

from sklearn import metrics

# Cargar la base de datos
base = pd.read_csv("sales_data_sample.csv", encoding = "ISO-8859-1", engine='python')

# Nombres de las columnas:
nombres_columns = base.columns.tolist()
print(nombres_columns)


# Quitar los 00:00 de la columna ORDERDATE (Limpiar los valores de la variable de la FECHA)
base['ORDERDATE'] = base['ORDERDATE'].replace("0:00", "", regex=True)
base['ORDERDATE'] = base['ORDERDATE'].replace("/", "-", regex=True)
base['ORDERDATE'] = base['ORDERDATE'].replace(" ", "", regex=True)

# Manera en que se arregla ORDERDATE
def date_var(variable):
    temp = dt.datetime.strptime(variable, '%m-%d-%Y').date()
    return temp

# Reordenamos los valores de la variable y la transformamos en un objeto tipo date (FECHA)
base['ORDERDATE'] = base['ORDERDATE'].apply(date_var)
# Creamos la columna tipo date
base['ORDERDATE'] = pd.to_datetime(base['ORDERDATE'])
# Creamos una columna con los días de la semana de la venta
base['dia_sem'] = base['ORDERDATE'].dt.dayofweek
#print(base['day_week'].value_counts())

# Extreamos la variables a utilizar en el modelo en una nueva base de datos.
new_base = base.loc[:, ['dia_sem', 'MONTH_ID', 'PRODUCTLINE', 'CITY', 'QUANTITYORDERED']]

# Transformamos nuestras variables categoricas independientes
dia_sem = pd.get_dummies(new_base['dia_sem'])
mes = pd.get_dummies(new_base['MONTH_ID'])
product = pd.get_dummies(new_base['PRODUCTLINE'])
city = pd.get_dummies(new_base['CITY'])

# Juntamos en una sola base (variable) nuestras variables transformadas
mod_base = pd.concat([dia_sem, mes, product, city], axis=1)

# Transformamos nuestra variable dependiente
respuesta = zscore(new_base['QUANTITYORDERED'])

x = mod_base.values
y = respuesta

# Dividimos nuestra base de datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creación del modelo
model = Sequential()
model.add(Dense(15, input_dim=x.shape[1], activation='softmax'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# "Compilación" del modelo
model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=2, epochs=600)

# Predicción
pred = model.predict(x_test)

# Evaluación del modelo
score = metrics.mean_squared_error(pred, y_test)
print("Log loss score, Error: {}".format(score))

