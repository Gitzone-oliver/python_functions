# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:19:52 2018

@author: OLIVER
"""
# 000000000000000000  PAQUETES 000000000000000000000000000000000000000
import numpy as np
#import numpy
import pandas as pd
import math
from sklearn.feature_selection import f_regression, mutual_info_regression, mutual_info_classif

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Funciones
# Calculates the entropy of the given data set for the target attribute.
# Data: Matriz de datos 
# target_attr: Atributo de valores
def entropy(data, target_attr):
    val_freq = []
    data_entropy = 0.0
    if len(target_attr) is not 0:
       list_attr=data[target_attr]
    else:
       list_attr=data
    
    # Calculate the frequency of each of the values in the target attr
    vals_attr=np.unique(list_attr, return_counts=True)
    val_freq=vals_attr[1] #[0] Son los valores [1] Son las frecuencias  
    # Calculate the entropy of the data for the target attribute
    for freq in val_freq:
        data_entropy += (-freq/len(list_attr)) * math.log(freq/len(list_attr), 2) 
        
    return data_entropy

#------------------------------------------------------------------------------
# Calculates the information gain (reduction in entropy) that would result by 
# splitting the data on the chosen attribute (attr).
def gain(data, target_attr, attr):
    #Target_attr= Vector de etiquetas
    #attr       = Nombre de atributo divido por la etiquetas de clasificacion 
    subset_entropy =[]
    list_attr=data[attr] # Vector de las atributos
    
    # Calculate the frequency of each of the values in the target attr
    vals_attr = np.unique(list_attr, return_counts=True)  
    #Calculate the frequency of each of the values in the target attr
    #vals_class = np.unique(target_attr, return_counts=True)
    
    # Calculate the sum of the entropy for each subset of records weighted by their probability 
    # of occuring in the training set.
    for val in range(0,len(vals_attr[0]),1): #[0] Son los valores [1] Son las frecuencias
        #print('------------------------------------------------')
        #print('Numero de valores en atributo',len(vals_attr[0]))
        #print('Indice del valor unico:',val)
        val_prob = vals_attr[1][val] / sum(vals_attr[1]) # Es la probabilidad del valor sobre todo el conjunto de instancias
        #print('Lista de valores atributo:', len(list_attr))
        #print('Valor especifico attr:', vals_attr[0][val])
        data_subset = target_attr[list_attr == vals_attr[0][val]] # Vector de las etiquetas
        
        #print('Subconjunto de etiquetas para el valor pi',data_subset)
        #print(val_prob)
        subset_entropy.append(val_prob * entropy(data_subset,[]))
        #print('%3.5f',subset_entropy)
 
    # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
    return (entropy(target_attr,[]) - sum(subset_entropy))

#------------------------------------------------------------------------------
golf_file = "Golf.csv"

# Open the file for reading and read in data
golf_file_handler = open(golf_file, "r")
golf_data = pd.read_csv(golf_file_handler, sep=",")
golf_file_handler.close()
golf_data=pd.DataFrame(golf_data)
# Imprime la base de datos en crudo
print(golf_data.head())
# Etiqueta las columnas por el tipo de variable
col_name=golf_data.columns
col_type=golf_data.iloc[0,:]
for i in range(0,len(golf_data.columns),1):
    golf_data[col_name[i]] = golf_data[col_name[i]].astype(col_type[i])
# Eliminar el vector extra que indica el tipo de variable
golf_data=golf_data.drop(0)
# Eliminar columnas con datos numericos
golf_data=golf_data.drop(['Temperature','Humidity'],axis=1)
# Convertir variables de categoricas a numericas
maps=[['overcast', 'rainy', 'sunny'],['cool','mild','hot'],['normal','high'],['FALSE','TRUE'],['yes','no']]
print('\n')
print(golf_data.dtypes)
print(golf_data.head(10))
j=0
for i in golf_data.columns:
    golf_data[i]=golf_data[i].factorize(maps[j])[0]
    print(maps[j])
    j=j+1
    
print(golf_data.head(10))


# -----------------------------------------------------------------------------

# LA PARTICION PARA EL TESTING
# Separar el vector de etiqueta del dataset
y=golf_data['Play'] # Etiqueta de clasificacion
#pd.factorize(tar_names)
df=golf_data.drop(['Play'],1) # El indice   0: Es la fila,   1 : columna
#print(y)
#print(df.columns)
entropia=[]
infogain=[]
for n_attrib in df.columns:
    entropia.append(entropy(df, n_attrib))
    infogain.append(gain(df, y, n_attrib))

print('------------------------   Valores de entropia   --------------------------------')
print(entropia)
print('\n')
print('------------------------   Valores de Info Gain   --------------------------------')
print(infogain)
print('\n')
print('------------------------   Nombres variables   --------------------------------')
print(df.columns)
print('\n\n')

#f_test, _ = f_regression(df, y)
#f_test /= np.max(f_test)

mi = mutual_info_classif(df, y, discrete_features='auto', n_neighbors=7)
print('------------------------   Informacion mutua (valores continuos)   --------------------------------')
print('# de valores: ', len(mi))
mi /= np.max(mi)
print('Valor normalizado:', mi)

# ------------------    Graficas   ----------------    Graficas   -------------
import matplotlib.pyplot as plt
 
# create plot
fig, ax = plt.subplots()
index = np.arange(len(df.columns))
bar_width = 0.4
opacity = 0.6
 
rects1 = plt.bar(index, entropia, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Entropia')
 
rects2 = plt.bar(index + bar_width, infogain, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Info gain')

rects3 = plt.bar(index + bar_width, mi, bar_width,
                 alpha=opacity,
                 color='k',
                 label='Info mutua')
 
plt.xlabel('Rasgos')
plt.ylabel('Scores')
plt.title('Scores por rasgo')
plt.xticks(index + bar_width, df.columns, rotation=90)
plt.legend()

plt.tight_layout()
plt.show()
