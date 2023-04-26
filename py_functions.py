#------------------------------------------------------------------------------
# La funcion indices tiene como 
#                 OBJETIVO : Buscar los indices del elemento especificado 
#                            dado por  element en una lista ( lst ).
# Muy parecido a la instruccion find de matlab
# --------------------------------------------
# Por otro lado la instruccion index solo encuentra el primer indice que se 
# cruce con el elemento
#------------------------------------------------------------------------------
def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError: # Esto se ejecuta hasta que encuentra un error 
        # probablemente cuando se acaba el arreglo si no salta a append
            return result
        result.append(offset)  
#------------------------------------------------------------------------------
# Esta funcion filtra el resultado de la lista vartofilter teniendo en cuenta
# los indices optenido de main_lst buscando el elemento (element)
# devuelve unicamente aquellos elementos que conciden con los elementos de la
# lista main_lst que son iguales a element
#
# sel: es una expresion bolleana 0 o 1 que decide si deja solamente los objetos
#      parecidos a element o los quita de la lista
#      0 --> Para quitar los elementos de la lista
#      1 --> Para dejar unicamente esos elementos en la lista devuelta
def filterdata_vec(vartofilter,main_lst,element,sel) :
    # NOTE: vartofilter have the same size that main_lst
    IDindices = list(indices(main_lst,element))
    #print('Tamaño de vector: ' + str(N.shape(IDindices)))
    if sel == 1:
       newvar=[]
       for i in IDindices:
           newvar.append(vartofilter[i])
       return newvar
    if sel == 0: # Entra aqui si sel=0
       newvar=[]
       dimvec=len(main_lst)
       nvec=list(N.arange(dimvec))
       #print(nvec)
       IDindices.reverse()
       #print(IDindices)
       for i in IDindices:
           nvec.remove(i)
       #print(nvec)
       for j in nvec:
           newvar.append(vartofilter[j])
       return newvar
    else:
       newvar=[] 
       return newvar      
#---------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Remueve los duplicados de la variable list (que es una lista) 
def remove_duplicateList(List):
    s = []
    for i in List:
       if i not in s:
          s.append(i)
    return s
#------------------------------------------------------------------------------
# Agrega una lista vectadd a una principal llamada MAINvec
def add_List(MAINvec,vectoadd):
    dimvec=len(vectoadd)
    nvec=list(N.arange(dimvec))# Crea unvector con los indices de vectoadd
    for i in nvec:
        MAINvec.append(vectoadd[i])
    return MAINvec
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

