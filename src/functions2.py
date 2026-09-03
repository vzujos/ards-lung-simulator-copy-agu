# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:47:03 2023

@author: Nibaldo
"""

from meshio import read
from meshio import write_points_cells
import numpy as np
import matplotlib.pyplot as plt

def give_zinteres(xyz,ien,id_elemento,xinteres,yinteres ):
    ############################################################################################################
    ###Finalidad :
        #El objetivo de esta funcion es obtener una coordenada z, dado x e y conocidos, y dado un deternimado elemento triangular (malla superficial) .
        #En el fondo, tomamos las coordendas del elemento triangular, calculamos la ecuacion de su plano, y utilizando
        #esta ecuacion, podemos calcular la coordenada z en dicho plano dado x (xinteres) e y (yinteres) conocidos.
    
    ###Inputs: 
        #xyz: (array) arreglo de coordenadas de malla 
        #ien: (array) arreglo de conectividad entre elementos
        # id:_elemento: (int) id del elemento que se desea calcular su plano 
        #xinteres: (float) coordenada x a evaluar (no necesariamente dentro del elemento)
        #yinteres: (float) coordenada y a evaluar (no necesariamente dentro del elemento)
        
    ###Outputs:
       # zinteres: (float) valor escalar indicando la coordenada z que se obtiene del plano dado por los vertices
       # del triangulo y las coordenadas x e y de interes.
    ############################################################################################################
    coordse=xyz[ien[id_elemento]]
    x1=coordse[0,0]
    x2=coordse[1,0]
    x3=coordse[2,0]
    
    y1=coordse[0,1]
    y2=coordse[1,1]
    y3=coordse[2,1]
    
    z1=coordse[0,2]
    z2=coordse[1,2]
    z3=coordse[2,2]
    
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1) #ecuacion de un plano
    zinteres=(-d-a*xinteres-b*yinteres)/c   

    return zinteres



    
def point_in_element(xyz,ien,id_elemento,xinteres,yinteres ):
    ############################################################################################################
    ###Finalidad :
        #El objetivo de esta funcion es determinar si un punto x, y esta dentro de un triangulo cuyo id es id_elemento. Para esto, 
        # se calcula el z interes con lo que se tendra un cuarto nodo.
        # Se compara las areas que se forman entre el nodos y el vertices de interes con el area del elemento.
    ###Inputs: 
        #xyz: (array) arreglo de coordenadas de malla 
        #ien: (array) arreglo de conectividad entre elementos
        # id:_elemento: (int) id del elemento a evaluar 
        #xinteres: (float) coordenada x a evaluar (no necesariamente dentro del elemento)
        #yinteres: (float) coordenada y a evaluar (no necesariamente dentro del elemento)
        
    ###Outputs:
       # coordsinteres: (list) lista con coordenadas del punto de interes [xinteres,yinteres,zinteres]
       #(round(area,2)==round(areasuma,2)): (bool) True si esta el punto de interes esta dentro del elemento, False en otro caso
    ############################################################################################################
    coordse=xyz[ien[id_elemento]]
    x1=coordse[0,0]
    x2=coordse[1,0]
    x3=coordse[2,0]
    y1=coordse[0,1]
    y2=coordse[1,1]
    y3=coordse[2,1]
    z1=coordse[0,2]
    z2=coordse[1,2]
    z3=coordse[2,2]
    zinteres=  give_zinteres(xyz,ien,id_elemento,xinteres,yinteres ) 
    area=abs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2) #area del triangulo vertices 1,2,3 (area elemento triangular)
    area1=(xinteres*(y2-y3)+x2*(y3-yinteres)+x3*(yinteres-y2))/2 #area triangulo que se forma con vertices 2,3 e interes
    area2=(x1*(yinteres-y3)+xinteres*(y3-y1)+x3*(y1-yinteres))/2 #area triangulo que se forma con vertices 1,3 e interes
    area3=(x1*(y2-yinteres)+x2*(yinteres-y1)+xinteres*(y1-y2))/2 #area triangulo que se forma con vertices 1,2 e interes
    areasuma=abs(area1)+abs(area2)+abs(area3) #suma de areas
    coordsinteres=[xinteres,yinteres,zinteres]
    return coordsinteres,(round(area,2)==round(areasuma,2)) #para que el punto de interes este dentro del triangulo la suma de las areas debe ser igual al area del elemento. Se redondea para evitar errores numericos


def campo_escalar_in_point(xyz,ien,elem_interes,coordsinteres,campo):
    ############################################################################################################
    ###Finalidad :
        #El objetivo de esta funcion es obtener el valor de un campo en un punto determinado. Se interpola con funciones P1.
        
    ###Inputs: 
        #xyz: (array) arreglo de coordenadas de malla 
        #ien: (array) arreglo de conectividad entre elementos
        #elem_interes: (int) id del elemento donde pertenece el punto de interes, cuyas coordenadas son coordsinteres. Es necesario para obtener las xyz de los nodos .
        #coordsinteres : (list)  lista con coordenadas de punto de interes [xinteres,yinteres,zinteres]        
        #campo :(array) campo escalar con dimensiones igual al numero de nodos
        
    ###Outputs:
       # campointeres: (float) valor escalar del campo en el punto de interes
    ############################################################################################################
    coordse=xyz[ien[elem_interes]] #coordendadas del elemento
    x1=coordse[0,0] #coordenada x nodo 1
    x2=coordse[1,0]
    x3=coordse[2,0]
    y1=coordse[0,1]
    y2=coordse[1,1]
    y3=coordse[2,1]
    z1=coordse[0,2]
    z2=coordse[1,2]
    z3=coordse[2,2]
    xinteres,yinteres,zinteres=coordsinteres    
    #area=((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2)
    area=0.5*((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
    N1=(1/(2*area))*(( x2*y3-x3*y2)+(y2-y3)*xinteres+(x3-x2)*yinteres) #funcion de forma N1 
    N2=(1/(2*area))*(( x3*y1-x1*y3)+(y3-y1)*xinteres+(x1-x3)*yinteres)
    N3=(1/(2*area))*(( x1*y2-x2*y1)+(y1-y2)*xinteres+(x2-x1)*yinteres)
    #area2=0.5*((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)) #DH, 
    #print(area, area2) #son idems
    campoe=campo[ien[elem_interes]]
    campointeres=N1*campoe[0]+N2*campoe[1]+N3*campoe[2] #Interpolacion lineal usando funciones de forma P1

    return campointeres


"""
mainfolder=''
file_name='ne4_surface1_midExp_unormal000000.vtu'
nmuestrasy=10
nmuestrasx=60
"""
def give_field__through_ydirection(file_name,side, nmuestrasx,nmuestrasy,surface,normalize=False, campo="DispField"):
    
    ############################################################################################################
    ###Finalidad :
        #El objetivo de esta funcion es entregar valores representativos del campo de interes 
        #a lo largo de la direccion y (se entregan listas con nmuestrasy elementos con distinta informacion)
        #En particular, se genera un arreglo de posibles coordenadas y a recorrer. Para cada yinteres,
        # se recorren sus xinteres, y usando esos xinteres se calcula el valor del campo en esas coordenadas. 
        # se obtiene por ejemplo una lista con valores del campo medio (o un determinado percentil),
        # a lo largo del eje y
        # Funcion bien comentada, ver detalle linea a linea
        
    ###Inputs: 
        #file_name: (str) nombre del archivo a leer 
        #nmuestrasx: (int) numero de puntos a muestrar en la direccion x. estos x se muestrean dado un y interes.
        #nmuestrasy: (int) numero de puntos a muestrar en la direccion y
        #surface: (str)  tipo de superficie (diagragma, anterior, lateral)
        
    ###Outputs:
        #En resumen, los outputs entregan informacion a lo largo del eje y, por lo tanto siempre tienen dimension igual al nmuestrasy
       # xmax_elementos_posibles_mismacoord_all: (list) lista con la cooredenada x maxima (donde hay elementos)  a lo largo del eje y
       # xmin_elementos_posibles_mismacoord_all: (list) lista con la cooredenada x minima (donde hay elementos)  a lo largo del eje y
       # yinteres_all: (array) arreglo con las coordenadas muestreadas en y
       # campo_all: (list) lista con los valores de los campos en todos los puntos (nmuestrasy elementos, y cada elemento tiene nmuestrasx) 
       #lower_all: (list) lista con umbral inferior representatvio del campo en direccion y
       #middle_all: (list) lista con valor medio  representatvio del campo en direccion y
       #upper_all: (list) lista con umbral superior representatvio del campo en direccion y
    ############################################################################################################
    
    #Se leen xyz y se gira en caso de ser necesario. La funcion esta predefinida para el diafragma, donde los valores 
    # y mas negativos son la parte anterior (ventral) y los y mas positivos son los posteriores (dorsal). 
    # coordenada y va desde anterior a posterior.
    # Note que por ejemplo, para la superficie anterior, para poder reutilizar esta funcion hay que intercambiar 
    # las coordenadas y con las z. Al hacer esto, las z mas bajas (mas basales) se transforman en los y menores,
    # y las z mayores (apicales), se transforman en los y mayores. Luego, en el caso de la superficie anterior,
    # y recorren desde la base al apice (en la direccion y).
    mesh = read(file_name)
    raw_ien = mesh.cells_dict['triangle']
    raw_xyz = mesh.points
    
    # retrieve cellwise diaphragm side mask
    diaphr_mask = mesh.cell_data['DiaphragmSide'][0]
    smask = diaphr_mask == side
    # isolate cells
    scells = raw_ien[smask]
    # isolate points ids
    spoint_ids = np.unique(scells)
    # isolated diaphragm side point list
    xyz = raw_xyz[spoint_ids]

    # generate a new list
    translator = {}
    for e,pid in enumerate(spoint_ids):
        translator.update({pid:e})
    
    # clean list of cells for the diaphragm side
    ien = []
    for cell in scells:
        p1,p2,p3 = cell
        ien += [[translator[p1],translator[p2],translator[p3]]]
    ien = np.array(ien)
    
    if campo == "DispField":
        normaldisp = mesh.point_data[campo][:,2][spoint_ids] 
    elif campo == "EE-shape":
        data = mesh.points[:,2][spoint_ids]
        Z_correction = np.quantile(data,0.975)
        normaldisp = data - Z_correction
    elif campo == "EI-shape":
        data = mesh.points[:,2][spoint_ids]+ mesh.point_data["DispField"][:,2][spoint_ids] 
        Z_correction = np.quantile(data,0.975)
        normaldisp = data - Z_correction
        
    if normalize: normaldisp /= np.max(np.abs(normaldisp))
    
    #Para cada elemento, se calculan los valores maximos y minimos de sus coordenadas x e y. Es decir,
    # como un elemento 'e' tiene 3 nodos, tendra 3 coordenadas x y el maximo  de esas coordenadas x se almancena
    # en la posicion 'e' del arreglo xmax_elems.
    xmin_elems=(xyz[:,0][ien]).min(axis=1) # dimension =(nelem,0)
    xmax_elems=(xyz[:,0][ien]).max(axis=1) # dimension =(nelem,0)
    ymin_elems=(xyz[:,1][ien]).min(axis=1) # dimension =(nelem,0)
    ymax_elems=(xyz[:,1][ien]).max(axis=1) # dimension =(nelem,0)
    
    #Se calculan el yminimo e ymaximo de tota la malla
    ymin_total=xyz[:,1].min()  #float
    ymax_total=xyz[:,1].max()  #float
    campo_all=[] # este es una donde se almacena el valor del campo en todos los puntos muestreadados. 
    #Por lo tanto tiene nmuestrasy elementos. Cada uno de sus elemtos tiene a su vez nmuestrasx elementos. 
    upper_all=[] # aca se almacena valores superiores representativos de cada coordenada y interes. Tiene nmuestrasy elementos. Por ejemplo, su primer elemento podria ser el percentile elemento del primer elemento de campo_all (elemento que tiene nmuestrasx elementos)
    middle_all=[]
    lower_all=[]
    #Se genera un arreglo de cooredenadas y interes donde se determinaran el desplazamiento en dichas alturas (coordenedas y)
    #la dimenision de y interes es de nmuestrasy. Note que los puntos iniciales y finales no se consideran ya que muchas veces estos puntos tienen errores 
    # ya sea en el registro o en la definicion de la superficie.
    yinteres_all=np.linspace(ymin_total,ymax_total,nmuestrasy+2)[1:-1]
    xmax_elementos_posibles_mismacoord_all=[] # se almacenan los x maximo posibles en cada y. esto me sirve por ejemplo para trazar las lineas de cada ROI
    xmin_elementos_posibles_mismacoord_all=[]
    
    
    for yinteres in yinteres_all:   #se recorren todas las coordendas y 
        #para un determinado yinteres:
        campo_linea=[]
        elementos_posibles_mismacoord=np.where((yinteres<ymax_elems) & (yinteres>ymin_elems)) #elementos posibles  que contienen  la coordenada yinteres. Se dice posible pq podria ser que el punto este afuera, pero esto se chequea en la funcion point_in_element
        xmax_elementos_posibles_mismacoord=xyz[:,0][ien[elementos_posibles_mismacoord]].max() #de esos elementos que posiblemente  contienen la coordenada yinteres, calculo el xmaximo y xminimo
        xmin_elementos_posibles_mismacoord=xyz[:,0][ien[elementos_posibles_mismacoord]].min()
        xinteres_linea=np.linspace(xmin_elementos_posibles_mismacoord,xmax_elementos_posibles_mismacoord,nmuestrasx+2)[1:-1] #se genera un arreglo que va desde el  minimo posible hasta el xmaximo posible (de un determinado yinteres). agregamos dos ya que no consideamos ni el primer ni el ultimo punto
        
        for xinteres in xinteres_linea: # se recorren las coordenadas x, dado un determinado y interes
            elementos_posibles=np.where((xinteres<xmax_elems) & (xinteres>xmin_elems) & (yinteres<ymax_elems) & (yinteres>ymin_elems)) #elementos posibles que contienen a los puntos xinteres, yinteres. Se dice posible pq podria ser que el punto este afuera, pero esto se chequea en la funcion point_in_element
            # se recorren los elementos posibles (en general son dos), y de esos solo uno cumple con res==True. Para el elemento que cumple esto se determina el campo de interes ui
            for i in np.arange(len(elementos_posibles[0])):
                elem=elementos_posibles[0][i]
                coordsinteres_try,res=point_in_element(xyz,ien,elem,xinteres,yinteres)
                if res==True:
                    elem_interes=elem
                    coordsinteres=coordsinteres_try
                    ui=campo_escalar_in_point(xyz,ien,elem_interes,coordsinteres,normaldisp)
                    campo_linea.append(ui)
    
        lower_all.append(np.percentile(campo_linea,25))
        middle_all.append(np.percentile(campo_linea,50))
        upper_all.append(np.percentile(campo_linea,75))
        campo_all.append(campo_linea)
        xmax_elementos_posibles_mismacoord_all.append(xmax_elementos_posibles_mismacoord)
        xmin_elementos_posibles_mismacoord_all.append(xmin_elementos_posibles_mismacoord)

    return xmax_elementos_posibles_mismacoord_all,xmin_elementos_posibles_mismacoord_all,yinteres_all,campo_all,lower_all,middle_all,upper_all
    


def give_field__through_ydirection_final(file_name, nmuestrasx,nmuestrasy,surface,normalize=False, campo="DispField"):
    
    ############################################################################################################
    ###Finalidad :
        #El objetivo de esta funcion es entregar valores representativos del campo de interes 
        #a lo largo de la direccion y (se entregan listas con nmuestrasy elementos con distinta informacion)
        #En particular, se genera un arreglo de posibles coordenadas y a recorrer. Para cada yinteres,
        # se recorren sus xinteres, y usando esos xinteres se calcula el valor del campo en esas coordenadas. 
        # se obtiene por ejemplo una lista con valores del campo medio (o un determinado percentil),
        # a lo largo del eje y
        # Funcion bien comentada, ver detalle linea a linea
        
    ###Inputs: 
        #file_name: (str) nombre del archivo a leer 
        #nmuestrasx: (int) numero de puntos a muestrar en la direccion x. estos x se muestrean dado un y interes.
        #nmuestrasy: (int) numero de puntos a muestrar en la direccion y
        #surface: (str)  tipo de superficie (diagragma, anterior, lateral)
        
    ###Outputs:
        #En resumen, los outputs entregan informacion a lo largo del eje y, por lo tanto siempre tienen dimension igual al nmuestrasy
       # xmax_elementos_posibles_mismacoord_all: (list) lista con la cooredenada x maxima (donde hay elementos)  a lo largo del eje y
       # xmin_elementos_posibles_mismacoord_all: (list) lista con la cooredenada x minima (donde hay elementos)  a lo largo del eje y
       # yinteres_all: (array) arreglo con las coordenadas muestreadas en y
       # campo_all: (list) lista con los valores de los campos en todos los puntos (nmuestrasy elementos, y cada elemento tiene nmuestrasx) 
       #lower_all: (list) lista con umbral inferior representatvio del campo en direccion y
       #middle_all: (list) lista con valor medio  representatvio del campo en direccion y
       #upper_all: (list) lista con umbral superior representatvio del campo en direccion y
    ############################################################################################################
    
    #Se leen xyz y se gira en caso de ser necesario. La funcion esta predefinida para el diafragma, donde los valores 
    # y mas negativos son la parte anterior (ventral) y los y mas positivos son los posteriores (dorsal). 
    # coordenada y va desde anterior a posterior.
    # Note que por ejemplo, para la superficie anterior, para poder reutilizar esta funcion hay que intercambiar 
    # las coordenadas y con las z. Al hacer esto, las z mas bajas (mas basales) se transforman en los y menores,
    # y las z mayores (apicales), se transforman en los y mayores. Luego, en el caso de la superficie anterior,
    # y recorren desde la base al apice (en la direccion y).
    mesh = read(file_name)
    raw_ien = mesh.cells_dict['triangle']
    raw_xyz = mesh.points
    
    # retrieve cellwise diaphragm side mask
    diaphr_mask = mesh.cell_data['Diaphragm'][0]
    smask = diaphr_mask == 1
    # isolate cells
    scells = raw_ien[smask]
    # isolate points ids
    spoint_ids = np.unique(scells)
    # isolated diaphragm side point list
    xyz = raw_xyz[spoint_ids]

    # generate a new list
    translator = {}
    for e,pid in enumerate(spoint_ids):
        translator.update({pid:e})
    
    # clean list of cells for the diaphragm side
    ien = []
    for cell in scells:
        p1,p2,p3 = cell
        ien += [[translator[p1],translator[p2],translator[p3]]]
    ien = np.array(ien)
    
    if campo == "DispField" or campo =='u':
        normaldisp = mesh.point_data[campo][:,2][spoint_ids] 
    elif campo == "EE-shape":
        data = mesh.points[:,2][spoint_ids]
        Z_correction = np.quantile(data,0.975)
        normaldisp = data - Z_correction
    elif campo == "EI-shape":
        data = mesh.points[:,2][spoint_ids]+ mesh.point_data["DispField"][:,2][spoint_ids] 
        Z_correction = np.quantile(data,0.975)
        normaldisp = data - Z_correction
        
    if normalize: normaldisp /= np.max(np.abs(normaldisp))
    
    #Para cada elemento, se calculan los valores maximos y minimos de sus coordenadas x e y. Es decir,
    # como un elemento 'e' tiene 3 nodos, tendra 3 coordenadas x y el maximo  de esas coordenadas x se almancena
    # en la posicion 'e' del arreglo xmax_elems.
    xmin_elems=(xyz[:,0][ien]).min(axis=1) # dimension =(nelem,0)
    xmax_elems=(xyz[:,0][ien]).max(axis=1) # dimension =(nelem,0)
    ymin_elems=(xyz[:,1][ien]).min(axis=1) # dimension =(nelem,0)
    ymax_elems=(xyz[:,1][ien]).max(axis=1) # dimension =(nelem,0)
    
    #Se calculan el yminimo e ymaximo de tota la malla
    ymin_total=xyz[:,1].min()  #float
    ymax_total=xyz[:,1].max()  #float
    campo_all=[] # este es una donde se almacena el valor del campo en todos los puntos muestreadados. 
    #Por lo tanto tiene nmuestrasy elementos. Cada uno de sus elemtos tiene a su vez nmuestrasx elementos. 
    upper_all=[] # aca se almacena valores superiores representativos de cada coordenada y interes. Tiene nmuestrasy elementos. Por ejemplo, su primer elemento podria ser el percentile elemento del primer elemento de campo_all (elemento que tiene nmuestrasx elementos)
    middle_all=[]
    lower_all=[]
    #Se genera un arreglo de cooredenadas y interes donde se determinaran el desplazamiento en dichas alturas (coordenedas y)
    #la dimenision de y interes es de nmuestrasy. Note que los puntos iniciales y finales no se consideran ya que muchas veces estos puntos tienen errores 
    # ya sea en el registro o en la definicion de la superficie.
    yinteres_all=np.linspace(ymin_total,ymax_total,nmuestrasy+2)[1:-1]
    xmax_elementos_posibles_mismacoord_all=[] # se almacenan los x maximo posibles en cada y. esto me sirve por ejemplo para trazar las lineas de cada ROI
    xmin_elementos_posibles_mismacoord_all=[]
    
    
    for yinteres in yinteres_all:   #se recorren todas las coordendas y 
        #para un determinado yinteres:
        campo_linea=[]
        elementos_posibles_mismacoord=np.where((yinteres<ymax_elems) & (yinteres>ymin_elems)) #elementos posibles  que contienen  la coordenada yinteres. Se dice posible pq podria ser que el punto este afuera, pero esto se chequea en la funcion point_in_element
        xmax_elementos_posibles_mismacoord=xyz[:,0][ien[elementos_posibles_mismacoord]].max() #de esos elementos que posiblemente  contienen la coordenada yinteres, calculo el xmaximo y xminimo
        xmin_elementos_posibles_mismacoord=xyz[:,0][ien[elementos_posibles_mismacoord]].min()
        xinteres_linea=np.linspace(xmin_elementos_posibles_mismacoord,xmax_elementos_posibles_mismacoord,nmuestrasx+2)[1:-1] #se genera un arreglo que va desde el  minimo posible hasta el xmaximo posible (de un determinado yinteres). agregamos dos ya que no consideamos ni el primer ni el ultimo punto
        
        for xinteres in xinteres_linea: # se recorren las coordenadas x, dado un determinado y interes
            elementos_posibles=np.where((xinteres<xmax_elems) & (xinteres>xmin_elems) & (yinteres<ymax_elems) & (yinteres>ymin_elems)) #elementos posibles que contienen a los puntos xinteres, yinteres. Se dice posible pq podria ser que el punto este afuera, pero esto se chequea en la funcion point_in_element
            # se recorren los elementos posibles (en general son dos), y de esos solo uno cumple con res==True. Para el elemento que cumple esto se determina el campo de interes ui
            for i in np.arange(len(elementos_posibles[0])):
                elem=elementos_posibles[0][i]
                coordsinteres_try,res=point_in_element(xyz,ien,elem,xinteres,yinteres)
                if res==True:
                    elem_interes=elem
                    coordsinteres=coordsinteres_try
                    ui=campo_escalar_in_point(xyz,ien,elem_interes,coordsinteres,normaldisp)
                    campo_linea.append(ui)
    
        lower_all.append(np.percentile(campo_linea,25))
        middle_all.append(np.percentile(campo_linea,50))
        upper_all.append(np.percentile(campo_linea,75))
        campo_all.append(campo_linea)
        xmax_elementos_posibles_mismacoord_all.append(xmax_elementos_posibles_mismacoord)
        xmin_elementos_posibles_mismacoord_all.append(xmin_elementos_posibles_mismacoord)

    return xmax_elementos_posibles_mismacoord_all,xmin_elementos_posibles_mismacoord_all,yinteres_all,campo_all,lower_all,middle_all,upper_all
    

   