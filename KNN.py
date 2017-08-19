#knn (Miguel Angel Sanchez Bravo)
 
import cv2
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
 
Clases = []
Clases.append('Tornillo')
Clases.append('Tuerca')
Clases.append('Estrella')
imagenes = []
imagenes.append('Clase1.jpg')
imagenes.append('Clase2.jpg')
imagenes.append('Clase3.jpg')
 
veces = 0
cont = 0
entrenamiento = [[0 for x in range(2)] for y in range(15)]
response_data =  [[0 for x in range(1)] for y in range(15)]
#Fase de entrenamiento
for i in imagenes:
    imagen = cv2.imread(i);
    img = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,127,255,0)
    contornos, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cont += 1
    for cnt in contornos[0:3]:
        area = float(cv2.contourArea(contornos[veces], 0))
        perimetro = float(cv2.arcLength(contornos[veces], 1))
        print("Area", int(area), "Perimetro", int(perimetro))
        entrenamiento[veces][0] =  perimetro
        entrenamiento[veces][1] = area
        response_data[veces][0] = cont
        veces = veces + 1
        cv2.drawContours(imagen,[cnt],0,(128, 255, 0), 3)
        cv2.imshow('img',imagen)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
print("El numero de figuras es:", len(contornos))
matrix_data = np.matrix(entrenamiento).astype('float32')
responses_data = np.matrix(response_data).astype('float32')
model = cv2.KNearest()
model.train(matrix_data, responses_data)
 
#Fase de entrenamiento
imgen = cv2.imread('Clasif.jpg')
imgris = cv2.cvtColor(imgen,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgris,127,255,0)
contornos, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
veces = 0
Matrix = [[0 for x in range(2)] for y in range(2)]
Matrix2= [[0 for x in range(1)] for y in range(33)]
for cnt in contornos:
    area = float(cv2.contourArea(contornos[veces], 0))
    perimetro = float(cv2.arcLength(contornos[veces], 1))
    Matrix[0][0] =  perimetro
    Matrix[0][1] = area
    matrix_data = np.matrix(Matrix).astype('float32')
    retval, results, neigh_resp, dists = model.find_nearest(matrix_data, k = 1)
    [x, y, w, h] = cv2.boundingRect(cnt) # Coordenadas para poner directo el texto
    cv2.drawContours(img,[cnt],0,(128, 255, 0), 3)
    print ("La pieza resaltada es", Clases[int(results[0][0])-1])
    cv2.putText(img, Clases[int(results[0][0])-1], (x, y+h), 0, 1, (128, 255, 255))
    veces = veces + 1
    cv2.imshow('img',img)
    cv2.waitKey(0)
