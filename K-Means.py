import cv2
 
import numpy as np
 
import matplotlib.pyplot as plt
 
imga = cv2.imread('Clasif.jpg')
imgris = cv2.cvtColor(imga,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgris,127,255,0)
img, contornos, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
w, h = 10, 2
 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
 
veces = 0
Matrix = [[0 for x in range(h)] for y in range(w)]
for cnt in contornos[0:10]:
    area = float(cv2.contourArea(contornos[veces], 0))
    perimetro = float(cv2.arcLength(contornos[veces], 1))
    Matrix[veces][0] =  perimetro
    Matrix[veces][1] = area
    veces = veces + 1
    print("Area", int(area), "Perimetro", int(perimetro))
    cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img,[cnt],0,(255, 0, 0), -1)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print("El numero de figuras es:", len(contornos))
 
Z = np.float32(Matrix)
compactness,labels,center = cv2.kmeans(Z,2,None,criteria,10,flags)
 
print ("Labels", labels, "Centers:", center)
 
print("Cluster [", labels.astype(int)[0],"]", "Is tornillo with center", center.astype(float)[0][0], center.astype(float)[0][1])
print("Cluster [", labels.astype(int)[1],"]", "Is tuerca with center", center.astype(float)[1][0], center.astype(float)[1][1])
 
count = 10
for cnt in contornos[10:24]:
    d0 = math.sqrt((pow(cv2.arcLength(contornos[count], 1) - center.astype(float)[0][0], 2)) + (pow(cv2.contourArea(contornos[count],False) - center.astype(float)[0][1], 2)))
    d1 = math.sqrt((pow(cv2.arcLength(contornos[count], 1) - center.astype(float)[1][0], 2)) + (pow(cv2.contourArea(contornos[count],False) - center.astype(float)[1][1], 2)))
    count = count + 1;
    print("D0", int(d0), "D1", int(d1))
 
    if d1 < d0:
        print("Es un Tuerca")
        cv2.drawContours(img,[cnt],-1,(255,0,0),3)
    else:
        print("Es una Tornillo")
        cv2.drawContours(img, [cnt],0,255, -1)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
