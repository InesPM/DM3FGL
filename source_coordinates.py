#!/usr/bin/python
# -*- coding: latin-1 -*-
import numpy as np
from math import *
from scipy.optimize import curve_fit
import random as rd
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.cm as cm
import matplotlib as mpl
from astropy.table import Table

list=fits.open('3FGL.fit')
header=fits.getheader('3FGL.fit')
data=fits.getdata('3FGL.fit')
t=Table(data)


list.close()
#print(t)
#print(data[1])

#Coordenadas galácticas
GLON=t[:]['GLON']
GLAT=t[:]['GLAT']

#Coordenadas ecuatoriales
RA=t[:]['RAJ2000']
DE=t[:]['DEJ2000']

#Trazado de objetos en coordenadas galácticas
plt.figure(1)
plt.xlabel('Galactic Longitude')
plt.ylabel('Galactic Latitude')
plt.title('3FGL objects, galactic coordinates')
plt.plot(GLON,GLAT,'g+')

#Trazado de objetos en coordenadas ecuatoriales
plt.figure(2)
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title('3FGL objects, equatorial coordinates')
plt.plot(RA,DE,'b+')

plt.show()
