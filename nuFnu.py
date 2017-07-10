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
import pylab as pl

def nu(source):	#Source=Source_Name, t=complete catalog matrix
	Fnu=['nuFnu100_300','nuFnu300_1000','nuFnu1000_3000','nuFnu3000_10000','nuFnu10000_100000']
					#Spectral energy distribution (MeV)
	F=['Flux100_300','Flux300_1000','Flux1000_3000','Flux3000_10000','Flux10000_100000']
	Func=['Unc_Flux100_300','Unc_Flux300_1000','Unc_Flux1000_3000','Unc_Flux3000_10000','Unc_Flux10000_100000']
	a,b=0,0
	nuFnu,flux,unc_fluxm,unc_fluxp,unc_num,unc_nup=[],[],[],[],[],[]
	while a<3034:	#3034 objects
		if name[a]==source:
			while b<len(F):
				nuFnu.append(t[a][Fnu[b]])
				flux.append(t[a][F[b]])
				unc_fluxm.append(-t[a][Func[b]][0])
				unc_fluxp.append(t[a][Func[b]][1])
				unc_num.append(-(t[a][Func[b]][0])*(t[a][Fnu[b]])/(t[a][F[b]]))
				unc_nup.append((t[a][Func[b]][1])*(t[a][Fnu[b]])/(t[a][F[b]]))
				#unc_nu=unc_flux*nuFnu/flux
				b=b+1
		a=a+1
	return (nuFnu,flux,unc_fluxm, unc_fluxp, unc_num, unc_nup)
	
	

##Apertura y cierre del catálogo##
list=fits.open('3FGL.fit')
header=fits.getheader('3FGL.fit')
data=fits.getdata('3FGL.fit')
t=Table(data)

list.close()
#print(t)
#print(data[1])

##Nombre de la fuente##
name=t[:]['Source_Name']
name=np.array(name)
Source = name[300]
print(Source)

##Flujo diferencial de energía 'nuFnu300_1000'##
nuFnu=t[:]['nuFnu300_1000']
nuFnu=np.array(nuFnu)
print(nuFnu)

(nuFnu,flux,unc_fluxm,unc_fluxp,unc_num,unc_nup) = nu(Source)
#print('\nflux=',flux,'\nunc_flux=',unc_flux)
E=np.array([sqrt(100*300),sqrt(300*1000),sqrt(1000*3000),sqrt(3000*10000),sqrt(10000*100000)])
#Logarithmic mid-point slope

#print('flujo diferencial a diferentes energías:',nu(Source))

fig=pl.figure()

ax=fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_xlim(2e-4, 60)
#ax.set_ylim(5e-22,1e-12)
ax.set_title(Source)
ax.set_xlabel('$E$ [TeV]')
ax.set_ylabel('$E^2 dN/dE$ [erg cm$^{-2}$ s$^{-1}$]')

print('cosa',unc_fluxm)
#ax.plot(E,flux,'g+',E,unc_flux,'cx')
ax.errorbar(E, nuFnu, yerr=[unc_num,unc_nup],fmt='--o')
#ax.plot(E, nuFnu, 'g+')


plt.show()