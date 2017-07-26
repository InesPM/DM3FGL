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
	
	E1    = np.array([sqrt(100*300),sqrt(300*1000),sqrt(1000*3000),sqrt(3000*10000),sqrt(10000*100000)])	#TeV
	Emin1 = E1-np.array([100,300,1000,3000,10000,])
	Emax1 = np.array([300,1000,3000,10000,100000])-E1
	
	E1    = E1*1e-6	#TeV
	Emin1 = Emin1*1e-6
	Emax1 = Emax1*1e-6
	#evals= E
	#logarithmic mid-point of the band
	
	a,b=0,0

	nuFnu = []; flux = []; unc_fluxm = []; unc_fluxp = []; unc_num = []; unc_nup = []
	E = []; Emin = []; Emax = []

	while a<3034:	#3034 objects
		if name[a]==source:
			while b<len(F):
				if np.isnan(-t[a][Func[b]][0]) == False :		
					if np.isinf(-t[a][Func[b]][0]) == False :

						nuFnu.append(t[a][Fnu[b]])
						flux.append(t[a][F[b]])
						unc_fluxm.append(-t[a][Func[b]][0])
						unc_fluxp.append(t[a][Func[b]][1])
						unc_num.append(-(t[a][Func[b]][0])*(t[a][Fnu[b]])/(t[a][F[b]]))
						unc_nup.append((t[a][Func[b]][1])*(t[a][Fnu[b]])/(t[a][F[b]]))

						E.append(E1[b])
						Emin.append(Emin1[b])
						Emax.append(Emax1[b])

						
				b=b+1
		a=a+1

	nuFnu     = np.array(nuFnu)
	flux      = np.array(flux)
	unc_fluxm = np.array(unc_fluxm)
	unc_fluxp = np.array(unc_fluxp)
	unc_num   = np.array(unc_num)
	unc_nup   = np.array(unc_nup)
	
	E         = np.array(E)
	Emin      = np.array(Emin)
	Emax      = np.array(Emax)

	return (nuFnu,flux,unc_fluxm, unc_fluxp, unc_num, unc_nup, E, Emin, Emax)
	

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
Source = name[8]
print(Source)

##Flujo diferencial de energía 'nuFnu300_1000'##
nuFnu=t[:]['nuFnu300_1000']
nuFnu=np.array(nuFnu)
print(nuFnu)

(nuFnu,flux,unc_fluxm,unc_fluxp,unc_num,unc_nup,E,Emin,Emax) = nu(Source)

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
ax.errorbar(E, nuFnu, xerr=[Emin,Emax], yerr=[unc_num,unc_nup],fmt='--o',linewidth=1)


plt.show()
