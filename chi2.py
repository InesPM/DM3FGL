import numpy as np
from math import *
from astropy.io import fits
from astropy.table import Table, Column
import pylab as pl
import scipy as sp
import bisect
import scipy.optimize as opt
from scipy.stats import chi2
from matplotlib import pyplot as plt
import datetime as dt

#################
### Functions ###
#################

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


############################
### Opening the fit file ###
############################

hdulist = fits.open('b3FGL.fit')
header  = hdulist[1].header
data    = hdulist[1].data
t       = Table(data)
hdulist.close()

datas    = np.array(data)
name     = np.array(data[:]['Source_Name'])
chi2     = np.array(data[:]['chi_square'])

a=0
x=[]
while a<3034 :
	Source = name[a]
	(nuFnu,flux,unc_fluxm,unc_fluxp,unc_num,unc_nup,E,Emin,Emax) = nu(Source)

	df = len(nuFnu)-1			#degrees of freedom

	#P>0.99
	if df == 1:
		if chi2[a]<1e-4:
			x.append(name)
	elif df == 2:
		if chi2[a]<2.01e-2:
			x.append(name)
	elif df == 3:
		if chi2[a]<0.1148:
			x.append(name)
	elif df == 4:
		if chi2[a]<0.2971:
			x.append(name)

	#if data[a]['chi_square'] <1 :
	#	x.append(data[a]['chi_square'])

	a=a+1	
x = np.array(x)
print('youpiiii:',len(x))
