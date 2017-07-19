import numpy as np
from math import *
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import Table, Column
import pylab as pl
import scipy as sp
import bisect
from scipy.interpolate import interp1d
from scipy.interpolate import spline
import scipy.optimize as opt
from matplotlib import pyplot as plt
import datetime as dt

#############
### Paths ###
#############

pl.rcParams['figure.figsize'] = (10.0, 7.0)
pl.rcParams['font.size'] = 18
pl.rcParams['font.family'] = 'serif'
pl.rcParams['lines.linewidth'] = 3


pathforfigs ='/home/ines/Desktop/DM/TFGmodif'
pathforaux='/home/ines/Desktop/DM/TFGmodif'
filename=pathforaux+'/CascadeSpectra/Spectra/AtProduction_gammas.dat'
path=pathforaux+'/sensitivities/'
path3FGL='/home/ines/Desktop/DM/TFGmodif'

###################
### Useful data ###
###################

Qe  = 1.602176462e-19
TeV = 1
GeV = 1e-3 * TeV
MeV = 1e-6 * TeV
erg = 0.624151 * TeV 
eV  = 1e-9 * GeV

###############
###Functions###
###############

##---------------------------------##
##- Generation of the DM spectrum -##
##---------------------------------##

def getDMspectrum(evals, mass=1000, Jboost=1):
	option     = 'e2'
	finalstate = 'b'
	channel    = None
	boost      = 1
	Jfactor    = Jboost*1.7e19
	#Options:
	#  e: outputs (E, dN/dE)
	#  e2: outputs (E, E**2 dN/dE)
	#  x: outputs (x,dN/dx)
	# mass in GeV
	# Jfactor in GeV2cm-5
	sigmav=3*1e-26 # annihilation cross section in cm3s-1
	data = np.genfromtxt (filename, names = True ,dtype = None,comments='#')

	massvals = data["mDM"]

	print(a,'- mass:',mass,', J:',Jfactor,'\n')
	

	#index = np.where(np.abs( (massvals - mass) / mass) < 1.e-3)
	#xvals = 10**(data["Log10x"][index])
	#print(option,mass,'shape xvals',xvals.shape,end=' ')
   
	
	if (mass < np.max(massvals) and mass > np.min(massvals)) :
		#print('min=',np.min(massvals),'max=',np.max(massvals))
		min = np.min(np.abs( (massvals - mass) /mass))
		index = np.where(np.abs( (massvals - mass) /mass) == min)
		#print('index',index)
		xvals = 10**(data["Log10x"][index])
		#print(option,mass,'shape xvals',xvals.shape,end=' ')
	else :
		print('\nError: mass out of range\n')
        
		

	def branchingratios(m_branon): #<sigmav>_particle / <sigmav>_total
	#PhysRevD.68.103505		#GeV/c**2
		m_top = 172.44
		m_W   = 80.4
		m_Z   = 91.2
		m_h   = 125.1
		m_c   = 1.275
		m_b   = 4.18
		m_tau = 1.7768
		if channel == None:
			if m_branon > m_top:
				c_0_top = 3.0 / 16 * m_branon ** 2 * m_top ** 2 * (m_branon ** 2 - m_top ** 2) * (1 - m_top ** 2 / m_branon ** 2) ** (1.0 / 2) 
			else:
				c_0_top = 0
			if m_branon > m_Z:
				c_0_Z = 1.0 / 64 * m_branon ** 2 * (1 - m_Z ** 2 / m_branon ** 2) ** (1.0 / 2) * (4 * m_branon ** 4 - 4 * m_branon ** 2 * m_Z ** 2 + 3 * m_Z ** 4)
			else:
				c_0_Z = 0
			if m_branon > m_W:
				c_0_W = 2.0 / 64 * m_branon ** 2 * (1 - m_W ** 2 / m_branon ** 2) ** (1.0 / 2) * (4 * m_branon ** 4 - 4 * m_branon ** 2 * m_W ** 2 + 3 * m_W ** 4)
			else:
				c_0_W = 0
			if m_branon > m_h:
				c_0_h = 1.0 / 64 * m_branon ** 2 * (2 * m_branon ** 2 + m_h ** 2) ** 2 * (1 - m_h ** 2 / m_branon ** 2) ** (1.0 / 2)
			else:
				c_0_h = 0
			if m_branon > m_c:
				c_0_c = 3.0 / 16 * m_branon ** 2 * m_c ** 2 * (m_branon ** 2 - m_c ** 2) * (1 - m_c ** 2 / m_branon ** 2) ** (1.0 / 2) 
			else:
				c_0_c = 0
			if m_branon > m_b:
				c_0_b = 3.0 / 16 * m_branon ** 2 * m_b ** 2 * (m_branon ** 2 - m_b ** 2) * (1 - m_b ** 2 / m_branon ** 2) ** (1.0 / 2) 
			else:
				c_0_b = 0
			if m_branon > m_tau:
				c_0_tau = 1.0 / 16 * m_branon ** 2 * m_tau ** 2 * (m_branon ** 2 - m_tau ** 2) * (1 - m_tau ** 2 / m_branon ** 2) ** (1.0 / 2) 
			else:
				c_0_tau = 0
			c_0_T  = c_0_top + c_0_Z + c_0_W + c_0_h + c_0_c + c_0_b + c_0_tau
			br_t   = (c_0_top / c_0_T)
			br_Z   = c_0_Z / c_0_T
			br_W   = c_0_W / c_0_T
			br_h   = c_0_h / c_0_T
			br_c   = c_0_c / c_0_T
			br_b   = c_0_b / c_0_T
			br_tau = c_0_tau / c_0_T
			#f.append((c_0_T/(3*10**(-26)*math.pi**2))**(1./8))
		else:
			if channel == 't':
				br_t,br_Z,br_W,br_h,br_c,br_b,br_tau=1,0,0,0,0,0,0
			if channel == 'Z':
				br_t,br_Z,br_W,br_h,br_c,br_b,br_tau=0,1,0,0,0,0,0
			if channel == 'W':
				br_t,br_Z,br_W,br_h,br_c,br_b,br_tau=0,0,1,0,0,0,0
			if channel == 'h':
				br_t,br_Z,br_W,br_h,br_c,br_b,br_tau=0,0,0,1,0,0,0
			if channel == 'c':
				br_t,br_Z,br_W,br_h,br_c,br_b,br_tau=0,0,0,0,1,0,0
			if channel == 'b':
				br_t,br_Z,br_W,br_h,br_c,br_b,br_tau=0,0,0,0,0,1,0
			if channel == 'tau':
				br_t,br_Z,br_W,br_h,br_c,br_b,br_tau=0,0,0,0,0,0,1
		return {'masas': m_branon, 't': br_t, 'Z': br_Z, 'W': br_W, 'h': br_h, 'c': br_c, 'b': br_b, 'Tau': br_tau}	
        
    
	#tau name modified in AtProduction_Gammas.dat
    
	if finalstate == "new":
		di = branchingratios(mass)
		flux_c   = data[list(di.keys())[1]][index]/(np.log(10)*xvals) 
		flux_tau = data[list(di.keys())[2]][index]/(np.log(10)*xvals) 
		flux_b   = data[list(di.keys())[3]][index]/(np.log(10)*xvals) 
		flux_t   = data[list(di.keys())[4]][index]/(np.log(10)*xvals) 
		flux_W   = data[list(di.keys())[5]][index]/(np.log(10)*xvals) 
		flux_Z   = data[list(di.keys())[7]][index]/(np.log(10)*xvals) 
		flux_h   = data[list(di.keys())[6]][index]/(np.log(10)*xvals) 

		print(' shape flux_h:',flux_h.shape,)
		print(' shape xvals :',xvals.shape,'min',np.min(xvals),'max',np.max(xvals))
      
		loadspec_h   = interp1d(xvals,flux_h)
		loadspec_Z   = interp1d(xvals,flux_Z)
		loadspec_t   = interp1d(xvals,flux_t)
		loadspec_W   = interp1d(xvals,flux_W)
		loadspec_b   = interp1d(xvals,flux_b)
		loadspec_c   = interp1d(xvals,flux_c)
		loadspec_tau = interp1d(xvals,flux_tau)

	else:
		flux = data[finalstate][index]/(np.log(10)*xvals) #data is given in dN/d(log10(X)) = x ln10 dN/dx
		#flux = data[finalstate][index] 
		loadspec = interp1d(xvals,flux)

	def dNdx(x):
		fluxval = loadspec(x)
		if (x>1 or fluxval<0):
			return 0
		else:
			return fluxval
      
	def dNdx_new(x,di):
		fluxval_h = loadspec_h(x)
		if (x>1 or fluxval_h<0):
			fluxval_h = 0

		fluxval_Z = loadspec_Z(x)
		if (x>1 or fluxval_Z<0):
			fluxval_Z = 0
        
		fluxval_t = loadspec_t(x)
		if (x>1 or fluxval_t<0):
			fluxval_t = 0
        
		fluxval_W = loadspec_W(x)
		if (x>1 or fluxval_W<0):
			fluxval_W = 0
        
		fluxval_b = loadspec_b(x)
		if (x>1 or fluxval_b<0):
			fluxval_b = 0
        
		fluxval_c = loadspec_c(x)
		if (x>1 or fluxval_c<0):
			fluxval_c = 0
        
		fluxval_tau = loadspec_tau(x)
		if (x>1 or fluxval_tau<0):
			fluxval_tau = 0
		return (list(di.values())[1]*fluxval_c + list(di.values())[2]*fluxval_tau + 
				list(di.values())[3]*fluxval_b + list(di.values())[4]*fluxval_t +
				list(di.values())[5]*fluxval_W + list(di.values())[7]*fluxval_Z +
				list(di.values())[6]*fluxval_h)

	vdNdx = []
	x2vdNdx = []
	dNde = []
	e2dNde = []
	#evals = []
	xvals2 = [] #aportacion mia
	if  option is 'e': #and boost > 1:
		#if mass == 5000:
		sigmavboost = sigmav * boost #no era necesario
		file1 = open("tabla"+str(mass)+str(finalstate)+str(sigmavboost)+".txt","w")

    #logxvalsnew = np.linspace(-8.9,0,10000)
    #xvalsnew = 10**logxvalsnew
	#print('evalsmin',np.min(evals),'evalsmax',np.max(evals))		
	xvalsnew = evals/(mass*GeV)
	#print('xvalsmin',np.min(xvalsnew),'xvalsmax',np.max(xvalsnew))	
	logxvalsnew = np.log10(xvalsnew)
	#print('logxmin',np.min(logxvalsnew),'logxmax',np.max(logxvalsnew))

	for i in range(len(evals)):
		if logxvalsnew[i]>-8.9 and logxvalsnew[i]<0 :
			x=xvalsnew[i]
			xvals2.append(x) #aportacion mia
		
			if finalstate == 'new':
				aux = dNdx_new(x,di)
			else:
				aux = dNdx(x)
			vdNdx.append(aux)
			x2vdNdx.append(x**2*aux)
			dNdeaux = aux*Jfactor*GeV**2*sigmav*boost/(8*np.pi*(mass*GeV)**3)
			dNde.append(dNdeaux)
			e2dNde.append((1/erg)*x**2*aux*Jfactor*GeV**2*sigmav*boost/(8*np.pi*mass*GeV))
        
			#evals.append(x*mass*GeV)
		
			if option is 'e': #and boost > 1:
				#if mass == 5000 and dNdeaux != 0:
				if dNdeaux != 0:
					file1.write(str(x*mass*10**3) + " " + str(dNdeaux/(10**6)) + "\n")
		else :
			vdNdx.append(0)
			x2vdNdx.append(0)
			dNde.append(0)
			e2dNde.append(0)


	if option is 'e':
        #if mass == 5000 and boost > 1:
		file1.write(str(x*mass*10**3+1) + " " + "1e-99" + "\n")
		file1.write(str(x*mass*10**3+5) + " " + "1e-99" + "\n")
		file1.write(str(x*mass*10**3+10) + " " + "1e-99" + "\n")
		file1.close()
		#return (evals,dNde)
		return dNde
	if option is 'e2':
		#return (evals,e2dNde)
		return e2dNde
	if option is 'x':
		#return (xvals2,vdNdx)
		return vdNdx
	if option is 'x2':
		#return (xvals2,x2vdNdx)
		return x2vdNdx
	else:
		print('Option '+str(option)+' not supported')
		
		
    #Options:
    #  e: outputs (E, dN/dE)
    #  e2: outputs (E, E**2 dN/dE)
    #  x: outputs (x,dN/dx)
    # mass in GeV
    # Jfactor in GeV2cm-5
	sigmav=3*1e-26 # annihilation cross section in cm3s-1
	data = np.genfromtxt (filename, names = True ,dtype = None,comments='#')

	massvals = data["mDM"]
	index = np.where(np.abs( (massvals - mass) / mass) < 0.045)
	xvals = 10**(data["Log10x"][index])
    #print(option,mass,'shape xvals',xvals.shape,end=' ')
    
	def branchingratios(m_branon): #<sigmav>_particle / <sigmav>_total
    #PhysRevD.68.103505
		m_top = 172.44
		m_W = 80.4
		m_Z = 91.2
		m_h = 125.1
		m_c = 1.275
		m_b = 4.18
		m_tau = 1.7768
		if m_branon > m_top:
			c_0_top = 3.0 / 16 * m_branon ** 2 * m_top ** 2 * (m_branon ** 2 - m_top ** 2) * (1 - m_top ** 2 / m_branon ** 2) ** (1.0 / 2) 
		else:
			c_0_top = 0
		if m_branon > m_Z:
			c_0_Z = 1.0 / 64 * m_branon ** 2 * (1 - m_Z ** 2 / m_branon ** 2) ** (1.0 / 2) * (4 * m_branon ** 4 - 4 * m_branon ** 2 * m_Z ** 2 + 3 * m_Z ** 4)
		else:
			c_0_Z = 0
		if m_branon > m_W:
			c_0_W = 2.0 / 64 * m_branon ** 2 * (1 - m_W ** 2 / m_branon ** 2) ** (1.0 / 2) * (4 * m_branon ** 4 - 4 * m_branon ** 2 * m_W ** 2 + 3 * m_W ** 4)
		else:
			c_0_W = 0
		if m_branon > m_h:
			c_0_h = 1.0 / 64 * m_branon ** 2 * (2 * m_branon ** 2 + m_h ** 2) ** 2 * (1 - m_h ** 2 / m_branon ** 2) ** (1.0 / 2)
		else:
			c_0_h = 0
		if m_branon > m_c:
			c_0_c = 3.0 / 16 * m_branon ** 2 * m_c ** 2 * (m_branon ** 2 - m_c ** 2) * (1 - m_c ** 2 / m_branon ** 2) ** (1.0 / 2) 
		else:
			c_0_c = 0
		if m_branon > m_b:
			c_0_b = 3.0 / 16 * m_branon ** 2 * m_b ** 2 * (m_branon ** 2 - m_b ** 2) * (1 - m_b ** 2 / m_branon ** 2) ** (1.0 / 2) 
		else:
			c_0_b = 0
		if m_branon > m_tau:
			c_0_tau = 1.0 / 16 * m_branon ** 2 * m_tau ** 2 * (m_branon ** 2 - m_tau ** 2) * (1 - m_tau ** 2 / m_branon ** 2) ** (1.0 / 2) 
		else:
			c_0_tau = 0
		c_0_T = c_0_top + c_0_Z + c_0_W + c_0_h + c_0_c + c_0_b + c_0_tau
		br_t = (c_0_top / c_0_T)
		br_Z = c_0_Z / c_0_T
		br_W = c_0_W / c_0_T
		br_h = c_0_h / c_0_T
		br_c = c_0_c / c_0_T
		br_b = c_0_b / c_0_T
		br_tau = c_0_tau / c_0_T
        #f.append((c_0_T/(3*10**(-26)*math.pi**2))**(1./8))
		return {'masas': m_branon, 't': br_t, 'Z': br_Z, 'W': br_W, 'h': br_h, 'c': br_c, 'b': br_b, 'Tau': br_tau}
    
    #tau name modified in AtProduction_Gammas.dat

	if finalstate == "new":
		di = branchingratios(mass)
		flux_c = data[list(di.keys())[1]][index]/(np.log(10)*xvals) 
		flux_tau = data[list(di.keys())[2]][index]/(np.log(10)*xvals) 
		flux_b = data[list(di.keys())[3]][index]/(np.log(10)*xvals) 
		flux_t = data[list(di.keys())[4]][index]/(np.log(10)*xvals) 
		flux_W = data[list(di.keys())[5]][index]/(np.log(10)*xvals) 
		flux_Z = data[list(di.keys())[7]][index]/(np.log(10)*xvals) 
		flux_h = data[list(di.keys())[6]][index]/(np.log(10)*xvals) 
		#print(' shape flux_h:',flux_h.shape)
        
		loadspec_h = interp1d(xvals,flux_h)
		loadspec_Z = interp1d(xvals,flux_Z)
		loadspec_t = interp1d(xvals,flux_t)
		loadspec_W = interp1d(xvals,flux_W)
		loadspec_b = interp1d(xvals,flux_b)
		loadspec_c = interp1d(xvals,flux_c)
		loadspec_tau = interp1d(xvals,flux_tau)
		#print('shape xvals',xvals.shape)
		#print('shape loadspec_h',loadspec_h.shape)
	else:
		flux = data[finalstate][index]/(np.log(10)*xvals) #data is given in dN/d(log10(X)) = x ln10 dN/dx
        #flux = data[finalstate][index] 
        
		loadspec = interp1d(xvals,flux)

	def dNdx(x):
		fluxval = loadspec(x)
		if (x>1 or fluxval<0):
			return 0
		else:
			return fluxval
        
	def dNdx_new(x,di):
		fluxval_h = loadspec_h(x)
		if (x>1 or fluxval_h<0):
			fluxval_h = 0
        
		fluxval_Z = loadspec_Z(x)
		if (x>1 or fluxval_Z<0):
			fluxval_Z = 0
        
		fluxval_t = loadspec_t(x)
		if (x>1 or fluxval_t<0):
			fluxval_t = 0
        
		fluxval_W = loadspec_W(x)
		if (x>1 or fluxval_W<0):
			fluxval_W = 0
        
		fluxval_b = loadspec_b(x)
		if (x>1 or fluxval_b<0):
			fluxval_b = 0
        
		fluxval_c = loadspec_c(x)
		if (x>1 or fluxval_c<0):
			fluxval_c = 0
        
		fluxval_tau = loadspec_tau(x)
		if (x>1 or fluxval_tau<0):
			fluxval_tau = 0
		return (list(di.values())[1]*fluxval_c + list(di.values())[2]*fluxval_tau + 
				list(di.values())[3]*fluxval_b + list(di.values())[4]*fluxval_t +
				list(di.values())[5]*fluxval_W + list(di.values())[7]*fluxval_Z +
				list(di.values())[6]*fluxval_h)

	vdNdx = []
	x2vdNdx = []
	dNde = []
	e2dNde = []
	#evals = []
	xvals2 = [] #aportacion mia
	if  option is 'e': #and boost > 1:
        #if mass == 5000:
		sigmavboost = sigmav * boost #no era necesario
		file1 = open("tabla"+str(mass)+str(finalstate)+str(sigmavboost)+".txt","w")

    #logxvalsnew = np.linspace(-8.9,0,10000) Lo convertimos en xdata para hacer el ajuste
	#xvalsnew = 10**logxvalsnew
	#xvalsnew = evals/(mass*GeV)

	for i in range(len(evals)):
		if evals[i]<mass*GeV and evals[i]>mass*GeV*10**-8.9 :
			x=evals[i]/(mass*GeV)
			xvals2.append(x) #aportacion mia
			if finalstate == 'new':
				aux = dNdx_new(x,di)
			else:
				aux = dNdx(x)
			vdNdx.append(aux)
			x2vdNdx.append(x**2*aux)
			dNdeaux = aux*Jfactor*GeV**2*sigmav*boost/(8*np.pi*(mass*GeV)**3)
			dNde.append(dNdeaux)
			e2dNde.append((1/erg)*x**2*aux*Jfactor*GeV**2*sigmav*boost/(8*np.pi*mass*GeV))
        
        
			#evals.append(x*mass*GeV)
			if option is 'e': #and boost > 1:
				#if mass == 5000 and dNdeaux != 0:
				if dNdeaux != 0:
					file1.write(str(x*mass*10**3) + " " + str(dNdeaux/(10**6)) + "\n")
		else :
			vdNdx.append(0)
			x2vdNdx.append(0)
			dNde.append(0)
			e2dNde.append(0)
    
	if option is 'e':
		file1.write(str(x*mass*10**3+1) + " " + "1e-99" + "\n")
		file1.write(str(x*mass*10**3+5) + " " + "1e-99" + "\n")
		file1.write(str(x*mass*10**3+10) + " " + "1e-99" + "\n")
		file1.close()
		#return (evals,dNde)
		return dNde
	if option is 'e2':
		#return (evals,e2dNde)
		return e2dNde
	if option is 'x':
		#return (xvals2,vdNdx)
		return vdNdx
	if option is 'x2':
		#return (xvals2,x2vdNdx)
		return x2vdNdx
	else:
		print('Option '+str(option)+' not supported')


##------------------------------##
##- Data from the 3FGL catalog -##
##------------------------------##

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
				b=b+1
		a=a+1
	nuFnu     = np.array(nuFnu)
	flux      = np.array(flux)
	unc_fluxm = np.array(unc_fluxm)
	unc_fluxp = np.array(unc_fluxp)
	unc_num   = np.array(unc_num)
	unc_nup   = np.array(unc_nup)
	
	'''
	i = np.isnan(unc_fluxm)
	unc_fluxm[i] = 0
	j = np.isnan(unc_num)
	unc_num[j] = 0
	'''	
	return (nuFnu,flux,unc_fluxm, unc_fluxp, unc_num, unc_nup)

##-----------------##
##- chi2 function -##
##-----------------##

def chi2(c) :			
	#E,nuFnu from the 3FGL catalog, Edm,nuFnudm from the getDMspectrum function, they must be given

	#mass    = c[1]
	#Jboost  = c[0]
	#Jfactor = Jboost*1.7e19

	nuFnudm = getDMspectrum(evals, mass=mass, Jboost=Jboost)
	#nuFnudm = np.array(Edm),np.array(nuFnudm)

	i,c2=0,0
	while i<len(E) :
		index = np.argmin(np.abs(evals - E[i]))
		delta = (nuFnu[i] - nuFnudm[index])
		sigma = (unc_nup[i] + unc_num[i]) / 2
		alpha = (unc_nup[i] - unc_num[i]) / 2
		A     = (alpha / sigma)
		chi   = (delta/sigma)**2 * (1 - 2*A*delta/sigma + 5*(A*delta/sigma)**2)
		
		c2    = c2 + chi
		i=i+1

	#print('\nmass',mass,'Jfactor',Jfactor,'chi2',c2)

	return c2


####################
### Main program ###
####################

timei = dt.datetime.now()

#chan = 'b'
#m0   = 20     #GeV

##----------------------------------##
##- Chosen points from the catalog -##
##----------------------------------##

##Opening and closing the catalog
hdulist = fits.open('3FGL.fit')
header  = hdulist[1].header
data    = hdulist[1].data
t       = Table(data)
hdulist.close()

##---------------##
##- Some arrays -##
##---------------##

#evals
evals = np.logspace(np.log10(6)-13,2,20000)

#new catalog parameters
mass_a     = np.empty((3034,))
unc_mass_a = np.empty((3034,))
J_a        = np.empty((3034,))
unc_J_a    = np.empty((3034,))
chi2_a     = np.empty((3034,))

##---------------------------------##
##- Analysis of different sources -##
##---------------------------------##

##Creation of a document containing mass, Jfactor and chi2 of each source
#data=open('fitdata.dat','w')

#name[2502]=3FGL J1924.8-1034


#Different sources

a=0
#a=2502
#while a<2503 :		#test
while a<3034 :

	##Source name
	name=t[:]['Source_Name']
	Source = name[a]
	print('\n\n\n\n\n','-------',a,'-------','\n',Source)

	##Spectral energy distribution##
	(nuFnu1,flux1,unc_fluxm1,unc_fluxp1,unc_num1,unc_nup1) = nu(Source)

	index = np.array(np.isfinite(unc_num1))

	#Energy bins definition 	
	
	E1    = np.array([sqrt(100*300),sqrt(300*1000),sqrt(1000*3000),sqrt(3000*10000),sqrt(10000*100000)])	#TeV
	Emin1 = E1-np.array([100,300,1000,3000,10000,])
	Emax1 = np.array([300,1000,3000,10000,100000])-E1
	
	E1    = E1*1e-6	#TeV
	Emin1 = Emin1*1e-6
	Emax1 = Emax1*1e-6
	#evals= E
	#logarithmic mid-point of the band

	#We don't consider the bins where unc_num=nan, the source is not significant enough

	i = 0
	nuFnu = []; flux = []; unc_fluxm = []; unc_fluxp = []; unc_num = []; unc_nup = []
	E = []; Emin = []; Emax = []
	
	while i<len(E1) :
		if np.isnan(unc_num1[i]) == False :

			nuFnu.append(nuFnu1[i])
			flux.append(flux1[i])
			unc_fluxm.append(unc_fluxm1[i])
			unc_fluxp.append(unc_fluxp1[i])
			unc_num.append(unc_num1[i])
			unc_nup.append(unc_nup1[i])

			E.append(E1[i])
			Emin.append(Emin1[i])
			Emax.append(Emax1[i])
				
		i = i+1 

	nuFnu     = np.array(nuFnu)
	flux      = np.array(flux)
	unc_fluxm = np.array(unc_fluxm)
	unc_fluxp = np.array(unc_fluxp)
	unc_num   = np.array(unc_num)
	unc_nup   = np.array(unc_nup)

	E         = np.array(E)
	Emin      = np.array(Emin)
	Emax      = np.array(Emax)

	print('\nSpectral energy distribution',nuFnu)
	print('\nError bars\n',unc_num,'\n',unc_nup)
	print('\n\n')

	##-------------##
	##- Curve fit -##
	##-------------##
	
	
	s = (unc_num+unc_nup)/2
	#print(s.shape)
	popt, pcov=curve_fit(getDMspectrum, xdata=E, ydata=nuFnu, p0=(40,1), sigma=s, absolute_sigma=True, bounds=[(5, 1e-3),(1e3, 1e2),])
	mass    = popt[0]
	Jboost  = popt[1]
	merr    = sqrt(pcov[0][0])
	Jerr    = sqrt(pcov[1][1])	
	Jfactor = Jboost*1.7e19
	Jferr   = Jerr*1.7e19

	X2 = chi2(getDMspectrum)

	#print('\npopt:',popt,'\nperr:',pcov)
	print('mass:',mass,', J:',Jfactor,'\nmerr:',merr,', Jerr:',Jferr,', chi2:',X2)

	mass_a[a]     = mass
	unc_mass_a[a] = merr
	J_a[a]        = Jfactor
	unc_J_a[a]    = Jferr
	chi2_a[a]     = X2

	#############
	### plots ###
	#############

	fig=pl.figure(num=a)

	comment = 'mass='+str(mass)+'$\pm$'+str(merr)+'GeV, J='+str(Jfactor)+'$\pm$'+str(Jferr)+', $\chi^2$:'+str(X2)
		
	ax=fig.add_subplot(111)
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlim(1e-4, 0.1)
	ax.set_ylim(5e-20,1e-10)
	plt.suptitle(Source,fontsize=18)
	ax.set_title(comment,fontsize=10)
	ax.set_xlabel('$E$ [TeV]')
	ax.set_ylabel('$E^2 dN/dE$ [erg cm$^{-2}$ s$^{-1}$]')

	ax.errorbar(E, nuFnu, xerr=[Emin,Emax], yerr=[unc_num,unc_nup], fmt='--o', linewidth=1, label="data")


	#(Edm1,Fdm1) = getDMspectrum('e2', 'b', mass=mass, channel=chan, Jfactor=Jfactor)
	Fdm = getDMspectrum(evals, *popt)
	ax.plot(evals, Fdm, label="fit", linewidth=1)
	plt.legend(loc=3, prop={'size':12})	

	time = dt.datetime.now()
	print('\nTime:',time-timei)

	a=a+1

#plt.show()


#####################
### New fits file ###
#####################

##------------##
##- Creation -##
##------------##

mass_c = Column(name = 'mass', data = mass_a, format = 'E')
unc_mass_c = Column(name = 'unc_mass', data = unc_mass_a, format = 'E')
J_c = Column(name = 'J_factor', data = J_a, format = 'E')
unc_J_c = Column(name = 'unc_J_factor', data = unc_J_a, format = 'E')
chi2_c = Column(name = 'chi_square', data = chi2_a, format = 'E')


t.add_columns([mass_c, unc_mass_c, J_c, unc_J_c, chi2_c])

#print(t[0]['new'])

t.write('2new3FGL.fit', overwrite=True)

##--------##
##- Test -##
##--------##

hdulist2 = fits.open('2new3FGL.fit')
header2 = hdulist[1].header
data2 = hdulist2[1].data
#print(data2[1]['Source_Name'],type(data2[0]['Source_Name']))
#print(hdulist2.info())
#print('mass from fits:', data2[2]['mass'])
#cols = hdulist2[1].columns
#print(cols.info())

#print('mass',data2[3]['mass'], 'J', data2[3]['J_factor'], 'chi2', data2[3]['chi_square'])
#print('mass',data2[4]['mass'], 'J', data2[4]['J_factor'], 'chi2', data2[4]['chi_square'])
#print('mass',data2[5]['mass'], 'J', data2[5]['J_factor'], 'chi2', data2[5]['chi_square'])

datas = np.array(data)
x = np.array(np.where(datas['chi_square']<0))
print('It does not work in the folliwing indexes:',x, x.shape)

hdulist2.close()

