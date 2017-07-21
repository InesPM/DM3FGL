import numpy as np
from math import *
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from astropy.io import fits
from astropy.table import Table
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

'''Generation of the DM spectrum'''

def getDMspectrum(option='e',finalstate='b',mass=1000,channel=None,Jfactor=1.7e19,boost=1):
    #Options:
    #  e: outputs (E, dN/dE)
    #  e2: outputs (E, E**2 dN/dE)
    #  x: outputs (x,dN/dx)
    # mass in GeV
    # Jfactor in GeV2cm-5
    sigmav=3*1e-26 # annihilation cross section in cm3s-1
    data = np.genfromtxt (filename, names = True ,dtype = None,comments='#')

    massvals = data["mDM"]
	

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
        print(' shape flux_h:',flux_h.shape)
        
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
    evals = []
    xvals2 = [] #aportacion mia
    if  option is 'e': #and boost > 1:
        #if mass == 5000:
        sigmavboost = sigmav * boost #no era necesario
        file1 = open("tabla"+str(mass)+str(finalstate)+str(sigmavboost)+".txt","w")

    logxvalsnew = np.linspace(-8.9,0,10000)
    xvalsnew = 10**logxvalsnew

    for i in range(len(xvalsnew)):
        x=xvalsnew[i]
        xvals2.append(x) #aportacion mia
        #vdNdx.append(dNdx(x))
        #x2vdNdx.append(x**2*dNdx(x))
        #dNde.append(dNdx(x)*Jfactor*GeV**2*sigmav*boost/(8*np.pi*(mass*GeV)**3))
        #e2dNde.append((1/erg)*x**2*dNdx(x)*Jfactor*GeV**2*sigmav*boost/(8*np.pi*mass*GeV))
        if finalstate == 'new':
            aux = dNdx_new(x,di)
        else:
            aux = dNdx(x)
        vdNdx.append(aux)
        x2vdNdx.append(x**2*aux)
        dNdeaux = aux*Jfactor*GeV**2*sigmav*boost/(8*np.pi*(mass*GeV)**3)
        dNde.append(dNdeaux)
        e2dNde.append((1/erg)*x**2*aux*Jfactor*GeV**2*sigmav*boost/(8*np.pi*mass*GeV))
        
        
        evals.append(x*mass*GeV)
        if option is 'e': #and boost > 1:
            #if mass == 5000 and dNdeaux != 0:
            if dNdeaux != 0:
                file1.write(str(x*mass*10**3) + " " + str(dNdeaux/(10**6)) + "\n")
                #print i
                #print(option, boost, mass, x*mass*10**3, dNdeaux/(10**6))
        #print(x, vdNdx[i], evals[i], e2dNde[i])
      #  if x == 1:
      #      break
    if option is 'e':
        '''#if mass == 5000 and boost > 1:
        file1.write(str(x*mass*10**3+1) + " " + "1e-99" + "\n")
        file1.write(str(x*mass*10**3+5) + " " + "1e-99" + "\n")
        file1.write(str(x*mass*10**3+10) + " " + "1e-99" + "\n")
        file1.close()'''
        return (evals,dNde)
    if option is 'e2':
        return (evals,e2dNde)
    if option is 'x':
        return (xvals2,vdNdx)
    if option is 'x2':
        return (xvals2,x2vdNdx)
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
	
	i = np.isnan(unc_fluxm)
	unc_fluxm[i] = 0
	j = np.isnan(unc_num)
	unc_num[j] = 0
	return (nuFnu,flux,unc_fluxm, unc_fluxp, unc_num, unc_nup)


##-----------------##
##- chi2 function -##
##-----------------##

'''def chi2(c) :			
	#E,nuFnu from the 3FGL catalog, Edm,nuFnudm from the getDMspectrum function, they must be given"""

	m  = c[1]
	Jm = c[0]*6e18
	J  = Jm*c[1]

	(Edm,nuFnudm) = getDMspectrum('e2','b',mass=m,channel=chan,Jfactor=J)
	Edm,nuFnudm=np.array(Edm),np.array(nuFnudm)

	i,c2=0,0
	while i<len(E) :
		index = np.argmin(np.abs(Edm - E[i]))
		delta = (nuFnu[i] - nuFnudm[index])
		sigma = (unc_nup[i] + unc_num[i]) / 2
		alpha = (unc_nup[i] - unc_num[i]) / 2
		A     = (alpha / sigma)
		chi   = (delta/sigma)**2 * (1 - 2*A*delta/sigma + 5*(A*delta/sigma)**2)
		
		c2    = c2 + chi
		i=i+1

	#print('mass',m,'Jfactor',J,'chi2',c2)
	print('Jm : J/m',Jm,'mass',m,'Jfactor',J,'chi2',c2)

	return c2'''


def chi2(c) :			
	#E,nuFnu from the 3FGL catalog, Edm,nuFnudm from the getDMspectrum function, they must be given

	mass    = c[1]
	Jboost  = c[0]
	Jfactor = Jboost*1.7e19

	(Edm,nuFnudm) = getDMspectrum('e2',finalstate='b',mass=mass,channel=chan, Jfactor=Jfactor)
	Edm,nuFnudm=np.array(Edm),np.array(nuFnudm)

	i,c2=0,0
	while i<len(E) :
		index = np.argmin(np.abs(Edm - E[i]))
		delta = (nuFnu[i] - nuFnudm[index])
		sigma = (unc_nup[i] + unc_num[i]) / 2
		alpha = (unc_nup[i] - unc_num[i]) / 2
		A     = (alpha / sigma)
		chi   = (delta/sigma)**2 * (1 - 2*A*delta/sigma + 5*(A*delta/sigma)**2)
		
		c2    = c2 + chi
		i=i+1

	print('\nmass',mass,'Jfactor',Jfactor,'chi2',c2)
	time = dt.datetime.now()
	print('Time:',time-timei)

	return c2


"""
def chi22(m) :			
	#E,nuFnu from the 3FGL catalog, Edm,nuFnudm from the getDMspectrum function, they must be given
	(Edm,nuFnudm) = getDMspectrum('e2','b',mass=m,channel=chan,Jfactor=Jfactor)
	Edm,nuFnudm=np.array(Edm),np.array(nuFnudm)
	i,c2 = 0,0
	while i<len(E) :
		index = np.argmin(np.abs(Edm - E[i]))
		delta = (nuFnu[i] - nuFnudm[index])
		sigma = (unc_nup[i] + unc_num[i]) / 2
		alpha = (unc_nup[i] - unc_num[i]) / 2
		A     = (alpha / sigma)
		chi   = (delta/sigma)**2 * (1 - 2*A*delta/sigma + 5*(A*delta/sigma)**2)
		
		c2    = c2 + chi
		i=i+1
	print('2 : mass',m[0],'chi2',c2[0])
	return c2
"""

##---------------------##
##- chi2 minimization -##
##---------------------##

def massresult(func) :


	#Generation of initial parameters:
	data = np.genfromtxt (filename, names = True ,dtype = None,comments='#')
	massvals = data["mDM"]
	mmin, mmax = np.min(massvals), np.max(massvals)

	bnds = [(0, 10),(mmin, 1e2)]
	

	#c0   = [1/0.121,12.46]	#c0, m0

	
	c0 = differential_evolution(func, bnds)
	c0 = c0.x
	mass = c0[1]
	Jboost = c0[0]

	massresult =  opt.minimize(func, c0, bounds=bnds, method='SLSQP',tol=1e-2)	#func=chi2

	
	#massresult =  opt.minimize(func, c0, bounds=bnds, method='L-BFGS-B',tol=1e-3)
	#massresult =  opt.minimize(func, c0, method='Nelder-Mead',tol=1e-2)

	mass    = massresult['x'][1]
	Jboost  = massresult['x'][0]
	Jfactor = Jboost*1.7e19
	chi2    = massresult['fun']

	print('mass',mass,'Jfactor',Jfactor,'chi2',chi2)
	return mass, Jfactor, chi2


"""
def massresult2(func) :
	m0    = [mass]
	bnds2 = [(0,1e6)]
	massresult2 = opt.minimize(func, m0, bounds=bnds2, method='SLSQP', tol=1e-6)	#func=chi22
	#massresult2 = opt.minimize(func, m0, method='Nelder-Mead')	#func=chi22
	mass2 = massresult2['x']
	chi2  = massresult2['fun']
	print('mass',mass2,'chi2',chi2)
	return mass2, chi2
"""

####################
### Main program ###
####################

timei = dt.datetime.now()

chan='b'
b,m0=0,20

##Chosen points from the catalog 

"""
#Test points:
E=np.array([3.e-4,5.e-4,3.e-3,1.e-2])		#TeV
F=np.array([5.e-13,7.e-13,1.e-13,5.e-16])
"""

##Opening and closing the catalog
list=fits.open('3FGL.fit')
header=fits.getheader('3FGL.fit')
data=fits.getdata('3FGL.fit')
t=Table(data)
list.close()

'''Analysis of different sources'''

##Creation of a document containing mass, Jfactor and chi2 of each source
data=open('fitdata.dat','w')



#name[2502]=3FGL J1924.8-1034


#Different sources

#a=2502
a=0

#while a<2503 :		#test
#while a<3034 :
while a<2 :

#Different m0


#while b<2 :


	##Source name
	name=t[:]['Source_Name']
	Source = name[a]
	print('\n\n\n\n\n','-------',a,'-------','\n',Source,'\n m0 =',m0)

	##Spectral energy distribution##
	#Ftot=t[:]['nuFnu300_1000']
	#Ftot=np.array(Ftot)
	(nuFnu,flux,unc_fluxm,unc_fluxp,unc_num,unc_nup) = nu(Source)

	E    = np.array([sqrt(100*300),sqrt(300*1000),sqrt(1000*3000),sqrt(3000*10000),sqrt(10000*100000)])	#TeV
	Emin = E-np.array([100,300,1000,3000,10000,])
	Emax = np.array([300,1000,3000,10000,100000])-E
	
	E    = E*1e-6	#TeV
	Emin = Emin*1e-6
	Emax = Emax*1e-6
	#logarithmic mid-point of the band

	print('\nSpectral energy distribution',nuFnu)
	print('\nError bars\n',unc_num,'\n',unc_nup)
	print('\n\n')

	#Minimization	
	mass, Jfactor, X2 = massresult(chi2)
	#Jfactor  = 1.7e19
	#mass     = 30
	#mass2, X2 = massresult2(chi22)
	print('\nMass result=',mass,'\nJfactor=',Jfactor,'\nchi2=',X2)
	write=Source+' '+str(chan)+' '+str(mass)+' '+str(Jfactor)+' '+str(X2)+'\n'
	data.write(write)


	#############
	### plots ###
	#############

	fig=pl.figure(num=a)
	comment = 'mass='+str(mass)+'GeV, Jfactor='+str(Jfactor)+'$GeV^2cm^{-5}$, $\chi^2$='+str(X2)

	#comment = 'mass='+str(mass)+'GeV, '+'$\chi^2$='+str(X2)
		
	ax=fig.add_subplot(111)
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlim(1e-4, 0.1)
	ax.set_ylim(5e-20,1e-10)
	plt.suptitle(Source,fontsize=18)
	ax.set_title(comment,fontsize=10)
	ax.set_xlabel('$E$ [TeV]')
	ax.set_ylabel('$E^2 dN/dE$ [erg cm$^{-2}$ s$^{-1}$]')

	#ax.errorbar(E, nuFnu, xerr=[Emin,Emax], yerr=[unc_num,unc_nup],fmt='--o',linewidth=1,label='data')

	ax.errorbar(E, nuFnu, xerr=[Emin,Emax], yerr=[unc_num,unc_nup],fmt='--o',linewidth=1)

	#if mass!=0 :
	#	(Edm1,Fdm1) = getDMspectrum('e2', 'b', mass=mass, channel=chan, Jfactor=Jfactor)
	#	ax.plot(Edm1, Fdm1, label=m0, linewidth=1)
	#plt.legend(loc=3, prop={'size':12})	
	#(Edm1,Fdm1) = getDMspectrum('e2','b',mass,chan)
	#ax.plot(Edm1, Fdm1, label="m = 0.01 TeV", color='red', linewidth=1)

	say='m0: '+str(m0)+', J: '+str(Jfactor)+', m: '+str(mass)+', $\chi^2$:'+str(X2)
	
	(Edm1,Fdm1) = getDMspectrum('e2', 'b', mass=mass, channel=chan, Jfactor=Jfactor)
	ax.plot(Edm1, Fdm1, label=say, linewidth=1)
	plt.legend(loc=3, prop={'size':8})	

	#a=a+1

	m0=m0+20
	b=b+1
	time = dt.datetime.now()
	print('\nTime:',time-timei)

data.close()

#ax.errorbar(E, nuFnu, xerr=[Emin,Emax], yerr=[unc_num,unc_nup],fmt='--o',linewidth=1,label='data')
	
plt.show()
