import numpy as np
from math import *
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import Table
import pylab as pl
import scipy as sp
import bisect
from scipy.interpolate import interp1d
from scipy.interpolate import spline
from matplotlib import pyplot as plt

###########
###Paths###
###########

pl.rcParams['figure.figsize'] = (10.0, 7.0)
pl.rcParams['font.size'] = 18
pl.rcParams['font.family'] = 'serif'
pl.rcParams['lines.linewidth'] = 3


pathforfigs =u'C:\\Users\In\xe9s\Documents\progs\TFGmodif'
pathforaux=u'C:\\Users\In\xe9s\Documents\progs\TFGmodif'
filename=pathforaux+'\CascadeSpectra\Spectra\AtProduction_gammas.dat'
path=pathforaux+'/sensitivities/'

#################
###Useful data###
#################

Qe = 1.602176462e-19
TeV = 1
GeV = 1e-3 * TeV
MeV = 1e-6 * TeV
erg = 0.624151 * TeV 
eV = 1e-9 * GeV

###############
###Functions###
###############

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
        print(option,mass,'shape xvals',xvals.shape,end=' ')
    else :
        print('\nError: mass out of range\n')
        
		

    def branchingratios(m_branon): #<sigmav>_particle / <sigmav>_total
    #PhysRevD.68.103505
        m_top = 172.44
        m_W = 80.4
        m_Z = 91.2
        m_h = 125.1
        m_c = 1.275
        m_b = 4.18
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
            c_0_T = c_0_top + c_0_Z + c_0_W + c_0_h + c_0_c + c_0_b + c_0_tau
            br_t = (c_0_top / c_0_T)
            br_Z = c_0_Z / c_0_T
            br_W = c_0_W / c_0_T
            br_h = c_0_h / c_0_T
            br_c = c_0_c / c_0_T
            br_b = c_0_b / c_0_T
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
        flux_c = data[list(di.keys())[1]][index]/(np.log(10)*xvals) 
        flux_tau = data[list(di.keys())[2]][index]/(np.log(10)*xvals) 
        flux_b = data[list(di.keys())[3]][index]/(np.log(10)*xvals) 
        flux_t = data[list(di.keys())[4]][index]/(np.log(10)*xvals) 
        flux_W = data[list(di.keys())[5]][index]/(np.log(10)*xvals) 
        flux_Z = data[list(di.keys())[7]][index]/(np.log(10)*xvals) 
        flux_h = data[list(di.keys())[6]][index]/(np.log(10)*xvals) 
        print(' shape flux_h:',flux_h.shape)
        
        loadspec_h = interp1d(xvals,flux_h)
        loadspec_Z = interp1d(xvals,flux_Z)
        loadspec_t = interp1d(xvals,flux_t)
        loadspec_W = interp1d(xvals,flux_W)
        loadspec_b = interp1d(xvals,flux_b)
        loadspec_c = interp1d(xvals,flux_c)
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
        #if mass == 5000 and boost > 1:
        file1.write(str(x*mass*10**3+1) + " " + "1e-99" + "\n")
        file1.write(str(x*mass*10**3+5) + " " + "1e-99" + "\n")
        file1.write(str(x*mass*10**3+10) + " " + "1e-99" + "\n")
        file1.close()
        return (evals,dNde)
    if option is 'e2':
        return (evals,e2dNde)
    if option is 'x':
        return (xvals2,vdNdx)
    if option is 'x2':
        return (xvals2,x2vdNdx)
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
    print(option,mass,'shape xvals',xvals.shape,end=' ')
    
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
        print(' shape flux_h:',flux_h.shape)
        
        loadspec_h = interp1d(xvals,flux_h)
        loadspec_Z = interp1d(xvals,flux_Z)
        loadspec_t = interp1d(xvals,flux_t)
        loadspec_W = interp1d(xvals,flux_W)
        loadspec_b = interp1d(xvals,flux_b)
        loadspec_c = interp1d(xvals,flux_c)
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
        #if mass == 5000 and boost > 1:
        file1.write(str(x*mass*10**3+1) + " " + "1e-99" + "\n")
        file1.write(str(x*mass*10**3+5) + " " + "1e-99" + "\n")
        file1.write(str(x*mass*10**3+10) + " " + "1e-99" + "\n")
        file1.close()
        return (evals,dNde)
    if option is 'e2':
        return (evals,e2dNde)
    if option is 'x':
        return (xvals2,vdNdx)
    if option is 'x2':
        return (xvals2,x2vdNdx)
    else:
        print('Option '+str(option)+' not supported')



fig=pl.figure(figsize=(15,10))


###########
###plots###
###########

chan=None

ax=fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(2e-4, 60)
ax.set_ylim(5e-22,1e-12)
ax.set_xlabel('$E$ [TeV]')
ax.set_ylabel('$E^2 dN/dE$ [erg cm$^{-2}$ s$^{-1}$]')


(Edm1,Fdm1) = getDMspectrum('e2','b',10,chan)
ax.plot(Edm1, Fdm1, label="m = 0.01 TeV", color='red', linewidth=1)

(Edm2,Fdm2) = getDMspectrum('e2','b',15,chan)
ax.plot(Edm2, Fdm2, label="m = 0.015 TeV", color='orange', linewidth=1)

#Chosen points
xdata=np.array([3.e-4,5.e-4,3.e-3,1.e-2])
ydata=np.array([5.e-13,7.e-13,1.e-13,5.e-16])
ax.plot(xdata, ydata, 'c+', label='some points')

#Chi2
i,chi2=0,0
while i<len(xdata) :
	index = np.argmin(np.abs(Edm1 - xdata[i]))
	chi=np.sum((ydata[i]-Fdm1[index])**2/Fdm1[index])
	chi2=chi2+chi
	i=i+1
print('\nchi2 1',chi2)

i,chi2=0,0
while i<len(xdata) :
	index = np.argmin(np.abs(Edm2 - xdata[i]))
	chi=np.sum((ydata[i]-Fdm2[index])**2/Fdm2[index])
	chi2=chi2+chi
	i=i+1
print('\nchi2 2',chi2)

#option, finalstate, channel, Jfactor, boost = 'e2', 'new', 'b', 1.7e19, 1
#(Edm,Fdm) = curve_fit(f, xdata, ydata)
#ax.plot(Edm, Fdm, label="fit to DM spectrum", color='red', linewidth=1)

plt.show()