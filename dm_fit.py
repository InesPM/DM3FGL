import numpy as np
from math import *
from astropy.io import fits
from astropy.table import Table, Column
import pylab as pl
import bisect
import scipy as sp
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import spline
from scipy.stats import chi2
from matplotlib import pyplot as plt
import datetime as dt


#############
### Paths ###
#############

pl.rcParams['figure.figsize'] = (10.0, 7.0)
pl.rcParams['font.size'] = 18
pl.rcParams['font.family'] = 'serif'
pl.rcParams['lines.linewidth'] = 3


pathforfigs ='~/home/ines/Desktop/DM/TFGmodif/whole/'
pathforaux='~/home/ines/Desktop/DM/TFGmodif/talos/TFGmodif'
#filename=pathforaux+'/CascadeSpectra/Spectra/AtProduction_gammas.dat'
filename='AtProduction_gammas.dat'
path=pathforaux+'/sensitivities/'
path3FGL='~/home/ines/Desktop/DM/TFGmodif'

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
	#channel    = 'W'
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

	print(a, finalstate, '- mass:',mass,', J:',Jfactor,'\n')   
	
	if (mass < np.max(massvals) and mass > np.min(massvals)) :
		min = np.min(np.abs( (massvals - mass) /mass))
		index = np.where(np.abs( (massvals - mass) /mass) == min)
		xvals = 10**(data["Log10x"][index])
	else :
		print('\nError: mass out of range\n')
        
		

	def branchingratios(m_branon): #<sigmav>_particle / <sigmav>_total
	#PhysRevD.68.103505		#GeV/c**2
		m_top = masses['t']
		m_W   = masses['W']
		m_Z   = masses['Z']
		m_h   = masses['h']
		m_c   = masses['c']
		m_b   = masses['b']
		m_tau = masses['Tau']
		#if channel == None:
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

      
		loadspec_h   = interp1d(xvals,flux_h)
		loadspec_Z   = interp1d(xvals,flux_Z)
		loadspec_t   = interp1d(xvals,flux_t)
		loadspec_W   = interp1d(xvals,flux_W)
		loadspec_b   = interp1d(xvals,flux_b)
		loadspec_c   = interp1d(xvals,flux_c)
		loadspec_tau = interp1d(xvals,flux_tau)

	else:
		flux = data[finalstate][index]/(np.log(10)*xvals) #data is given in dN/d(log10(X)) = x ln10 dN/dx
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
	xvals2 = []
	if  option is 'e': #and boost > 1:
		#if mass == 5000:
		sigmavboost = sigmav * boost #no era necesario
		file1 = open("tabla"+str(mass)+str(finalstate)+str(sigmavboost)+".txt","w")
		
	xvalsnew = evals/(mass*GeV)
	logxvalsnew = np.log10(xvalsnew)

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
		return dNde
	if option is 'e2':
		return e2dNde
	if option is 'x':
		return vdNdx
	if option is 'x2':
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

##-----------------##
##- chi2 function -##
##-----------------##

def chi2(c) :			
	#E,nuFnu from the 3FGL catalog, Edm,nuFnudm from the getDMspectrum function, they must be given

	nuFnudm = getDMspectrum(evals, mass=mass, Jboost=Jboost)

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

	return c2


####################
### Main program ###
####################

timei = dt.datetime.now()

##----------------------------------##
##- Chosen points from the catalog -##
##----------------------------------##

##Opening and closing the catalog
hdulist = fits.open('b3FGL.fit')
header  = hdulist[1].header
data    = hdulist[1].data
t       = Table(data)
hdulist.close()

hdulist2 = fits.open('new3FGL.fit')
header2  = hdulist2[1].header
data2    = hdulist2[1].data
t2       = Table(data2)
hdulist2.close()

mass_b       = t[:]['mass']
unc_mass_b   = t[:]['unc_mass']
J_b          = t[:]['J_factor']
unc_J_b      = t[:]['unc_J_factor']
chi_square_b = t[:]['chi_square']

mass_new       = t2[:]['mass']
unc_mass_new   = t2[:]['unc_mass']
J_new          = t2[:]['J_factor']
unc_J_new      = t2[:]['unc_J_factor']
chi_square_new = t2[:]['chi_square']

name=t[:]['Source_Name']

##--------------##
##- Dictionary -##
##--------------##

masses = {'W': 80.4, 'Tau': 1.7768, 'b': 4.18, 't': 172.44,  'Z': 91.2, 'h': 125.1, 'c': 1.275}

fst = ['W','Tau','t','Z','h','c']		#finalstate values

##---------------##
##- Some arrays -##
##---------------##

##evals
evals = np.logspace(np.log10(6)-13,2,20000)

##new catalog parameters
mass_a     = np.empty((len(fst)+2,3034,))
unc_mass_a = np.empty((len(fst)+2,3034,))
J_a        = np.empty((len(fst)+2,3034,))
unc_J_a    = np.empty((len(fst)+2,3034,))
chi2_a     = np.empty((len(fst)+2,3034,))
P_a        = np.empty((len(fst)+2,3034,))

##---------------------------------##
##- Analysis of different sources -##
##---------------------------------##

##Creation of a document containing mass, Jfactor and chi2 of each source
f = open('DM3FGL.dat','a')

a=0

while a<3034:
	##Source name
	Source = name[a]
	print('\n\n\n\n\n','-------',a,'-------','\n',Source)

	##Spectral energy distribution##
	(nuFnu,flux,unc_fluxm,unc_fluxp,unc_num,unc_nup,E,Emin,Emax) = nu(Source)

	df = len(nuFnu)-1
	
	print('\nSpectral energy distribution',nuFnu)
	print('\nError bars\n',unc_num,'\n',unc_nup)
	print('\n\n')

	f.write(Source)

	if len(nuFnu)>0:

		df = int(len(nuFnu)-1)	

		fig=pl.figure(num=a)

		ax=fig.add_subplot(111)
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.set_xlim(1e-4, 0.1)
		ax.set_ylim(5e-20,1e-10)
		plt.suptitle(Source,fontsize=18)
		#ax.set_title(comment,fontsize=10)
		ax.set_xlabel('$E$ [TeV]')
		ax.set_ylabel('$E^2 dN/dE$ [erg cm$^{-2}$ s$^{-1}$]')
	
		ax.errorbar(E, nuFnu, xerr=[Emin,Emax], yerr=[unc_num,unc_nup], fmt='--o', linewidth=1, label="data")
	
		finalstate = 'b'
		comment = 'b: mass = $%.2f\pm%.2f$ GeV, J = $%.2e\pm%.2e$, $\chi^2 = %.4e$' % (mass_b[a], unc_mass_b[a], J_b[a], unc_J_b[a], chi_square_b[a])
		Fdm = getDMspectrum(evals, mass_b[a], J_b[a]/(1.7e19))
		ax.plot(evals, Fdm, label=comment, linewidth=1)
		plt.legend(loc=3, prop={'size':10})

	#finalstate = 'new'
	#comment = 'BWS: mass = $%.2f\pm%.2f$ GeV, J = $%.2e\pm%.2e$, $\chi^2 = %.4e$' % (mass_new[a], unc_mass_new[a], J_new[a], unc_J_new[a], chi_square_new[a])
	#Fdm = getDMspectrum(evals, mass_new[a], J_new[a]/(1.7e19))
	#ax.plot(evals, Fdm, label=comment, linewidth=1)
	#plt.legend(loc=3, prop={'size':10})	

	##-------------##
	##- Curve fit -##
	##-------------##
	
	g=0

	#while g<len(fst):

	while g<len(fst):
		finalstate = str(fst[g])
		

		if len(nuFnu)>0 :
			s = (unc_num+unc_nup)/2
			#print(s.shape)

			
			
			#mmin = float(np.maximum(5,masses[finalstate]))
			bnds = [(np.maximum(5,masses[finalstate]), 1),(1e3, 1e5)]
			CI = (masses[finalstate]+40,1)
			#CI = differential_evolution(getDMspectrum, [(np.maximum(5,masses[finalstate]), 1e3),(1, 1e3)], seed=3)
	
			popt, pcov = curve_fit(getDMspectrum, xdata=E, ydata=nuFnu, p0=CI, sigma=s, absolute_sigma=True, bounds=bnds)
			mass    = popt[0]
			Jboost  = popt[1]
			merr    = sqrt(pcov[0][0])
			Jerr    = sqrt(pcov[1][1])	
			Jfactor = Jboost*1.7e19
			Jferr   = Jerr*1.7e19

			X2      = float(chi2(getDMspectrum))
			print('type X2:',type(X2),'type df:',type(df))
			P       = chi2.sf(X2,df)

			#############
			### plots ###
			#############
	

			comment = finalstate+': mass = $%.2f\pm%.2f$ GeV, J = $%.2e\pm%.2e$, $\chi^2 = %.4e$' % (mass, merr, Jfactor, Jferr, X2)
	
			Fdm = getDMspectrum(evals, *popt)
			ax.plot(evals, Fdm, label=comment, linewidth=1)
			plt.legend(loc=3, prop={'size':10})	
	
		else :
			mass    = 0
			merr    = 0
			Jfactor = 0
			Jferr   = 0
	
			X2      = inf
			P       = 0
	
			print('Not fittable')
			
		mass_a[g][a]     = mass
		unc_mass_a[g][a] = merr
		J_a[g][a]        = Jfactor
		unc_J_a[g][a]    = Jferr
		chi2_a[g][a]     = X2
		P_a[g][a]        = P
	
		print('mass: ',mass_a[g][a],', J:',J_a[g][a],'\nmerr:',unc_mass_a[g][a],', Jerr:',unc_J_a[g][a],', chi2:',chi2_a[g][a])
	
		f.write(' %13.10f %13.10f %17.10e %17.10e %13.10f' % (mass, merr, Jfactor, Jferr, X2))	#+str(mass)+' '+str(merr)+' '+str(Jfactor)+' '+str(Jferr)+' '+str(X2))
	
		time = dt.datetime.now()
		print('\nTime:',time-timei)
	
		g=g+1

	#b and new probabilities

	P_a[len(fst)+0][a] = chi2.sf(chi_square_b[a],df)
	P_a[len(fst)+1][a] = chi2.sf(chi_square_new[a],df)

	#plt.savefig('Figure_%.0f.png' % (a))

	f.write('\n')
	a=a+1


#plt.show()

f.close()



#####################
### New fits file ###
#####################

##------------##
##- Creation -##
##------------##

## b annihilation source (already in the catalog)

t.rename_column('mass', 'mass_b')
t.rename_column('unc_mass', 'unc_mass_b')
t.rename_column('J_factor', 'J_factor_b')
t.rename_column('unc_J_factor', 'unc_J_factor_b')
t.rename_column('chi_square', 'chi_square_b')

P_c = Column(name = 'Probability_b', data = P_a[len(fst)][:], format = 'E')

t.add_columns([P_c])



## Annihilation sources

g=0
while g<len(fst):

	finalstate = str(fst[g])
	#print(finalstate)
	
	mass_c     = Column(name = 'mass_'         +finalstate, data = mass_a[g][:],     format = 'E')
	unc_mass_c = Column(name = 'unc_mass_'     +finalstate, data = unc_mass_a[g][:], format = 'E')
	J_c        = Column(name = 'J_factor_'     +finalstate, data = J_a[g][:],        format = 'E')
	unc_J_c    = Column(name = 'unc_J_factor_' +finalstate, data = unc_J_a[g][:],    format = 'E')
	chi2_c     = Column(name = 'chi_square_'   +finalstate, data = chi2_a[g][:],     format = 'E')
	P_c        = Column(name = 'Probability_'  +finalstate, data = P_a[g][:],        format = 'E')

	t.add_columns([mass_c, unc_mass_c, J_c, unc_J_c, chi2_c, P_c])

	mass_c     = None
	unc_mass_c = None
	J_c        = None
	unc_J_c    = None
	chi2_c     = None
	P_c        = None

	g=g+1

## New annihilation source

mass_c     = Column(name = 'mass_BWS',         data = np.array(mass_new),       format = 'E')
unc_mass_c = Column(name = 'unc_mass_BWS',     data = np.array(unc_mass_new),   format = 'E')
J_c        = Column(name = 'J_factor_BWS',     data = np.array(J_new),          format = 'E')
unc_J_c    = Column(name = 'unc_J_factor_BWS', data = np.array(unc_J_new),      format = 'E')
chi2_c     = Column(name = 'chi_square_BWS',   data = np.array(chi_square_new), format = 'E')
P_c        = Column(name = 'Probability_BWS',  data = P_a[len(fst)+1][:],       format = 'E')

t.add_columns([mass_c, unc_mass_c, J_c, unc_J_c, chi2_c, P_c])

t.write('DM3FGL.fit', overwrite=True)

'''
##--------##
##- Test -##
##--------##

hdulist2 = fits.open('DM3FGL.fit')
header2 = hdulist[1].header
data2 = hdulist2[1].data

a=15

print('b',  'mass',data2[a]['mass_b'],   'J', data2[a]['J_factor_b'],   'chi2', data2[a]['chi_square_b'],   'P', data2[a]['Probability_b'])
print('W',  'mass',data2[a]['mass_W'],   'J', data2[a]['J_factor_W'],   'chi2', data2[a]['chi_square_W'],   'P', data2[a]['Probability_W'])
print('Tau','mass',data2[a]['mass_Tau'], 'J', data2[a]['J_factor_Tau'], 'chi2', data2[a]['chi_square_Tau'], 'P', data2[a]['Probability_Tau'])
print('t',  'mass',data2[a]['mass_t'],   'J', data2[a]['J_factor_t'],   'chi2', data2[a]['chi_square_t'],   'P', data2[a]['Probability_t'])
print('Z',  'mass',data2[a]['mass_Z'],   'J', data2[a]['J_factor_Z'],   'chi2', data2[a]['chi_square_Z'],   'P', data2[a]['Probability_Z'])
print('h',  'mass',data2[a]['mass_h'],   'J', data2[a]['J_factor_h'],   'chi2', data2[a]['chi_square_h'],   'P', data2[a]['Probability_h'])
print('c',  'mass',data2[a]['mass_c'],   'J', data2[a]['J_factor_c'],   'chi2', data2[a]['chi_square_c'],   'P', data2[a]['Probability_c'])
print('BWS','mass',data2[a]['mass_BWS'], 'J', data2[a]['J_factor_BWS'], 'chi2', data2[a]['chi_square_BWS'], 'P', data2[a]['Probability_BWS'])


hdulist2.close()

'''
