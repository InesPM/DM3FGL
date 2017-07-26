import numpy as np
from math import *
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import Table, Column
import pylab as pl
import scipy as sp
from scipy.stats import chi2
import bisect
from scipy.interpolate import interp1d
from scipy.interpolate import spline
import scipy.optimize as opt
from matplotlib import pyplot as plt
import datetime as dt

############################
### Opening the fit file ###
############################

hdulist = fits.open('b3FGL2.fit')
header  = hdulist[1].header
data    = hdulist[1].data
t       = Table(data)
hdulist.close()

evals = np.logspace(np.log10(6)-13,2,20000)

datas    = np.array(data)
name     = np.array(data[:]['Source_Name'])
chisq    = np.array(data[:]['chi_square'])
masses   = np.array(data[:]['mass'])
unc_mass = np.array(data[:]['unc_mass'])
Jf       = np.array(data[:]['J_factor'])
unc_Jf   = np.array(data[:]['unc_J_factor'])
assoc1   = np.array(data[:]['ASSOC1'])
assoc2   = np.array(data[:]['ASSOC2'])
var      = np.array(data[:]['Variability_Index'])
glat     = np.array(data[:]['GLAT'])
P        = np.array(data[:]['Probability'])

a,b=0,0
newP=[]; newchisq=[]
while a<3034 :
	if np.isnan(P[a]) == False: # and np.isinf(chisq[a]) == False:
		newP.append(P[a])
		#newchisq.append(chisq[a])
	a = a+1
newP     = np.array(newP)
newchisq = np.array(newchisq)

######################
### Histogram plot ###
######################


n_bins = 100

fig, axes = plt.subplots(nrows=1, ncols=1)
ax = axes
#colors = ['red', 'tan', 'lime']
ax.hist(newP, n_bins, normed=False, histtype='step', stacked=True, fill=True,label='probability')
#ax.legend(prop={'size': 10})
ax.set_title('Probability value distribution')

fig.tight_layout()


plt.show()
