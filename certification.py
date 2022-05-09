import numpy as np
import scipy 
from scipy import signal
import math
import matplotlib.pyplot as plt
from sympy import symbols, solve, erf
files=np.linspace(2,20,7,dtype=int)
d= np.loadtxt("en_All.csv",dtype=float,delimiter=',',skiprows=13)
c1=d[:,1]
c1=c1-np.mean(c1)
c2=d[:,2]
ENOB=8.8
h=6.63e-34
c=3e8
print('input impedence= ')
R=int(input())
if(R==50):
    BW1=0.1e9
else:
    BW1=0.1e9/R
BW2=0.1e9;
etac=0.9;
etad=0.9;
Gc=3.9e3;
Gd=39000;
wl=1550e-9;
r1=0.09
dV=0.08/2**ENOB
alphac=h*c*BW1*etac*Gc/wl;
alphad=h*c*BW2*etad*Gd/wl;
error=float(input('error probability= '))
t = symbols('t')
fun1 = 1-erf(t)-error
out=solve(fun1)
print(out)
mu_en,sigma_en=scipy.stats.norm.fit(c1)
lambda_bar=2**(0.5)*sigma_en*(out[0]);

l1=dV/(2*alphad)
l2=-(dV/(2*alphad))-1
pmax=np.zeros(len(files))
nrm=np.zeros(len(files))
ncm=np.zeros(len(files))
Mean_P=np.zeros(len(files))
Pguess=np.zeros(len(files))
f,psd_en=scipy.signal.welch(c1, fs=125e6, window='hann', nperseg=3000, noverlap=0.5, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
s= (7,48)
Var_P=np.zeros(s)
VarP=np.zeros(len(files))
stdP=np.zeros(len(files))

for i in files:
    print(i)
    a = np.loadtxt(str(i)+"_All.csv",dtype=float,delimiter=',',skiprows=13)
    b1=a[:,1]
    b2=a[:,2]
    b1=b1-np.mean(b1)
    b2=b2-np.mean(c2)
    ncm[int((i/2)-1)]=(np.mean(b2)-lambda_bar)/alphac;
    if ncm[int((i/2)-1)]<0:
        nrm[int((i/2)-1)]=0
        ncm[int((i/2)-1)]=0
    else:
        nrm[int((i/2)-1)]=4*ncm[int((i/2)-1)]*r1+4*r1**2-4*ncm[int((i/2)-1)]*r1**2-np.log10(0.5*error)+(-np.log10(0.5*error))**0.5*(8*ncm[int((i/2)-1)]*r1-np.log(0.5*error))**0.5/4*r1**2;

    f,psd_b=scipy.signal.welch(b1, fs=125e6, window='hann', nperseg=3000, noverlap=0.5, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
   
   
    for j in np.arange(1,48,1):
        Var_P[int((i/2)-1),j]=np.var(b1[int(np.ceil(len(b1)/1000)*(j-1)+1):int(np.ceil(len(b1)/1000)*j)])-np.var(c2[int(np.ceil(len(b1)/1000)*(j-1)+1):int(np.ceil(len(b1)/1000)*j)])
    VarP[int((i/2)-1)]=np.mean(Var_P[int((i/2)-1),0:48]);
    stdP[int((i/2)-1)]=np.std(Var_P[int((i/2)-1),0:48]);
    Mean_P[int((i/2)-1)]=np.mean(b2)
    if i==2:
        psd_ref=psd_b
        ref2=b2;
    SNL=(psd_b-psd_en)/(psd_ref-psd_en)
    SNL_the=np.zeros(len(f))+np.mean(b2)/np.mean(ref2)    
    x_values=np.arange(np.min(b1),np.max(b1),(np.max(b1)-np.min(b1))/2**ENOB) 
    mu,sigma=scipy.stats.norm.fit(b1)
    l3=(nrm[int((i/2)-1)]/4)**0.5;
    Pguess[int((i/2)-1)]=0.5*(erf(l1/l3)-erf(l2/l3));
    best_fit_line = scipy.stats.norm.pdf(x_values, mu, sigma)
    
    pmax[int((i/2)-1)]=np.max(best_fit_line)*(x_values[2]-x_values[1])
    plt.figure(2)
    plt.plot(x_values, best_fit_line,label="{:.3f} mW".format(np.mean(b2)))
    plt.legend(loc="best")
    plt.figure(1)
    plt.plot(f,10*np.log10(SNL))
    plt.plot(f,10*np.log10(SNL_the))
    plt.legend(["SNL", "SNL_the"])
   
    plt.xlim(0,50e6)
plt.savefig('SNL.png', dpi=600)
plt.figure(3)
plt.plot(Mean_P, -np.log2(Pguess),'o',label='$\\kappa$')
plt.plot(Mean_P, -np.log2(pmax),'+',label='$H_{min}(X|E)$')
plt.legend(loc="upper right")
plt.figure(4)
plt.plot(Mean_P, nrm,'o',label='$n_{rm}$')
plt.show()
A = np.polyfit(np.sort(Mean_P[1:8]),np.sort(VarP[1:8]), 1)
plt.figure(5)
plt.plot(np.sort(Mean_P[1:8]),A[0]*np.sort(Mean_P[1:8])+A[1])
plt.errorbar(np.sort(Mean_P[1:8]),np.sort(VarP[1:8]),yerr=np.sort(stdP[1:8]),xerr=None,fmt='o')
