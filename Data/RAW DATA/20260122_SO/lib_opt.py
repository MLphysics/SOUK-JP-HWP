import math
import cmath
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from natsort import natsorted
from glob import glob
from tqdm import tqdm

def trans_ref_multilayer(freq, n, losstan, d, angle_in=0., incpol=1):

    pi = np.pi
    c = 299792458.         # [m/s]
    ep0 = 8.85418782e-12   # [Fm^-1]=[C^2 N^-1 m^-2] dielectric const in vaccum
    mu0 = 4 * pi * 1e-7    # [N A-2]
    Z0 = np.sqrt(mu0/ep0)  # Impedance of free space
    # ------------------------------------------------------------------------------------
    # Transmittance and Reflection caluculation assuming the configuration of | vacuum | sample | vacuum |
    # n: reflective index (array)
    # losstan: loss tangent (array)
    # d: tickness [m] (array)
    # freq: frequency [Hz] (array)
    # angle_in: incident angle [rad.] (not array)
    # incpol: incident polarization, 1=s-wave, -1=p-wave 
    # ------------------------------------------------------------------------------------
    
    # the number of layer
    num=len(d) 
    
    n.insert(0,1.)
    n.append(1.)
    losstan.insert(0,0.)
    losstan.append(0.)

    # define refraction angle
    angle = np.zeros(num+2)
    angle[0] = angle_in
    for i in range(0,num+1): 
        angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

    # define reflective index
    n_comparr = np.zeros(len(n),'complex')
    n_comparr[0] = complex(n[0], -0.5*n[0]*losstan[0])
    n_comparr[num+1] = complex(n[num+1], -0.5*n[num+1]*losstan[num+1])

    # define effective thickness
    h = np.zeros(num,'complex')

    # define output
    l = len(freq)
    output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)

    # frequency loop
    for j in range(0,l):
        # loop for each layer
        for i in range(0,num): 
            n_comparr[i+1] = complex(n[i+1], -0.5*n[i+1]*losstan[i+1])
            h[i] = n_comparr[i+1]*d[i]*np.cos(angle[i+1])

        f = freq[j]
        k = 2.*np.pi*f/c
        
        # ===========================================
        # Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
        Y = np.zeros(num+2,'complex')
        for i in range(0,num+2):
            if (incpol == 1):
                Y[i] = (1/Z0)*n_comparr[i]*np.cos(angle[i])
                cc = 1.
            if (incpol == -1):
                Y[i] = (1/Z0)*n_comparr[i]/np.cos(angle[i])
                cc = np.cos(angle[num+1])/np.cos(angle[0])

        # ===========================================
        # define matrix for each layer
        m = np.identity((2),'complex') # dot matrix
        me = np.zeros((2,2),'complex') # matrix for each layer 
        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
            me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
            me[1,1] = complex(np.cos(k*h[i]), 0.)
            m = np.dot(m,me)

        r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        
        output[0,j] = f+0.j 
        output[1,j] = r
        output[2,j] = t

    return output

def trans_ref_3layer(freq, n1, n2, n3, losstan1, losstan2, losstan3, d1, d2, d3, angle_in=0., incpol=1):

    n = [n1, n2, n3, n2, n1]
    losstan = [losstan1, losstan2, losstan3, losstan2, losstan1]
    d = [d1, d2, d3, d2, d1]
    
    pi = np.pi
    c = 299792458.         # [m/s]
    ep0 = 8.85418782e-12   # [Fm^-1]=[C^2 N^-1 m^-2] dielectric const in vaccum
    mu0 = 4 * pi * 1e-7    # [N A-2]
    Z0 = np.sqrt(mu0/ep0)  # Impedance of free space
    # ------------------------------------------------------------------------------------
    # Transmittance and Reflection caluculation assuming the configuration of | vacuum | sample | vacuum |
    # n: reflective index (array)
    # losstan: loss tangent (array)
    # d: tickness [m] (array)
    # freq: frequency [Hz] (array)
    # angle_in: incident angle [rad.] (not array)
    # incpol: incident polarization, 1=s-wave, -1=p-wave 
    # ------------------------------------------------------------------------------------
    
    # the number of layer
    num=len(d) 
    
    n.insert(0,1.)
    n.append(1.)
    losstan.insert(0,0.)
    losstan.append(0.)

    # define refraction angle
    angle = np.zeros(num+2)
    angle[0] = angle_in
    for i in range(0,num+1): 
        angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

    # define reflective index
    n_comparr = np.zeros(len(n),'complex')
    n_comparr[0] = complex(n[0], -0.5*n[0]*losstan[0])
    n_comparr[num+1] = complex(n[num+1], -0.5*n[num+1]*losstan[num+1])

    # define effective thickness
    h = np.zeros(num,'complex')

    # define output
    l = len(freq)
    output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)

    # frequency loop
    for j in range(0,l):
        # loop for each layer
        for i in range(0,num): 
            n_comparr[i+1] = complex(n[i+1], -0.5*n[i+1]*losstan[i+1])
            h[i] = n_comparr[i+1]*d[i]*np.cos(angle[i+1])

        f = freq[j]
        k = 2.*np.pi*f/c
        
        # ===========================================
        # Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
        Y = np.zeros(num+2,'complex')
        for i in range(0,num+2):
            if (incpol == 1):
                Y[i] = (1/Z0)*n_comparr[i]*np.cos(angle[i])
                cc = 1.
            if (incpol == -1):
                Y[i] = (1/Z0)*n_comparr[i]/np.cos(angle[i])
                cc = np.cos(angle[num+1])/np.cos(angle[0])

        # ===========================================
        # define matrix for each layer
        m = np.identity((2),'complex') # dot matrix
        me = np.zeros((2,2),'complex') # matrix for each layer 
        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
            me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
            me[1,1] = complex(np.cos(k*h[i]), 0.)
            m = np.dot(m,me)

        r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        
        output[0,j] = f+0.j 
        output[1,j] = r
        output[2,j] = t

    return output


def fit_transmittance_singlelayer(freq, n, losstan, d):
    output = trans_ref_multilayer(freq, [n], [losstan], [d])
    return np.abs(output[2])**2

def fit_transmittance_singlelayer_angle(freq, n, losstan, d, angle_in):
    output = trans_ref_multilayer(freq, [n], [losstan], [d], angle_in)
    return np.abs(output[2])**2

def fit_reflectance_singlelayer(freq, n, losstan, d,angle_in=0):
    output = trans_ref_multilayer(freq, [n], [losstan], [d],angle_in)
    return np.abs(output[1])**2

def fit_reflectance_twolayer(freq, n1, n2, losstan1, losstann2, d1, d2, angle_in=0):
    output = trans_ref_multilayer(freq, [n1,n2], [losstan1,losstan2], [d1,d2], angle_in)
    return np.abs(output[1])**2

def save_excel(freq,data,savef):
    df = pd.DataFrame([freq,data,np.power(10,np.array(data)/10)],index = ['freq','data','trans'])
    df_inv = df.T
    df_inv.to_excel(savef,header = True)



class Fit_data_lib:
    def __init__(self,incpol,i_angle,thickness,n_init,los_init,bound,d_arr,f_arr,npzname,datanum):
        self.incpol= incpol # (1: s-pol, -1: p-pol)
        self.i_angle = i_angle
        self.thickness = np.array([thickness]) # mm
        self.inc_angle = i_angle # deg.
        self.n_init = n_init # Refractive index
        self.los_init = los_init # Loss tangent
        self.bound = bound
        self.pas = 'C:/Users/IPMU/Analysis/VNA/test'
        self.d_arr = d_arr
        self.f_arr = f_arr
        self.npzname = npzname
        self.datanum = datanum
        
    def RMS(self,data):
        rms = np.sqrt(np.sum(data**2)/float(len(data)))
        return rms
    
    
    def data_load(self):
        freq = np.array([])
        trans = np.array([])
    
        for i in range(0,len(self.f_arr)):
            m = np.load(self.pas+'/'+self.d_arr[i]+'/'+self.f_arr[i])
            xylab = m.files
            freq = np.hstack((freq,m[xylab[0]]))
            trans = np.hstack((trans,10**(m[xylab[1]]/10.)))

        trans_data = trans[np.argsort(freq)]
        freq_data= np.sort(freq)*1e+9
        return freq_data,trans_data
    
    def data_load_all_format(self):
        freq = np.array([])
        trans = np.array([])
        phase = np.array([])
        real = np.array([])
        imag = np.array([])
    
        for i in range(0,len(self.f_arr)):
            m = np.load(self.pas+'/'+self.d_arr[i]+'/'+self.f_arr[i]+'.npz')
            xylab = m.files
            freq = np.hstack((freq,m['freq']))
            trans = np.hstack((trans,m['trans']))
            phase = np.hstack((phase, m['phase']))
            real = np.hstack((real, m['real']))
            imag = np.hstack((imag, m['imag']))
        freq_order = np.argsort(freq)
        freq_data= np.sort(freq)*1e+9
        
        trans_data = trans[freq_order]
        phase_data = phase[freq_order]
        real_data = real[freq_order]
        imag_data = imag[freq_order]
        return freq_data,trans_data, phase_data,real_data,imag_data
    
    def Func(self, n, losstan, d, freq_in, angle_i, incpol):
        c = 2.9979e8
        num=len(d) #; the number of layer not including two ends
        const = np.sqrt((8.85e-12)/(4.*np.pi*1e-7)) #SI unit sqrt(dielectric const/permiability)
        # ;-----------------------------------------------------------------------------------
        # ; angle of refraction
        angle = np.zeros(num+2)          # ; angle[0]=incident angle
        angle[0] = angle_i
        for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])
        # ;-----------------------------------------------------------------------------------
        # ; define the frequency span
        l = len(freq_in)
        output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)
        # ;-----------------------------------------------------------------------------------
        # ; define the effective thickness of each layer
        h = np.zeros(num,'complex')
        n_comparr = np.zeros(len(n),'complex')
        n_comparr[0] = complex(n[0], -0.5*n[0]*losstan[0])
        n_comparr[num+1] = complex(n[num+1], -0.5*n[num+1]*losstan[num+1])
        # ;-----------------------------------------------------------------------------------
        # ; for loop for various thickness of air gap between each layer
        for j in range(0,l):
            for i in range(0,num): 
                n_comparr[i+1] = complex(n[i+1], -0.5*n[i+1]*losstan[i+1])
                h[i] = n_comparr[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer

            freq = freq_in[j]
            k = 2.*np.pi*freq/c

            # ;===========================================
            # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
            Y = np.zeros(num+2,'complex')
            for i in range(0,num+2):
                if (incpol == 1):
                    Y[i] = const*n_comparr[i]*np.cos(angle[i])
                    cc = 1.
                if (incpol == -1):
                    Y[i] = const*n_comparr[i]/np.cos(angle[i])
                    cc = np.cos(angle[num+1])/np.cos(angle[0])

            # ;===========================================
            # ; define matrix for single layer
            m = np.identity((2),'complex')    # ; net matrix
            me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
            for i in range(0,num):
                me[0,0] = complex(np.cos(k*h[i]), 0.)
                me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
                me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
                me[1,1] = complex(np.cos(k*h[i]), 0.)
                m = np.dot(m,me)

            r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
            t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

            output[0,j] = freq+0.j #; unit of [Hz]
            output[1,j] = r
            output[2,j] = t

        return output

    def Transmittance_FitIndex_1layer_incloss(self, freq, par0, par1): #[Hz]
        if par0 < 1: return 1e30
        if par1 < 0: return 1e30
        n = np.array([1.,par0,1.])
        losstan = np.array([0.,par1,0.])
        angle_i = np.radians(self.inc_angle)
        RT = self.Func( n, losstan, self.thickness, freq, angle_i, self.incpol)
        return np.abs(RT[2])**2
    
    def Transmittance_FitIndex_1layer_incloss_all_format(self, freq, par0, par1): #[Hz]
        if par0 < 1: return 1e30
        if par1 < 0: return 1e30
        n = np.array([1.,par0,1.])
        losstan = np.array([0.,par1,0.])
        angle_i = np.radians(self.inc_angle)
        RT = self.Func( n, losstan, self.thickness, freq, angle_i, self.incpol)
        return np.real(RT[2]),np.imag(RT[2]),abs(RT[2])**2


    def Data_plot(self):
        m = np.load(self.pas+'/'+self.d_arr[0]+'/'+self.npzname+'.npz')
        freq_data = m['freq_data']
        trans_data = m['trans_data']
        freq_fit = m['freq_fit']
        trans_fit = m['trans_fit']

        fit_n = m['fit_n']
        fit_los = m['fit_los']
        fit_n_err = m['fit_n_err']
        fit_los_err = m['fit_los_err']
        
        trans_diff = m['trans_diff']
        rms = self.RMS(trans_diff)


        fig = plt.figure(figsize = (16,8))
        ax = fig.add_subplot(211)
        ax.plot(freq_data,trans_data,'.',
                color = 'r',
                label = '$n=%.5f \\pm$'%fit_n+'$%.5f$\n'%fit_n_err
                +'$tan{\delta}=%.7f\pm$'%fit_los+'$%.7f$'%fit_los_err)                                          #label 表示桁　変更
        ax.plot(freq_fit,trans_fit,'-',color = 'b')

        ax.set_xlabel('Frequency [GHz]',fontsize = 18)
        ax.set_ylabel('Transmittance',fontsize = 18)
        ax.tick_params(labelsize = 15)
        ax.grid(True)
        ax.set_ylim(0.0,1.05)
        #ax.set_xlim(97,100)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=15)


        ax = fig.add_subplot(212)
        ax.plot(freq_data,trans_diff,'.',
                color = 'r',
                label = '$n=%.3f \\pm$'%fit_n+'$%.3f$\n'%fit_n_err
                +'$tan{\delta}=%.5f\pm$'%fit_los+'$%.5f$'%fit_los_err)
        ax.text(0.01,0.01,'RMS = %.4f'%(rms),transform = ax.transAxes,fontsize = 15)
        ax.set_xlabel('Frequency [GHz]',fontsize = 18)
        ax.set_ylabel('Data $-$ fit',fontsize = 18)
        ax.tick_params(labelsize = 15)
        ax.grid(True)
        #ax.set_ylim(0.0,1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=15)

        plt.tight_layout()
        plt.savefig(self.pas+'/'+self.d_arr[0]+'/'+self.npzname+'.png',dpi = 300)
        plt.show()
        
    def Test_datafit(self):
        #data_load = self.data_load()
        #freq_data = data_load[0]
        #trans_data = data_laod[1]
        
        # ========== import data ==========
        freq_data, trans_data = self.data_load()
        # =================================
        
        # ================= Fit data ================
        par_index=[self.n_init, self.los_init]
        popt, pcov = curve_fit(self.Transmittance_FitIndex_1layer_incloss, 
                               np.array(freq_data), 
                               np.array(trans_data), 
                               p0=par_index, 
                               sigma=0.03,
                               absolute_sigma=True,
                               maxfev=100000,
                               bounds=self.bound)
        print('fit n, loss, angle = {0:.4f}, {1:.4f}'.format(popt[0],popt[1]))
        print('fit error = {0:.4f}, {1:.4f}'.format(np.sqrt(np.diag(pcov))[0],np.sqrt(np.diag(pcov))[1]))
        freq_fit=np.linspace(np.min(freq_data),np.max(freq_data),self.datanum)
        trans_fit=self.Transmittance_FitIndex_1layer_incloss(freq_fit, popt[0], popt[1])
        # ===========================================
        
        # ======= data - fit =========
        trans_diff=np.array(trans_data)-self.Transmittance_FitIndex_1layer_incloss(np.array(freq_data), popt[0], popt[1])
        # =================================
        
        # ========== Save_npz =============
        np.savez(self.pas+'/'+self.d_arr[0]+'/'+self.npzname+'.npz', 
                 freq_data = freq_data*1e-9,
                 trans_data = trans_data,
                 freq_fit = freq_fit*1e-9,
                 trans_fit = trans_fit,
                 fit_n = popt[0],
                 fit_los = popt[1],
                 fit_n_err = np.sqrt(np.diag(pcov))[0], 
                 fit_los_err = np.sqrt(np.diag(pcov))[1], 
                 trans_diff = trans_diff)
        # =================================
        
        # ========== data for excel ========
        df = pd.DataFrame([freq_data*1e-9,trans_data,freq_fit*1e-9,trans_fit,np.array([popt[0]]),np.array([popt[1]]),np.array([np.sqrt(np.diag(pcov))[0]]),np.array([np.sqrt(np.diag(pcov))[1]]), trans_diff], 
                          index = ['freq_data','trans_data','freq_fit','trans_fit','fit_n','fit_los','fit_n_err','fit_los_err','trans_diff'])
        df_inv = df.T
        df_inv.to_excel(self.pas+'/'+self.d_arr[0]+'/'+self.npzname+'.xlsx',header = True)
        
        # =========== Plot (ex) ===========
        self.Data_plot()
        # =================================
        
        print('Index: [0]freq data(GHz), [1]trans_data, [2]freq fit(GHz), [3]trans fit, [4]fit n, [5]fit los, [6]fit n err, [7]fit los err, [8]trans diff')
        return freq_data*1e-9, trans_data, freq_fit*1e-9, trans_fit, popt[0], popt[1], np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1], trans_diff
    
    def Test_datafit_all_format(self):
        #data_load = self.data_load()
        #freq_data = data_load[0]
        #trans_data = data_laod[1]
        
        # ========== import data ==========
        freq_data, trans_data, phase_data, real_data,imag_data = self.data_load_all_format()
        # =================================
        
        # ================= Fit data ================
        par_index=[self.n_init, self.los_init]
        popt, pcov = curve_fit(self.Transmittance_FitIndex_1layer_incloss, 
                               np.array(freq_data), 
                               np.array(trans_data), 
                               p0=par_index,                                                                  #ここのsigmaを変える
                               absolute_sigma=False,
                               maxfev=100000,
                               bounds=self.bound)
        print('fit n, loss, angle = {0:.4f}, {1:.4f}'.format(popt[0],popt[1]))
        print('fit error = {0:.4f}, {1:.4f}'.format(np.sqrt(np.diag(pcov))[0],np.sqrt(np.diag(pcov))[1]))
        freq_fit=np.linspace(np.min(freq_data),np.max(freq_data),self.datanum)
        real_fit,imag_fit,trans_fit = self.Transmittance_FitIndex_1layer_incloss_all_format(freq_fit, popt[0], popt[1])
        
        # ===========================================
        
        # ======= data - fit =========
        trans_diff=np.array(trans_data)-self.Transmittance_FitIndex_1layer_incloss(np.array(freq_data), popt[0], popt[1])
        # =================================
        
        # ========== Save_npz =============
        np.savez(self.pas+'/'+self.d_arr[0]+'/'+self.npzname, 
                 freq_data = freq_data*1e-9,
                 trans_data = trans_data,
                 phase_data = phase_data, 
                 real_data = real_data,
                 imag_data = imag_data,
                 freq_fit = freq_fit*1e-9,
                 trans_fit = trans_fit,
                 real_fit = real_fit,
                 imag_fit = imag_fit,
                 fit_n = popt[0],
                 fit_los = popt[1],
                 fit_n_err = np.sqrt(np.diag(pcov))[0], 
                 fit_los_err = np.sqrt(np.diag(pcov))[1], 
                 trans_diff = trans_diff)
        # =================================
        
        # ========== data for excel ========
        df = pd.DataFrame([freq_data*1e-9,trans_data,freq_fit*1e-9,trans_fit, 
                           phase_data, real_data,imag_data,real_fit,imag_fit,
                           np.array([popt[0]]),np.array([popt[1]]),np.array([np.sqrt(np.diag(pcov))[0]]),np.array([np.sqrt(np.diag(pcov))[1]]), trans_diff], 
                          index = ['freq_data','trans_data','freq_fit','trans_fit', 
                                   'phase_data', 'real_data','imag_data','real_fit','imag_fit',
                                   'fit_n','fit_los','fit_n_err','fit_los_err','trans_diff'])
        df_inv = df.T
#        df_inv.to_excel(self.pas+'/'+self.d_arr[0]+'/'+self.npzname+'.xlsx',header = True)                            # put "#" if you don't need excel file 
        
        # =========== Plot (ex) ===========
        self.Data_plot()
        # =================================
        
        print('Index: [0]freq data(GHz), [1]trans_data, [2]freq fit(GHz), [3]trans fit, [4]fit n, [5]fit los, [6]fit n err, [7]fit los err, [8]trans diff')
        return freq_data*1e-9, trans_data, freq_fit*1e-9, trans_fit, popt[0], popt[1], np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1], trans_diff    
    
    
def Plot_measurement_result_all_format(df,sname):
    fig = plt.figure(figsize = (15,8))

    form_label = ['trans','REAL','IMAG','PHAS']
    form = ['trans','Real [mU]','Imag [mU]','Phase [degrees]']

    for i in range(0,4):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(df['freq'],df[form_label[i]],'b.-')
        ax.set_ylabel(form[i],fontsize = 15)
        ax.set_xlabel('Frequency [GHz]',fontsize = 15)
        ax.tick_params(labelsize = 13)
        ax.grid(True)
        #ax.legend()
    fig.tight_layout()
    plt.savefig(sname+'.png',dpi = 300)
    plt.show()
    plt.close()
    return df





class Dataanalysis_lib:
    def __init__(self,global_dir,save_dir,data_dir,f_name,freq_num,rot_num):
        self.global_dir = global_dir
        self.data_dir = data_dir
        self.f_name = f_name
        self.freq_num = freq_num
        self.rot_num = rot_num
        self.rot_angle = np.linspace(0,np.pi*2,self.rot_num)
        self.rot_angle_high_res = np.linspace(0,np.pi*2,10001)
        self.save_dir = global_dir + save_dir
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
    def Data_load(self):
        """
        output: 
            - frequency [GHz], 1D arr
            - transmittance, 2D arr [rotation angle, frequency]
        """
        f_arr = natsorted(glob(self.global_dir+self.data_dir+self.f_name + '_MLOG_*.npz'))
        M = np.empty([self.rot_num,self.freq_num])
        for i in range(0,len(f_arr)):
            m = np.load(f_arr[i])
            M[i] = 10**(m['data']/10.)
            if i == 0:
                freq = m['freq']
        M = np.transpose(M)
        return freq, M
    
    def Fit_function_f8(self,x,
                        a0,a1,a2,a3,a4,a5,a6,a7,a8,
                        b1,b2,b3,b4,b5,b6,b7,b8):
        """
        output: 
            - modulated signal, 1D arr
        """
        return  a0 +\
                a1*np.cos(1.*x + 1.*b1) +\
                a2*np.cos(2.*x + 2.*b2) +\
                a3*np.cos(3.*x + 3.*b3) +\
                a4*np.cos(4.*x + 4.*b4) +\
                a5*np.cos(5.*x + 5.*b5) +\
                a6*np.cos(6.*x + 6.*b6) +\
                a7*np.cos(7.*x + 7.*b7) +\
                a8*np.cos(8.*x + 8.*b8)
    
    def Calculate_modulated_signal_multiple_freq(self,rot_angle,fit):
        """
        output: 
            - modulated signal, 2D arr
        """
        trans_fit_arr = np.empty([self.freq_num,len(rot_angle)])
        for i in tqdm(range(0,self.freq_num),desc = 'Calculate modulated signal at measured frequencies...'):
            trans_fit_arr[i] = self.Fit_function_f8(rot_angle, 
                                                 fit[i][0],fit[i][1],fit[i][2],fit[i][3],fit[i][4],
                                                 fit[i][5],fit[i][6],fit[i][7],fit[i][8],fit[i][9],
                                                 fit[i][10],fit[i][11],fit[i][12],fit[i][13],
                                                 fit[i][14],fit[i][15],fit[i][16])
        return trans_fit_arr
    
    def Fit_result(self,x_angle,ydata):
        """
        output:
            - fit result, 1D arr
            - fit err, 1D arr
        """
        degrad = np.radians(0)
        b_phase = np.radians(20)
        pa = [0.5,0,0.01,0,0.5,0,0,0,0, degrad,degrad,degrad,degrad,degrad,degrad,degrad,degrad]
        popt, pcov = curve_fit(self.Fit_function_f8,x_angle,ydata,p0 = pa,maxfev=100000,
                               bounds = ((-1,-1,-1,-1,-1,-1,-1,-1,-1,-b_phase,-b_phase,-b_phase,-b_phase,-b_phase,-b_phase,-b_phase,-b_phase),
                                         (1,1,1,1,1,1,1,1,1,b_phase,b_phase,b_phase,b_phase,b_phase,b_phase,b_phase,b_phase)))
        err = np.sqrt(np.diag(pcov))
        return popt,err
    
    def Fit_result_multiple_freq(self,x_angle,ydata):
        """
        output:
            - fit result, 2D arr [frequency, a and b]
            - fit err, 1D arr [frequency, a and b]
        """
        fit_arr = np.empty([self.freq_num,17])
        err_arr = np.empty([self.freq_num,17])
        for i in tqdm(range(0,self.freq_num),desc = 'Fitting result at measured frequencies...'):
            fit_res,err_res = self.Fit_result(x_angle,ydata[i])
            fit_arr[i] = fit_res
            err_arr[i] = err_res
        return fit_arr,err_arr
    
    def Plot_fit_result_single_case(self,freq,fit_res,fit_err,save_name):
        a_lab = [r'$a_0$',r'$a_1$',r'$a_2$',r'$a_3$',r'$a_4$',r'$a_5$',r'$a_6$',r'$a_7$',r'$a_8$']        
        b_lab = [r'$\phi_1$',r'$\phi_2$',r'$\phi_3$',r'$\phi_4$',r'$\phi_5$',r'$\phi_6$',r'$\phi_7$',r'$\phi_8$']
        
        fig =plt.figure(figsize = (15,9))
        for i in range(0,9):
            ax = fig.add_subplot(3,3,i+1)
            ax.errorbar(freq,fit_res[:,i],yerr = fit_err[:,i],fmt = 'b.',capsize = 1)
            ax.set_ylabel(a_lab[i])
            if i == 7:
                ax.set_xlabel('Frequency [GHz]')
        fig.tight_layout()
        plt.savefig(self.save_dir + save_name + '_fit_a.png',dpi = 300)
        plt.show()
        plt.close()  
        
        fig = plt.figure(figsize = (15,9))        
        for i in range(0,8):
            ax = fig.add_subplot(3,3,i+2)
            ax.errorbar(freq,np.degrees(fit_res[:,i+9]),yerr = np.degrees(fit_err[:,i+9]),fmt = 'b.',capsize = 1)
            ax.set_ylabel(b_lab[i]+' [deg.]')
            if i == 7:
                ax.set_xlabel('Frequency [GHz]')
        fig.tight_layout()
        plt.savefig(self.save_dir + save_name + 'fit_phi.png',dpi = 300)
        plt.show()
        plt.close()


























