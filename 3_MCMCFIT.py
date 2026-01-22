import numpy as np
import multiprocessing
import time
import emcee
import corner
import matplotlib.pyplot as plt
import matplotlib as plt1
import pandas as pd
#
from scipy.constants import speed_of_light as c
#
import tmm ## may have trouble importing this one 
"""
from ARCS.tmm import ARrecipe
from ARCS.schema import RefInd as RI
from ARCS.schema import DataSchema
"""
#
#################################################################
#-


class DUT:
    
    # Sapphire Plate Thicknesses
    MF1 = 3.7434
    MF2 = 3.7399
    MF3 = 3.7462
    MF4 = 3.7869
    MF5 = 3.7475
    MF6 = 3.774894035
    MF7 = 3.8038
    MF8 = 3.7650
    
    # Sapphire Plate Thicknesses err
    MF1 = 0.0177
    MF2 = 0.0091
    MF3 = 0.0124
    MF4 = 0.0030
    MF5 = 0.0030
    MF6 = 0.0069
    MF7 = 0.0175
    MF8 = 0.0075


    """
    LF1 =
    LF2 = 
    LF3 = 
    LF4 = 
    """

class DataSchema:
    ks = "wave_num"
    freq = 'freq_data' 
    trans = 'trans_data' 
    phase = 'phase_data' 
    realdata = 'real_data' 
    imagdata = 'imag_data' 
    freqfit = 'freq_fit' 
    transfit = 'trans_fit' 
    realfit = 'real_fit' 
    imagfit ='imag_fit' 
    n = 'fit_n' 
    loss = 'fit_los' 
    nerr = 'fit_n_err'
    losserr = 'fit_los_err'
    transdiff = 'trans_diff'
    
    
    
#### setting some params 
dut = DUT.MF2
    
    
def ARrecipe(d_list, n_list, ks, title, angle, plot):
    ### INPUT
    # d_list: array, list of thicknesses for each layer
    # n_list: array, list of refractive indices for each layer 
    # ks: array, target wave numbers for analysis - BEWARE Of UNITS! cm^-1 is probably best
    # title: string, ID for the data being modelled 
    # angle: float, radians (?? check this), angle of incidence of light on surface from perpendicular
    # plot: Boolean, do you want to see a visual representation of the data??
    ### OUTPUT
    # df: data frame containing: 
    #       ks: array, your wave numbers saved in a convienent place
    #       Rcomp: Reflection component 
    #       Tcomp: Transmission component 
   
    # conversion of wave number if neccessary: 
    ks1 = ks                              # wave number in m^-1 
   
    # wavelength range in nm
    # initialize lists of y-values to plot
    Rcomp = np.empty(len(ks1))
    Tcomp = np.empty(len(ks1))

    #degree = np.pi/180 

    for i, k in enumerate(ks1):
        # For normal incidence, s and p polarizations are identical.
		# I arbitrarily decided to use 's'.
        cohtmm = tmm.coh_tmm('s', n_list, d_list, angle, 1/k)
        Rcomp[i] = cohtmm['R']
        Tcomp[i] = cohtmm['T']
    
        #Rcomp.append(tmm.unpolarized_RT( n_list, d_list, 0, 1/k)['R'])
        #Tcomp.append(tmm.unpolarized_RT( n_list, d_list, angle*degree, 1/k)['T'])
    

    
    if plot == True: 
        ticks = 7
        fig, ax1 = plt.subplots()
        ax1.plot(ks, Tcomp, 'blue',alpha=0.5)
        
        
        plt.grid()
        
        ax2 = ax1.twiny()
        
        
        #ax2.invert_xaxis()
        ax2.plot(np.linspace(0, max(ks)-min(ks),len(Tcomp)), Tcomp, alpha=0, c="pink")
        
        
        ax1.set_xlabel(DataSchema.ks)
        ax1.set_ylabel(DataSchema.trans)
        
        ax2.set_xlabel(DataSchema.freq)
        
        # align the plots
        ax1.set_xlim(left=ks[0], right=ks[-1])
        ax2.set_xlim(left=0, right=max(ks)-min(ks))
        ax1.set_ylim(0.6,1.02)
        
        ksticks = np.linspace(min(ks), max(ks), ticks) # make k sticks out of cm^-1
        ax1.set_xticks(ksticks)
        
        
        labels = [int(round((x*(0.299792458/10^-2)),3)) for x in ksticks] # for Frequency in GHz
                      
        ax2.set_xticklabels(labels)
        ax2.xaxis.set_major_locator(plt1.ticker.FixedLocator(np.linspace(0, max(ks)-min(ks),ticks)))
        
        plt.title(title+" from AR recipe func.", fontweight= "bold")
        plt.show()
        
    
    df = pd.DataFrame({DataSchema.ks : ks,
                       DataSchema.R  : Rcomp,
                       DataSchema.T  : Tcomp})
        
    return df



### MODEL

# function for preparing your thetas to be read by the model 
def ARlayer(Theta):
    ### converts theta parameters into recipe format for AR model
    # IMPORTANT NOTE this is optimised for 2 material alternating layers, if you want something more specific you will have to write your own
    ### INPUT
    # Theta = theta parameters as a dictionary 
    ### OUTPUT
    #thickness, refindex : arrays of recipes    
    layers = len(Theta) # given the specific format for the recipes they can be identified by the number of parameters. 
    if layers == 3:
        thickness = (np.inf, Theta[0]                   , np.inf)
        refindex  = (1 , complex(Theta[1],Theta[2]) , 1)
    return np.array(thickness), np.array(refindex)
# TMM model in a format for the MCMC
def model(theta, x):
    ### INPUT 
    # theta: dictionary of theta params 
    # x: array of desired frequency band to work over 
    ### OUTPUT 
    # recipefit: df=[freqs, reflection, transmission] 
    ks_ = x*10**9/c
    
    #start = time.time()
    t_list, n_list = ARlayer(theta) 
    #print("Time per model call:", time.time() - start) 
    recipefit = ARrecipe(t_list, n_list, ks_, "-",angle=0, plot=False)
    return recipefit
### MODEL 
#-
### Def L_inlike - how good is the fit
# weighted error 
def lnlike(theta, x,y,yerr):
    

    
    ymodel_df = model(theta, x=x)
    
    if DataSchema.T not in ymodel_df:
        return -np.inf
       
    ymodel = ymodel_df[DataSchema.T].values
    if not np.all(np.isfinite(ymodel)):
        return -np.inf
    ln = -0.5 * np.sum(((y - ymodel) / yerr) ** 2)
    
    ### === debug    
    #print("lnlike ======================")
    #print(ln)
    #print(f"ln: {ln}, type: {type(ln)}")
    #print("lnlike ======================")
    ### === debug    
        
    return ln
### Def L_inlike


### Def Lnprior - IMPORTANT: will need to be changed depending on theta 
def lnprior(theta):
    # unwrap theta 
    tl, nlr, nli = theta
  
    # limits for prior
    tllim   = [9e-3, 12e-3]
    nlrlim  = [1.50, 1.55]
    nlilim  = [0.5e-4, 15e-4]

    #conditions of prior implicit
    if       tllim[0] <=  tl <=  tllim[1]\
        and nlrlim[0] <= nlr <= nlrlim[1]\
        and nlilim[0] <= nli <= nlilim[1]: #apply conditions on priors here
        return 0.0
    else:
        return -np.inf  
    
### Def Lnprior

### Def Inprob
def lnprob(theta, x, y , yerr):
    lp = lnprior(theta) #call lnprior
    
    if not isinstance(lp, (int, float, complex, str)):
        print("lp from your lpprior is not scalar")
        print("check1", lp )
        return -np.inf   
    
    
    ll = lp + lnlike(theta, x, y, yerr) #recall if lp not -inf, its 0, so this just returns likelihood
    
    ### === debug
    if not isinstance(lp, (int, float, complex, str)):
        print("check2")
        print("Non-scalar log-prob returned!", ll)
        return -np.inf    
    #print("lnprob ======================")
    #print(f"ll: {ll}, type: {type(ll)}")
    #print("lnprob ======================")
    ### === debug    
    #print(ll)
    return ll
### Def Inprob

### Run the MCMC
def mainMCMC(p0,nwalkers,niter,ndim,lnprob,data):
    F       = data[DataSchema.F].values
    T       = data[DataSchema.T].values
    Terr    = data[DataSchema.T_err].values  # unpack the tuple returned by prepdata
    dTuple = (F, T, Terr)
    
    
    Para= True # easily switch between parallel processing and not for debuging 
    if Para == True:
        with multiprocessing.Pool(processes=20) as pool: ## with parallel sampling
        
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=dTuple, pool=pool)
            #
            start = time.time()
            print("burn in ...")
            p0, _, _ = sampler.run_mcmc(p0, 50,**{'skip_initial_state_check':True}) # burn in 
            sampler.reset() # reset the sampler so burn-in steps don't contaminate production stats
            print("burn-in Time:", time.time() - start) 
            # 
            start = time.time()
            print("Production ...")
            pos, prob, state = sampler.run_mcmc(p0, niter,**{'skip_initial_state_check':True})
            print("Production Time:", time.time() - start) 
            #
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=dTuple)
        #
        start = time.time()
        print("burn in ...")
        p0, _, _ = sampler.run_mcmc(p0, 5,**{'skip_initial_state_check':True}) # burn in 
        sampler.reset() # reset the sampler so burn-in steps don't contaminate production stats
        print("burn-in Time:", time.time() - start) 
        # 
        start = time.time()
        print("Production ...")
        pos, prob, state = sampler.run_mcmc(p0, niter,**{'skip_initial_state_check':True})
        print("Production Time:", time.time() - start) 
        #
    return sampler, pos, prob, state
### Run the MCMC

### PLOT - corner plot + data with fit
def plotter(sampler,x,y,labels,randsam= False):
    ############################################ RESULTS PLOT 
    plt.figure(figsize=(20,10))
    # plot the fit data
    plt.plot(x,y,label='Fit data')
    # A shortcut for accessing chain flattened along the zeroth (walker) axis
    samples = sampler.flatchain
    # plot 10 random possible samples // optional as it adds a lot of noise extra data to the graph
    if randsam == True:
        for theta in samples[np.random.randint(len(samples), size=10)]:
            dfmodel = model(theta, x)
            plt.plot(x, dfmodel[DataSchema.T], color="r", alpha=0.1)
    else:
        pass
    # determine the best fitting theta by selecting the best set of thetas as the one with the largest probability of fit
    # ((flatlnprobability==A shortcut to return the equivalent of lnprobability but aligned to flatchain rather than chain.))
    theta_max  = samples[np.argmax(sampler.flatlnprobability)]
    # Run the best thetas in the model for plotting 
    best_fit_model = model(theta_max,x)
    # plot the best fit 
    plt.plot(x,best_fit_model[DataSchema.T],label='Highest Likelihood Model', color="orange", alpha=1)
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Transmission (1)')
    plt.legend()
    plt.show()
    ############################################ RESULTS PLOT 
    ############################################ CORNER PLOT 
    corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
    plt.show()
    ############################################ CORNER PLOT 
    
    return theta_max

## trace plot for walkers values vs iterations for any number of theta values 
def plot_trace(sampler, labels):
    
    

    fig, axes = plt.subplots(len(labels), figsize=(12, 7), sharex=True)
    samples = sampler.get_chain()  # shape: (nsteps, nwalkers, ndim)

    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.5)
        ax.set_ylabel(labels[i])
        ax.axvline(0, color='k', linestyle='--')  # burn-in start
        ax.grid()
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.show() 
    
      
def generate_intervals(start, end, bandwidth):
    intervals = []
    
    interv = start
    for i in range(1, int(abs(end-start)/bandwidth)+1):
        intervals.append([interv,interv+bandwidth])
        interv += bandwidth
    return np.array(intervals)


def params_errs(sampler):
    ### extract max likelhood/ Max. a posteriori
    samples = sampler.get_chain(flat=True)
    MLE_thetaRESULT  = samples[np.argmax(sampler.flatlnprobability)] ## Max. Likelhood = Max. a Posteriori  
    ### extract median wih errors 
    Med_theta_estimates = []
    for i in range(samples.shape[1]):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)  # 1σ uncertainties
        Med_theta_estimates.append(np.array((mcmc[1], q[0], q[1])))  # (median, -err, +err)
    return MLE_thetaRESULT, Med_theta_estimates
    


### PLOT
def main(dut):
    ##### 1. GET THE DATA!!
    ## == data = prepdata(USERbool:bool, SYS, CYCLE, SAMPLE, FRANGE, SMOFAC)
    
    data = prepdata(USERbool=True, SYS="Data_Terascan", CYCLE="Cycle 3", SAMPLE="BVI_A3_nocoat", FRANGE="89-225GHz", SMOFAC="smooth150")
    #FRANGE = "89-225GHz" #FRANGE = "166-448GHz" #FRANGE = "70-180GHz" #FRANGE = "20-1360GHz" FRANGE = "50-500GHz"
    ##### 2. CHECK YOUR PRIORS!! & MCMC INPUTS !!
    #set nwalkers - number of test particle guesses
    nwalkers = 128
    #set niter - length of the chains
    niter = int(nwalkers*4)
    #theta taken as: [t1, tg, tlens, n1, ng, nlens]
    initial = [DUT.dut, # tlens
               1.526      , # nlens RE
               1e-4       ] # nlens IM
    #
    initialerr = [0.01e-3,  # error on tlens - comes from stdev on measurements 
                  0.001  ,  # error in path length calculation 
                  0.5e-4  ] # by eye estimate - hence the large error!
    ndim = len(initial)
    # determine how the walkers walk here - drawn from random normal dist
    #    *mean           * error or standard dev   * rand number from normal dist
    p0 = [np.array(initial) + np.array(initialerr) * np.random.normal(size=ndim) for i in range(nwalkers)]

    ##### 3. RUN MCMC!!
    sampler1, pos, prob, state = MCMC.mainMCMC(p0,nwalkers,niter,ndim,MCMC.lnprob,data)
    ##### 4. PLOT YOUR RESULT!! & GET IDEAL THETAS!!
    # general plot with corner plot
    labels = ['T', 'Re{n}', 'Im{n}']
    thetaRESULT = MCMC.plotter(sampler=sampler1,x=data[DataSchema.F],y=data[DataSchema.T], labels=labels, randsam=False)
    # trace plot 
    MCMC.plot_trace(sampler1, labels)
    
    
    print(thetaRESULT)
    #return thetaRESULT
    


if __name__ == '__main__':
    startTime = time.time()
    multiprocessing.freeze_support()  # optional, safe to include
    main()
    print("TOTAL RUN TIME: ", time.time() - startTime)