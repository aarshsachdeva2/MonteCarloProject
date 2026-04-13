from data import MonteCarloEngine
from VMC import payout
from VMC import payout
import numpy as np
from my_config import CONFIG
from scipy.stats import norm
import pandas as pd
from scipy.stats import norm
       
class CV:
    def __init__(self,data_engine,payout_engine,strike_ratio):
        self.data_engine=data_engine
        self.payout_engine=payout_engine
        
        self.Y=self.payout_engine.final_payout
        
        self.R=self.data_engine.R
        self.r=self.data_engine.r
        self.d=self.data_engine.d
        self.T_N=self.data_engine.T_N
        self.S0=self.data_engine.S0
        self.sig_yearly=self.data_engine.sig_yearly
        
        self.strike_ratio=strike_ratio
        self.strike_price=self.strike_ratio*self.S0
        
        self.X=None
        self.EX=None
        self.beta=None
        self.Y_CV=None

    def cv(self):
        S_N=self.R[:,-1,:]
        discounted_terminal_price=self.S0*S_N*np.exp(-self.r*self.T_N) #CV1
        european_call=np.maximum(self.S0*S_N-self.strike_price,0) #CV2
        european_put=np.maximum((self.strike_ratio-S_N)*self.S0,0) #CV3
        self.X=np.column_stack([
            discounted_terminal_price.reshape(len(self.Y),-1) ,
            european_put.reshape(len(self.Y),-1)
        ])
        return self.X
    
    
    def ecv(self):
        d1=(np.log(self.S0/self.strike_price)+(self.r-self.d+0.5*self.sig_yearly**2)*self.T_N)/(self.sig_yearly*np.sqrt(self.T_N))
        d2=d1-self.sig_yearly*np.sqrt(self.T_N)
        ecv1=self.S0*np.exp(-self.d*self.T_N)
        C=self.S0*np.exp(-self.d*self.T_N)*norm.cdf(d1)-self.strike_price*np.exp(-self.r*self.T_N)*norm.cdf(d2)
        P=self.strike_price*np.exp(-self.r*self.T_N)*norm.cdf(-d2)-self.S0*np.exp(-self.d*self.T_N)*norm.cdf(-d1)
        self.EX=np.concatenate([
            ecv1 ,
             P
        ])
        return self.EX

    
    def beta_cal(self):
        self.X_centered=self.X-self.X.mean(axis=0)
        self.Y_centered=self.Y-self.Y.mean(axis=0)
        XTX = self.X.T@self.X
        XTY = self.X.T@self.Y
        self.beta=np.linalg.lstsq(self.X_centered,self.Y_centered,rcond=None)[0]
        
        return self.beta
    
    
    def apply_cv(self):
        H=self.X-self.EX
        self.Y_CV=self.Y-(H)@self.beta
        self.mean=np.mean(self.Y_CV)
        self.se=np.sqrt(np.var(self.Y_CV) /self.data_engine.N) 
        return self.mean, self.se, self.Y_CV
    
    def execute(self):
        self.cv()
        self.ecv()
        self.beta_cal()
        self.apply_cv()
        return self

