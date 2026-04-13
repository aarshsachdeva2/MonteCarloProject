import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from my_config import CONFIG
from data import MonteCarloEngine
'''

'''

class payout:
    def __init__(self,engine):
        self.engine=engine
        self.R=engine.R
        self.barrier_bw=engine.barrier_bw
        self.check_idx=engine.checkpt_idx()
        self.autocall_checkpt=engine.autocall_checkpt
        
        self.n=engine.n
        self.r=engine.r
        self.N=engine.N
        self.IV=engine.IV
        self.PA=engine.PA
        self.T_N=engine.T_N
        self.t_c=engine.t_c
    
    def autocall(self):
        check = (self.R[:, self.check_idx, :] > 1).all(axis=2)
        payout_list = np.where(check.any(axis=1),check.argmax(axis=1) + 1,0)
        discounted_check=np.exp(-self.r*self.autocall_checkpt)*self.IV
        discounted_check=np.insert(discounted_check,0,0)
        discounted_coupons=self.IV*self.PA*self.autocall_checkpt*np.exp(-self.r*self.autocall_checkpt)
        discounted_coupons=np.insert(discounted_coupons,0,0)
        self.autocall_payout=np.array(discounted_check[payout_list]+discounted_coupons[payout_list])
    
    def barrier(self):
        terminal_min=self.R[:,-1,:].min(axis=1)
        terminal_price=np.minimum(terminal_min,1)
        barrier_ts=(self.R<0.59).any(axis=(1,2))
        barrier_bwts=(self.barrier_bw<0.59).any(axis=(1,2))
        barrier_breach=(barrier_ts|barrier_bwts)
        barrier_breach=np.where(barrier_breach,terminal_price*self.IV,self.IV)
        coupons=self.PA*self.IV*self.T_N*np.exp(-self.r*self.T_N)
        self.barrier_payout=barrier_breach+coupons
    
    def metrics(self):
        self.final_payout=np.where(self.autocall_payout>0,self.autocall_payout,self.barrier_payout)
        self.mean=np.mean(self.final_payout)
        self.SE=np.sqrt(np.var(self.final_payout)/self.N)
        return self.mean,self.SE,self.final_payout
        
    def execute(self):
        self.autocall()
        self.barrier()
        return self.metrics()
    
    

               
         
         
        
        