

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from my_config import CONFIG
from scipy.stats import norm,qmc
class MonteCarloEngine:
    def __init__(self,
                 N: int,
                 m: int,
                 n: int,
                 S0: np.ndarray,
                 d: np.ndarray,
                 r: float,
                 corr_matrix: np.ndarray,
                 sig: np.ndarray,
                 IV: float,
                 PA: np.ndarray,
                 days_trade_yr: int,
                 T_N: float,
                 t_c: float,
                 autocall_checkpt: np.ndarray):
        self.N=N #No. of Simulation
        self.m=m #No. of time steps
        self.n=n #No. of stocks 
        self.S0=S0 #Intial stock price array 
        self.d=d #Dividend payed per annum
        self.r=r #Return
        self.corr_matrix=corr_matrix #Stock Correlation matrix
        self.sig=sig #Daily Volatility of the stocks
        self.IV=IV #Investment amount
        self.PA=PA #Coupons paid as ratio of IV
        self.days_trade_yr=days_trade_yr #No. of trading days in a year
        self.T_N=T_N #Time of maturity in yrs
        self.t_c=t_c #Fraction in years for coupon payment
        self.autocall_checkpt=autocall_checkpt
        self.sig_yearly=self.sig*np.sqrt(self.days_trade_yr)
        
        
    def simulate_Z(self):
        Z0=np.random.randn(self.N,self.m,self.n)
        L=np.linalg.cholesky(self.corr_matrix)
        self.Z=Z0@L.T
        self.U=np.random.rand(self.N,self.m,self.n)
        return self.Z,self.U
    

    def simulate_stratified_tensor(self):
        
        # ===== Step 1: Correlation =====
        L = np.linalg.cholesky(self.corr_matrix)
        
        # ===== Step 2: Stratify terminal shocks =====
        U = (np.arange(self.N) + np.random.rand(self.N)) / self.N
        Z_terminal_1d = norm.ppf(U)   # (N,)
        
        # ===== Step 3: Expand to assets =====
        if self.n > 1:
            Z_rest = np.random.randn(self.N, self.n - 1)
            Z_terminal = np.column_stack([Z_terminal_1d, Z_rest])  # (N,n)
        else:
            Z_terminal = Z_terminal_1d.reshape(-1,1)
        
        # ===== Step 4: Apply correlation =====
        Z_terminal = Z_terminal @ L.T   # (N,n)
        
        # ===== Step 5: Build full Brownian path =====
        Z_full = np.random.randn(self.N, self.m, self.n)  # (N,m,n)
        
        # ===== Step 6: Replace ONLY terminal step =====
        Z_full[:, -1, :] = Z_terminal
        
        # ===== Step 7: Independent uniforms (for barrier etc) =====
        U_barrier = np.random.rand(self.N, self.m, self.n)
        
        return Z_full, U_barrier
        
    def simulate_path(self,Z):
        self.dt=self.T_N/self.m
        diff=self.sig_yearly*np.power(self.dt,0.5)*Z
        drift=(self.r-self.d-0.5*np.power(self.sig_yearly,2))*self.dt
        logR=diff+drift
        self.R=np.exp(np.cumsum(logR,axis=1))
        return self.R
    
    def checkpt_idx(self):
        self.check_idx=((np.array(self.autocall_checkpt)*self.m/self.T_N)-1).astype(int)
        return self.check_idx
    
    def brownian_bridge(self,U):
        R_roll=np.roll(self.R,shift=-1,axis=1)
        log_sum=0.5*(np.log(self.R)+np.log(R_roll))
        log_dif=np.power(0.5*(np.log(R_roll)-np.log(self.R)),2)
        log_=np.power(self.sig_yearly,2)*self.dt*np.log(U)
        log_term=log_sum-np.power(log_dif-log_,0.5)
        self.barrier_bw=np.exp(log_term)
        self.barrier_bw=self.barrier_bw[:,:-1,:]
        return self.barrier_bw      
        
    def execute(self,method="vmc"):
        if method=="vmc":
            self.Z,self.U=self.simulate_Z()
        if method=="sobol":
            self.Z,self.U=self.simulate_stratified_tensor()
        self.R=self.simulate_path(self.Z)
        self.check_idx=self.checkpt_idx()
        self.barrier_bw=self.brownian_bridge(self.U)
        return self
        
