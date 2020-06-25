
"""
Created on Thu Jun 25 10:18:34 2020

@author: Sivaraman Sivaraj
"""
import numpy as np

sway = np.load('sway.npy')
sway_ramp = np.load("sway_ramp.npy")
yaw = np.load("yaw.npy")
yaw_ramp = np.load("yaw_ramp.npy")


Cov_zero = np.zeros((15,15))

def cov_diagonal(cov, hd_l):
    op = cov
    for i in range((14)):
        op[i][i] = int(hd_l[i])
    return op



def upper(matrix, row, col): 
  
    for i in range(0, row): 
      
        for j in range(0, col): 
          
            if (i > j): 
               matrix[i][j] = int(0)
    return matrix



sway_d = cov_diagonal(Cov_zero, sway)
yaw_d = cov_diagonal(Cov_zero, yaw)
sway_d_ramp = cov_diagonal(Cov_zero, sway_ramp)
yaw_d_ramp = cov_diagonal(Cov_zero, yaw_ramp)


_cov_sway = np.cov(sway_d)
_cov_yaw = np.cov(yaw_d)
_cov_sway_ramp = np.cov(sway_d_ramp)
_cov_yaw_ramp = np.cov(yaw_d_ramp)



cov_sway = upper(_cov_sway, 14,14)
cov_yaw = upper(_cov_yaw, 14,14)
cov_sway_ramp = upper(_cov_sway_ramp, 14,14)
cov_yaw_ramp = upper(_cov_yaw_ramp, 14,14)

# np.savetxt("sway_covariance.csv",cov_sway)
# np.savetxt("yaw_covariance.csv",cov_yaw)
# np.savetxt("sway_covariance_ramp.csv",cov_sway_ramp)
# np.savetxt("yaw_covariance_ramp.csv",cov_yaw_ramp)


