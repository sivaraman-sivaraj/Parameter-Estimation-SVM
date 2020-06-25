
"""
Created on Sun May 17 13:06:45 2020

@author: Sivaraman Sivaraj, Suresh Rajendran
"""


print(__doc__)

import hydrodynamics as hdy

"""
surgeXC, yawRC, swayYC, nU, nV,nR # importing variables from hydrodynamics

u,v,r,U,rac - input data also importable
"""

import numpy as np
from sklearn.model_selection import train_test_split

surgeXC = hdy.surgeXC
swayYC = hdy.swayYC
yawRC = hdy.yawRC

nU = hdy.nU
nV = hdy.nV
nR = hdy.nR

    
nU_ramp = hdy.nU_ramp
nV_ramp = hdy.nV_ramp
nR_ramp = hdy.nR_ramp


from sklearn.svm import LinearSVR

linear_svm = LinearSVR(C=1e08,fit_intercept = True, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(surgeXC, nU)
coefLinear = linear_svm.coef_

linear_svm_ramp = LinearSVR(C=1,fit_intercept = True, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, verbose = 0).fit(surgeXC, nU_ramp)
coefLinear_ramp = linear_svm_ramp.coef_
print("Train set accuracy of Surge on LinearSVR method: {:.2f}".format(linear_svm.score(surgeXC,nU)))



linear_svm1 = LinearSVR(C=1e08,fit_intercept = True, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(swayYC,nV)

coefLinear1 = linear_svm1.coef_
linear_svm1_ramp = LinearSVR(C=1e08,fit_intercept = True, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(swayYC,nV_ramp)

coefLinear1_ramp = linear_svm1_ramp.coef_

print("Train set accuracy of Sway on LinearSVR method: {:.2f}".format(linear_svm1.score(swayYC,nV)))



linear_svm2 = LinearSVR(C=1e08,fit_intercept = True, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(yawRC,nR)

coefLinear2 = linear_svm2.coef_

linear_svm2_ramp = LinearSVR(C=1e08,fit_intercept = True, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(yawRC,nR_ramp)

coefLinear2_ramp = linear_svm2_ramp.coef_
print("Train set accuracy of Yaw on LinearSVR method: {:.2f}".format(linear_svm2.score(yawRC,nR)))


np.save('surge_regression_coeff.npy', coefLinear)
np.save('sway_regression_coeff.npy', coefLinear1)
np.save('yaw_regression_coeff.npy', coefLinear2)

np.save('surge_regression_coeff_ramp.npy', coefLinear_ramp)
np.save('sway_regression_coeff_ramp.npy', coefLinear1_ramp)
np.save('yaw_regression_coeff_ramp.npy', coefLinear2_ramp)








