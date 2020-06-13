
"""
Created on Sun May 17 13:06:45 2020

@author: Sivaraman Sivaraj, Suresh Rajendran
"""


print(__doc__)

import hydrodynamics as hdy

"""
TV, RAC, surgeXC, yawRC, swayYC, nU, nV,nR # importing variables from hydrodynamics

x,X,y,Y,z,Z,pi,pi1,theta,theta1,shi,\
shi1,epn1, epn2, epn3,epn4, epn5, epn6,u,v,w,p,q,r,rudderAngle, rpm
"""

import numpy as np
from sklearn.model_selection import train_test_split

surgeXC_train = hdy.surgeXC
swayYC_train = hdy.swayYC
yawRC_train = hdy.yawRC

_nU_train = hdy.nU
_nV_train = hdy.nV
_nR_train = hdy.nR



# surgeXC_train, surgeXC_test, nU_train, nU_test = train_test_split(hdy.surgeXC,hdy.nU,random_state = 0)
# swayYC_train, swayYC_test, nV_train, nV_test = train_test_split(hdy.swayYC,hdy.nV, random_state = 0)     
# yawRC_train, yawRC_test, nR_train, nR_test = train_test_split(hdy.yawRC,hdy.nR, random_state = 0)   

def ramp_signal(ip,Lambda):
    op = []
    for i in range(len(ip)):
        temp = ip[i]+Lambda
        op.append(temp)
    return op
        
nU_train = ramp_signal(_nU_train, 0.01)
nV_train = ramp_signal(_nV_train, 0.01)
nR_train = ramp_signal(_nR_train, 0.01)
    
# nU_test = ramp_signal(_nU_test, 0.01)
# nV_test = ramp_signal(_nV_test, 0.01)
# nR_test = ramp_signal(_nR_test, 0.01)


from sklearn.svm import LinearSVR

linear_svm = LinearSVR(C=1e08,fit_intercept = False, dual = True ,
                       epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                       max_iter = 10000, random_state = None, tol = 0.000001,
                       verbose = 0).fit(surgeXC_train, nU_train)
coefLinear = linear_svm.coef_
print("Train set accuracy of Surge on LinearSVR method: {:.2f}".format(linear_svm.score(surgeXC_train,nU_train)))



linear_svm1 = LinearSVR(C=1e08,fit_intercept = False, dual = True ,
                       epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                       max_iter = 10000, random_state = None, tol = 0.000001,
                       verbose = 0).fit(swayYC_train,nV_train)

coefLinear1 = linear_svm1.coef_
print("Train set accuracy of Sway on LinearSVR method: {:.2f}".format(linear_svm1.score(swayYC_train,nV_train)))



linear_svm2 = LinearSVR(C=1e08,fit_intercept = False, dual = True ,
                       epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                       max_iter = 10000, random_state = None, tol = 0.000001,
                       verbose = 0).fit(yawRC_train,nR_train)

coefLinear2 = linear_svm2.coef_
print("Train set accuracy of Yaw on LinearSVR method: {:.2f}".format(linear_svm2.score(yawRC_train,nR_train)))



np.save('surge_regression_coeff.npy', coefLinear)
np.save('sway_regression_coeff.npy', coefLinear1)
np.save('yaw_regression_coeff.npy', coefLinear2)








