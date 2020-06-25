import scipy.io
import numpy as np
import matplotlib.pyplot as plt



mat = scipy.io.loadmat('OUTPUT.mat')

"""
7*1 format

0. time
1.surge_speed
2.sway_speed
3.yaw_rate
4.yaw_angle
5.total_speed
6.rudder_angle

"""

data = mat["OUTPUT"]

time = data[:,0][1:][::20]
_u = data[:,1][1:][::20]# raw values
_v = data[:,2][1:][::20]
_r = data[:,3][1:][::20]
_U = data[:,5][1:][::20]
__rac = data[:,6][1:][::20]




def radian(var):
    op = np.pi *var/180
    return op

_rac = radian(__rac) # changing the values to radian mode

def delta_val(var): # x is initial value
    D =[]
    for i in range(1,len(var)):
        temp = var[i]-var[0]
        D.append(temp)
    return D


u = delta_val(_u)
v = delta_val(_v)
r = delta_val(_r)
rac = delta_val(_rac) # fifth element total veocity may take as it is.

# def align_by_one(var):
#     op = var[1:]
#     return op

# U = align_by_one(_U)

U = _U


"""
surgeXC = first governing equation
swayYC = second governing equation
yawRC = third governing equation

"""

def Create_Surgecomponents(u,v, yawRate, RAC, TV,L):
    """
    

    Parameters
    ----------
    u : surge velocity-delta
    v : sway velocity_delta
    yawRate : delta
    RAC : rudder angle change in radian
    TV : total velocity
    L : length of ship

    Returns
    -------
    Surge_Components : list of 11 surege components

    """
    Surge_Components = []
    for i in range(len(u)):
        c1 = u[i]
        c2 = u[i]*TV[i]
        c3 = u[i]**2
        c4 = (u[i]**3)/TV[i]
        c5 = v[i]**2
        c6 = (yawRate[i]**2)#*(L**2)
        c7 = (RAC[i]**2)*(TV[i]**2)
        c8 = (RAC[i]**2)*u[i]*TV[i]
        c9 = v[i]*yawRate[i]#*L
        c10 = v[i]*RAC[i]*TV[i]
        c11 = v[i]*RAC[i]*u[i]
        
        # cb = TV[i]**2 #bias term
        
        temp = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]
        Surge_Components.append(temp)
    Surge_Components.pop()# for last element, we can't have (k+1) component  
    return Surge_Components

surgeXC = Create_Surgecomponents(u, v, r, rac, U,171.8)

def Create_Swaycomponents(U,V,R,RAC,TV,L):
    """
    

    Parameters
    ----------
    u : surge velocity-delta
    v : sway velocity_delta
    yawRate : delta
    RAC : rudder angle change in radian
    TV : total velocity
    L : length of ship

    Returns
    -------
    Sway_Components : list of 16 sway components


    """
    Sway_Components = []
    for i in range(len(U)):
        # c1 = V[i]
        # c2b = TV[i]**2 #bias term
        c3 = U[i]*TV[i]
        c4 = U[i]**2
        c5 = V[i]*TV[i]
        c6 = R[i]*TV[i]#*L
        c7 = RAC[i]*(TV[i]**2)
        c8 = (V[i]**3)/TV[i]
        c9 = (RAC[i]**3)*(TV[i]**2)
        c10 = (V[i]**2)*R[i]*1/TV[i]#1 by L
        c11 = (V[i]**2)*RAC[i]
        c12 = V[i]*(RAC[i]**2)*TV[i]
        c13 = RAC[i]*U[i]*TV[i]
        c14 = V[i]*U[i]
        c15 = R[i]*U[i]#*L
        c16 = RAC[i]*(U[i]**2)
        
        temp=[c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16]
        Sway_Components.append(temp)
    Sway_Components.pop()
    return Sway_Components

swayYC = Create_Swaycomponents(u, v, r, rac, U, 171.8)

def Create_Yawcomponents(U,V,R,RAC,TV,L):
    """
    

    Parameters
    ----------
    u : surge velocity-delta
    v : sway velocity_delta
    yawRate : delta
    RAC : rudder angle change in radian
    TV : total velocity
    L : length of ship

    Returns
    -------
    yaw_Components : list of 16 yaw components

    """
    Yaw_Components = []
    for i in range(len(U)):
        # c1 = R[i]
        # c2b = U[i]**2#bias term
        c3 = U[i]*TV[i]
        c4 = U[i]**2
        c5 = V[i]*TV[i]
        c6 = R[i]*TV[i]#*L
        c7 = RAC[i]*(TV[i]**2)
        c8 = V[i]**3 / TV[i]
        c9 = (RAC[i]**3)*(TV[i]**2)
        c10 = (V[i]**2)*R[i]*1/TV[i]# 1 by L
        c11 = (V[i]**2)*RAC[i]
        c12 = V[i]*(RAC[i]**2)*TV[i]
        c13 = RAC[i]*U[i]*TV[i]
        c14 = V[i]*U[i]
        c15 = R[i]*U[i]#*L
        c16 = RAC[i]*(U[i]**2)
        
        temp=[c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16]
        Yaw_Components.append(temp)
    Yaw_Components.pop()
    return Yaw_Components

yawRC = Create_Yawcomponents(u, v, r, rac, U, 171.8)


def EqL_next(var):# left hand side of the equation
    var_next = []
    for i in range(len(var) -1):
        temp = var[i+1]
        var_next.append(temp)
    return var_next

def fin_pro(u,nU):
    op = []
    for i in range(298):
        temp = nU[i]-u[i]
        op.append(temp)
    return op

_nU = EqL_next(u)
_nV = EqL_next(v)
_nR = EqL_next(r)

nU = fin_pro(u,_nU)
nV = fin_pro(v, _nV)
nR = fin_pro(r, _nR)

# surgeXC = Create_Surgecomponents(u, v, r, rac, U)
# yawRC = Create_Yawcomponents(u, v, r, rac, U)
# swayYC = Create_Swaycomponents(u, v, r, rac, U)


# # tx = swayYC[3500]
# # plt.figure()
# # plt.plot(tx)
# # plt.title('sway', loc = 'center')
# # plt.show()




# """
# TV, RAC, surgeXC, yawRC, swayYC, nU, nV,nR # importing variables from hydrodynamics

# x,X,y,Y,z,Z,pi,pi1,theta,theta1,shi,\
# shi1,epn1, epn2, epn3,epn4, epn5, epn6,u,v,w,p,q,r,rudderAngle, rpm
# """

# import numpy as np
# from sklearn.model_selection import train_test_split

# surgeXC_train = surgeXC[::20]
# swayYC_train = swayYC[::2]
# yawRC_train = yawRC[::20]

# _nU_train = nU[::20]
# _nV_train = nV[::2]
# _nR_train = nR[::20]



# # surgeXC_train, surgeXC_test, nU_train, nU_test = train_test_split(hdy.surgeXC,hdy.nU,random_state = 0)
# # swayYC_train, swayYC_test, nV_train, nV_test = train_test_split(hdy.swayYC,hdy.nV, random_state = 0)     
# # yawRC_train, yawRC_test, nR_train, nR_test = train_test_split(hdy.yawRC,hdy.nR, random_state = 0)   

def ramp_signal(ip,Lambda):
    op = []
    for i in range(len(ip)):
        temp = ip[i]+Lambda
        op.append(temp)
    return op
        
# nU_train = ramp_signal(_nU_train, 0.01)
# nV_train = ramp_signal(_nV_train, 0.01)
# nR_train = ramp_signal(_nR_train, 0.01)
    
nU_ramp = ramp_signal(nU, 0.01)
nV_ramp = ramp_signal(nV, 0.01)
nR_ramp = ramp_signal(nR, 0.01)


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
linear_svm1_ramp = LinearSVR(C=1e01,fit_intercept = True, dual = True ,
                        epsilon = 1e-4, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(swayYC,nV_ramp)

coefLinear1_ramp = linear_svm1_ramp.coef_

print("Train set accuracy of Sway on LinearSVR method: {:.2f}".format(linear_svm1.score(swayYC,nV)))



linear_svm2 = LinearSVR(C=1e08,fit_intercept = True, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(yawRC,nR)

coefLinear2 = linear_svm2.coef_

linear_svm2_ramp = LinearSVR(C=1e01,fit_intercept = True, dual = True ,
                        epsilon = 1e-4, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(yawRC,nR_ramp)

coefLinear2_ramp = linear_svm2_ramp.coef_
print("Train set accuracy of Yaw on LinearSVR method: {:.2f}".format(linear_svm2.score(yawRC,nR)))



from tabulate import tabulate
X = coefLinear
X_ramp = coefLinear_ramp
Y = coefLinear1
R = coefLinear2
Y_ramp = coefLinear1_ramp
R_ramp = coefLinear2_ramp

"""
End of features description
"""
h = 2 # time step size
L = 171.8 # length of ship
Xg = 0#-2300#-0.002300 # Longitutional co-ordinate of ship center of gravity 
m = 798 # mass of ship
IzG = 39.2 # Moment of inertia of ship around center of gravity

Xau= -42 # accelaration derivative of surge force with respect to u 

Yav = -748 # accelaration derivative of sway force with respect to v
Yar = -9.354 # accelaration derivative of sway force with respect to r

Nav = 4.646 # Yaw moment derivative with respect to sway velocity
Nar = -43.8 # Yaw moment derivative with respect to rudder angle

S = ((IzG-Nar)*(m-Yav))-(((m*Xg)-Yar)*((m*Xg)-Nav))


"""

surgre solution and derivatives
# """

# def term1(h,L,m,Xau):
#     return (L*(m-Xau))/h
# CC1 = term1(h,L,m,Xau)

# def surge_derivatives(CC1,var):
#     t1 = var[1:]
#     surget = [i for i in map(lambda x : CC1*x, t1)]
#     surge = []
#     for elem in surget:
#         surge.append(int(elem))
#     return surge
    

# surge = surge_derivatives(CC1,X)
# surge_ramp = surge_derivatives(CC1, X_ramp)

# sl_no = np.arange(1,11)
# surge_hydrodynamic_derivatives = ['X`u', 'X`uu', 'X`uuu', 'X`vv', 'X`rr',
#                                   'X`ðð', 'X`ððu', 'X`vr', 'X`vð', 'X`vðu']

# Actual_Value = [-184,-110,-215,-899,18,
#                     -95,-190,798,93,93]

# table1 = zip(sl_no,surge_hydrodynamic_derivatives,Actual_Value,surge, surge_ramp)
# headers1 = ['sl_no','Surge_hydrodynamic_derivatives','Original','case 1(C = 10^8)','case 2(C= 10^4, ramp added)']

# surge_table = tabulate(table1, headers1, tablefmt="pretty")
# print(surge_table)

"""

sway and yaw solution
"""

def term2(h,IzG,Nav,S,L):
    op = h*(IzG-Nar)/(S*L)
    return op

def term3(h,m,Xg,Yar,S,L):
    op = (-h)*((m*Xg)-Yar)/(S*L)
    return op

def term4(h,m,Xg,Nav,S,L):
    op = (-h)*((m*Xg)-Nav)/(S*(L**2))
    return op

def term5(h,m,Yav,S,L):
    op = h*(m-Yav)/(S*(L**2))
    return op

M11= term2(h,IzG,Nav,S,L)
M12= term3(h,m,Xg,Yar,S,L)
M21= term4(h,m,Xg,Nav,S,L)
M22= term5(h,m,Yav,S,L)

solMatrix = np.array([[M11,M12],[M21,M22]])


def two_one_Matrix(t2,t3):
    List = []
    for i in range(14):
        temp = np.array([[t2[i]],[t3[i]]])
        List.append(temp)
    return List

c = two_one_Matrix(Y,R)
c_ramp = two_one_Matrix(Y_ramp,R_ramp)

def SNsolution(M,c): #sway and yaw moment solution
    List = []
    im = np.linalg.inv(M)
    for i in range(len(c)):
        temp = im.dot(c[i])
        List.append(temp)
    return List

Sway_Yaw_derivatives = SNsolution(solMatrix,c)
Sway_Yaw_derivatives_ramp = SNsolution(solMatrix,c_ramp)


def separation(M):
    sway_components = []
    yaw_components = []
    for i in M:
        sway_components.append(int(i[0][0]))
        yaw_components.append(int(i[1][0]))
    return sway_components,yaw_components

sway, yaw = separation(Sway_Yaw_derivatives)
sway_ramp, yaw_ramp = separation(Sway_Yaw_derivatives_ramp)

sl_no = np.arange(1,15)
sway_hydrodynamic_derivatives = ['Y`ou','Y`ouu','Y`v','Y`r',
                                  'Y`ð','Y`vvv','Y`ððð','Y`vvr','Y`vvð',
                                  'Y`vðð','Y`ðu','Y`vu','Y`ru','Y`ðuu',
                                  ]

yaw_hydrodynamic_derivatives = ['N`ou','N`ouu','N`v','N`r',
                                'N`ð','N`vvv','N`ððð','N`vvr','N`vvð',
                                'N`vðð','N`ðu','N`vu','N`ru','N`ðuu',
                                ]

sway_original = [-8,-4,-1160,-499,
                  278,-8078,-90,15356,1190,
                  -4,556,-1160,-499,278]

yaw_original = [6,3,-264,-166,
                  -139,1636,45,-5483,-489,
                  13,-278,-264,0,-139]

table1 = zip(sl_no,yaw_hydrodynamic_derivatives,yaw_original,yaw, yaw_ramp)
headers1 = ['sl_no','yaw_hydrodynamic_derivatives','Original','case 1(C = 10^8)','case 2(C= 10^4, ramp added)' ]

yaw_table = tabulate(table1, headers1, tablefmt="pretty")
print(yaw_table)

table2 = zip(sl_no,sway_hydrodynamic_derivatives,sway_original,sway,sway_ramp )
headers2 = ['sl_no','sway_hydrodynamic_derivatives','Original','case 1(C = 10^8)','case 2(C= 10^4, ramp added)']

sway_table = tabulate(table2, headers2, tablefmt="pretty")
print(sway_table)

