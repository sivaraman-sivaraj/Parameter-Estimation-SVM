import scipy.io
import numpy as np



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

time = data[:,0]
_u = data[:,1]# raw values
_v = data[:,2]
_r = data[:,3]
_U = data[:,5]
__rac = data[:,6]




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
rac = delta_val(_rac)

def align_by_one(var):
    op = var[1:]
    return op

U = align_by_one(_U)



"""
surgeXC = first governing equation
swayYC = second governing equation
yawRC = third governing equation

"""

def Create_Surgecomponents(u,v, yawRate, RAC, TV):
    Surge_Components = []
    for i in range(len(u)):
        c1 = u[i]
        c2 = u[i]*TV[i]
        c3 = u[i]**2
        c4 = (u[i]**3)/TV[i]
        c5 = v[i]**2
        c6 = yawRate[i]**2
        c7 = (RAC[i]**2)*(TV[i]**2)
        c8 = (RAC[i]**2)*u[i]*TV[i]
        c9 = v[i]*yawRate[i]
        c10 = v[i]*RAC[i]*TV[i]
        c11 = v[i]*RAC[i]*u[i]
        c12 = u[i]*(v[i]**2)/TV[i]
        c13 = u[i]*(yawRate[i]**2)/TV[i]
        c14 = u[i]*v[i]*yawRate[i]/ TV[i]
        c15 = yawRate[i]*RAC[i]*TV[i]
        c16 = u[i]*yawRate[i]*RAC[i]
        c17 = TV[i]**2
        temp = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17]
        Surge_Components.append(temp)
    Surge_Components.pop()
    return Surge_Components

def Create_Swaycomponents(U,V,R,RAC,TV):
    Sway_Components = []
    for i in range(len(U)):
        c1 = V[i]
        c2 = TV[i]**2
        c3 = U[i]*TV[i]
        c4 = U[i]**2
        c5 = V[i]*TV[i]
        c6 = R[i]*TV[i]
        c7 = RAC[i]*(TV[i]**2)
        c8 = (V[i]**3)/TV[i]
        c9 = (RAC[i]**3)*(TV[i]**2)
        c10 = (V[i]**2)*R[i]/TV[i]
        c11 = (V[i]**2)*RAC[i]
        c12 = V[i]*(RAC[i]**2)*TV[i]
        c13 = RAC[i]*U[i]*TV[i]
        c14 = V[i]*U[i]
        c15 = R[i]*U[i]
        c16 = RAC[i]*(U[i]**2)
        c17 = (R[i]**3)/TV[i]
        c18 = V[i]*(R[i]**2)/TV[i]
        c19 = V[i]*(U[i]**2)/TV[i]
        c20 = R[i]*(U[i]**2)/TV[i]
        c21 = R[i]*(RAC[i]**2)*TV[i]
        c22 = (R[i]**2)*RAC[i]
        c23 = R[i]*V[i]*RAC[i]
        temp=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,
              c18,c19,c20,c21,c22,c23]
        Sway_Components.append(temp)
    Sway_Components.pop()
    return Sway_Components

def Create_Yawcomponents(U,V,R,RAC,TV):
    Yaw_Components = []
    for i in range(len(U)):
        c1 = R[i]
        c2 = U[i]**2
        c3 = U[i]*TV[i]
        c4 = U[i]**2
        c5 = V[i]*TV[i]
        c6 = R[i]*TV[i]
        c7 = RAC[i]*(TV[i]**2)
        c8 = V[i]**3 / TV[i]
        c9 = (RAC[i]**3)*(TV[i]**2)
        c10 = (V[i]**2)*R[i]/TV[i]
        c11 = (V[i]**2)*RAC[i]
        c12 = V[i]*(RAC[i]**2)*TV[i]
        c13 = RAC[i]*U[i]*TV[i]
        c14 = V[i]*U[i]
        c15 = R[i]*U[i]
        c16 = RAC[i]*(U[i]**2)
        c17 = (R[i]**3)/TV[i]
        c18 = V[i]*(R[i]**2)/TV[i]
        c19 = V[i]*(U[i]**2)/TV[i]
        c20 = R[i]*(U[i]**2)/TV[i]
        c21 = R[i]*(RAC[i]**2)*TV[i]
        c22 = (R[i]**2)*RAC[i]
        c23 = R[i]*V[i]*RAC[i]
        temp=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,
              c17,c18,c19,c20,c21,c22,c23]
        Yaw_Components.append(temp)
    Yaw_Components.pop()
    return Yaw_Components


def EqL_next(var):# left hand side of the equation
    var_next = []
    for i in range(len(var) -1):
        temp = var[i+1]
        var_next.append(temp)
    return var_next

EqL_u = EqL_next(u[::20])
EqL_v = EqL_next(v[::20])
EqL_r = EqL_next(r[::20])

surgeXC = Create_Surgecomponents(u[::20], v[::20], r[::20], rac[::20], U[::20])
yawRC = Create_Yawcomponents(u[::20], v[::20], r[::20], rac[::20], U[::20])
swayYC = Create_Swaycomponents(u[::20], v[::20], r[::20], rac[::20], U[::20])





