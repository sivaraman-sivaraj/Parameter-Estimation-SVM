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


def ramp_signal(ip,Lambda):
    op = []
    for i in range(len(ip)):
        temp = ip[i]+Lambda
        op.append(temp)
    return op

nU_ramp = ramp_signal(nU, 0.01)
nV_ramp = ramp_signal(nV, 0.01)
nR_ramp = ramp_signal(nR, 0.01)














