import numpy as np
import time
from tabulate import tabulate

Y = np.load('sway_regression_coeff.npy')
R = np.load('yaw_regression_coeff.npy')
Y_ramp = np.load('sway_regression_coeff_ramp.npy')
R_ramp = np.load('sway_regression_coeff_ramp.npy')


"""
start of declaration of ship features

"""
h = 2 # time step size
L = 171.8 # length of ship
Xg = -0.002300 # Longitutional co-ordinate of ship center of gravity 
m = 798 # mass of ship
IzG = 39.2 # Moment of inertia of ship around center of gravity
lenthofShip = 7.0


Xau= -42 # accelaration derivative of surge force with respect to u 

Yav = -748 # accelaration derivative of sway force with respect to v
Yar = -9.354 # accelaration derivative of sway force with respect to r

Nav = 4.646 # Yaw moment derivative with respect to sway velocity
Nar = -43.8 # Yaw moment derivative with respect to rudder angle

S = ((IzG-Nar)*(m-Yav))-(((m*Xg)-Yar)*((m*Xg)-Nav))


"""
End of features description
"""

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

sl_no = np.arange(1,23)
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
