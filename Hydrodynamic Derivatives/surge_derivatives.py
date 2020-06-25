# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:59:46 2020

@author: Sivaraman Sivaraj
"""


import numpy as np
import time
from tabulate import tabulate


X = np.load('surge_regression_coeff.npy')
X_ramp = np.load('surge_regression_coeff_ramp.npy')


print(X)


"""
start of declaration of ship features

"""


h = 2 # time step size
L = 171.8 # length of ship
Xg = -2300 # Longitutional co-ordinate of ship center of gravity 
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

surgre solution and derivatives
"""

def term1(h,L,m,Xau):
    return (L*(m-Xau))/h
CC1 = term1(h,L,m,Xau)

def surge_derivatives(CC1,var):
    t1 = var[1:]
    surget = [i for i in map(lambda x : CC1*x, t1)]
    surge = []
    for elem in surget:
        surge.append(int(elem))
    return surge
    

surge = surge_derivatives(CC1,X)
surge_ramp = surge_derivatives(CC1, X_ramp)

sl_no = np.arange(1,11)
surge_hydrodynamic_derivatives = ['X`u', 'X`uu', 'X`uuu', 'X`vv', 'X`rr',
                                  'X`ðð', 'X`ððu', 'X`vr', 'X`vð', 'X`vðu']

Actual_Value = [-184,-110,-215,-899,18,
                    -95,-190,798,93,93]

table1 = zip(sl_no,surge_hydrodynamic_derivatives,Actual_Value,surge, surge_ramp)
headers1 = ['sl_no','Surge_hydrodynamic_derivatives','Original','case 1(C = 10^8)','case 2(C= 10^4, ramp added)']

surge_table = tabulate(table1, headers1, tablefmt="pretty")
print(surge_table)


# np.savetxt('surge.txt', surge)
# text_file=open("surge.csv","w")
# text_file.write(str(surge_table))
# text_file.close()



