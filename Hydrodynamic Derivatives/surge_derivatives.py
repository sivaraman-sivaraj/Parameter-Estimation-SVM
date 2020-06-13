# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:59:46 2020

@author: srama
"""


import numpy as np
import time
from tabulate import tabulate

X = np.load('surge_regression_coeff.npy')


print(X)


"""
start of declaration of ship features

"""
# h = 0.05 # time step size
# L = 7.0 # length of ship
# Xg = 0.25 # Longitutional co-ordinate of ship center of gravity 
# mt = 3.27*1025 # mass of ship
# m = (2*mt)/(1025*(7**3))
# IzG_ = mt*((0.25*L)**2) # Moment of inertia of ship around center of gravity
# IzG = (2*IzG_) / (1025*(7**5))
# lenthofShip = 7.0


# Xau= (2*174.994)/(1025*(7**3)) # accelaration derivative of surge force with respect to u 

# Yav = (2*1702.661)/(1025*(7**3)) # accelaration derivative of sway force with respect to v
# Yar = (2*1273.451)/(1025*(7**4)) # accelaration derivative of sway force with respect to r

# Nav = (2*1273.451)/(1025*(7**4)) # Yaw moment derivative with respect to sway velocity
# Nar = (2*9117.302)/ (1025*(7**5)) # Yaw moment derivative with respect to rudder angle

# S = ((IzG-Nar)*(m-Yav))-(((m*Xg)-Yar)*((m*Xg)-Nav))

"""
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
    return h/(L*(m-Xau))

CC1 = term1(h,L,m,Xau)
t1 = X[1:]

surget = [i for i in map(lambda x : CC1*x, t1)]

surge = []
for elem in surget:
    surge.append(int(elem))


sl_no = np.arange(1,16)
surge_hydrodynamic_derivatives = ['Xu', 'Xuu', 'Xuuu', 'Xvv', 'Xrr',
                                 'Xðð', 'Xððu', 'Xvr', 'Xvð', 'Xvðu',
                                 'Xuvv', 'Xurr', 'Xuvr','Xrð', 'Xurð']

surge_predicted = [-184,-110,-215,-899,18,
                   -95,-190,798,93,93,
                   0,0,0,0,0]

table1 = zip(sl_no,surge_hydrodynamic_derivatives,surge,surge_predicted)
headers1 = ['sl_no','Surge_hydrodynamic_derivatives','Predicted_value','Actual_Value']

surge_table = tabulate(table1, headers1, tablefmt="pretty")
print(surge_table)


np.savetxt('surge.txt', surge)
text_file=open("surge.csv","w")
text_file.write(str(surge_table))
text_file.close()


