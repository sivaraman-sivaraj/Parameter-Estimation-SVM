import numpy as np
import time
from tabulate import tabulate

Y = np.load('sway_regression_coeff.npy')
R = np.load('yaw_regression_coeff.npy')


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
End of features description
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

sway and yaw solution
"""

def term2(h,IzG,Nav,S,L):
    return h*(IzG-Nav)/(S*L)

def term3(h,m,Xg,Yar,S,L):
    return (-h)*((m*Xg)-Yar)/(S*L)

def term4(h,m,Xg,Nav,S,L):
    return (-h)*(m*Xg-Nav)/(S*L**2)

def term5(h,m,Yav,S,L):
    return h*(m-Yav)/(S*L**2)

M11= term2(h,IzG,Nav,S,L)
M12= term3(h,m,Xg,Yar,S,L)
M21= term4(h,m,Xg,Nav,S,L)
M22= term5(h,m,Yav,S,L)

solMatrix = np.array([[M11,M12],[M21,M22]])


t2 = Y[1:]
t3 = R[1:]


def two_one_Matrix(t2,t3):
    List = []
    for i in range(len(t2)):
        temp = np.array([[t2[i]],[t3[i]]])
        List.append(temp)
    return List

c = two_one_Matrix(t2,t3)

def SNsolution(M,c): #sway and yaw moment solution
    List = []
    im = np.linalg.inv(M)
    for i in range(len(c)):
        temp = im.dot(c[i])
        List.append(temp)
    return List

Sway_Yaw_derivatives = SNsolution(solMatrix,c)

def separation(M):
    sway_components = []
    yaw_components = []
    for i in M:
        sway_components.append(int(i[0][0]))
        yaw_components.append(int(i[1][0]))
    return sway_components,yaw_components

sway, yaw = separation(Sway_Yaw_derivatives)

sl_no = np.arange(1,23)
sway_hydrodynamic_derivatives = ['Yo','You','Youu','Yv','Yr',
                                 'Yð','Yvvv','Yððð','Yvvr','Yvvð',
                                 'Yvðð','Yðu','Yvu','Yru','Yðuu',
                                 'Yrrr','Yvrr','Yvuu','Yruu','Yrðð',
                                 'Yrrð','Yrvð']

yaw_hydrodynamic_derivatives = ['No','Nou','Nouu','Nv','Nr',
                                'Nð','Nvvv','Nððð','Nvvr','Nvvð',
                                'Nvðð','Nðu','Nvu','Nru','Nðuu',
                                'Nrrr','Nvrr','Nvuu','Nruu','Nrðð',
                                'Nrrð','Nrvð']

sway_original = [-4,-8,-4,-1160,-499,
                  278,-8078,-90,15356,1190,
                  -4,556,-1160,-499,278,
                  0.008,-0.391,0,0,0,
                  0,0,0]

yaw_original = [3,6,3,-264,-166,
                  -139,1636,45,-5483,-489,
                  13,-278,-264,0,-139,
                  -0.013,0.055,0,-166,0,
                  0,0,0]

table1 = zip(sl_no,yaw_hydrodynamic_derivatives,yaw,yaw_original)
headers1 = ['sl_no','yaw_hydrodynamic_derivatives','Predicted_value','Actual_Value']

yaw_table = tabulate(table1, headers1, tablefmt="pretty")
print(yaw_table)

table2 = zip(sl_no,sway_hydrodynamic_derivatives,sway,sway_original )
headers2 = ['sl_no','sway_hydrodynamic_derivatives','Predicted_value','Actual_Value']

sway_table = tabulate(table2, headers2, tablefmt="pretty")
print(sway_table)

np.savetxt("yaw_hdy.csv",yaw)
np.savetxt("sway_hdy.csv",sway)

# print(solMatrix)



Cov_zero = np.zeros((15,15))

def cov_diagonal(cov, hd_l):
    op = cov
    for i in range((15)):
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

_cov_sway = np.cov(sway_d)
_cov_yaw = np.cov(yaw_d)

cov_sway = upper(_cov_sway, 15,15)
cov_yaw = upper(_cov_yaw, 15,15)

np.savetxt("sway_covariance.csv",cov_sway)
np.savetxt("yaw_covariance.csv",cov_yaw)

# print(cov_sway)












