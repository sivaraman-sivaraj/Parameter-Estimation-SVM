import scipy.io
import numpy as np
from tabulate import tabulate


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
u = data[:,1]# raw values
v = data[:,2]
r = data[:,3]
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


# u = delta_val(_u,7.7175)
# v = delta_val(_v,-8.81109e-05)
# r = delta_val(_r, 0.000467568)
rac = delta_val(_rac)

def align_by_one(var):
    op = var[1:]
    return op

U = align_by_one(_U)

"Different number of samples"
surge_50 = u[::120]
surge_100 = u[::60]
surge_300 = u[::20]
surge_600 = u[::10]

delu_s50 = delta_val(surge_50)
delu_s100 = delta_val(surge_100)
delu_s300 = delta_val(surge_300)
delu_s600 = delta_val(surge_600)


std_u50 = np.std(delu_s50)
std_u100 = np.std(delu_s100)
std_u300 = np.std(delu_s300)
std_u600 = np.std(delu_s600)


# print(std_u50,std_u100,std_u300, std_u600 )


sway_50 = v[::120]
sway_100 = v[::60]
sway_300 = v[::20]
sway_600 = v[::10]

delv_s50 = delta_val(sway_50)
delv_s100 = delta_val(sway_100)
delv_s300 = delta_val(sway_300)
delv_s600 = delta_val(sway_600)


std_v50 = np.std(delv_s50)
std_v100 = np.std(delv_s100)
std_v300 = np.std(delv_s300)
std_v600 = np.std(delv_s600)


# print(std_v50,std_v100,std_v300,std_v600 )

yaw_50 = r[::120]
yaw_100 = r[::60]
yaw_300 = r[::20]
yaw_600 = r[::10]

delv_r50 = delta_val(yaw_50)
delv_r100 = delta_val(yaw_100)
delv_r300 = delta_val(yaw_300)
delv_r600 = delta_val(yaw_600)

std_r50 = np.std(delv_r50)
std_r100 = np.std(delv_r100)
std_r300 = np.std(delv_r300)
std_r600 = np.std(delv_r600)


# print(std_r50,std_r100,std_r300,std_r600 )

N = [50,100, 300, 600]
SD_surge = [round(std_u50,8),round(std_u100,8),round(std_u300,8), round(std_u600,8)]
SD_sway = [round(std_v50,8),round(std_v100,8),round(std_v300,8),round(std_v600,8)]
SD_yaw = [round(std_r50,8),round(std_r100,8),round(std_r300,8),round(std_r600,8) ]


table1 = zip(N, SD_surge,SD_sway, SD_yaw)
headers1 = ['N', 'SD(surge)', 'SD(sway)', 'SD(yaw)']

SD_table = tabulate(table1, headers1, tablefmt="pretty")
print(SD_table)

