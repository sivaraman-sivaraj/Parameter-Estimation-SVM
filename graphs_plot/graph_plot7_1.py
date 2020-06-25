import scipy.io
import matplotlib.pyplot as plt
import numpy as np



mat = scipy.io.loadmat('OUTPUT.mat')

"""
7*1 format

1. time
2.surge_speed
3.sway_speed
4.yaw_rate
5.yaw_angle
6.total_speed
7.rudder_angle
"""

ss = mat["OUTPUT"]

u = ss[:,1]
v = ss[:,2]
r = ss[:,3]
yaw_angle = ss[:,4]
U = ss[:,5]
rac = ss[:,6]



def graph13():
    plt.figure(figsize=(15,12))
    plt.subplot(311)
    x212 = np.arange(0,len(u))
    plt.plot(x212,u,'g')
    plt.xlabel('time in 0.1 seconds increment')
    plt.ylabel('Surge Speed')
    plt.axvline(x=1500,color = 'r')
      
    plt.subplot(312)
    x211 = np.arange(0,len(v))
    plt.plot(x211,v,'m')
    plt.xlabel('time in 0.1 seconds increment')
    plt.ylabel('Sway Speed')
    plt.axvline(x=1500,color = 'r')
    
    plt.subplot(313)
    x313 = np.arange(0,len(r))
    plt.plot(x313,r,'b')
    plt.xlabel('time in 0.1 seconds increment')
    plt.ylabel('yaw rate')
    plt.axvline(x=1500,color = 'r')    
    # plt.savefig('plot_1.jpg')
    plt.show()
    
    
    
graph13()


def graph46():
    plt.figure(figsize=(15,12))
    plt.subplot(211)
    x212 = np.arange(0,len(yaw_angle))
    plt.plot(x212,yaw_angle,'g')
    plt.plot(x212,rac,'m')
    plt.xlabel('time in 0.1 seconds increment')
    plt.ylabel('Yaw Angle')
    plt.title('Heading angle (Ψ) vs Rudder Angle(𝛿)')
    plt.axvline(x=1500,color = 'r')
      
       
    plt.subplot(212)
    x313 = np.arange(0,len(U))
    plt.plot(x313,U,'b')
    plt.xlabel('time in 0.1 seconds increment')
    plt.ylabel('Total Speed')
    plt.axvline(x=1500,color = 'r')
    # plt.savefig('plot_2.jpg')    
    plt.show()


graph46()
