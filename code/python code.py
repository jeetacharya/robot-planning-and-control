#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import numpy as np
import math
import modern_robotics as mr
import matplotlib.pyplot as plt
ci=[]
print('Specify the initial configuration of the cube having 3 elements of the vector : x,y,theta')
for m in range(0,3):
    ele1=float(input())
    ci.append(ele1)                                                                                               # receiving initial cube configuration as input
cf=[]
print('Specify the final configuration of the cube having 3 elements of the vector : x,y,theta')
for r in range(0,3):
    ele2=float(input())
    cf.append(ele2)                                                                                               # receiving final cube configuration as input
rid=[]
print('Specify the initial desired configuration of the youbot having 12 elements of the vector : transformation matrix with no last row')
for s in range(0,12):
    ele3=float(input())
    rid.append(ele3)                                                                                              # receiving initial youbot configuration as input in the form of transformation matrix
def TrajectoryGenerator():                                # function to create the desired trajectory or the reference trajectory
    tmatrix=pd.DataFrame(columns=['r11','r12','r13','r21','r22','r23','r31','r32','r33','px','py','pz','gripper'])                     
    tsei1=np.array([[rid[0],rid[1],rid[2],rid[3]],[rid[4],rid[5],rid[6],rid[7]],[rid[8],rid[9],rid[10],rid[11]],[0, 0, 0, 1]])         # Tse at initial stage of process 1
    tsee1=np.array([[0,0,1,ci[0]],[math.sin(ci[2]),math.cos(ci[2]),0,ci[1]],[-math.cos(ci[2]),math.sin(ci[2]),0,0.25],[0,0,0,1]])      # Tse at end stage of process 1
    tsei2=np.array([[0,0,1,ci[0]],[math.sin(ci[2]),math.cos(ci[2]),0,ci[1]],[-math.cos(ci[2]),math.sin(ci[2]),0,0.25],[0,0,0,1]])
    tsee2=np.array([[0,0,1,ci[0]],[math.sin(ci[2]),math.cos(ci[2]),0,ci[1]],[-math.cos(ci[2]),math.sin(ci[2]),0,0.025],[0,0,0,1]])
    tsei4=np.array([[0,0,1,ci[0]],[math.sin(ci[2]),math.cos(ci[2]),0,ci[1]],[-math.cos(ci[2]),math.sin(ci[2]),0,0.025],[0,0,0,1]])
    tsee4=np.array([[0,0,1,ci[0]],[math.sin(ci[2]),math.cos(ci[2]),0,ci[1]],[-math.cos(ci[2]),math.sin(ci[2]),0,0.25],[0,0,0,1]])
    tsei5=np.array([[0,0,1,ci[0]],[math.sin(ci[2]),math.cos(ci[2]),0,ci[1]],[-math.cos(ci[2]),math.sin(ci[2]),0,0.25],[0,0,0,1]])
    tsee5=np.array([[0,-math.sin(cf[2]),math.cos(cf[2]),cf[0]],[math.sin(ci[2]),(math.cos(ci[2])*math.cos(cf[2])),(math.cos(ci[2])*math.sin(cf[2])),cf[1]],[-math.cos(ci[2]),(math.sin(ci[2])*math.cos(cf[2])),(math.sin(ci[2])*math.sin(cf[2])),0.25],[0,0,0,1]])
    tsei6=np.array([[0,-math.sin(cf[2]),math.cos(cf[2]),cf[0]],[math.sin(ci[2]),(math.cos(ci[2])*math.cos(cf[2])),(math.cos(ci[2])*math.sin(cf[2])),cf[1]],[-math.cos(ci[2]),(math.sin(ci[2])*math.cos(cf[2])),(math.sin(ci[2])*math.sin(cf[2])),0.25],[0,0,0,1]])
    tsee6=np.array([[0,-math.sin(cf[2]),math.cos(cf[2]),cf[0]],[math.sin(ci[2]),(math.cos(ci[2])*math.cos(cf[2])),(math.cos(ci[2])*math.sin(cf[2])),cf[1]],[-math.cos(ci[2]),(math.sin(ci[2])*math.cos(cf[2])),(math.sin(ci[2])*math.sin(cf[2])),0.025],[0,0,0,1]])
    tsei8=np.array([[0,-math.sin(cf[2]),math.cos(cf[2]),cf[0]],[math.sin(ci[2]),(math.cos(ci[2])*math.cos(cf[2])),(math.cos(ci[2])*math.sin(cf[2])),cf[1]],[-math.cos(ci[2]),(math.sin(ci[2])*math.cos(cf[2])),(math.sin(ci[2])*math.sin(cf[2])),0.025],[0,0,0,1]])
    tsee8=np.array([[0,-math.sin(cf[2]),math.cos(cf[2]),cf[0]],[math.sin(ci[2]),(math.cos(ci[2])*math.cos(cf[2])),(math.cos(ci[2])*math.sin(cf[2])),cf[1]],[-math.cos(ci[2]),(math.sin(ci[2])*math.cos(cf[2])),(math.sin(ci[2])*math.sin(cf[2])),0.25],[0,0,0,1]])
    tsei=[tsei1,tsei2,tsei4,tsei5,tsei6,tsei8]
    tsee=[tsee1,tsee2,tsee4,tsee5,tsee6,tsee8]
    for i in range(len(tsei)):
        lr11=lr12=lr13=lrpx=lr21=lr22=lr23=lrpy=lr31=lr32=lr33=lrpz=[]
        if i==2:                                  # grasping the cube requires 63 lines of code with all constant values of T matrix with different gripper state
            lr11=[0]*63
            lr12=[0]*63
            lr13=[1]*63
            lrpx=[ci[0]]*63
            lr21=[math.sin(ci[2])]*63
            lr22=[math.cos(ci[2])]*63
            lr23=[0]*63
            lrpy=[ci[1]]*63
            lr31=[-math.cos(ci[2])]*63
            lr32=[math.sin(ci[2])]*63
            lr33=[0]*63
            lrpz=[0.025]*63
            lrgr=[1]*63
            extra={'r11':lr11,'r12':lr12,'r13':lr13,'r21':lr21,'r22':lr22,'r23':lr23,'r31':lr31,'r32':lr32,'r33':lr33,'px':lrpx,'py':lrpy,'pz':lrpz,'gripper':lrgr}
            tmatrix=tmatrix.append(pd.DataFrame(extra))
        elif i==5:                                # releasing the cube requires 63 lines of code with all constant values of T matrix with different gripper state
            lr11=[0]*63
            lr12=[-math.sin(cf[2])]*63
            lr13=[math.cos(cf[2])]*63
            lrpx=[cf[0]]*63
            lr21=[math.sin(ci[2])]*63
            lr22=[(math.cos(ci[2])*math.cos(cf[2]))]*63
            lr23=[(math.cos(ci[2])*math.sin(cf[2]))]*63
            lrpy=[cf[1]]*63
            lr31=[-math.cos(ci[2])]*63
            lr32=[(math.sin(ci[2])*math.cos(cf[2]))]*63
            lr33=[(math.sin(ci[2])*math.sin(cf[2]))]*63
            lrpz=[0.025]*63
            lrgr=[0]*63    
            extra={'r11':lr11,'r12':lr12,'r13':lr13,'r21':lr21,'r22':lr22,'r23':lr23,'r31':lr31,'r32':lr32,'r33':lr33,'px':lrpx,'py':lrpy,'pz':lrpz,'gripper':lrgr}
            tmatrix=tmatrix.append(pd.DataFrame(extra))        
        Xstart=tsei[i]
        Xend=tsee[i]
        if i==0 or i==3:                          # motions corresponding to movement of chassis requires more time
            Tf=8
        else:                                     # motions corresponding to no movement of chassis requires less time
            Tf=2
        N=Tf/0.01
        method=5                                  # quintic polynomial trajectory motion
        if i==0 or i==3:
            trajectory=mr.ScrewTrajectory(Xstart, Xend, Tf, N, method)             # screw motion
        else:    
            trajectory=mr.CartesianTrajectory(Xstart, Xend, Tf, N, method)         # straight line motion
        a=np.array(trajectory)
        lr11=lr12=lr13=lrpx=lr21=lr22=lr23=lrpy=lr31=lr32=lr33=lrpz=[]
        lr11=a[:,0,0]
        lr12=a[:,0,1]
        lr13=a[:,0,2]
        lrpx=a[:,0,3]
        lr21=a[:,1,0]
        lr22=a[:,1,1]
        lr23=a[:,1,2]
        lrpy=a[:,1,3]
        lr31=a[:,2,0]
        lr32=a[:,2,1]
        lr33=a[:,2,2]
        lrpz=a[:,2,3]
        if i==0 or i==1 or i==5:
            lrgr=[0]*len(lrpz)
        else:
            lrgr=[1]*len(lrpz)
        extra={'r11':lr11,'r12':lr12,'r13':lr13,'r21':lr21,'r22':lr22,'r23':lr23,'r31':lr31,'r32':lr32,'r33':lr33,'px':lrpx,'py':lrpy,'pz':lrpz,'gripper':lrgr}
        tmatrix=tmatrix.append(pd.DataFrame(extra))
    tmatrix.index = np.arange(1,len(tmatrix)+1)
    tmatrix.to_csv(r'tmatrix.csv', sep=',', header=True, mode='w')
trajectorygenerator=TrajectoryGenerator()                                                                                # creating the csv file for trajectory generation matrices
xdmatrix=pd.read_csv(r'tmatrix.csv',index_col=0)
print('Specify the joint velocity limit for joint 1')                          # limiting joint 1 velocity as input
dthetalimit1=float(input())
print('Specify the joint velocity limit for joint 2')
dthetalimit2=float(input())
print('Specify the joint velocity limit for joint 3')
dthetalimit3=float(input())
print('Specify the joint velocity limit for joint 4')
dthetalimit4=float(input())
print('Specify the joint velocity limit for joint 5')
dthetalimit5=float(input())
print('Specify the wheel velocity limit for wheel 1')                          # limiting wheel 1 velocity as input
ulimit1=float(input())
print('Specify the wheel velocity limit for wheel 2')
ulimit2=float(input())
print('Specify the wheel velocity limit for wheel 3')
ulimit3=float(input())
print('Specify the wheel velocity limit for wheel 4')
ulimit4=float(input())
print('Specify the timestep between successive configurations')                # timestep=dt as input
dt=float(input())
row1=[]
print('Specify the initial actual configuration of the youBot having 13 elements of the vector')
for l in range(0,13):
    ele=float(input())
    row1.append(ele)                                                           # 13-element vector of the initial actual configuration of the youBot as input
configuration=pd.DataFrame(columns=['phi','x','y','j1','j2','j3','j4','j5','w1','w2','w3','w4','gripper'])
configuration.loc[len(configuration)] = row1
configuration.index = np.arange(1,len(configuration)+1)
tbo=np.array([[1,0,0,0.1662],[0,1,0,0],[0,0,1,0.0026],[0,0,0,1]])              # Tb0
moe=np.array([[1,0,0,0.033],[0,1,0,0],[0,0,1,0.6546],[0,0,0,1]])               # M0e
b=np.array([[0,0,1,0,0.033,0],[0,-1,0,-0.5076,0,0],[0,-1,0,-0.3526,0,0],[0,-1,0,-0.2176,0,0],[0,0,1,0,0,0]]).T             # list of screws
xerrtotal=0
xerror=pd.DataFrame(columns=['col1','col2','col3','col4','col5','col6'])
u1,u2,u3,u4,dtheta1,dtheta2,dtheta3,dtheta4,dtheta5=0,0,0,0,0,0,0,0,0          # initializing to 0 for first iteration only
for v in range(2524):
    for i in range(len(configuration)+1,len(configuration)+2):
        if abs(dtheta1)<=dthetalimit1:
            configuration['j1'][i]=configuration['j1'][i-1] + dtheta1*dt
        else:
            if dtheta1<=0:
                configuration['j1'][i]=configuration['j1'][i-1] + dthetalimit1*(-1)*dt
            else:
                configuration['j1'][i]=configuration['j1'][i-1] + dthetalimit1*dt
        if abs(dtheta2)<=dthetalimit2:
            configuration['j2'][i]=configuration['j2'][i-1] + dtheta2*dt
        else:
            if dtheta2<=0:
                configuration['j2'][i]=configuration['j2'][i-1] + dthetalimit2*(-1)*dt
            else:
                configuration['j2'][i]=configuration['j2'][i-1] + dthetalimit2*dt
        if abs(dtheta3)<=dthetalimit3:
            configuration['j3'][i]=configuration['j3'][i-1] + dtheta3*dt
        else:
            if dtheta3<=0:
                configuration['j3'][i]=configuration['j3'][i-1] + dthetalimit3*(-1)*dt
            else:
                configuration['j3'][i]=configuration['j3'][i-1] + dthetalimit3*dt
        if abs(dtheta4)<=dthetalimit4:
            configuration['j4'][i]=configuration['j4'][i-1] + dtheta4*dt
        else:
            if dtheta4<=0:
                configuration['j4'][i]=configuration['j4'][i-1] + dthetalimit4*(-1)*dt
            else:
                configuration['j4'][i]=configuration['j4'][i-1] + dthetalimit4*dt
        if abs(dtheta5)<=dthetalimit5:
            configuration['j5'][i]=configuration['j5'][i-1] + dtheta5*dt
        else:
            if dtheta5<=0:
                configuration['j5'][i]=configuration['j5'][i-1] + dthetalimit5*(-1)*dt
            else:
                configuration['j5'][i]=configuration['j5'][i-1] + dthetalimit5*dt
        if abs(u1)<=ulimit1:
            configuration['w1'][i]=configuration['w1'][i-1] + u1*dt
            angle1=u1*dt
        else:
            if u1<=0:
                configuration['w1'][i]=configuration['w1'][i-1] + ulimit1*(-1)*dt
                angle1=ulimit1*(-1)*dt
            else:
                configuration['w1'][i]=configuration['w1'][i-1] + ulimit1*dt
                angle1=ulimit1*dt
        if abs(u2)<=ulimit2:
            configuration['w2'][i]=configuration['w2'][i-1] + u2*dt
            angle2=u2*dt
        else:
            if u2<=0:
                configuration['w2'][i]=configuration['w2'][i-1] + ulimit2*(-1)*dt
                angle2=ulimit2*(-1)*dt
            else:
                configuration['w2'][i]=configuration['w2'][i-1] + ulimit2*dt
                angle2=ulimit2*dt
        if abs(u3)<=ulimit3:
            configuration['w3'][i]=configuration['w3'][i-1] + u3*dt
            angle3=u3*dt
        else:
            if u3<=0:
                configuration['w3'][i]=configuration['w3'][i-1] + ulimit3*(-1)*dt
                angle3=ulimit3*(-1)*dt
            else:
                configuration['w3'][i]=configuration['w3'][i-1] + ulimit3*dt
                angle3=ulimit3*dt
        if abs(u4)<=ulimit4:
            configuration['w4'][i]=configuration['w4'][i-1] + u4*dt
            angle4=u4*dt
        else:
            if u4<=0:
                configuration['w4'][i]=configuration['w4'][i-1] + ulimit4*(-1)*dt
                angle4=ulimit4*(-1)*dt
            else:
                configuration['w4'][i]=configuration['w4'][i-1] + ulimit4*dt
                angle4=ulimit4*dt
        configuration['gripper'][i]=configuration['gripper'][i-1]
        omegabz=0.0308*(((-1)*angle1)+(angle2)+(angle3)-(angle4))               # steps for odometry
        vbx=0.0118*(angle1+angle2+angle3+angle4)
        vby=0.0118*(((-1)*angle1)+angle2-(angle3)+angle4)
        if omegabz==0:
            deltaphib=0
            deltaxb=vbx
            deltayb=vby
        else:
            deltaphib=omegabz
            deltaxb=(((vbx)*math.sin(omegabz))+((vby)*(math.cos(omegabz)-1)))/(omegabz)
            deltayb=(((vby)*math.sin(omegabz))+((vbx)*(1-math.cos(omegabz))))/(omegabz)
        configuration['phi'][i]=configuration['phi'][i-1]+omegabz
        configuration['x'][i]=configuration['x'][i-1]+(deltaxb*math.cos(configuration['phi'][i-1]))-(deltayb*math.sin(configuration['phi'][i-1]))
        configuration['y'][i]=configuration['y'][i-1]+(deltaxb*math.sin(configuration['phi'][i-1]))+(deltayb*math.cos(configuration['phi'][i-1]))
        configuration['gripper'][i]=xdmatrix['gripper'][i]
        row = [configuration['phi'][i],configuration['x'][i],configuration['y'][i],configuration['j1'][i],configuration['j2'][i],configuration['j3'][i],configuration['j4'][i],configuration['j5'][i],configuration['w1'][i],configuration['w2'][i],configuration['w3'][i],configuration['w4'][i],configuration['gripper'][i]]
        configuration.loc[len(configuration)+1] = row
    configuration.to_csv(r'configuration.csv', sep=',', header=True, mode='w')
    xmatrix=pd.read_csv(r'configuration.csv',index_col=0)
    tsb=np.array([[math.cos(xmatrix['phi'][len(xmatrix)]),-math.sin(xmatrix['phi'][len(xmatrix)]),0,xmatrix['x'][len(xmatrix)]],[math.sin(xmatrix['phi'][len(xmatrix)]),math.cos(xmatrix['phi'][len(xmatrix)]),0,xmatrix['y'][len(xmatrix)]],[0,0,1,0.0963],[0,0,0,1]])
    thetal1=np.array([xmatrix['j1'][len(xmatrix)],xmatrix['j2'][len(xmatrix)],xmatrix['j3'][len(xmatrix)],xmatrix['j4'][len(xmatrix)],xmatrix['j5'][len(xmatrix)]])
    M=moe
    Blist=b
    thetalist=thetal1
    toe=mr.FKinBody(M, Blist, thetalist)                              # calculating T0e
    tso=tsb.dot(tbo)
    x=tso.dot(toe)                                                    # calculating Tse which is also X (actual configuration of youbot)
    xd=np.array([[xdmatrix['r11'][len(xmatrix)],xdmatrix['r12'][len(xmatrix)],xdmatrix['r13'][len(xmatrix)],xdmatrix['px'][len(xmatrix)]],[xdmatrix['r21'][len(xmatrix)],xdmatrix['r22'][len(xmatrix)],xdmatrix['r23'][len(xmatrix)],xdmatrix['py'][len(xmatrix)]],[xdmatrix['r31'][len(xmatrix)],xdmatrix['r32'][len(xmatrix)],xdmatrix['r33'][len(xmatrix)],xdmatrix['pz'][len(xmatrix)]],[0,0,0,1]])
    xdnext=np.array([[xdmatrix['r11'][len(xmatrix)+1],xdmatrix['r12'][len(xmatrix)+1],xdmatrix['r13'][len(xmatrix)+1],xdmatrix['px'][len(xmatrix)+1]],[xdmatrix['r21'][len(xmatrix)+1],xdmatrix['r22'][len(xmatrix)+1],xdmatrix['r23'][len(xmatrix)+1],xdmatrix['py'][len(xmatrix)+1]],[xdmatrix['r31'][len(xmatrix)+1],xdmatrix['r32'][len(xmatrix)+1],xdmatrix['r33'][len(xmatrix)+1],xdmatrix['pz'][len(xmatrix)+1]],[0,0,0,1]])
    im1=mr.TransInv(x)                                                # all variable starting with im are intermediate matrices used for calculations only
    im2=im1.dot(xd)
    im3=mr.MatrixLog6(im2)
    xerr=mr.se3ToVec(im3)
    xerrtotal=xerrtotal+(xerr*dt)                                     # adding all errors dynamically
    im4=mr.TransInv(xd)
    im5=im4.dot(xdnext)
    im6=mr.MatrixLog6(im5)
    im7=im6/dt
    nud=mr.se3ToVec(im7)
    im8=mr.Adjoint(im2)
    kpmat=np.diag(np.full(6,2))                                       # proportional gain
    kimat=np.diag(np.full(6,1))                                       # integral gain
    nu=(im8.dot(nud))+(kpmat.dot(xerr))+(kimat.dot(xerrtotal))        # task space motion control
    jarm=mr.JacobianBody(Blist, thetalist)
    teo=mr.TransInv(toe)
    tob=mr.TransInv(tbo)
    im9=teo.dot(tob)
    im10=mr.Adjoint(im9)
    f6mat=np.array([[0,0,0,0],[0,0,0,0],[-0.0308,0.0308,0.0308,-0.0308],[0.0118,0.0118,0.0118,0.0118],[-0.0118,0.0118,-0.0118,0.0118],[0,0,0,0]])
    jbase=im10.dot(f6mat)
    je=np.concatenate((jbase,jarm),axis=1)                            # whole jacobian
    utheta=np.dot(np.linalg.pinv(je),nu)                              # list of u and theta values
    u1=utheta[0]
    u2=utheta[1]
    u3=utheta[2]
    u4=utheta[3]
    dtheta1=utheta[4]
    dtheta2=utheta[5]
    dtheta3=utheta[6]
    dtheta4=utheta[7]
    dtheta5=utheta[8]
    xerror.loc[v]=xerr                                                 # saving all errors
    xerror.to_csv(r'error.csv', sep=',', header=True, mode='w')
x = list(np.linspace(0,25.24,2524,endpoint=False))                                                             # list of x coordinates from 0s to 25.23s with steps of 0.01s
errormat=pd.read_csv(r'error.csv')      # reading the csv file containing 6-dimensional errors that have step size of 0.01s
y1 = list(errormat.loc[:,'col1'])                                                                              # list of y coordinates indicating first column of errors
y2 = list(errormat.loc[:,'col2'])                                                                              
y3 = list(errormat.loc[:,'col3'])                                                                              
y4 = list(errormat.loc[:,'col4'])                                                                              
y5 = list(errormat.loc[:,'col5'])
y6 = list(errormat.loc[:,'col6'])
plt.plot(x, y1, color='green',label='line1')                                                                   # error values corresponding to first column in error file has green colour and has been labelled as 'line1'
plt.plot(x, y2, color='red',label='line2') 
plt.plot(x, y3, color='blue',label='line3') 
plt.plot(x, y4, color='orange',label='line4') 
plt.plot(x, y5, color='yellow',label='line5') 
plt.plot(x, y6, color='black',label='line6') 
plt.ylim(-2,2)                                                                                                 # range of values on y-axis for the graph
plt.xlim(0,26)                                                                                                 # range of values on x-axis for the graph
plt.xlabel('time')                                                                                             # x-axis shows time
plt.ylabel('error')                                                                                            # y-axis shows the corresponding error
plt.title('Error plot')                                                                                        # title of the plot
plt.legend()                                                                                                   # legend is given on plot
plt.savefig(r'errorplot.pdf')           # save the plot at this location in pdf format


# In[ ]:




