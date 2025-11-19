






#!/usr/bin/python2

## UR5/UR10 Inverse Kinematics - Ryan Keating Johns Hopkins University


# ***** lib
import numpy as np
from numpy import linalg
import mujoco

import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

global mat
mat=np.matrix


# ****** Coefficients ******
UR10E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

global d1, a2, a3, a7, d4, d5, d6
d1 =  0.1807
a2 = -0.612
a3 = -0.5723
a7 = 0.075
d4 =  0.17415
d5 =  0.11985
d6 =  0.11655

global d, a, alph

#d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823]) ur5
d = mat([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])#ur10 mm
# a =mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0]) ur5
a =mat([0 ,-0.612 ,-0.5723 ,0 ,0 ,0])#ur10 mm
#alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])  #ur5
alph = mat([pi/2, 0, 0, pi/2, -pi/2, 0 ]) # ur10


# ************************************************** FORWARD KINEMATICS

def AH( n,th,c  ):

  T_a = mat(np.identity(4), copy=False)
  T_a[0,3] = a[0,n-1]
  T_d = mat(np.identity(4), copy=False)
  T_d[2,3] = d[0,n-1]

  Rzt = mat([[cos(th[n-1,c]), -sin(th[n-1,c]), 0 ,0],
	         [sin(th[n-1,c]),  cos(th[n-1,c]), 0, 0],
	         [0,               0,              1, 0],
	         [0,               0,              0, 1]],copy=False)
      

  Rxa = mat([[1, 0,                 0,                  0],
			 [0, cos(alph[0,n-1]), -sin(alph[0,n-1]),   0],
			 [0, sin(alph[0,n-1]),  cos(alph[0,n-1]),   0],
			 [0, 0,                 0,                  1]],copy=False)

  A_i = T_d * Rzt * T_a * Rxa
	    

  return A_i

def HTrans(th,c ):  
  A_1=AH( 1,th,c  )
  A_2=AH( 2,th,c  )
  A_3=AH( 3,th,c  )
  A_4=AH( 4,th,c  )
  A_5=AH( 5,th,c  )
  A_6=AH( 6,th,c  )
      
  T_06=A_1*A_2*A_3*A_4*A_5*A_6

  return T_06

# ************************************************** INVERSE KINEMATICS 

def invKine(desired_pos, model, data):# T60
  th = mat(np.zeros((6, 8)))
  P_05 = (desired_pos * mat([0,0, -d6, 1]).T-mat([0,0,0,1 ]).T)
  joint_ids = [int(model.joint(name).dofadr) for name in UR10E_JOINTS]
  q = np.array([data.qpos[jid] for jid in joint_ids])
  # **** theta1 ****
  site_id = model.site("attachment_site").id
  psi = atan2(P_05[2-1,0], P_05[1-1,0])
  phi = acos(d4 /sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]))
  #The two solutions for theta1 correspond to the shoulder
  #being either left or right
  th[0, 0:4] = pi/2 + psi + phi
  th[0, 4:8] = pi/2 + psi - phi
  th = th.real
  
  # **** theta5 ****
  
  cl = [0, 4]# wrist up or down
  for i in range(0,len(cl)):
    c = cl[i]
    T_10 = linalg.inv(AH(1,th,c))
    T_16 = T_10 * desired_pos
    th[4, c:c+2] = + acos((T_16[2,3]-d4)/d6);
    th[4, c+2:c+4] = - acos((T_16[2,3]-d4)/d6);

  th = th.real
  
  # **** theta6 ****
  # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

  cl = [0, 2, 4, 6]
  for i in range(0,len(cl)):
    c = cl[i]
    T_10 = linalg.inv(AH(1,th,c))
    T_16 = linalg.inv( T_10 * desired_pos )
    th[5, c:c+2] = atan2((-T_16[1,2]/sin(th[4, c])),(T_16[0,2]/sin(th[4, c])))
		  
  th = th.real

  # **** theta3 ****
  cl = [0, 2, 4, 6]
  for i in range(0,len(cl)):
    c = cl[i]
    T_10 = linalg.inv(AH(1,th,c))
    T_65 = AH( 6,th,c)
    T_54 = AH( 5,th,c)
    T_14 = ( T_10 * desired_pos) * linalg.inv(T_54 * T_65)
    P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T
    t3 = cmath.acos((linalg.norm(P_13)**2 - a2**2 - a3**2 )/(2 * a2 * a3)) # norm ?
    th[2, c] = t3.real
    th[2, c+1] = -t3.real

  # **** theta2 and theta 4 ****

  cl = [0, 1, 2, 3, 4, 5, 6, 7]
  for i in range(0,len(cl)):
    c = cl[i]
    T_10 = linalg.inv(AH( 1,th,c ))
    T_65 = linalg.inv(AH( 6,th,c))
    T_54 = linalg.inv(AH( 5,th,c))
    T_14 = (T_10 * desired_pos) * T_65 * T_54
    P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T
    
    # theta 2
    th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(a3* sin(th[2,c])/linalg.norm(P_13))
    # theta 4
    T_32 = linalg.inv(AH( 3,th,c))
    T_21 = linalg.inv(AH( 2,th,c))
    T_34 = T_32 * T_21 * T_14
    th[3, c] = atan2(T_34[1,0], T_34[0,0])
  th = th.real

  return th[:,0].A1

#
#
#   HOLD VERSION!!!
#
#


import numpy as np
import mujoco

# UR10e joint names
UR10E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

def inverse_kinematics(target_pos, model, data_model, 
                                 target_rot=np.eye(3), damping=1e-1, 
                                 max_iter=50, tol=1e-3):
    """
    Iterative IK for UR10e that avoids going to zero angles.
    Starts from current joint positions in data_model.
    """
    # Numeric IDs of joints
    joint_ids = [int(model.joint(name).dofadr) for name in UR10E_JOINTS]
    site_id = model.site("attachment_site").id

    # Take current joint positions as initial guess
    q = np.array([data_model.qpos[jid] for jid in joint_ids])

    for it in range(max_iter):


        # Current TCP pose
        current_pos = data_model.site_xpos[site_id]
        pos_err = target_pos - current_pos

        # Orientation error
        current_rot = data_model.site_xmat[site_id].reshape(3,3)
        rot_err = 0.5 * (np.cross(current_rot[:,0], target_rot[:,0]) +
                         np.cross(current_rot[:,1], target_rot[:,1]) +
                         np.cross(current_rot[:,2], target_rot[:,2]))

        # Task-space error
        err = np.concatenate([pos_err, rot_err])

        #print(f"Iteration {it}:")
        #print("  Current pos:", current_pos)


        # Converged?
        if np.linalg.norm(err) < tol:
            print("Converged!")
            break

        # Jacobian pseudo-inverse step
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data_model, Jp, Jr, site_id)
        J = np.vstack([Jp[:, joint_ids], Jr[:, joint_ids]])  # 6x6
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
        dq = J_pinv @ err

        # Update joint angles incrementally
        q += 0.005*dq
                # Update data_model with current q
        #for i, jid in enumerate(joint_ids):
        #    data_model.qpos[jid] = q[i]
        #mujoco.mj_fwdPosition(model, data_model)
    return q
