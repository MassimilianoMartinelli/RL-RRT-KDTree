






#!/usr/bin/python2

## UR5/UR10 Inverse Kinematics - Ryan Keating Johns Hopkins University


# ***** lib
import numpy as np
from numpy import linalg

from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

import numpy as np
from mujoco import MjModel, MjData, mj_step


# UR10e joint names
UR10E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

def check_joint_limits(q, model):
    """Clamp q to joint limits."""
    clamped_q = []
    for i, name in enumerate(UR10E_JOINTS):
        joint = model.joint(name)
        q_min, q_max = joint.range
        clamped_q.append(np.clip(q[i], q_min, q_max))
    return np.array(clamped_q)

def forward_kinematics(q, model, data, site_name="attachment_site"):
    """Return current end-effector position and rotation matrix."""
    joint_ids = [int(model.joint(name).dofadr) for name in UR10E_JOINTS]
    for i, jid in enumerate(joint_ids):
        data.qpos[jid] = q[i]
    mujoco.mj_forward(model, data)
    site_id = model.site(site_name).id
    pos = data.site_xpos[site_id]
    rot = data.site_xmat[site_id].reshape(3,3)
    return pos, rot

def inverse_kinematics_gradient(goal_pos, model, data,
                                q_init=None, step_size=0.7,
                                alpha=1.0, tol=1e-2, max_iter=150):
    """
    Gradient descent IK using Jacobian Transpose.
    """
    joint_ids = [int(model.joint(name).dofadr) for name in UR10E_JOINTS]
    
    # Initial guess
    if q_init is None:
        q = np.array([data.qpos[jid] for jid in joint_ids])
    else:
        q = np.array(q_init)

    for it in range(max_iter):
        # Current pose
        current_pos, _ = forward_kinematics(q, model, data)
        e = goal_pos - current_pos
        error_norm = np.linalg.norm(e)
        
        if error_norm < tol:
            print(f"Converged in {it} iterations, error {error_norm}")
            break

        # Jacobian
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        site_id = model.site("attachment_site").id
        mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
        J = Jp[:, joint_ids]

        # Gradient descent update
        dq = alpha * J.T @ e
        q += step_size * dq

        # Clamp to joint limits
        q = check_joint_limits(q, model)
    print("error norm is :",error_norm)
    return q


def forward_kinematics_2(q, model, data, site_name="attachment_site"):
    """Return current end-effector position and rotation matrix."""
    joint_ids = [int(model.joint(name).dofadr) for name in UR10E_JOINTS]
    for i, jid in enumerate(joint_ids):
        data.qpos[jid] = q[i]
    mujoco.mj_forward(model, data)
    site_id = model.site(site_name).id
    pos = data.site_xpos[site_id]
    rot = data.site_xmat[site_id].reshape(3,3)
    return pos, rot

def inverse_kinematics_gradient_2(goal_pos, model, data,
                                q_init=None, step_size=0.7,
                                alpha=1.0, tol=1e-2, max_iter=150):
    """
    Gradient descent IK using Jacobian Transpose.
    """
    joint_ids = [int(model.joint(name).dofadr) for name in UR10E_JOINTS]
    
    # Initial guess
    if q_init is None:
        q = np.array([data.qpos[jid] for jid in joint_ids])
        q_0 = q
    else:
        q = np.array(q_init)

    for it in range(max_iter):
        # Current pose
        current_pos, _ = forward_kinematics_2(q, model, data)
        e = goal_pos - current_pos
        error_norm = np.linalg.norm(e)
        
        if error_norm < tol:
            print(f"Converged in {it} iterations, error {error_norm}")
            break

        # Jacobian
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        site_id = model.site("attachment_site").id
        mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
        J = Jp[:, joint_ids]

        # Gradient descent update
        dq = alpha * J.T @ e
        q += step_size * dq

        # Clamp to joint limits
        q = check_joint_limits(q, model)
    for i, jid in enumerate(joint_ids):
        data.qpos[jid] = q_0[i]
    return q
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
                                 target_rot=np.eye(3), damping=1e-2, 
                                 max_iter=20, tol=1e-3):
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
        print("the error is : ",err)
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
        #J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
        J_pinv = np.linalg.pinv(J,damping)
        dq = J_pinv @ err
        print("dq is : ",dq)
        # Update joint angles incrementally
        q += 0.01*dq
        for i, jid in enumerate(joint_ids):
          data_model.qpos[jid] = q[i]
        mujoco.mj_step(model,data_model)
                # Update data_model with current q
        #mujoco.mj_fwdPosition(model, data_model)
    return q
