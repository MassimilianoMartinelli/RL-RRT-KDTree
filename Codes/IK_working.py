
import numpy as np
import mujoco
import gymnasium
from QLearning3D import *
import time
np.random.seed(42)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
import pyrender
import numpy as np
import gymnasium
import skvideo.io
from base64 import b64encode
from IPython.display import HTML
from gymnasium.envs.registration import *
from mujoco import viewer
# UR10e joint names
UR10E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

#!/usr/bin/env python3
import time
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
                       max_iter=50, tol=1e-4):
    """
    Iterative IK for UR10e that uses current joint positions in data_model
    as initial guess and returns q (6,) in radians.
    """
    # Numeric IDs of joints (fix Deprecation issue using [0])
    joint_ids = [int(model.joint(name).dofadr[0]) for name in UR10E_JOINTS]
    site_id = model.site("attachment_site").id

    # initial guess from data_model
    q = np.array([data_model.qpos[jid] for jid in joint_ids], dtype=np.float64)

    for _ in range(max_iter):
        # write q into data_model
        #for i, jid in enumerate(joint_ids):
        #    data_model.qpos[jid] = float(q[i])

        # update forward kinematics
        #mujoco.mj_fwdPosition(model, data_model)

        # position error
        current_pos = np.array(data_model.site_xpos[site_id])
        pos_err = target_pos - current_pos

        # orientation error (axis-angle approximation)
        current_rot = np.array(data_model.site_xmat[site_id]).reshape(3, 3)
        rot_err = 0.5 * (np.cross(current_rot[:, 0], target_rot[:, 0]) +
                         np.cross(current_rot[:, 1], target_rot[:, 1]) +
                         np.cross(current_rot[:, 2], target_rot[:, 2]))

        # task-space error as 6x1 vector
        err = np.concatenate([pos_err, rot_err]).astype(np.float64)  # shape (6,)

        if np.linalg.norm(err) < tol:
            break

        # compute site Jacobian
        Jp = np.zeros((3, model.nv), dtype=np.float64)
        Jr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data_model, Jp, Jr, site_id)

        # select columns corresponding to the robot joints
        J = np.vstack([Jp[:, joint_ids], Jr[:, joint_ids]])  # 6x6

        # damped pseudo-inverse robust computation
        A = J @ J.T + damping * np.eye(6)
        # solve for J_pinv @ err  via: dq = J.T @ inv(A) @ err
        try:
            A_inv_err = np.linalg.solve(A, err)   # shape (6,)
            dq = J.T @ A_inv_err                  # shape (6,)
        except np.linalg.LinAlgError:
            # fallback small step if singular
            dq = 1e-3 * err[:6]

        # step size
        q = q + 0.005 * dq

    return q


# ================= MAIN ===================
if __name__ == "__main__":
    # --- modifica questo percorso al tuo XML se necessario ---
    model_path = "/home/roboticlab/Workspace_Massimiliano/code/RL-RRT-KDTree/ManiSkill-UR10e-main/scene.xml"
    path ="/home/roboticlab/Workspace_Massimiliano/code/RL-RRT-KDTree/ManiSkill-UR10e-main/scene.xml"
    env_id = 'Environment'
    # carica modello e dati
    model = mujoco.MjModel.from_xml_path(model_path)
    data_model = mujoco.MjData(model)
    register(
    id='Environment',
    entry_point='env:Environment',
    kwargs={
        'model_path': path
    }
)
    # target posizione costante (metri)
    target_pos = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # opzione: impostare una posizione "home" iniziale (qpos indices)
    joint_ids = [int(model.joint(name).dofadr[0]) for name in UR10E_JOINTS]
    home_qpos = [0.01, -0.88, 1.73, -0.57, 3.14, 0.0]
    for i, jid in enumerate(joint_ids):
        data_model.qpos[jid] = float(home_qpos[i])

    # aggiorna FK iniziale
    mujoco.mj_fwdPosition(model, data_model)
    env = gymnasium.make(env_id, model_path=path, render_mode='rgb_array', camera_id=5)
    env.reset()
    # lancia il viewer in context manager (safe)
    try:
        with mujoco.viewer.launch_passive(model, data_model) as v:
            # loop fino a chiusura viewer
            time.sleep(5)
            while v.is_running():
                # calcola IK usando la data_model corrente
                q_sol = inverse_kinematics(target_pos, model, data_model)
                
                # applica la soluzione a data_model (senza chiamare env.step)
                #for i, jid in enumerate(joint_ids):
                #    data_model.qpos[jid] = float(q_sol[i])
                env.step(q_sol)
                # aggiorna FK e avanzamento simulazione
                mujoco.mj_fwdPosition(model, data_model)
                mujoco.mj_step(model, data_model)

                # aggiorna viewer
                v.sync()

                # piccolo sleep per non saturare la CPU
                time.sleep(0.01)

    except Exception as e:
        print("Errore durante l'esecuzione del viewer:", e)
