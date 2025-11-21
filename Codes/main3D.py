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

path = '/home/roboticlab/Workspace_Massimiliano/code/RL-RRT-KDTree/ManiSkill-UR10e-main/UR10E_Scenario2.xml'
env_id = 'Environment'
model = mujoco.MjModel.from_xml_path(path)
data_model = mujoco.MjData(model)

register(
    id='Environment',
    entry_point='env:Environment',
    kwargs={
        'model_path': path
    }
)

env = gymnasium.make(env_id, model_path=path, render_mode='rgb_array', camera_id=5)
env.reset()
# UR10E joint names e posizione "home"
UR10E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]
home_qpos = [0.01 , -0.88 , 1.73 , -0.570 , 3.14 , 0]

# Estrai gli indici numerici dei joint (dofadr)
joint_ids = [int(model.joint(name.encode()).dofadr) for name in UR10E_JOINTS]

# Assegna la configurazione "home"
for i, jid in enumerate(joint_ids):
    data_model.qpos[jid] = home_qpos[i]



mujoco.mj_forward(model, data_model)
mujoco.mj_step(model, data_model) 

v = viewer.launch_passive(model, data_model)
time.sleep(5)
start=[-0.07 , 0.75 , 1.105]
goal=[0.8 , 0.75 , 1.305]
n = 400

x = (np.random.rand(n) - 0.5) * 2.4 # [-0.7, 0.7]
y = (np.random.rand(n) - 0.5) * 1.4  # [-0.7, 0.7]
z = np.random.rand(n) * 1.7           # [0, 1.7]

data = np.stack([x, y, z], axis=1)
depth2D=3
depthZ=2
T_RRT=[]
T_RL=[]
SR=[]

medians2D = collect_medians_of_splits(data, depth2D)
print("medians 2D", medians2D)

mediansZ = collect_medians_z(data, depthZ, depth=1, medians_z=None)
print("medians 3D", mediansZ)
# -------- FIX: CREATE ZONES FIRST --------
zones = create_zone(data, depth2D, depthZ, boundry=2.0)

# -------- FIX: PASS ZONES INTO Final_zone --------
startZone = Final_zone(start, medians2D, mediansZ, depth2D, depthZ)
goalZone  = Final_zone(goal,  medians2D, mediansZ, depth2D, depthZ)

print("startZone, goalZone", startZone, goalZone)

adjacency_matrix = get_adjacency_matrix(zones, depth2D)
print("here is the adjacency matrix", adjacency_matrix)

DofZ = get_DofZ(data, medians2D, mediansZ, depth2D, depthZ, zones)
dist = distances(zones, goal, normalized=True)
AllStatesActions = setReward(adjacency_matrix, DofZ, dist, depth2D)

gamma=0.9
episodes=2000
epsilon=0.9
alpha=0.1

startRL=time.time()
Q, policy = train(depth2D, episodes, alpha, gamma, epsilon, AllStatesActions, goalZone)
endRL=time.time()

policy = get_final_policy(depth2D, Q, policy)
print("policy is here", policy)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def check_success(path, goal, time, tolerance=1e-2):
    if not path:
        return 0
    
    final_state = path[-1]
    print("final state",final_state)
    check= np.allclose(final_state, goal, atol=tolerance)
    print("check",check)
    if check:
        if time > 4:
            print("Hi")
            return 0
        else:
            print("hi")
            return 1
    return 0

for i in range(1):
    Time, path, done, iteration_count = simulate3D(zones, policy, data, start, startZone, goal, goalZone,env, model, data_model, v,home_qpos)   
    time.sleep(0.15)
    print("iteration : ", i)
    T_RRT.append(Time)
    SR.append(check_success(path, goal, Time, tolerance=1e-2))

    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]
    z_coords = [p[2] for p in path]
    ax.plot(x_coords, y_coords, z_coords, label=f'Path {i+1}')

ax.scatter(start[0], start[1], start[2], c='red', marker='o', label='Start')
ax.scatter(goal[0], goal[1], goal[2], c='green', marker='x', label='Goal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('3D Path Planning')
ax.legend()

plt.show()

print(SR)
print(f"success rate is {SR.count(1)/len(SR)*100}%")
print(f"RL Time", endRL-startRL)
print(f"mean RRT Time", sum(T_RRT)/len(T_RRT))
print(f"number of iteration is {iteration_count}")