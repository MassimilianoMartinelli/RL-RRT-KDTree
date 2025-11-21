import ompl.base as ob
import ompl.geometric as og
from ompl import util as ou
import numpy as np
import time
from inverse_kinematics_UR10e import *
import mujoco
from gymnasium.envs.registration import register
import gymnasium
from ikpy.chain import Chain
UR10E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]
def is_state_valid_3d(state, env, model, data,viewer):
    for point in state[1:]:
        joint_ids = [int(model.joint(name).dofadr) for name in UR10E_JOINTS]
        
        print("target : ",point)
        q_0 =np.array([data.qpos[jid] for jid in joint_ids])
        for i in range(70):
            x, y, z = point
            path =[x,y,z]
            if i == 0 :
                act = inverse_kinematics_gradient_2(path, model , data)
                for i, jid in enumerate(joint_ids):
                    data.qpos[jid] = q_0[i]
            data.ctrl[:] = act
            mujoco.mj_step(model,data)
            viewer.sync()  
            time.sleep(0.05)
            current_pos = data.site_xpos[model.site("attachment_site").id]
            print("ee current pos : ", current_pos)


def is_point_too_close_to_obstacles(point, obstacle_list, min_distance):
    
    for obstacle in obstacle_list:
        if obstacle.point_in_rect(point, min_distance):
            return True
    return False

def is_state_valid_rect(state, obstacles, min_distance=0):

    for obs in obstacles:
        if obs.point_in_rect(state, min_distance):
            return False
        # if not is_point_too_close_to_obstacles(state, obstacles, min_distance):
        #     return True
    return True


def is_state_valid_box(state, env, model, data,viewer):
    env = env.unwrapped
    state[2] = state[2]
    print("states :", state[0],state[1],state[2])
    for i in range(1):

        q_act = [data.qpos[0],data.qpos[1],data.qpos[2],data.qpos[3],data.qpos[4],data.qpos[5]]
        state_T = np.array([
            [1, 0, 0, state[0]],
            [0, 1, 0, state[1]],
            [0, 0, 1, state[2]],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        state = [state[0], state[1],state[2]]
        act = inverse_kinematics_gradient(state,model,data)
        print("act : ",act)
        current_pos = data.site_xpos[model.site("attachment_site").id]
        joint_ids = [int(model.joint(name).dofadr) for name in UR10E_JOINTS]
        #data.ctrl[:] = act
        for i, jid in enumerate(joint_ids):
          data.qpos[jid] = act[i]
        #for i, jid in enumerate(joint_ids):
        #    print("joint values : ",data.qpos[jid])
        #print("current pos : ", current_pos)
        #viewer.sync()  
        #time.sleep(0.01)
        #env.reset()
    contact_list = []
    for i in range(env.data.ncon):
        contact = env.data.contact[i]    
        geom1 = contact.geom1
        geom2 = contact.geom2
        geom1_name = env.model.geom(geom1).name
        geom2_name = env.model.geom(geom2).name
        contact_list.append((geom1_name, geom2_name))
    if len(contact_list) == 0:
        return True
    else:
        return False


def RRT2D(start, subgoal, zone_start, zone_next, obstacles):
    space = ob.RealVectorStateSpace(2)
    
    lower_bound = [min(zone_start[0], zone_next[0]), min(zone_start[1], zone_next[1])]
    upper_bound = [max(zone_start[2], zone_next[2]), max(zone_start[3], zone_next[3])]
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, lower_bound[0])
    bounds.setLow(1, lower_bound[1])
    bounds.setHigh(0, upper_bound[0])
    bounds.setHigh(1, upper_bound[1])
    
    space.setBounds(bounds)
    
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(lambda state: is_state_valid_rect(state, obstacles)))
    
    start_state = ob.State(space)
    start_state()[0] = float(start[0])
    start_state()[1] = float(start[1])
    
    goal_state = ob.State(space)
    goal_state()[0] = float(subgoal[0])
    goal_state()[1] = float(subgoal[1])
    
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)
    
    planner = og.RRT(si)
    planner.setRange(12) 
    planner.setProblemDefinition(pdef)
    planner.setup()
    startRRT=time.time()
    solved = planner.solve(2)  # Allow 10 seconds to solve
    endRRT=time.time()
    iteration_count = 1
    print("endRRT-startRRT",endRRT-startRRT)
    if endRRT- startRRT <2:
        path = pdef.getSolutionPath()
        new_path = [(state[0], state[1]) for state in path.getStates()]
        planning_time = endRRT-startRRT  # Get the planning time
        return planning_time, new_path, 1,iteration_count
    else:
        return 10, [], 0,iteration_count  # Return the max time if not solved   

def RRT3D(start, subgoal, zone_start, zone_next, obstacles, env, model, data, viewer):
    space = ob.RealVectorStateSpace(3)
    
    lower_bound = [min(zone_start[0], zone_next[0]), min(zone_start[1], zone_next[1]), min(zone_start[2], zone_next[2])]
    upper_bound = [max(zone_start[3], zone_next[3]), max(zone_start[4], zone_next[4]), max(zone_start[5], zone_next[5])]
    print("lowerbound : ",lower_bound)
    print("upper_bound",upper_bound)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, lower_bound[0])
    bounds.setLow(1, lower_bound[1])
    bounds.setLow(2, lower_bound[2])
    bounds.setHigh(0, upper_bound[0])
    bounds.setHigh(1, upper_bound[1])
    bounds.setHigh(2, upper_bound[2])
    
    space.setBounds(bounds)
    
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(lambda state: is_state_valid_box(state, env, model, data, viewer)))
    
    start_state = ob.State(space)
    start_state()[0] = start[0]
    start_state()[1] = start[1]
    start_state()[2] = start[2]
    goal_state = ob.State(space)
    goal_state()[0] = subgoal[0]
    goal_state()[1] = subgoal[1]
    goal_state()[2] = subgoal[2]
    print("subgoal : ",subgoal[0],subgoal[1], subgoal[2])
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)
    planner = og.RRT(si)
    planner.setRange(3) 
    planner.setProblemDefinition(pdef)
    planner.setup()
    startRRT=time.time()
    print("--------------")
    solved = planner.solve(1.5)  # Allow 10 seconds to solve
    endRRT=time.time()
    iteration_count = 1
    print("endRRT-startRRT",endRRT-startRRT)
    if endRRT- startRRT <1.5:
        path = pdef.getSolutionPath()
        new_path = [(state[0], state[1], state[2]) for state in path.getStates()]
        print("questo è il nuovo path : ",new_path)
        planning_time = endRRT-startRRT  # Get the planning time
        return planning_time, new_path, 1,iteration_count
    else:
        return 100, [], 0,iteration_count
    
    
def path_moving(state,env , model, data,viewer):
    is_state_valid_3d(state, env, model, data, viewer)
      
        