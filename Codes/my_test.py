import numpy as np
import gymnasium
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt

path = './ManiSkill-UR10e-main/ur10e.xml'
env_id = 'Environment'

register(
    id=env_id,
    entry_point='env:Environment',   # must point to your class!
    kwargs={'model_path': path}
)

env = gymnasium.make(env_id, model_path=path, render_mode='rgb_array', camera_id=5)

obs, info = env.reset()

# Desired joint configuration (6 DOF)
q_target = np.array([-1.0, -1.2, 1.8, -1.0, -1.5, 0.3])
print("Target joints:", q_target)

# Apply this target for multiple steps so the controller can converge
num_steps = 200
for t in range(num_steps):
    # Action = desired joint positions (thanks to your general actuator PD setup)
    obs, reward, terminated, truncated, info = env.step(q_target)

    # Read current joint positions from the simulator
    q_current = env.unwrapped.data.qpos[:6].copy()

    if terminated or truncated:
        break

# After motion: print final joint configuration
q_final = env.unwrapped.data.qpos[:6].copy()
print("Final joints:", q_final)

# Render one final image
img = env.render()
plt.imshow(img)
plt.axis("off")
plt.show()

env.close()
