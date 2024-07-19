import model.DDPG as DDPG
import gym
import rsoccer_gym
import cv2
import time

def create_video(source, fps=60, output_name='output'):
    out = cv2.VideoWriter(output_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (source[0].shape[1], source[0].shape[0]))
    for i in range(len(source)):
        out.write(source[i])
    out.release()


env = gym.make('SSLGoToBallIR2-v0')
# state dimension
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high
min_action = env.action_space.low

policy = DDPG.DDPG(env.observation_space.shape[0],
                   env.action_space.shape[0],
                   env.action_space.high,
                   env.action_space.low)

# Load the policy 
policy.load("./weight/goToBall")

state, done = env.reset(), False
frames = []
for episode in range(10):
    total_reward = 0
    done = False
    t = 0
    while not done and t < 600:
        t += 1
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        frames.append(env.render(mode="rgb_array"))
        time.sleep(0.02)

    print(f"Episode: {episode}, Reward: {total_reward}")
    state = env.reset()
env.close()
create_video(frames, 60, 'output')