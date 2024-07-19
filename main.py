import random
import numpy as np
import torch
import argparse
import os

import gym
import rsoccer_gym

from model.ReplayBuffer import ReplayBuffer
import model.DDPG as DDPG

def evaluation(env, policy, eval_num):
    state, done = env.reset(), False
    total_reward = 0.0
    eps = 0
    count = 0
    while eps < eval_num:
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        count += 1

        if done or count >= args.limit_steps:
            state, done = env.reset(), False
            eps += 1
            count = 0

    return total_reward/eval_num

def add_on_policy_mc(transitions):
    r = 0
    dis = 0.99
    for i in range(len(transitions)-1, -1, -1):
        r = transitions[i]["reward"]+dis*r
        transitions[i]["n_step"] = r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="goToBall")               # Environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Environment, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1000, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--eval_num", default=100, type=int)        # Eval Number
    parser.add_argument("--max_episode", default=5e4, type=int)     # Max time steps to run environment
    parser.add_argument("--batch_size", default=64, type=int)       # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.0001)                    # Target network update rate
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file name
    parser.add_argument("--limit_steps", default=600, type=int)     # Max time steps for each episode (600 equals 10 seconds)
    args = parser.parse_args()

    file_name = f"{args.name}_{args.seed}"
    print("---------------------------------------")
    print(f"Run Name: {args.name}, Seed: {args.seed}")
    print("---------------------------------------")

    # Create Environment
    env = gym.make('SSLGoToBallIR2-v0')

    # Set Seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize policy and replay buffer
    policy = DDPG.DDPG(env.observation_space.shape[0],
                       env.action_space.shape[0],
                       env.action_space.high,
                       env.action_space.low,
                       args.discount,
                       args.tau)
    replay_buffer = ReplayBuffer(env.observation_space.shape[0],
                                 env.action_space.shape[0])
    
    # Save model and optimizer parameters
    if args.save_model and not os.path.exists("./weights"):
        os.makedirs("./weights")

    # Load model and optimizer parameters
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./weights/{policy_file}")

    # Initialize
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    transitions = []
    high_eval = -1200
    timestep = 0
    # save rewards for plotting
    rewards = []

    # Training Loop
    while episode_num < args.max_episode:
        # Select action randomly or according to policy
        eps_rnd = random.random()
        dec = min(max(0.1, 1.0 - float(timestep - args.start_timesteps) * 0.00009), 1)

        if eps_rnd < dec or timestep < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state)

        next_state, reward, done, _ = env.step(action)
        done_bool = float(done)

        transitions.append({"state": state,
                            "action": action,
                            "next_state": next_state,
                            "reward": reward,
                            "done": done_bool})
        
        state = next_state
        episode_reward += reward

        timestep += 1
        episode_timesteps += 1

        if done or episode_timesteps >= args.limit_steps:
            add_on_policy_mc(transitions)
            for t in transitions:
                replay_buffer.add(t["state"],
                                t["action"],
                                t["next_state"],
                                t["reward"],
                                t["n_step"],
                                t["done"])
            if timestep > args.start_timesteps:
                for i in range(int(episode_timesteps/10)):
                        policy.train(replay_buffer, args.batch_size)

            print(f"Episode Num: {episode_num+1} Reward: {episode_reward} Timestep: {timestep}")
            rewards.append(episode_reward)

            state, done = env.reset(), False
            episode_reward = 0
            transitions = []
            episode_num += 1
            episode_timesteps = 0

            # Evaluate episode and save model
            if (episode_num+1) % args.eval_freq == 0:
                eval = evaluation(env, policy, args.eval_num)
                print(f"Episode Num: {episode_num+1} Evaluation: {eval:.3f}")
                if eval > high_eval:
                    high_eval = eval
                    if args.save_model: 
                        policy.save(f"./weights/{file_name+str(episode_num+1)}")
                        print('saved in ', episode_num+1)
                state, done = env.reset(), False