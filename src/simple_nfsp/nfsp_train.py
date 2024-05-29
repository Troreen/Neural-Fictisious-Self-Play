import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from simple_nfsp.agents.eval_agents import RandomAgent
from simple_nfsp.agents.nfsp_agent import NFSPAgent
from simple_nfsp.games.kuhn_poker import KuhnPoker


def evaluate_agent(env, agent, num_episodes=10):
    total_rewards = 0
    eval_agent = RandomAgent()
    for i in range(num_episodes):
        rl_turn = 0 if i % 2 == 0 else 1
        state = env.reset()
        done = False
        last_move_rl = False
        while not done:
            legal_actions = env.legal_actions()
            if env.current_player == rl_turn:
                # No exploration during evaluation
                last_move_rl = True
                action = agent.select_action(state, legal_actions, epsilon=0)
            else:
                last_move_rl = False
                try:
                    action = eval_agent.select_action(legal_actions)
                except:
                    print(env.get_info_state(), env.current_player, env.legal_actions())
                    raise ValueError("No legal actions available")
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if reward < -2:
                print(env.current_player, env.legal_actions())
                print("Played an illegal move")
        reward = reward if last_move_rl else -reward
        total_rewards += reward
    average_reward = total_rewards / num_episodes
    return average_reward


def save_checkpoint(agent, save_dir):
    """Saves the state of the agent, including all networks and optimizers.

    Args:
        agent (NFSPAgent): The agent to save.
        save_dir (str): The file path to save the agent state.
    """
    path = save_dir.rsplit("/", 1)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {
        "br_net_state_dict": agent.br_net.state_dict(),
        "as_net_state_dict": agent.as_net.state_dict(),
        "br_optimizer_state_dict": agent.br_optimizer.state_dict(),
        "as_optimizer_state_dict": agent.as_optimizer.state_dict(),
        "target_br_net_state_dict": (
            agent.target_br_net.state_dict()
            if hasattr(agent, "target_br_net")
            else None
        ),
    }
    torch.save(checkpoint, save_dir)


def nfsp_runner(
    env,
    anticipatory_param=0.1,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    batch_size=64,
    num_episodes=4000,
    eval_freq=2000,
    eval_episodes=1000,
    target_update_interval=100,
    avg_reward_window_size=500,
    save_freq=500,
    save_dir="models",
    save_prefix="nfsp",
    plot_folder="plots",
):
    """Run the NFSP training process.

    Train the NFSP agent on the given environment. Creates plots for evaluation scores and moving
    average rewards.

    Args:
        env (Game): The environment to train the agent on.
        anticipatory_param (float): The NFSP anticipatory parameter.
        epsilon_start (float): Starting value of epsilon for epsilon-greedy action selection.
        epsilon_end (float): Final value of epsilon.
        epsilon_decay (float): Factor to decay epsilon after each episode.
        batch_size (int): Batch size for training.
        num_episodes (int): Number of episodes to train.
        eval_freq (int): Frequency of evaluation.
        eval_episodes (int): Number of episodes to evaluate.
        target_update_interval (int): Frequency to update the target network.
        avg_reward_window_size (int): Window size for calculating the moving average reward (for plot).
        save_freq (int): Frequency to save the model.
        save_dir (str): Directory to save models.
        save_prefix (str): Prefix for the saved models.
        plot_folder (str): Folder to save plots.

    Returns:
        None
    """
    # Initialize the NFSP Agent
    agent = NFSPAgent(env.state_space, env.action_space, anticipatory_param, device="cuda")

    # Initialize epsilon for epsilon-greedy action selection
    epsilon = epsilon_start

    # Initialization for plotting
    evaluation_scores = []
    average_rewards = []
    total_rewards = []

    now = time.time()  # Get the current time for saving the files

    # Main training loop
    for episode in tqdm(range(1, num_episodes + 1)):
        # Reset the environment for a new episode
        state = env.reset()
        total_ep_rewards = 0  # Track rewards accumulated in the episode
        done = False

        while not done:  # Continue until the episode is done
            legal_actions = env.legal_actions()
            if len(legal_actions) == 0:
                print(env.state, env.current_player, env.legal_actions())
                raise ValueError("No legal actions available")
            action = agent.select_action(state, legal_actions, epsilon)

            # Environment executes action, returns next state and reward
            next_state, reward, done, _ = env.step(action)
            total_ep_rewards += reward

            # Agent stores experience in memory
            agent.store_experience_rl(state, action, reward, next_state, done, legal_actions)
            if agent.last_action_best_response:
                agent.store_experience_sl(state, action, legal_actions)

            # Move to the next state
            state = next_state

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Train the agent after each episode
        agent.train(batch_size)

        # Periodically update the target network
        if episode % target_update_interval == 0:
            agent.update_target_network()

        # Logging
        total_rewards.append(total_ep_rewards)

        if episode % eval_freq == 0:
            tqdm.write("Evaluation in progress...")
            evaluation_score = evaluate_agent(env, agent, eval_episodes)
            evaluation_scores.append(evaluation_score)
            tqdm.write(f"Evaluation score after episode {episode}: {evaluation_score}")

        # Save the model
        if episode % save_freq == 0:
            save_checkpoint(
                agent, f"{save_dir}/{now}/{save_prefix}_episode_{episode}.pt"
            )
            tqdm.write("\033[1;32m" + f"Saved model at episode {episode}" + "\033[0m")

    # Calculate Moving Average
    for i in range(10, num_episodes):
        if i >= avg_reward_window_size:
            window_avg = np.mean(total_rewards[i - avg_reward_window_size : i])
        else:
            window_avg = np.mean(total_rewards[:i + 1])
        average_rewards.append(window_avg)                                    

    # Ensure the output directory exists
    output_dir = f"{plot_folder}/{now}"
    os.makedirs(output_dir, exist_ok=True)

    # Plotting the evaluation scores (no changes needed here)
    plt.figure(figsize=(10, 6))  # Optionally specify the figure size
    plt.plot(range(eval_freq, num_episodes + 1, eval_freq), evaluation_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("NFSP Agent Evaluation Scores")
    plt.savefig(f"{output_dir}/nfsp_evaluation_scores.png")
    plt.clf()  # Clear the figure for the next plot

    # Plotting the smoothed moving average rewards
    plt.figure(figsize=(10, 6))  # Optionally specify the figure size
    plt.plot(range(10, num_episodes), average_rewards)
    plt.xlabel("Episodes")
    plt.ylabel(f"Moving Average Reward (Window={avg_reward_window_size})")
    plt.title("NFSP Agent Training - Moving Average Reward")
    plt.savefig(f"{output_dir}/nfsp_moving_avg_rewards.png")


if __name__ == "__main__":
    nfsp_runner(
        env=KuhnPoker(),
    )
