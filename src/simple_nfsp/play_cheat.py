import os
import torch
from simple_nfsp.agents.nfsp_agent import NFSPAgent
from simple_nfsp.games.cheat import CheatGame

def load_checkpoint(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    agent.br_net.load_state_dict(checkpoint["br_net_state_dict"])
    agent.as_net.load_state_dict(checkpoint["as_net_state_dict"])
    agent.br_optimizer.load_state_dict(checkpoint["br_optimizer_state_dict"])
    agent.as_optimizer.load_state_dict(checkpoint["as_optimizer_state_dict"])
    if checkpoint["target_br_net_state_dict"]:
        agent.target_br_net.load_state_dict(checkpoint["target_br_net_state_dict"])


cards_showing = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
cards_showing = " |".join([str(card).rjust(2) for card in cards_showing])
def play_against_agent(env, agent):
    done = False
    human_turn = 0 #if input("Do you want to play first? (y/n): ").strip().lower() == 'y' else 1
    state = env.reset()

    while not done:
        print(f"Round {env.num_rounds}")
        print(f"                 {cards_showing}")
        print('-' * 71)
        p1_display = " |".join([str(card).rjust(2) for card in env.players[0].hand[1:]])
        print(f"Player 1's hand: {p1_display}")
        p2_display = " |".join([str(card).rjust(2) for card in env.players[1].hand[1:]])
        print(f"Player 2's hand: {p2_display}")
        print('-' * 71)
        print(f"Current Player: Player {env.current_player + 1}")

        if env.current_player == human_turn:
            print(f"Legal actions: {env.legal_actions()} | Action type: {env.action_types[env.next_action]}")
            action = int(input("Enter your action: "))
        else:
            action = agent.select_action(state, env.legal_actions(), epsilon=0)
            print(f"Agent played: {action}")

        next_state, reward, done, _ = env.step(action)
        state = next_state

        if done:
            print(f"\nGame Over! Player {env.current_player + 1} {'won' if reward > 0 else 'lost'} the game.")

if __name__ == "__main__":
    env = CheatGame()
    agent = NFSPAgent(env.state_space, env.action_space, anticipatory_param=0.1, device="cuda")

    # Load the trained model checkpoint
    checkpoint_path = "models/1715725212.8200085/nfsp_episode_10000.pt"
    load_checkpoint(agent, checkpoint_path)

    # Play against the trained agent
    play_against_agent(env, agent)