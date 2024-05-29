# To test all the functionalities of the Cheat Game
from simple_nfsp.games.cheat import CheatGame


cards_showing = [1,2,3,4,5,6,7,8,9,10,11,12,13]
cards_showing = " ".join([str(card) for card in cards_showing])

game = CheatGame(num_rounds=10)
done = False
while not done:
    print(f"Round {game.num_rounds}")
    print(f"                  {cards_showing}")
    print(f"Player 1's hand: {game.players[0].hand}")
    print(f"Player 2's hand: {game.players[1].hand}")

    print(f"Player {game.current_player + 1}'s turn")
    print(f"Legal actions: {game.legal_actions()} | Action type: {game.action_types[game.next_action]}")
    action = input("Enter action: ")
    info_state, reward, done, _ = game.step(int(action))
    print(info_state, reward, done)