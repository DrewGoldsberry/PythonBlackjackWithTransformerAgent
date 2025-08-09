# main.py

import pygame
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, GREEN, FPS
from game_ui import BlackjackUI
from agent_player import AgentPlayer
from transformer_agent import TransformerAgent
from player import Player
if __name__ == "__main__":
   #defaults to TransormerAgent being used to play the game
   agent = TransformerAgent.load("./models/blackjack_agent_ep.pt")
   
   #run the game with the agent
   player = AgentPlayer("Jimmy", agent)
   ui = BlackjackUI(player=player)

   #uncomment to play as a human
   #player= Player("Human")
   # ui = BlackjackUI(agent=player, is_agent=False)
   ui.run()