# main.py

import pygame
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, GREEN, FPS
from game_ui import BlackjackUI
from agent_player import AgentPlayer
from transformer_agent import TransformerAgent
from player import Player
if __name__ == "__main__":
   # ui = BlackjackUI()
   # ui.run()
    #agent = TransformerAgent.load("./models/blackjack_agent_ep_dumb.pt")
    agent = TransformerAgent.load("./models/blackjack_agent_ep.pt")
    player = AgentPlayer("Jimmy", agent)
    #player= Player("Human")
    ui = BlackjackUI(agent=player, is_agent=True)
    ui.run()