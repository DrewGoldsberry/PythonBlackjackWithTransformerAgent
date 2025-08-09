# game_ui.py

import pygame
from constants import *
from blackjack_env import BlackjackEnv
from agent_player import AgentPlayer
import time
class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.font = pygame.font.SysFont('arial', 20)
        self.enabled = True

    def draw(self, screen):
        color = BLUE if self.enabled else (100, 100, 100)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if self.enabled and event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.callback()
       

class BlackjackUI:
    def __init__(self, player=None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Blackjack")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 24)
        self.env = BlackjackEnv(player)
        self.env.reset()
        self.running = True
        self.buttons = []
        self.build_buttons()
        self.message = ""
        self.round_over=False
        self.bet_input_active = False
        self.bet_input_text = "10"
        self.bet_input_font = pygame.font.SysFont('arial', 24)
        self.player = player
        self.last_action_time = time.time()
        self.round_end_time = None
    def build_buttons(self):
        spacing = 120
        start_x = SCREEN_WIDTH // 2 - 2 * spacing
        actions = ["Hit", "Stand", "Double", "Split"]
        callbacks = [self.hit, self.stand, self.double, self.split]
        for i, (label, cb) in enumerate(zip(actions, callbacks)):
            btn = Button(start_x + i * spacing, SCREEN_HEIGHT - 100, 100, 40, label, cb)
            self.buttons.append(btn)

    def run(self):
        while self.running:
            self.screen.fill(GREEN)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                for button in self.buttons:
                    button.handle_event(event)


                 # Handle text input for the bet box
                if event.type == pygame.MOUSEBUTTONDOWN:
                    input_box = pygame.Rect(20, SCREEN_HEIGHT - 90, 100, 35)
                    if input_box.collidepoint(event.pos):
                        self.bet_input_active = True
                    else:
                        self.bet_input_active = False

                elif event.type == pygame.KEYDOWN and self.bet_input_active:
                    if event.key == pygame.K_RETURN:
                        self.bet_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.bet_input_text = self.bet_input_text[:-1]
                    elif event.unicode.isdigit():
                        self.bet_input_text += event.unicode


            self.render()
            if not self.round_over and isinstance(self.player, AgentPlayer):
                now = time.time()
                if now - self.last_action_time >= 4.0:
                    self.agent_act()
            if self.round_over and isinstance(self.player, AgentPlayer):
                if self.round_end_time and (time.time() - self.round_end_time >= 3):
                    self.next_round()
                    self.round_end_time = None 
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def render(self):
        # Draw player
        player = self.env.players[0]
        hand = player.current_hand()
        self.draw_hand(hand.cards, 100, SCREEN_HEIGHT - 250, label="Player")

        dealer_hand = self.env.dealer.current_hand()
        self.draw_hand(dealer_hand.cards, 100, 100, label="Dealer", hide_second_card=not self.round_over)
        #next button
        btn = Button(SCREEN_WIDTH - 160, SCREEN_HEIGHT - 100, 140, 40, "Next Round", self.next_round)
        self.buttons.append(btn)

        # Draw bankroll
        bankroll_text = self.font.render(f"Bankroll: ${player.bankroll}", True, WHITE)
        self.screen.blit(bankroll_text, (20, SCREEN_HEIGHT - 40))
        #draw message
        if self.message:
            msg_surface = self.font.render(self.message, True, YELLOW)
            self.screen.blit(msg_surface, (SCREEN_WIDTH // 2 - msg_surface.get_width() // 2, SCREEN_HEIGHT - 60))
        
        if isinstance(self.player, AgentPlayer):
            pass  # don't draw hit/stand/etc.
        else:
            for button in self.buttons:
                button.draw(self.screen)

                    # Draw bet input box
        input_box = pygame.Rect(20, SCREEN_HEIGHT - 90, 100, 35)
        pygame.draw.rect(self.screen, WHITE, input_box, 0 if self.bet_input_active else 2)
        bet_surface = self.bet_input_font.render(self.bet_input_text, True, BLACK)
        self.screen.blit(bet_surface, (input_box.x + 5, input_box.y + 5))
        label = self.font.render("Bet:", True, WHITE)
        self.screen.blit(label, (20, SCREEN_HEIGHT - 125))

    def draw_hand(self, cards, x, y, label="", hide_second_card=False):
        spacing = 70
        label_surface = self.font.render(label, True, WHITE)
        self.screen.blit(label_surface, (x, y - 30))

        for i, card in enumerate(cards):
            rect = pygame.Rect(x + i * spacing, y, 60, 90)
            pygame.draw.rect(self.screen, WHITE, rect)

            if i == 1 and hide_second_card:
                # Draw red card back for hidden second card
                pygame.draw.rect(self.screen, RED, rect.inflate(-10, -10))
                pygame.draw.line(self.screen, BLACK, rect.topleft, rect.bottomright, 2)
                pygame.draw.line(self.screen, BLACK, rect.topright, rect.bottomleft, 2)
            else:
                # Show actual card
                card_text = self.font.render(str(card), True, BLACK)
                self.screen.blit(card_text, (rect.x + 10, rect.y + 30))

    def hit(self):
        if not self.round_over:
            player = self.env.players[0]
            hand = player.current_hand()
            card = self.env.deck.draw()
            hand.add_card(card)
            val = hand.get_values()
            if hand.is_busted():
                self.message = f"{player.name} busted with {val}!"
                self.round_over = True
            if player.current_hand().get_values() == 21:
                self.round_over = True
                self.env.play_round()
    
    def stand(self):
        if not self.round_over:
            print("STAND")
            self.env.play_round()       # <- Skip dealer replay
            self.set_end_round_message()
            self.round_over = True
            self.update_button_states()
    
    def double(self):
        print("DOUBLE")
        if not self.round_over:
            self.round_over = True
            player = self.env.players[0]
            hand = player.current_hand()
            card = self.env.deck.draw()
            hand.add_card(card)
            player.double_down()
            self.env.play_round()
            self.set_end_round_message()
            self.update_button_states()

    def split(self):
        print("SPLIT")
        player = self.env.players[0]
        if player.current_hand().can_split():
            player.split_hand()
    
    def update_button_states(self):
        for button in self.buttons:
            if button.text == "Next Round":
                button.enabled = self.round_over
            else:
                button.enabled = not self.round_over
    def set_end_round_message(self):
        player = self.env.players[0]
        hand = player.current_hand()
        dealer_val = self.env.dealer.current_hand().get_values()
        player_val = hand.get_values()

        if hand.is_busted():
            self.message = f"{player.name} busted with {player_val}."
        elif dealer_val > 21 or player_val > dealer_val:
            self.message = f"{player.name} wins! {player_val} vs Dealer {dealer_val}"
        elif dealer_val == player_val:
            self.message = f"{player.name} pushes. {player_val} vs Dealer {dealer_val}"
        else:
            self.message = f"{player.name} loses. {player_val} vs Dealer {dealer_val}"

    def next_round(self):
        if self.round_over:
            bet_amount=0
            try:
                bet_amount = float(self.bet_input_text)
            except ValueError:
                bet_amount = 10
            self.env.reset(bet_amount=bet_amount)
            self.round_over = False
            self.message = ""
            self.update_button_states()

    def agent_act(self):
        hand = self.player.current_hand()
        dealer_card = self.env.dealer.current_hand().cards[0]

        if hand.is_blackjack() or hand.is_busted():
            self.round_over = True
            self.env.play_round(skip_dealer=False)
            self.set_end_round_message()
            self.update_button_states()
            self.round_end_time = time.time()
            return

        action = self.player.decide_action(dealer_card)

        if action == "hit":
            card = self.env.deck.draw()
            self.env.deck.discard(card)
            hand.add_card(card)
            self.last_action_time = time.time()

            if hand.is_busted():
                self.round_over = True
                self.env.play_round()
                self.set_end_round_message()
                self.update_button_states()

        elif action == "stand":
            self.round_over = True
            self.env.play_round()
            self.set_end_round_message()
            self.update_button_states()
            self.last_action_time = time.time()


        elif action == "double":
            if self.player.bankroll >= hand.bet:
                self.player.double_down()
                card = self.env.deck.draw()
                self.env.deck.discard(card)
                hand.add_card(card)
                self.round_over = True
                self.env.play_round()
                self.set_end_round_message()
                self.update_button_states()

        elif action == "split":
            if hand.can_split():
                self.player.split_hand()
        if self.round_over:
            self.round_end_time = time.time()

        