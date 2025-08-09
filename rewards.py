from constants import AGENT_STARTING_BANKROLL
import math
class Reward:
    def __init__(self, label,reward, bool_function ):
        self.label = label
        self.bool_function = bool_function
        self.reward=reward

REWARDS_BINDINGS = [
    #Reward(,,lambda env, player:),
    Reward("Agent Hit on 21", -2, lambda env, player: not player.current_hand().is_blackjack() 
                                                and player.current_hand().get_original_delt_values() == 21),
    
    #Reward("Agent has Blackjack", 2, lambda env, player: player.current_hand().is_blackjack()),
    
    Reward("Agent stood_below 17 and card showing was greater than 7",-5, lambda env, player: player.current_hand().stood_below_17 
                                                                                    and env.dealer.current_hand().get_first_card_value() >= 7),
    
    Reward("Agent Hit above 17", -5,lambda env, player: player.current_hand().hit_above_17),
    
    Reward("Agent had a good double", 2,lambda env, player: player.current_hand().has_doubled 
                                                    and player.current_hand().get_original_delt_values() <= 11 
                                                    and player.current_hand().get_original_delt_values()>8 
                                                    and env.dealer.current_hand().get_first_card_value() <= 8),
    
    Reward("Agent didnt bust but shouldnt have hit",-2,lambda env, player: player.current_hand().get_values() <=21 
                                        and env.dealer.current_hand().hit_above_17),
    
    Reward("Agent didnt double when it should have",-5,lambda env, player: player.current_hand().get_original_delt_values()>= 8 
                                                                and player.current_hand().get_original_delt_values() <= 11 
                                                                and env.dealer.current_hand().get_first_card_value() < 8
                                                                and not player.current_hand().has_doubled),
    
    Reward("Agent didnt hit when it was below 17 and dealer showing 7 or higher",-5,lambda env, player: env.dealer.current_hand().get_first_card_value() >= 7 
                                                                                            and player.current_hand().get_values() < 17),
    
    Reward("Agent busted on a double",-5, lambda env, player: player.current_hand().has_doubled 
                                                        and player.current_hand().is_busted()),
    
    Reward("Agent Doubled when hand had to high of value",-5,lambda env, player: player.current_hand().has_doubled 
                                                                    and player.current_hand().get_original_delt_values() > 11
                                                                    and not player.current_hand().ace_in_original_hand),
    
    Reward("Agent Hit when its hand was under 17 and dealer showing high card",2,lambda env, player: not player.current_hand().hit_above_17 
                                                                                                and len(player.current_hand().cards)> 2
                                                                                                and player.current_hand().get_original_delt_values()<= 11
                                                                                                and env.dealer.current_hand().get_first_card_value()>= 7),

    Reward("Agent stays on delt high cards",2,lambda env, player: len(player.current_hand().cards) == 2
                                                                    and player.current_hand().get_values()>=17),

    Reward("Agent stood when hand started with over 11 and dealer showing low card",2,lambda env, player: player.current_hand().get_values()>11
                                                                    and len(player.current_hand().cards)> 2
                                                                    and env.dealer.current_hand().get_first_card_value()<=6
                                                                    and not player.current_hand().hit_above_17),

    Reward("Agent doubled when original value was greater than 11",-5,lambda env, player: player.current_hand().has_doubled
                                                                    and player.current_hand().get_original_delt_values()> 11
                                                                    )                                                                                                                                         

    #Reward("Agent Won round",2,lambda env, player: player.current_hand().get_values()<=21 
    #                                            and (env.dealer.current_hand().get_values() < player.current_hand().get_values() 
    #                                                 or env.dealer.current_hand().is_busted())),
    
    #Reward("Add rewards for balance changes", lambda env, player: max(-5, min(math.log(player.bankroll / AGENT_STARTING_BANKROLL), 5)), lambda env, player: True)
    #Reward(,,lambda env, player:),
]
