"""Reward definitions for agent training.

This module defines small helper callables that evaluate the game state and
assign scalar rewards.  The rewards are intentionally dense to provide more
training signal than a simple win/loss outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from constants import AGENT_STARTING_BANKROLL
import math


# ---------------------------------------------------------------------------
# Reward container
# ---------------------------------------------------------------------------

@dataclass
class Reward:
    """Container describing a training reward."""

    label: str
    value: Union[float, Callable]
    condition: Callable[["BlackjackEnv", "AgentPlayer"], bool]


# ---------------------------------------------------------------------------
# Helper functions used by reward conditions/values
# ---------------------------------------------------------------------------


def _balance_change(env, player):
    """Reward based on bankroll growth using a log scale."""
    ratio = max(player.bankroll, 1) / AGENT_STARTING_BANKROLL
    return max(-5.0, min(math.log(ratio), 5.0))


def _agent_won(env, player):
    """Return ``True`` if the agent won the round."""
    hand = player.current_hand()
    dealer = env.dealer.current_hand()
    if hand.is_busted():
        return False
    dealer_val = dealer.get_values()
    player_val = hand.get_values()
    return dealer_val > 21 or player_val > dealer_val


def _agent_push(env, player):
    """Return ``True`` if the round resulted in a push."""
    hand = player.current_hand()
    dealer_val = env.dealer.current_hand().get_values()
    return not hand.is_busted() and hand.get_values() == dealer_val


# ---------------------------------------------------------------------------
# Reward bindings
# ---------------------------------------------------------------------------

REWARDS_BINDINGS = [
    # Round outcome rewards
    Reward("Agent has blackjack", 3, lambda env, player: player.current_hand().is_blackjack()),
    Reward(
        "Agent hit on 21",
        -2,
        lambda env, player: not player.current_hand().is_blackjack()
        and player.current_hand().get_original_delt_values() == 21,
    ),
    Reward("Agent busted", -5, lambda env, player: player.current_hand().is_busted()),
    Reward("Agent won round", 5, _agent_won),
    Reward("Agent pushed round", 1, _agent_push),
    Reward(
        "Agent lost round",
        -5,
        lambda env, player: not _agent_won(env, player) and not _agent_push(env, player),
    ),
    Reward("Bankroll change", _balance_change, lambda env, player: True),

    # Tactical play rewards/penalties
    Reward(
        "Agent stood below 17 against high dealer card",
        -5,
        lambda env, player: player.current_hand().stood_below_17
        and env.dealer.current_hand().get_first_card_value() >= 7,
    ),
    Reward("Agent hit above 17", -5, lambda env, player: player.current_hand().hit_above_17),
    Reward(
        "Agent had a good double",
        2,
        lambda env, player: player.current_hand().has_doubled
        and 8 < player.current_hand().get_original_delt_values() <= 11
        and env.dealer.current_hand().get_first_card_value() <= 8,
    ),
    Reward(
        "Agent didn't double when it should have",
        -5,
        lambda env, player: 8 <= player.current_hand().get_original_delt_values() <= 11
        and env.dealer.current_hand().get_first_card_value() < 8
        and not player.current_hand().has_doubled,
    ),
    Reward(
        "Agent didn't hit vs high dealer card",
        -5,
        lambda env, player: env.dealer.current_hand().get_first_card_value() >= 7
        and player.current_hand().get_values() < 17,
    ),
    Reward(
        "Agent busted on a double",
        -5,
        lambda env, player: player.current_hand().has_doubled
        and player.current_hand().is_busted(),
    ),
    Reward(
        "Agent doubled with high starting value",
        -5,
        lambda env, player: player.current_hand().has_doubled
        and player.current_hand().get_original_delt_values() > 11
        and not player.current_hand().ace_in_original_hand,
    ),
    Reward(
        "Agent hit correctly vs high dealer card",
        2,
        lambda env, player: not player.current_hand().hit_above_17
        and len(player.current_hand().cards) > 2
        and player.current_hand().get_original_delt_values() <= 11
        and env.dealer.current_hand().get_first_card_value() >= 7,
    ),
    Reward(
        "Agent stayed on dealt high cards",
        2,
        lambda env, player: len(player.current_hand().cards) == 2
        and player.current_hand().get_values() >= 17,
    ),
    Reward(
        "Agent stood after drawing above 11 with low dealer card",
        2,
        lambda env, player: player.current_hand().get_values() > 11
        and len(player.current_hand().cards) > 2
        and env.dealer.current_hand().get_first_card_value() <= 6
        and not player.current_hand().hit_above_17,
    ),
]

