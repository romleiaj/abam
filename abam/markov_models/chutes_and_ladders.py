# Using chutes and ladders as a toy example to explore markov models with
# Based off this article: https://math.uchicago.edu/~may/REU2014/REUPapers/Hochman.pdf
import numpy as np
from numpy.typing import ArrayLike
from typing import Iterable, Dict
from pydantic import BaseModel, PositiveInt
from functools import reduce

# Question: What is the expected number of steps to reach an (the) absorption state?

class ChutesAndLadders(BaseModel):
    die_sides: int 
    chutes_ladders: Dict[PositiveInt, PositiveInt]
    num_states: int

    def transition_function(self,
                            current_state: int,
                            next_state: int) -> float:
        """This function answers the question, from where we're standing
           (current_state), what is the probability that we'll end up at
           next_state after one transition?

        Args:
            current_state (int): Index of current state.
            next_state (int): Index of next state.

        Returns:
            float: Probability of reaching next state after one transition.
        """
        if current_state in chutes_ladders:
            # if we're somehow standing on a chute/ladder start point,
            # transition to the end. This is so columns sum to 1
            if next_state == chutes_ladders[current_state]:
                return 1.0
            else:
                return 0.0
        # if the last state, return 1
        if current_state == next_state and current_state == (self.num_states - 1):
            return 1.0
        # if in the last die_side states, it's possible to remain in the same state
        if current_state == next_state and current_state >= (self.num_states - self.die_sides):
            return (self.die_sides - (self.num_states - current_state - 1)) / self.die_sides
        # normally, just return our dice roll probability
        visible_states = self.get_visible_states(current_state)
        if next_state in visible_states:
            # a single state can be visible multiple times if its length is
            # less than k, so account for that here.
            num_times_visible = sum([1. for s in visible_states if next_state == s])
            return num_times_visible / self.die_sides
        # but most of the time, it will be just 0
        return 0.

    def get_visible_states(self, current_state: int) -> Iterable[int]:
        """What states can we actually see (have the possibility to
        transition to) from where we're standing?

        Args:
            current_state (int): Current column index.

        Returns:
            Iterable[int]: Column indices we could transition to.
        """
        visible = []
        if current_state < (self.num_states - self.die_sides):
            visible  = list(range(current_state + 1, current_state + self.die_sides + 1))
        else:
            visible = list(range(current_state + 1, self.num_states))
        for i, state in enumerate(visible):
            # have to account for chutes and ladders
            if state in self.chutes_ladders:
                visible[i] = self.chutes_ladders[state]
        return visible

    def get_transition_matrix(self) -> ArrayLike:
        P = np.zeros((self.num_states, self.num_states)) # transition matrix (NxN of zeros)
        for i in range(self.num_states):
            for j in range(self.num_states):
                # iterate through columns first
                P[i, j] += self.transition_function(j, i)
        return P

    def get_expected_number_of_turns(self,
                                     P: ArrayLike,
                                     start_state: int,
                                     final_state: int) -> float:
        """Given a starting state and an end state, report the expected
        number of turns it will require to reach the end state.

        Args:
            P (ArrayLike): Transition matrix.
            current_state (int): Index of current state.
            final_state (int): Index of end state.

        Returns:
            float: Expected number of turns to reach end state from start
            state give a certain transition matrix.
        """
        if final_state < start_state:
            return 0.0
        if final_state == start_state:
            return 1.0
        # since P is an absorption matrix, need to define a submatrix Q
        # that removes all absorption states so it is invertible.
        Q = P[:(num_states - 1), :(num_states - 1)]
        I = np.identity(num_states - 1)
        # N is the fundamental matrix for P
        N = np.linalg.inv(I - Q)
        # if we start at state j, N(j,i) is the expected number of times
        # that state i is reached. This includes the starting state, so N(j,j)
        # is always >= 1 
        # This mean to find the expected value to get to a specific state, sum that
        # transition states
        return np.sum(N[start_state:(final_state + 1), start_state])

    def is_valid_transition_matrix(self, P: ArrayLike) -> bool:
        """Evalated whether transition matrix is valid.

        Args:
            P (ArrayLike): Transition matrix.

        Returns:
            bool: Whether it is valid or not.
        """
        total_eps = 1e-4
        S = np.sum(np.sum(P, axis=0))
        if ( S - num_states ) > total_eps:
            return False
        return True
        
        
if __name__ == "__main__":
    die_sides = 6
    board_height = 10
    board_width = 10
    num_states = board_width * board_height
    # add figurative "0" start state
    num_states += 1
    # indices of chutes and ladders
    # ladders are represented by a tuple going from lower (ind0) to higher (ind1)
    # chutes are represented by a tuple going from higher to lower
    chutes_ladders = dict([(1, 38), (4, 14), (9, 31), (21, 42), (28, 84), (36, 44),
                            (51, 67), (71, 91), (80, 100), (16, 6), (47, 26), (49, 11),
                            (56, 53), (62, 19), (64, 60), (87, 24), (93, 73), (95, 75),
                            (98, 78)])
    CaL = ChutesAndLadders(die_sides=die_sides,
                           chutes_ladders=chutes_ladders,
                           num_states=num_states)
    P = CaL.get_transition_matrix()
    is_valid = CaL.is_valid_transition_matrix(P)
    print(f"Valid transition matrix? {is_valid}")

    v = np.zeros(101)
    # starting square
    v[0] = 1
    v = np.atleast_2d(v).T
    it = 20
    ret = reduce(lambda x, y: x @ y, [P] * it) @ v
    # After starting at square 0, after 20 dice rolls, what do the probabilities
    # look like?
    print(ret.T)

    s = 0
    e = 100
    n = CaL.get_expected_number_of_turns(P, s, e)
    print(f"Expected number of turns from {s} to {e}: {n}")