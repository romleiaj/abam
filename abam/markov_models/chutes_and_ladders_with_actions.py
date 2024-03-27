# Using chutes and ladders as a toy example to explore markov models with
# Based off this article: https://math.uchicago.edu/~may/REU2014/REUPapers/Hochman.pdf
# and this video for MDP https://www.youtube.com/watch?v=9g32v7bK3Co
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Iterable, Dict, List
from pydantic import BaseModel, PositiveInt
from functools import reduce

# Question: What is the optimal die to roll at any give square?

class ChutesAndLadders(BaseModel):
    actions: List[int]
    chutes_ladders: Dict[PositiveInt, PositiveInt]
    num_states: int
    discount_factor: float

    def reward_function(self, state: int) -> float:
        """Maps a given state to a reward.

        Args:
            state (int): State you're evaluating.

        Returns:
            float: Reward.
        """
        if state == (self.num_states - 1):
            return 1.
       # experiments with adding reward at the ladders and chutes
       # if current_state in chutes_ladders:
       #     if chutes_ladders[current_state] > current_state: # ladder
       #         return 1.
       #     else:
       #         return -1.
        return 0

    def transition_function(self,
                            current_state: int,
                            next_state: int,
                            action: int) -> float:
        """This function answers the question, from where we're standing
           (current_state), what is the probability that we'll end up at
           next_state after one transition under a given action?

        Args:
            current_state (int): Index of current state.
            next_state (int): Index of next state.
            die_sides (int): Number of sides to the die.
            action (int): Current action to take.

        Returns:
            float: Probability of reaching next state after one transition.
        """
        die_sides = action
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
        if current_state == next_state and current_state >= (self.num_states - die_sides):
            return (die_sides - (self.num_states - current_state - 1)) / die_sides
        # normally, just return our dice roll probability
        visible_states = self.get_visible_states(current_state,
                                                 action=die_sides)
        if next_state in visible_states:
            # a single state can be visible multiple times if its length
            # includes both ends of a chute/ladder, so account for that here.
            num_times_visible = sum([1. for s in visible_states if
                                     next_state == s])
            return num_times_visible / die_sides
        # but most of the time, it will be just 0
        return 0.

    def get_visible_states(self,
                           current_state: int,
                           action: int) -> Iterable[int]:
        """What states can we actually see (have the possibility to
        transition to) from where we're standing, under a given action?

        Args:
            current_state (int): Current column index.
            die_sides (int): Number of sides to the die.
            action (int): Current action to take.

        Returns:
            Iterable[int]: Column indices we could transition to.
        """
        die_sides = action
        visible = []
        if current_state < (self.num_states - die_sides):
            visible  = list(range(current_state + 1, current_state + die_sides + 1))
        else:
            visible = list(range(current_state + 1, self.num_states))
        for i, state in enumerate(visible):
            # have to account for chutes and ladders
            if state in self.chutes_ladders:
                visible[i] = self.chutes_ladders[state]
        return visible

    def get_transition_matrix(self) -> NDArray[np.float_]:
        """Calculate and return the transition matrix.
        
        Returns:
            NDArray[np.float_]: Transition probabilities for every state.
        """
        # transition matrix (mxNxN of zeros)
        Pa = np.zeros((len(self.actions), self.num_states, self.num_states)) 
        for i, die in enumerate(self.actions):
            for j in range(self.num_states):
                for k in range(self.num_states):
                    # iterate through columns first
                    Pa[i, j, k] = self.transition_function(k, j,
                                                          action=die)
        return Pa

    def get_expected_number_of_turns(self,
                                     P: ArrayLike,
                                     start_state: int,
                                     final_state: int) -> float:
        """Given a transition matrix, starting state, and an end state,
        report the expected number of turns it will require to reach
        the end state.

        Args:
            P (ArrayLike): Transition matrix.
            start_state (int): Index of starting state.
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

    def is_valid_transition_matrix(self, P: NDArray[np.float_]) -> bool:
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
            
    def bellmans(self, state: int, action: int,
                 V: NDArray) -> float:
        """ When you have the reward and the probability for getting that
            reward, you can use Bellman's to get the expected value,
            discounting the future value at V[visible_state] by
            some float < 1 so the policy converges

        Args:
            state (int): Current state you're assessing to determine
                         the best action to take.
            action (int): A given action you're evaluating.
            V (NDArray): The current value matrix.

        Returns:
            float: The amount of expected value to be gained by this
                   action at this state.
        """
        value = []
        for visible_state in self.get_visible_states(state, action):
            r = self.reward_function(visible_state)
            prob = self.transition_function(state, visible_state, action)
            q = prob * (r + self.discount_factor * V[visible_state])
            value.append(q)
        return sum(value)

    def value_iteration(self) -> tuple[Dict, float]:
        """Find the optimal policy for this instance of
        Chutes and Ladders.

        Returns:
            tuple[Dict, int]: Returns the optimal policy, mapping state
            to action, and the final loss for analyzing the error.
        """
        V = np.zeros(self.num_states) # values for each state
        pi = None
        it = 0
        max_it = 1000
        while it < max_it:
            newV = np.zeros(self.num_states)
            for state in range(self.num_states):
                if state == (self.num_states - 1):
                    # if we're at the end state, no reward to be gained
                    newV[state] = 0.
                else:
                    newV[state] = max(self.bellmans(state, action, V)
                                      for action in self.actions)
            # check for convergence
            loss = max(abs(V[state] - newV[state]) for state in
                       range(self.num_states))
            if loss < 1e-10:
                break
            # Set our previous value matrix to the new one, and run it again
            V = newV
            it += 1
        print(f"Converged after {it} iterations.")
        # Now that we have our final converged value matrix, we run it 
        # through Bellman's once last time to get the final mapping from
        # state to action.
        pi = np.zeros(self.num_states)
        for state in range(self.num_states):
            if state == self.num_states - 1:
                pi[state] = None
            else:
                pi[state] = max((self.bellmans(state, action, V), action)
                                for action in self.actions)[1]
        return pi, loss
        

if __name__ == "__main__":
    # number of sides is the value
    dice = [6, 8]
    discount_factor = 0.9
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
    CaL = ChutesAndLadders(actions=dice,
                           chutes_ladders=chutes_ladders,
                           num_states=num_states,
                           discount_factor=discount_factor)
    pi, loss = CaL.value_iteration()
    # being explicit about pi's functionality
    state_to_action = pi
    print(pi)