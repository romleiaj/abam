import numpy as np
from functools import reduce

k = 8 
board_height = 10
board_width = 10
N = board_width * board_height
# add figurative state 0, "off board"
N += 1
P = np.zeros((N, N))
chutes_ladders = [(1, 38), (4, 14), (9, 31), (21, 42), (28, 84), (36, 44),
                        (51, 67), (71, 91), (80, 100), (16, 6), (47, 26), (49, 11),
                        (56, 53), (62, 19), (64, 60), (87, 24), (93, 73), (95, 75),
                        (98, 78)]

# going from 0 to N - k, pattern is the same. 1/k probability of 
# landing on the next k squares
for i in range(0, N - k):
    P[i+1:i+k+1, i] = 1.0 / k
    
# On the last k rows, slightly different formulation
for i in range(N - k, N - 1):
    # There's is now a likelihood of remaining in the same place (at i,i)
    # since multiple numbers can result in that outcome in this last row
    P[i, i] = (k - (N - i - 1)) / k
    # The remaining numbers in between are the usual formula
    P[(i+1):, i] = 1 / k

# The final row/col is the absorption state, so just a 1
P[N-1, N-1] = 1

# Now our gameboard is naively setup. Time to add some chutes and ladders!
for cl in chutes_ladders:
    # The first index is the start (f), the second is the end(g). The algorithm
    # is the same regardless if it's a chute or ladder
    # The probability of going to g is now the probability of going to f
    # plus what the probability of going to g is (usually 0, unless within k)
    f, g = cl
    # In every state that can end up at state g (which is all of them, technically)
    # Add the probability of ending up at state f to state g
    P[g, :] += P[f, :]
    # And now remove the probability that we can actually end up at state f
    P[f, :] = 0
    # State f can't go anywhere except g, but you never reach it in practicality
    P[:, f] = 0
    # if you're ever in state g, automatically transition to f
    P[g, f] = 1

print(np.sum(P, axis=0))
print(len(np.sum(P, axis=0)))

v = np.zeros(101)
# starting square
v[0] = 1
v = np.atleast_2d(v).T
it = 20
ret = reduce(lambda x, y: x @ y, [P] * it) @ v
# After starting at square 0, after 20 dice rolls, what do the probabilities
# look like?
print(ret.T)
        
start_state = 0
final_state = 100
# since P is an absorption matrix, need to define a submatrix Q
# that removes all absorption states so it is invertible.
Q = P[:(N - 1), :(N - 1)]
I = np.identity(N - 1)
# N is the fundamental matrix for P
N = np.linalg.inv(I - Q)
# if we start at state j, N(j,i) is the expected number of times
# that state i is reached. This includes the starting state, so N(j,j)
# is always >= 1 
# This mean to find the expected value to get to a specific state, sum that
# transition states
print(np.sum(N[start_state:(final_state + 1), start_state]))


# now say we have 2 dice, a d6 and a d8, and can choose between each die
# on each roll. How do we determine an optimal roll at any particular 
# state?

# To do this, we need to define a `policy`, which tells us what policy
# to select that maximizes the `reward`, in this case something we define
# that brings us closer to our goal (reaching the absorption state, square 0)

# Since we can calculate the number of step left to the end with the above
# formula, it intuitively makes sense that we want to optimize for that.
# Naively, then, I think 2 transition matrices need to be created, and then
# the die should be selected such that the state most likely to land in 
# has least number of expected turns to the end.

# So, reward when E[s,n] goes down, penalize when E[s,n] goes up, where s is your 
# current state, n is the absorption state.

# Using https://www.cs.cmu.edu/afs/cs/academic/class/15780-s16/www/slides/mdps.pdf
# as a guide.

# ??? something I need, to be determined what this actually is
discount_factor = 0.9
# Reward values, maps each state to a reward value
R = np.zeros(N-1)
# Only one absorption state, so set that as the max reward?
R[-1] = 100.
# Should probably set something for chutes and ladders too
for cl in chutes_ladders: 
    if cl[1] > cl[0]: # ladder
        R[cl[0]] = 1
    else: # chute
        R[cl[0]] = -1

# Transition matrix P is now supposed to define the probability
# distribution over the next states given the current state
# _and_ the current action.
# so:
# for i in actions:
#   for j in N:
#       for k in N:
#           P[k, j, i] = transition_function(k, i) 
#           etc...

# Now let's (try) to use Bellman's optimality equation!
# We need a value function for a policy, which gives
# the expected sum of discounted rewards under that policy.
# So in our case, with 2 dice, we need 2 value functions.

# I'll explicitly defined our 2 value functions as V6 and V8,
# Vpi for the general case.
# Vpi(current_state) = Reward(current_state) + discount_factor * \
#                             sum(P[next_state, :]*Vpi[next_state,:])
# Let vpi be a vector of values for each state, r be a vector of 
# rewards for each state. Ppi is what we've traditionally had, the 
# transition values for each function. l is the discount factor.
# Bellman equation in vector form is thus:
# vpi = r + l*Ppi*vpi
# algebra....
# vpi = (I - l*Ppi)^-1 * r
# AH-HAH! I've seen this before. That's N with a different pair of shoes.

Q = P[:(N - 1), :(N - 1)]
I = np.identity(N - 1)
# N is the fundamental matrix for P
vpi = np.linalg.inv(I - Q*discount_factor) * R

Vpi = np.zeros((N-1, N-1))
# 1. initialize an estimate for the value function randomly

# Vpi(current_state) = np.linalg.inv(I - Q*discount_factor)[current_state] * R[current_state]