import numpy as np
from functools import reduce

k = 6 
dice = [6] # sides to the dice
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