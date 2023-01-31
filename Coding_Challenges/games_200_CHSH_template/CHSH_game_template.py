#! /usr/bin/python3

import sys
import math
import random
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # normalize

    normalization_constant = 1 / math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha = alpha * normalization_constant
    beta = beta * normalization_constant

    # rotate

    theta = 2 * math.acos(alpha)

    qml.RY(theta, wires=[0])

    # entangle

    qml.CNOT(wires=[0, 1])

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    assert x in [0, 1]
    assert y in [0, 1]

    # Can't edit the final line. So, want to rotate back *onto* the computational basis state.

    if x == 0:
        qml.RY(- 2 * theta_A0, wires=[0])
    else:
        qml.RY(- 2 * theta_A1, wires=[0])

    if y == 0:
        qml.RY(- 2 * theta_B0, wires=[1])
    else:
        qml.RY(- 2 * theta_B1, wires=[1])

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # find a and b based on chsh_circuit and return prob(x * y == a + b mod 2)

    prob_wins = 0

    for (x, y) in [(0, 0), (0 ,1), (1, 0), (1, 1)]:
        res = chsh_circuit(params[0], params[1], params[2], params[3], x, y, alpha, beta)

        prob_a_0 = res[0] + res[1]
        prob_b_0 = res[0] + res[2]
        prob_a_1 = 1 - prob_a_0
        prob_b_1 = 1 - prob_b_0

        if x == 0 and y == 0:
            prob_wins += prob_a_0 * prob_b_0
        elif x == 0 and y == 1:
            prob_wins += prob_a_0 * prob_b_1
        elif x == 1 and y == 0:
            prob_wins += prob_a_1 * prob_b_0
        elif x == 1 and y == 1:
            prob_wins += prob_a_1 * prob_b_1

    return prob_wins / 4

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""

        return 1 - winning_prob(params, alpha, beta)

    # QHACK #

    #Initialize parameters, choose an optimization method and number of steps
    init_params = np.array([0.01, math.pi / 4, 0.01, math.pi / 4], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.8)
    steps = 10

    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters

        params = opt.step(cost, params)
        params = np.clip(opt.step(cost, params), - 2 * np.pi, 2 * np.pi)

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")
