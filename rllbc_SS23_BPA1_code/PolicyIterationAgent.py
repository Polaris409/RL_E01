import numpy as np
from agent import Agent


# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)

        self.V = {}
        for state in all_states:
            self.V[state] = 0.0


        # Policy initialization
        # ******************
        # TODO 1.1.a)
        # self.V = ...
        self.V = {s: 0.0 for s in states}  # Initializing the state value function to zero for each state


        self.pi = {}


        for s in states:
            a = self.mdp.getPossibleActions(s)
            if a:
                self.pi[s] = a[-1]
        else:
            self.pi[s] = None


        counter = 0
        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else 'None' for s in states}


        counter = 0

        self.V = {s: 0.0 for s in states}

        while True:
            # Policy evaluation
            for i in range(iterations):
                for s in states:
                    if self.pi[s] is None:
                       self.pi[s] = self.mdp.getPossibleActions(s)[-1]

                newV = {}
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # TODO 1.1.b)
                    # if...
                    #
                    # else:...

                    # update value estimate
                    # self.V=...
                    if self.mdp.isTerminal(s):
                        newV[s] = 0.0  # Set the state value of terminal states to 0.0
                    else:
                        transitions = self.mdp.getTransitions(s, a)
                        expected_value = 0.0
                        for next_state, reward, probability in transitions:
                            if next_state is not None:
                                expected_value += probability * (reward + discount * self.V.get(next_state, 0.0))


                        newV[s] = expected_value  # Update the value of non-terminal states
                        self.V = newV  # Update the current value function with the new value function

                # Update value estimate
                # ******************

            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    # self.pi[s] = ...

                    # policy_stable =
                best_action = None
                best_value = float('-inf')
                for a in actions:
                    transitions = self.mdp.getTransitions(s, a)
                    expected_value = 0.0
                    for next_state, reward, probability in transitions:
                        expected_value += probability * (reward + discount * self.V[next_state])
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = a
                self.pi[s] = best_action  # Update the policy with the greedy best action

                if old_action != self.pi[s]:
                    policy_stable = False  # Check if the policy has changed
                    # ****************
            counter += 1

            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.
        return self.V[state]
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 1.3.
        q_value = 0.0
        transitions = self.mdp.getTransitions(state, action)
        for next_state, reward, probability in transitions:
            q_value += probability * (reward + self.discount * self.V[next_state])
        return q_value
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        # **********
        # TODO 1.4.
        return self.pi[state]
        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
