using POMDPs
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater, has_consistent_distributions, alphavectors
using QuickPOMDPs: QuickPOMDP
# using SARSOP: SARSOPSolver
using NativeSARSOP: SARSOPSolver
using Statistics
using Plots
using Random

# ASEN 5264 Final Project: Formulating Blackjack as an MDP and POMDP and solving in Julia
# Date Created: 04/23/2026
# Authors: Lucas Becker and Theodore Kishner


#########################################
# Define the Blackjack MDP
