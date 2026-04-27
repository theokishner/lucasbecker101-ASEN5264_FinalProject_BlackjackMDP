using POMDPs
using POMDPTools: SparseCat, Deterministic
using QuickPOMDPs: QuickPOMDP
using Statistics
using Random

############################
# STATE
############################
struct BJState
    player_sum::Int
    dealer_upcard::Int
    usable_ace::Bool
    deck_counts::NTuple{10, Int}  # [A,2,...,10]
    terminal::Bool
end

const ACTIONS = [:hit, :stick]

############################
# HELPERS
############################
function draw_card_dist(deck_counts)
    total = sum(deck_counts)
    probs = [c / total for c in deck_counts]
    return SparseCat(1:10, probs)
end

function remove_card(deck, card)
    return ntuple(i -> i == card ? deck[i] - 1 : deck[i], 10)
end

function update_sum(sum_val, usable_ace, card)
    if card == 1
        if sum_val + 11 <= 21
            return sum_val + 11, true
        else
            return sum_val + 1, usable_ace
        end
    else
        return sum_val + card, usable_ace
    end
end

function adjust_for_ace(sum_val, usable_ace)
    if sum_val > 21 && usable_ace
        return sum_val - 10, false
    else
        return sum_val, usable_ace
    end
end

############################
# DEALER (FIXED — NO SHADOWING sum)
############################
function dealer_outcomes(dealer_upcard, deck)
    outcomes = Dict{Int, Float64}()

    function recurse(dealer_sum, usable, deck, prob)
        dealer_sum, usable = adjust_for_ace(dealer_sum, usable)

        if dealer_sum >= 17
            outcomes[dealer_sum] = get(outcomes, dealer_sum, 0.0) + prob
            return
        end

        total = sum(deck)

        for card in 1:10
            if deck[card] == 0
                continue
            end

            p = deck[card] / total
            new_deck = remove_card(deck, card)

            new_sum, new_usable = update_sum(dealer_sum, usable, card)
            recurse(new_sum, new_usable, new_deck, prob * p)
        end
    end

# correct dealer initialization
    init_sum = dealer_upcard == 1 ? 11 : dealer_upcard
    init_usable = dealer_upcard == 1 ? true : false

    recurse(init_sum, init_usable, deck, 1.0)
    return outcomes
end

############################
# TRANSITION
############################
function bj_transition(s::BJState, a)
    if s.terminal
        return Deterministic(s)
    end

    deck = s.deck_counts

    # HIT
    if a == :hit
        dist = draw_card_dist(deck)

        states = BJState[]
        probs = Float64[]

        for card in 1:10
            p = pdf(dist, card)
            if p == 0.0
                continue
            end

            new_deck = remove_card(deck, card)

            new_sum, usable = update_sum(s.player_sum, s.usable_ace, card)
            new_sum, usable = adjust_for_ace(new_sum, usable)

            terminal = new_sum > 21

            push!(states, BJState(
                new_sum,
                s.dealer_upcard,
                usable,
                new_deck,
                terminal
            ))
            push!(probs, p)
        end

        return SparseCat(states, probs)
    end

    # STICK
    if a == :stick
    dealer_out = dealer_outcomes(s.dealer_upcard, s.deck_counts)

    states = BJState[]
    probs = Float64[]

        for (dealer_sum, p) in dealer_out
            terminal_state = BJState(
             s.player_sum,
             dealer_sum,   # store final dealer result
             s.usable_ace,
             s.deck_counts,
             true
        )
            push!(states, terminal_state)
            push!(probs, p)
        end

        return SparseCat(states, probs)
    end

    return Deterministic(s)
end

############################
# REWARD
############################
function bj_reward(s::BJState, a, sp::BJState)
    # 1. If we are ALREADY in a terminal state, no more rewards are given.
    if s.terminal
        return 0.0
    end

    # 2. If the NEXT state is not terminal, no reward is given yet.
    if !sp.terminal
        return 0.0
    end

    # 3. Calculate final win/loss/draw
    if sp.player_sum > 21
        return -1.0
    end

    dealer_sum = sp.dealer_upcard  # overloaded as final result

    if dealer_sum > 21 || sp.player_sum > dealer_sum
        return 1.0
    elseif sp.player_sum == dealer_sum
        return 0.0
    else
        return -1.0
    end
end

############################
# INITIAL STATE
############################
function build_initial_dist()
    init_deck = (4,4,4,4,4,4,4,4,4,16)
    total_cards = sum(init_deck)
    
    states = BJState[]
    probs = Float64[]
    
    for card in 1:10
        # Probability of drawing this card as the upcard
        p = init_deck[card] / total_cards 
        
        # Remove it from the deck
        new_deck = remove_card(init_deck, card)
        
        # Player sum starts at 0, dealer has their upcard
        push!(states, BJState(0, card, false, new_deck, false))
        push!(probs, p)
    end
    
    return SparseCat(states, probs)
end

initial_dist = build_initial_dist()

############################
# MDP
############################
############################
# MDP
############################
m_onedeck = QuickPOMDP(
    actions = ACTIONS,
    discount = 1.0,

    transition = bj_transition,
    reward = bj_reward,
    initialstate = initial_dist,

    observation = (a, sp) -> Deterministic(sp),
    obstype = BJState,
    
    # ADD THIS LINE:
    isterminal = s -> s.terminal 
)

############################
# POLICY
############################
function baseline_policy(m, s)
    if s.player_sum > 15
        return :stick
    else
        return :hit
    end
end

############################
# ROLLOUT
############################
function rollout(mdp, policy_function, s0, max_steps)
    r_total = 0.0
    disc = 1.0
    γ = discount(mdp)
    s = s0

    for t in 1:max_steps
        if isterminal(mdp, s)
            break
        end

        a = policy_function(mdp, s)

        dist = bj_transition(s, a)
        sp = rand(dist)

        r = bj_reward(s, a, sp)

        r_total += disc * r
        disc *= γ
        s = sp
    end

    return r_total
end

############################
# SIMULATION
############################
baseline_score = [
    rollout(m_onedeck, baseline_policy, rand(initial_dist), 10)
    for _ in 1:10000
]

println(baseline_score[1:10])  # print first 10 rewards for sanity check
println("Mean reward: ", mean(baseline_score))
println("Std error: ", std(baseline_score) / sqrt(length(baseline_score)))

# ----------- Q-LEARNING Implementation ----------- #
function qlearning_episode!(Q, mdp; ϵ=0.10, γ=1.0, α=0.2)
    start = time()
    
    # Epsilon-greedy policy (Compatible with all Julia versions)
    function policy(s)
        if rand() < ϵ
            return rand(ACTIONS)
        else
            # Safely get Q-values and find the index of the max
            q_vals = [get(Q, (s, a), 0.0) for a in ACTIONS]
            return ACTIONS[argmax(q_vals)]
        end
    end

    # 1. Safe Initialization (bypasses the Deterministic fallback bug)
    init_dist = initialstate(mdp)
    s = init_dist isa Deterministic ? init_dist.val : rand(init_dist)
    a = policy(s)
    
    hist = typeof(s)[]
    push!(hist, s)

    while !s.terminal
        # 2. Transition and Reward (Using POMDPs.jl API, not CommonRLInterface)
        dist = transition(mdp, s, a)
        
        # Safely extract state to avoid the "not callable" error
        sp = rand(transition(mdp, s, a))
        
        r = reward(mdp, s, a, sp)

        # 3. Update Q-Table safely
        best_next_q = maximum([get(Q, (sp, a_next), 0.0) for a_next in ACTIONS])
        current_q = get(Q, (s, a), 0.0)
        Q[(s, a)] = current_q + α * (r + γ * best_next_q - current_q)

        # 4. Step forward
        s = sp
        a = policy(sp)
        push!(hist, sp)
    end

    return (hist=hist, Q=copy(Q), time=time()-start)
end

function qlearning!(mdp; n_episodes = 10000, α=0.2)
    # Start with an empty Dict, typed to your specific State/Action pairs
    Q = Dict{Tuple{BJState, Symbol}, Float64}()
    episodes = []
    
    for i in 1:n_episodes
        push!(episodes, qlearning_episode!(Q, mdp;
                                           ϵ=max(0.1, 1 - i/n_episodes), α=α))
    end
    
    return episodes
end

# ----------- Evaluation Function ----------- #
# This replaces your old evaluate function to work with POMDPs instead of CommonRLInterface
function evaluate(mdp, Q; n_episodes=1000)
    function greedy_policy(m, s)
        q_vals = [get(Q, (s, a), 0.0) for a in ACTIONS]
        return ACTIONS[argmax(q_vals)]
    end
    
    returns = Float64[]
    for _ in 1:n_episodes
        init_dist = POMDPs.initialstate(mdp)
        s0 = init_dist isa Deterministic ? init_dist.val : rand(init_dist)
        push!(returns, rollout(mdp, greedy_policy, s0, 10))
    end
    
    return returns
end

# Run Q-Learning
num_episodes = 10000
qlearning_episodes = qlearning!(m_onedeck, n_episodes=num_episodes, α=0.075)
episodes_dict = Dict("Q-Learning" => qlearning_episodes)

# ----------- Plotting Learning Curves --------- # 
function learning_curve_steps(episodes_dict, mdp)
    p = plot(xlabel="Steps in environment", ylabel="Avg return", legend=:bottomright)
    n = 1000
    stop = num_episodes
    
    for (name, eps) in episodes_dict
        xs = [0]
        # Evaluate empty Q-table
        ys = [mean(evaluate(mdp, Dict{Tuple{BJState, Symbol}, Float64}()))] 
        
        for i in n:n:min(stop, length(eps))
            newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
            push!(xs, last(xs) + newsteps)
            
            Q = eps[i].Q
            push!(ys, mean(evaluate(mdp, Q)))
        end    
        plot!(p, xs, ys, label=name)
    end
    return p
end

p_steps = learning_curve_steps(episodes_dict, m_onedeck)
display(p_steps)

# ----------- Final Evaluation & Comparison ----------- #

# 1. Grab the final Q-table from the very last episode of training
final_Q = qlearning_episodes[end].Q

# 2. Evaluate the learned policy for 10,000 episodes (same as baseline)
println("Evaluating final Q-learning policy...")
q_scores = evaluate(m_onedeck, final_Q, n_episodes=10000)

# 3. Calculate means and standard errors
mean_baseline = mean(baseline_score)
se_baseline = std(baseline_score) / sqrt(length(baseline_score))

mean_q = mean(q_scores)
se_q = std(q_scores) / sqrt(length(q_scores))

# 4. Print a nice comparison table
println("\n" * "="^40)
println("     BASELINE VS. Q-LEARNING RESULTS    ")
println("="^40)
println("Baseline (Hit if <= 15):")
println("  Mean Return: ", round(mean_baseline, digits=4))
println("  Std Error:   ", round(se_baseline, digits=4))
println("-"^40)
println("Q-Learning (10,000 episodes):")
println("  Mean Return: ", round(mean_q, digits=4))
println("  Std Error:   ", round(se_q, digits=4))
println("="^40)

