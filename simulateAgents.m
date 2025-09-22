function [agent_results, agents] = simulateAgents(expt, agents, stimuli, true_category, perceived_category, ~)
% SIMULATEAGENTS
% Run agents A–D on a 2afc threat categorisation task,
% agent D uses TBI-modulated adaptive updating (others non-adaptive).

n_trials = expt.n_trials;
mu_T     = expt.mu_threat;
mu_N     = expt.mu_nonthreat;
sigma    = expt.sigma;

agent_names = {'A','B','C','D'};

% preallocate agent outputs 
for i = 1:numel(agent_names)
    nm = agent_names{i};
    agent_results.(nm).posterior = zeros(n_trials,1);
    agent_results.(nm).decision  = zeros(n_trials,1);
    agent_results.(nm).fitness   = zeros(n_trials,1);
    agent_results.(nm).fit_total = 0;
end

% Agent D extra channels
agent_results.D.update_cost   = zeros(n_trials,1);


%  explicit per-trial vectors requested
agent_results.D.alpha_eff_vec = zeros(n_trials,1);   % effective alpha  
agent_results.D.Hq_vec        = nan(n_trials,1);     % posterior entropy (0..1)

%  priors 
prior_A = 0.5 * ones(n_trials+1,1);
prior_B = zeros(n_trials+1,1); prior_B(1) = agents.B.prior(1);
prior_C = zeros(n_trials+1,1); prior_C(1) = 0.5;
prior_D = zeros(n_trials+1,1); prior_D(1) = 0.3;

% Agent D state
momentum  = 0;
last_sign = 0;
alpha_base = expt.alpha_base; % base step size (scaled inside agent logic for D).


for t = 1:n_trials
    x_t = stimuli(t);

    % likelihoods
    like_T = max(normpdf(x_t, mu_T, sigma), 1e-6);
    like_N = max(normpdf(x_t, mu_N, sigma), 1e-6);

    %%  Agent A 
    pA = prior_A(t);
    [postA, decA] = agent_logic('A', pA, like_T, like_N, expt, false, 0, 0, 0, 0);
    agent_results.A.posterior(t) = postA;
    agent_results.A.decision(t)  = decA;
    agent_results.A.fitness(t)   = payoff(agent_results.A.decision(t), true_category(t));
    agent_results.A.fit_total    = agent_results.A.fit_total + agent_results.A.fitness(t);
    prior_A(t+1) = pA;

    %%  Agent B
    pB = prior_B(t);
    pB = min(max(pB, 0.05), 0.95);
    window = max(1,t-15):t; idx_window = window(window >= 1);
    if ~isempty(idx_window)
        trend = mean(perceived_category(idx_window) - pB);
        pB = pB + 0.01 * trend;
    end
    pB = min(max(pB, 0.45), 0.55);
    [postB, decB] = agent_logic('B', pB, like_T, like_N, expt, false, 0, 0, 0, 0);
    agent_results.B.posterior(t) = postB;
    agent_results.B.decision(t)  = decB;
    if rand < 0.03, agent_results.B.decision(t) = ~agent_results.B.decision(t); end
    agent_results.B.fitness(t) = payoff(agent_results.B.decision(t), true_category(t));
    agent_results.B.fit_total  = agent_results.B.fit_total + agent_results.B.fitness(t);
    prior_B(t+1) = pB;

    %% Agent C 
    pC = prior_C(t);
    aff_surprise = tanh(abs(pC - perceived_category(t)).^1.5);
    if mean(perceived_category(max(1,t-19):t)) > 0.5
        p_valence = 0.73;
    else
        p_valence = 0.20;
    end
    target  = p_valence + expt.aff_bias_base + expt.aff_gain * aff_surprise;
    next_pC = expt.aff_decay * pC + (1 - expt.aff_decay) * target;
    next_pC = min(max(next_pC, 0.05), 0.95);
    [postC, decC] = agent_logic('C', pC, like_T, like_N, expt, false, 0, 0, 0, 0);
    agent_results.C.posterior(t) = postC;
    agent_results.C.decision(t)  = decC;
    agent_results.C.fitness(t)   = payoff(agent_results.C.decision(t), true_category(t));
    agent_results.C.fit_total    = agent_results.C.fit_total + agent_results.C.fitness(t);
    prior_C(t+1) = next_pC;

    %%  Agent D 
    pD = prior_D(t);

    [postD, decD, updated_pD, update_cost, adapt, ~] = agent_logic( ...
    'D', pD, like_T, like_N, expt, true, alpha_base, 0, momentum, last_sign, t);

    momentum  = adapt.momentum;
    last_sign = adapt.last_sign;

    agent_results.D.posterior(t)   = postD;
    agent_results.D.decision(t)    = decD;
    agent_results.D.update_cost(t) = update_cost;
    agent_results.D.fitness(t)     = payoff(decD, true_category(t));
    agent_results.D.fit_total      = agent_results.D.fit_total + agent_results.D.fitness(t);
  

    % per-trial vectors
    agent_results.D.alpha_eff_vec(t) = adapt.alpha_eff_used;   % λ_tbi x base alpha 
  
    p = min(max(postD, 1e-12), 1-1e-12);
    agent_results.D.Hq_vec(t) = -( p*log(p) + (1-p)*log(1-p) ) / log(2);

    prior_D(t+1) = updated_pD;
end

% expose priors 
agent_results.A.p_threat = prior_A(1:n_trials);
agent_results.B.p_threat = prior_B(1:n_trials);
agent_results.C.p_threat = prior_C(1:n_trials);
agent_results.D.p_threat = prior_D(1:n_trials);
agents.B.p_threat_traj   = prior_B;
agents.C.p_threat_traj   = prior_C;
agents.D.p_threat_traj   = prior_D;

end

%% payoff helper
function u = payoff(decision, true_cat)
if true_cat == 1
    if decision == 1, u = 2.0; else, u = -2.0; end
else
    if decision == 0, u = 0.25; else, u = -0.25; end
end
end