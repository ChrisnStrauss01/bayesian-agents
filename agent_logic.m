function [posterior, decision, updated_prior, update_cost, adaptive_out, TBI_score] = agent_logic( ...
    agent_name, p_prior, likelihood_T, likelihood_N, expt, is_adaptive, alpha_eff, PE_smooth, momentum_in, last_sign_in, t)

%                                     AGENT LOGIC
%   Computes one-trial Bayesian update and decision for an agent.
% Agents A–C use a fixed path (posterior + symmetric decision rule at 0.5).
% Agent D adds an adaptive update with gain modulation (TBI score), leak,
% momentum, and a bounded step to update the prior.


%  defaults  
if nargin < 12 || isempty(t),            t = 1;  end
if nargin < 10 || isempty(momentum_in),  momentum_in  = 0; end
if nargin < 11 || isempty(last_sign_in), last_sign_in = 0; end
if nargin < 9  || isempty(PE_smooth),    PE_smooth    = 0; end

% Initialise adaptive-out container
adaptive_out = struct('momentum',momentum_in,'last_sign',last_sign_in,'kappa',0,'step',0,'alpha_eff_used',alpha_eff);
TBI_score = 0;

%% Posterior (single-trial update)
% guards against degenerate likelihoods (standard Bayes)
likelihood_T = max(likelihood_T, 1e-12);
likelihood_N = max(likelihood_N, 1e-12);
denom        = p_prior .* likelihood_T + (1 - p_prior) .* likelihood_N;   % p(x_t)
denom        = max(denom, 1e-12);
posterior    = (p_prior .* likelihood_T) ./ denom;



%% decision rule (symmetrical threshold)
decision = (posterior >= 0.5);


%% defaults for non-adaptive agents 
updated_prior = p_prior;   
update_cost   = 0;


%% Agent D: adaptive path only
if strcmp(agent_name,'D') && is_adaptive

    % leak /integration factor k_t from smoothed PE
    % Higher PE_smooth -> smaller k_t 
    kappa = 0.80 - 0.30 * PE_smooth;
    kappa = min(max(kappa, 0.50), 0.95);
    raw_d = kappa * (posterior - p_prior);   % pull toward current posterior

    % soft momentum 
    momentum = momentum_in;
    sign_d   = sign(posterior - p_prior);
    if t == 1
        momentum = 0;
    elseif sign_d == last_sign_in
        momentum = min(momentum + 0.04, 0.12);   % ramp when direction persists
    else
        momentum = max(momentum - 0.08, 0);      % decay when direction flips
    end
    if PE_smooth < 0.02
        momentum = momentum * 0.85;              %damp momentum when very stable
    end


    % predicted step (clamped) 
    if ~isfield(expt,'max_step') || isempty(expt.max_step), expt.max_step = 0.25; end
    step_pred = alpha_eff * (1 + momentum) * raw_d;
    step_pred = max(min(step_pred, expt.max_step), -expt.max_step);
    step_cost = abs(step_pred);                  % simple per-step complexity proxy

    % TBI score: utility − (KL + surprise + entropy penalty + step cost)
    EU_threat    = posterior *  2.0  + (1 - posterior) * (-0.25);
    EU_nonthreat = posterior * (-2.0) + (1 - posterior) *  0.25;
    E_U = max(EU_threat, EU_nonthreat);

  
    p0 = min(max(p_prior,   1e-12), 1-1e-12);
    p1 = min(max(posterior, 1e-12), 1-1e-12);
    KL = p0*log(p0/p1) + (1-p0)*log((1-p0)/(1-p1));   % Complexity 
    PredErr = -log(denom);                             % Surprise 
    H_t    = -( p1*log(p1) + (1-p1)*log(1-p1) );  %  entropy in bits
    H_norm = H_t / log(2);                       

    % light defaults
  defaults = {'beta_utility',2.0; 'delta_entropy',0.1; ...
            'lambda_update_cost',0; 'entropy_scale',4.0};
for i = 1:size(defaults,1)
    f = defaults{i,1}; v = defaults{i,2};
    if ~isfield(expt,f) || isempty(expt.(f))
        expt.(f) = v;
    end
end

 
    TBI_score = (1 + expt.beta_utility) * E_U ...
              - KL ...
              - PredErr ...
              - expt.delta_entropy * expt.entropy_scale * H_norm ...
              - expt.lambda_update_cost * step_cost;

    % gentle deterrent when PE is small
    eta  = 0.01;
    gate = 1 ./ (1 + exp(-40*(0.04 - PE_smooth)));
    TBI_score = TBI_score - eta * (step_pred.^2) .* gate;

    % map TBI -> multiplicative (min max) 
    lambda_min = 0.35; lambda_max = 0.85; k_map = 2;
    lambda_tbi = lambda_min + (lambda_max - lambda_min) ./ (1 + exp(-k_map * TBI_score));
  

    % apply step and log few internals
    step          = lambda_tbi * step_pred;
    updated_prior = min(max(p_prior + step, 0.02), 0.98);
    update_cost   = abs(step);
    adaptive_out.momentum        = momentum;
    adaptive_out.last_sign       = sign_d;
    adaptive_out.kappa           = kappa;
    adaptive_out.step            = step;
    adaptive_out.alpha_eff_base  = alpha_eff;         
    adaptive_out.alpha_eff_used  = alpha_eff * lambda_tbi; 

end
