function expt = config_bayesian_categorisation(varargin)


% CONFIGURATION for BAYESIAN CATEGORISATION TASK: creates experiment and sets parameters
   % config with overridable fields at the end


% Defaults
expt.n_trials   = 1000;     % total number of trials in a run
expt.hazard     = 1/80;     % context change hazard (per-trial prob of block switch)
expt.gamma_T    = 1.40;     % exponention for threat (prior reweighting strength)
expt.gamma_N    = 1.00;     % Bs exponent for Non-threat
expt.theta      =  0.05; 
% TBI relevent weights (defaults; can be overridden for sensitivity checks)
expt.delta_entropy = 0.10;  % δ: entropy penalty weight 
expt.beta_utility  = 2.00;  % β: utility emphasis 


% Likelihood / generative parameters
expt.mu_threat    =  1.25;  % mean of stimulus under threat
expt.mu_nonthreat = -1.25;  % mean of stimulus under non-threat
expt.sigma        =  1.0;   % observation noise SD


% high/low anchor probs (schedule / baselines)
expt.p_H = 0.9;             % base rate P(Threat) in "high" context
expt.p_L = 0.1;             % base rate P(Threat) in "low" context


%%  Agent parameters:

% Agent C: affective modulation params
expt.aff_decay     = 0.55;  % EMA decay for affect signal (closer to 1 = slower)
expt.aff_bias_base = 0.15;  % baseline affective push toward threat
expt.aff_gain      = 0.13;  % gain from "affective surprise" to target prior


%% Agent D !!(adaptive / 'coherence' preserving)
expt.alpha_base      = 0.05; % base learning rate when stable
expt.max_step        = 0.18; % clamp on single-step prior change (≤ max_step)
expt.k_map           = 2; % mapping sharpness: used by Ds utility mapping














% name value overrides (single pass)
if ~isempty(varargin)
    assert(mod(numel(varargin),2)==0,'use name–value pairs.');
    for i = 1:2:numel(varargin)
        key = char(varargin{i});
        val = varargin{i+1};
        if isfield(expt, key)
            expt.(key) = val;
        else
            warning('config_bayesian_categorisation:UnknownField',...
                    'unknown field "%s" ignored.', key);
        end
    end
end
end