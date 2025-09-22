function agents = initAgents(expt, stimuli)
%                   Initialise priors for Agents A–D
% where B uses a simple schema prior from stimulus thresholding, then exponent mapp
% C starts neutral and D conservative (adapted elsewhere)
stimuli  = stimuli(:);                
n_trials = numel(stimuli);             

% Agent A - flat prior
agents.A.prior = 0.5 * ones(n_trials,1);

% Agent B - schema prior + exponent mapping
if isfield(expt,'schema_threshold') && ~isempty(expt.schema_threshold)
    thr = expt.schema_threshold;
else
    thr = 1.5; 
end
pi_T                = 0.5 * ones(n_trials,1);
pi_T(stimuli > thr) = 0.7; 
pi_T(stimuli < thr) = 0.3;

% exponent map (with safe defaults)
if ~isfield(expt,'gamma_T') || isempty(expt.gamma_T), expt.gamma_T = 1.0; end
if ~isfield(expt,'gamma_N') || isempty(expt.gamma_N), expt.gamma_N = 1.0; end
num           = pi_T.^expt.gamma_T;
den           = num + (1 - pi_T).^expt.gamma_N;
agents.B.prior = num ./ den;

% Agent C: neutral start (affective updates happen elsewhere)
agents.C.prior    = zeros(n_trials,1);
agents.C.prior(1) = 0.5;

% Agent D: conservative start; carry only useful scalars
agents.D.prior      = zeros(n_trials,1);
agents.D.prior(1)   = 0.3;
agents.D.alpha_base = expt.alpha_base;  % downstream

% Optional passthroughs 
if isfield(expt,'alpha_max'),          agents.D.alpha_max          = expt.alpha_max; end
if isfield(expt,'beta_utility'),       agents.D.beta_utility       = expt.beta_utility; end
if isfield(expt,'delta_entropy'),      agents.D.delta_entropy      = expt.delta_entropy; end
if isfield(expt,'lambda_update_cost'), agents.D.lambda_update_cost = expt.lambda_update_cost; end
if isfield(expt,'max_step'),           agents.D.max_step           = expt.max_step; end

end