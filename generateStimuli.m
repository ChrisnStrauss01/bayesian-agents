
function [stimuli, true_category, perceived_category, contexts] = generateStimuli(expt)
%       GENERATE STIMULI
% Build stimuli, true/perceived labels, and context probs for n_trials.
% Uses restored schedule if present; otherwise falls back to a simple hazard process.

%safe defaults (only used if not set upstream)
if ~isfield(expt,'n_trials'),      expt.n_trials      = 1000;  end
if ~isfield(expt,'mu_threat'),     expt.mu_threat     = +1.25; end
if ~isfield(expt,'mu_nonthreat'),  expt.mu_nonthreat  = -1.25; end
if ~isfield(expt,'sigma'),         expt.sigma         = 1.0;   end
if ~isfield(expt,'p_L'),           expt.p_L           = 0.1;   end
if ~isfield(expt,'p_H'),           expt.p_H           = 0.9;   end
if ~isfield(expt,'hazard'),        expt.hazard        = 0.05;  end

n_trials = expt.n_trials;

    %Contexts:
   contexts = expt.p_threat_true(:);

% Sample true category from Bernoulli (contexts), 
true_category = rand(n_trials,1) < contexts; 

% Means given true category
mu_T = expt.mu_threat;
mu_N = expt.mu_nonthreat;
mu   = mu_T * true_category + mu_N * (1 - true_category);

% Deceptive block mid-swap (after local trial 30)
if isfield(expt,'block_starts') && ~isempty(expt.block_starts) && ...
   isfield(expt,'block_ends')   && ~isempty(expt.block_ends)   && ...
   isfield(expt,'block_types')  && ~isempty(expt.block_types)

    swap_mask = false(n_trials,1);
    for b = 1:numel(expt.block_starts)
        if strcmpi(expt.block_types{b}, 'deceptive')  % case-insensitive match
            idx = expt.block_starts(b):expt.block_ends(b);
            if numel(idx) > 30
                swap_mask(idx(31:end)) = true;  % local > 30
            end
        end
    end

    if any(swap_mask)
        tc_swap       = true_category(swap_mask);
        mu(swap_mask) = mu_T * (1 - tc_swap) + mu_N * tc_swap;  % swap means
    end
end

% Draw stimuli 
stimuli = mu + expt.sigma * randn(n_trials,1);

% Perceived category (noisy internal evidence) 
perceived_category = true_category + 0.1 * randn(n_trials,1);
perceived_category = min(max(perceived_category,0),1); %clamped

% Inject rare independent noise spikes (~5% of trials)
noise_prob   = 0.05;
noise_trials = rand(n_trials,1) < noise_prob;
if any(noise_trials)
    n_noise = nnz(noise_trials);
    %Adds extra stimulus noise and occasionally flips perceived category
    stimuli(noise_trials) = stimuli(noise_trials) + 2 * expt.sigma * randn(n_noise,1);
    flip_mask = rand(n_noise,1) > 0.5;
    if any(flip_mask)
        idx = find(noise_trials);
        perceived_category(idx(flip_mask)) = 1 - perceived_category(idx(flip_mask));
    end
end

% explicit cast expected downstream
true_category = double(true_category);
end
