function metrics = computeMetrics(results, true_category, expt, stimuli)
% COMPUTE METRICS!
% Computes: accuracy, fitness payoff, relative advantage (oracle – realised), RMSE (full/transition),
% Brier score, CSI (and time series), KL-based disruption indices, ROC AUC,
% and a simple precision–adaptivity cost (Σ|Δ prior|) with payoff/adaptivity AUC.
% Summaries are returned globally and per block type.  

epsilon      = 1e-6;
delta_EMA    = 0.60; % CSI smoothing parameter
CSI_WINDOW   = 50;   % window used for CSI / moving variance
MIN_STABLE_N = 20;   % for "stable" diagnostics

n_trials    = numel(true_category);
agent_names = fieldnames(results);

%  Norm block arrays & contexts 
starts = expt.block_starts(:);
ends   = expt.block_ends(:);
assert(numel(starts)==numel(ends), 'block_starts and block_ends size mismatch');
block_types = expt.block_types(:);
if isfield(expt,'contexts') && ~isempty(expt.contexts)
    contexts_vec = expt.contexts(:);
else
    error('expt.contexts missing: assign expt.contexts = contexts from generateStimuli().');
end

% Oracle posterior + decision 
oracle_posterior = zeros(n_trials,1);
oracle_decision  = zeros(n_trials,1);
for t = 1:n_trials
    pT  = contexts_vec(t); % true base rate
    L_T = normpdf(stimuli(t), expt.mu_threat,    expt.sigma);
    L_N = normpdf(stimuli(t), expt.mu_nonthreat, expt.sigma);
    denom = max(epsilon, pT*L_T + (1-pT)*L_N);
    oracle_posterior(t) = (pT*L_T) / denom;
    oracle_decision(t)  = oracle_posterior(t) >= 0.5;
end
oracle_fitness = payoff_vector(oracle_decision, true_category);

% Block type per trial 
block_types_per_trial = cell(n_trials,1);
for t = 1:n_trials
    bi = find(t <= ends, 1, 'first');
    if isempty(bi), bi = numel(ends); end
    block_types_per_trial{t} = block_types{bi};
end
unique_types  = {'stable','volatile','deceptive'};
change_points = starts(2:end);  


% Loop agents
for a = 1:numel(agent_names)
    name = agent_names{a};
    post = results.(name).posterior(:);   % p(T)
    dec  = results.(name).decision(:);    % 0/1
    fit  = results.(name).fitness(:);     % payoff (per trial) — used below

    % Accuracy
    metrics.(name).accuracy = mean(dec == true_category);

    % Fitness Payoff
    metrics.(name).fitness_payoff = sum(fit);

    % Relative advantage (regret) (oracle – realised payoff); lower = better
    metrics.(name).regret = sum(oracle_fitness - fit);

    % RMSE (full) & transition RMSE (±15)
    metrics.(name).rmse_full = sqrt(mean((post - oracle_posterior).^2));
    trans_vals = [];
    for cp = change_points(:).'
        w1 = max(1, cp-15); w2 = min(n_trials, cp+15);
        trans_vals(end+1) = sqrt(mean((post(w1:w2) - oracle_posterior(w1:w2)).^2)); %#ok<AGROW>
    end
    metrics.(name).rmse_transition = mean(trans_vals);

    % Brier Score
    metrics.(name).brier_score = mean((post - true_category).^2);

    % Coherence Stability Index (CSI) accuracy EMA reweighted by (norm inverse variance
    acc_bin = double(dec == true_category);
    ema_acc = zeros(n_trials,1); ema_acc(1) = acc_bin(1);
    for t = 2:n_trials
        ema_acc(t) = delta_EMA * acc_bin(t) + (1 - delta_EMA) * ema_acc(t-1);
    end
    if exist('movvar','file') == 2
        v_t = movvar(post, CSI_WINDOW, 0, 'Endpoints','shrink'); % length of n_trials
    else
        v_t = nan(n_trials,1);
        for t = 1:n_trials
            lo = max(1, t-CSI_WINDOW+1); v_t(t) = var(post(lo:t));
        end
    end
    var_floor = 1e-3;
    max_var   = max([var(post), var_floor]);
    v_t       = max(v_t, var_floor) / max_var;

    CSI_t = ema_acc ./ (v_t + epsilon);           % per‑trial series
    metrics.(name).csi        = mean(CSI_t(~isnan(CSI_t) & CSI_t>0));
    metrics.(name).csi_series = CSI_t;          
    metrics.(name).csi_t      = metrics.(name).csi_series;  

   
    %  stability in vol blocks & post-shift disruption
    KL_SPIKE_THR_INIT  = 0.05;
    POST_WIN           = 15;
    STABLE_BUFFER_INIT = 10;
    TRIM_RADIUS_INIT   = 2;

    % per-trial KL (between consecutive posteriors)
    KL_t = nan(n_trials,1);
    for t = 2:n_trials
        p_prev = min(max(post(t-1), epsilon), 1-epsilon);
        p_curr = min(max(post(t),   epsilon), 1-epsilon);
        KL_t(t) = p_prev*log(p_prev/p_curr) + (1-p_prev)*log((1-p_prev)/(1-p_curr));
    end

    % Per change summaries
    W_kl   = 15;  lag_f = 0.90;
    kl_peak_per_change = nan(numel(change_points),1);
    kl_auc_per_change  = nan(numel(change_points),1);
    lag_per_change     = nan(numel(change_points),1);
    u_t = payoff_vector(dec, true_category);

    for i = 1:numel(change_points)
        cp = change_points(i);
        lo = max(2, cp - W_kl); hi = min(n_trials, cp + W_kl);
        seg = KL_t(lo:hi);
        kl_peak_per_change(i) = max(seg, [], 'omitnan');
        kl_auc_per_change(i)  = nansum(seg);

        if i < numel(change_points), block_end = change_points(i+1) - 1;
        else,                        block_end = n_trials;
        end
        if block_end > cp
            Lb = block_end - cp + 1;
            tail_len  = max(5, floor(0.3 * Lb));
            tail_lo   = max(cp, block_end - tail_len + 1);
            steady_mu = mean(u_t(tail_lo:block_end), 'omitnan');
            rmean = cumsum(u_t(cp:block_end)) ./ (1:Lb)';
            target = lag_f * steady_mu;
            if steady_mu >= 0, k = find(rmean >= target, 1, 'first');
            else,              k = find(rmean <= target, 1, 'first');
            end
            if ~isempty(k), lag_per_change(i) = k; end
        end
    end
    metrics.(name).kl_peak_per_change = kl_peak_per_change;
    metrics.(name).kl_auc_per_change  = kl_auc_per_change;
    metrics.(name).lag_per_change     = lag_per_change;

    %  CSI and KL convergence checks:

    %  Post-shift KL spike counts (forward window)
    spike_counts = nan(numel(change_points),1);
    for i = 1:numel(change_points)
        cp = change_points(i);
        hi = min(n_trials, cp + POST_WIN);
        if hi >= cp+1
            spike_counts(i) = sum(KL_t(cp+1:hi) > KL_SPIKE_THR_INIT, 'omitnan');
        end
    end
    metrics.(name).kl_spikes_postchange_window = POST_WIN;
    metrics.(name).kl_spikes_postchange        = spike_counts;
    metrics.(name).kl_spikes_postchange_mean   = mean(spike_counts,'omitnan');

    %  KL AUC around each change
    W_AUC = 15;
    kl_auc_pm = nan(numel(change_points),1);
    for i = 1:numel(change_points)
        cp = change_points(i);
        lo = max(2, cp - W_AUC);
        hi = min(n_trials, cp + W_AUC);
        kl_auc_pm(i) = nansum(KL_t(lo:hi));
    end
    metrics.(name).kl_auc_pm      = kl_auc_pm;
    metrics.(name).kl_auc_pm_mean = mean(kl_auc_pm,'omitnan');

    %  Coherence stability in "stable" blocks 
    have_stable = any(strcmp(block_types,'stable'));

    % Build adaptive stable mask (relax buffer to keep enough points)
    stable_mask = false(n_trials,1);
    if have_stable
        buf = STABLE_BUFFER_INIT;
        while true
            tmp = false(n_trials,1);
            for b = 1:numel(block_types)
                if strcmp(block_types{b}, 'stable')
                    s = starts(b);
                    e = ends(b);
                    s2 = max(1, s + buf);
                    e2 = min(n_trials, e - buf);
                    if s2 <= e2, tmp(s2:e2) = true; end
                end
            end
            if nnz(tmp) >= MIN_STABLE_N || buf == 0
                stable_mask = tmp; break;
            end
            buf = max(buf-2,0);
        end
        if nnz(stable_mask) == 0
            for b = 1:numel(block_types)
                if strcmp(block_types{b}, 'stable')
                    s = starts(b); e = ends(b);
                    if s <= e, stable_mask(s:e) = true; end
                end
            end
        end
    end

    % Untrimmed stability (raw var and CV)
    if nnz(stable_mask) >= 1
        csi_vals = CSI_t(stable_mask);
        csi_vals = csi_vals(~isnan(csi_vals) & csi_vals>0);
        if ~isempty(csi_vals)
            mu_csi = mean(csi_vals); sd_csi = std(csi_vals,1);
            metrics.(name).csi_stable_mean = mu_csi;
            metrics.(name).csi_stable_var  = var(csi_vals,1);
            metrics.(name).csi_stable_cv   = sd_csi / max(mu_csi, eps);
        else
            metrics.(name).csi_stable_mean = NaN;
            metrics.(name).csi_stable_var  = NaN;
            metrics.(name).csi_stable_cv   = NaN;
        end
    else
        metrics.(name).csi_stable_mean = NaN;
        metrics.(name).csi_stable_var  = NaN;
        metrics.(name).csi_stable_cv   = NaN;
    end

    % Spike-trimmed CV (CSI):
    trim_mask = stable_mask; % default (also used below for postvar CV)
    if nnz(stable_mask) >= 1
        for trim_r = TRIM_RADIUS_INIT:-1:0
            tmp = stable_mask;
            spk = find(KL_t > KL_SPIKE_THR_INIT);
            for s = spk'
                lo = max(1, s-trim_r); hi = min(n_trials, s+trim_r);
                tmp(lo:hi) = false;
            end
            if nnz(tmp) >= MIN_STABLE_N || trim_r == 0
                trim_mask = tmp; break;
            end
        end
        csi_trim = CSI_t(trim_mask);
        csi_trim = csi_trim(~isnan(csi_trim) & csi_trim>0);
        if ~isempty(csi_trim)
            mu_ct = mean(csi_trim); sd_ct = std(csi_trim,1);
            metrics.(name).csi_stable_cv_trim = sd_ct / max(mu_ct, eps);
        else
            metrics.(name).csi_stable_cv_trim = NaN;
        end
    else
        metrics.(name).csi_stable_cv_trim = NaN;
    end

    % Posterior-variance stability (CV of moving variance) 
    if exist('movvar','file') == 2
        post_var_local = movvar(post, CSI_WINDOW, 0, 'Endpoints','shrink');
    else
        post_var_local = nan(n_trials,1);
        for t = 1:n_trials
            lo = max(1, t-CSI_WINDOW+1); post_var_local(t) = var(post(lo:t));
        end
    end
    if nnz(trim_mask) >= 1
        pv = post_var_local(trim_mask);
        pv = pv(~isnan(pv) & pv > 0);
        if ~isempty(pv)
            mu_pv = mean(pv); sd_pv = std(pv,1);
            metrics.(name).postvar_stable_cv = sd_pv / max(mu_pv, eps);
        else
            metrics.(name).postvar_stable_cv = NaN;
        end
    else
        metrics.(name).postvar_stable_cv = NaN;
    end


    % Coherence Disruption (global spike count)
    KL = zeros(n_trials-1,1);
    for t = 2:n_trials
        p_prev = min(max(post(t-1), epsilon), 1-epsilon);
        p_curr = min(max(post(t),   epsilon), 1-epsilon);
        KL(t-1) = p_prev*log(p_prev/p_curr) + (1-p_prev)*log((1-p_prev)/(1-p_curr));
    end
    metrics.(name).deltaC = KL;
    metrics.(name).coherence_spike_count = sum(KL > KL_SPIKE_THR_INIT);

    %  ROC / AUC
    [~,~,~,auc] = perfcurve(true_category, post, 1);
    metrics.(name).roc_auc = auc;

    % Precision adaptivity cost (Σ|Δπ|)
    if isfield(results.(name),'p_threat') && ~isempty(results.(name).p_threat)
        prior_path = results.(name).p_threat(:);
        adapt_cost = sum(abs(diff(prior_path)));        % Σ|Δπ| proxy from prior path (zero if prior never moves)
        cum_payoff = cumsum(fit);
        cum_adapt  = cumsum([0; abs(diff(prior_path))]);
        curve_y    = cum_payoff ./ (cum_adapt + epsilon);
        cost_benefit_auc = trapz(1:n_trials, curve_y);
    else
        adapt_cost = NaN; cost_benefit_auc = NaN;
    end
    metrics.(name).precision_adaptivity_cost = adapt_cost;
    metrics.(name).cost_benefit_auc          = cost_benefit_auc;

   
    % Per block-type breakdown (existing)
    for bt = unique_types
        bt_str = bt{1};
        idx = strcmp(block_types_per_trial, bt_str);
        if any(idx)
            metrics.(name).accuracy_per_type.(bt_str)        = mean(dec(idx) == true_category(idx));
            metrics.(name).fitness_payoff_per_type.(bt_str)  = sum(fit(idx));
            metrics.(name).regret_per_type.(bt_str)          = sum(oracle_fitness(idx) - fit(idx));
            metrics.(name).rmse_per_type.(bt_str)            = sqrt(mean((post(idx) - oracle_posterior(idx)).^2));
            metrics.(name).brier_score_per_type.(bt_str)     = mean((post(idx) - true_category(idx)).^2);
            csi_sub = CSI_t(idx);
            metrics.(name).csi_per_type.(bt_str)             = mean(csi_sub(csi_sub>0));
            idx_kl = idx(2:end) & idx(1:end-1);
            metrics.(name).coherence_spike_count_per_type.(bt_str) = sum(KL(idx_kl) > KL_SPIKE_THR_INIT);
            try
                [~,~,~,auc_bt] = perfcurve(true_category(idx), post(idx), 1);
            catch
                auc_bt = NaN;
            end
            metrics.(name).roc_auc_per_type.(bt_str) = auc_bt;

            if exist('prior_path','var') && ~isnan(adapt_cost)
                idx_inds   = find(idx);
                prior_seg  = prior_path(idx_inds);
                adapt_cost_bt = sum(abs(diff(prior_seg)));
                cum_fit_bt    = cumsum(fit(idx));
                cum_adapt_bt  = cumsum([0; abs(diff(prior_seg))]);
                curve_bt      = cum_fit_bt ./ (cum_adapt_bt + epsilon);
                auc_bt_cost   = trapz(1:numel(prior_seg), curve_bt);
                metrics.(name).precision_adaptivity_cost_per_type.(bt_str) = adapt_cost_bt;
                metrics.(name).cost_benefit_auc_per_type.(bt_str)          = auc_bt_cost;
            else
                metrics.(name).precision_adaptivity_cost_per_type.(bt_str) = NaN;
                metrics.(name).cost_benefit_auc_per_type.(bt_str)          = NaN;
            end
        else
            metrics.(name).accuracy_per_type.(bt_str) = NaN;
            metrics.(name).fitness_payoff_per_type.(bt_str) = NaN;
            metrics.(name).regret_per_type.(bt_str) = NaN;
            metrics.(name).rmse_per_type.(bt_str) = NaN;
            metrics.(name).brier_score_per_type.(bt_str) = NaN;
            metrics.(name).csi_per_type.(bt_str) = NaN;
            metrics.(name).coherence_spike_count_per_type.(bt_str) = NaN;
            metrics.(name).roc_auc_per_type.(bt_str) = NaN;
            metrics.(name).precision_adaptivity_cost_per_type.(bt_str) = NaN;
            metrics.(name).cost_benefit_auc_per_type.(bt_str) = NaN;
        end
    end
    % Agent-D summaries: mean alpha and entropy penalty 
    if strcmp(name,'D')
        % mean effective learning-rate actually used (per-trial)
        if isfield(results.(name),'alpha_eff_vec') && ~isempty(results.(name).alpha_eff_vec)
            metrics.(name).mean_alpha_eff = mean(results.(name).alpha_eff_vec, 'omitnan');
        end

        % mean posterior entropy (0..1) and its contribution to δ·H penalty
        if isfield(results.(name),'Hq_vec') && ~isempty(results.(name).Hq_vec)
            mH = mean(results.(name).Hq_vec, 'omitnan');    % 0..1
            metrics.(name).mean_entropy_Hq = mH;

            % optional scaling knob (defaults to 1.0 if not present)
            if isfield(expt,'entropy_scale') && ~isempty(expt.entropy_scale)
                ent_scale = expt.entropy_scale;
            else
                ent_scale = 1.0;
            end

            metrics.(name).mean_entropy_penalty = expt.delta_entropy * ent_scale * mH;
        end
    end
end
end

%  Helper: per-trial fitness vector 
function u = payoff_vector(decisions,true_cat)
n = numel(true_cat);
u = zeros(n,1);
for i = 1:n
    if true_cat(i) == 1
        u(i) =  2.0*(decisions(i)==1) + (-2.0)*(decisions(i)==0);
    else
        u(i) =  0.25*(decisions(i)==0) + (-0.25)*(decisions(i)==1);
    end
end
end