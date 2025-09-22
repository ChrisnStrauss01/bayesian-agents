# Bayesian Categorisation Agents (For MSc Dissertation)

[![DOI](https://zenodo.org/badge/1061586509.svg)](https://doi.org/10.5281/zenodo.17173916)

This repository reproduces the simulations reported in the MSc dissertation.  
Agents A–D are evaluated on a 2AFC “threat vs. non-threat” task with asymmetric payoffs.  
Note. Relative advantage in the code is stored under the field name `regret`.

## Quick start (MATLAB):
1. Clone this repo and open it in MATLAB.
2. Ensure these files are on the MATLAB path:
   - `config_bayesian_categorisation.m`  
   - `restoreOGSchedule.m`  
   - `generateStimuli.m`  
   - `initAgents.m`  
   - `simulateAgents.m`  
   - `agent_logic.m`  
   - `computeMetrics.m`  
   - `run_repro.m`  
   - `seedList.m`
     
## Run:
To produce all results across full 50-seeds:
```matlab
run_repro('multi') 
```

## Outputs:
 (per-seed .mat files and a summary .csv) saved in the results/ folder.


## If you use this code, please cite:

Christopher Strauss. (2025). Bayesian Agents Simulation Code (MSc Dissertation) [Mac OS 15.6]. Zenodo. https://doi.org/10.5281/zenodo.17173916
