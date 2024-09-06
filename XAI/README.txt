
A. GWR_and_Baseline_LIME_Explainer.py: The code used to execute GW-LIME and LIME experiments. Details are provided in Sections 3 and 4 of the paper.

B. senstivity_analysis_experiment: Code and results related to the sensitivity analysis experiment. Details of the experiment are in Section 4.2 of the paper.

C. lime: Updated baseline LIME package. The original code is sourced from the ofiicial LIME package (https://github.com/marcotcr/lime). The files explanation.py, lime_base.py, and lime_tabular.py are the updated versions customized for this study. These updates were necessary to ensure that LIME uses the original data points for neighborhood definition, rather than the randomly perturbed dataset (refer to Section 4.1 of the paper for further details).
