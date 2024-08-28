A. Datasets in the data folder:

1. mexico_migration_explanation_input_with_crime_no_implicit_target_num_migrants_df: The main tabular dataset consisting of socio-demographic and satellite principal components, obtained from the graph construction stage of the GraphSAGE training.
   
2. predictions_target_with_crime_no_implicit_target_num_migrants: Predicted values from the forward pass of GraphSAGE, serving as the target variable for the XAI models.

3. GWR_Weights_adaptive_bisquare_target_num_migrants_crime_variables_no_implicit_PCA*: Neighborhood weights calculated by GWR.

4. mexico_migration_with_crime_no_implicit_variable_grouping: A table detailing the category to which each socio-demographic variable belongs.

5. Explanations and Performance Indicators: Contains the top-K most important features identified by the XAI models, along with the performance indicators for local fidelity experiments.

B. GWR_and_Baseline_LIME_Explainer.py: The code used to execute GW-LIME and LIME experiments. Details are provided in Sections 3 and 4 of the paper.

C. senstivity_analysis_experiment: Code and results related to the sensitivity analysis experiment. Details of the experiment are in Section 4.2 of the paper.

D. lime: Updated baseline LIME package. The original code is sourced from the ofiicial LIME package (https://github.com/marcotcr/lime). The files explanation.py, lime_base.py, and lime_tabular.py are the updated versions customized for this study. These updates were necessary to ensure that LIME uses the original data points for neighborhood definition, rather than the randomly perturbed dataset (refer to Section 4.1 of the paper for further details).