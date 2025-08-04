# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, r2_score, mean_absolute_error, mean_squared_error,
    auc
)

# Configuration esthétique pour les graphiques
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
from scipy import stats
import math
from equipy.fairness import FairWasserstein
from equipy.metrics import unfairness
from scipy.stats import kstest, ks_2samp, norm
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from models_and_metrics import *

class Data : 

    def generate_synthetic_data(nb_features, nb_obs, tho, additional_s, t_unfair, T_mean, T_std, t_correl, random_seed):
        """Generate synthetic dataset parametrized py (t_unfair, T_mean, T_std, t_correl).
        This function enables to reproduce our results.

        Args:
            nb_features (_type_): _description_
            nb_obs (_type_): _description_
            tho (_type_): _description_
            additional_s (_type_): _description_
            t_unfair (_type_): _description_
            T_mean (_type_): _description_
            T_std (_type_): _description_
            t_correl (_type_): _description_
            random_seed (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Simulation of S
        param = {}
        param['q'] = 1 - norm.pdf(tho, loc=0, scale=1)
        param['S'] = np.random.binomial(1, param['q'], (nb_obs)) + 1 + additional_s * np.random.binomial(1, param['q'], (nb_obs))
        data = pd.DataFrame({"S": param['S']})

        # Generate means for each group
        n_binomial, p_binomial = 3, np.random.uniform(low=0.0, high=1.0, size=nb_features)
        mu_1, mu_2 = np.zeros(nb_features), np.zeros(nb_features)
        
        if T_mean == 0: 
            mu_1 += np.random.binomial(n_binomial, p_binomial, nb_features)
            mu_2 = mu_1
        else:
            p_binomial_1 = np.random.uniform(low=0.0, high=1.0, size=nb_features)
            mu_1 += np.random.binomial(n_binomial, p_binomial_1, nb_features)
            mu_2 = mu_1 + T_mean * np.ones(nb_features)
        
        # Generate covariance matrices for each group
        if t_correl == 0:
            # Independent features case (diagonal covariance)
            if T_std == 0:
                std_1 = np.random.uniform(low=0.0, high=2.0) * np.ones(nb_features)
                std_2 = std_1
            else:
                std_1 = np.random.uniform(low=0.0, high=2.0, size=nb_features)
                std_2 = std_1 + np.sqrt(T_std)* np.ones(nb_features)
                
            cov_1 = np.diag(std_1**2)
            cov_2 = np.diag(std_2**2)
        else:
            # Code for correlated cases remains unchanged
            if T_std == 0:
                std_1 = np.random.uniform(low=0.0, high=2.0) * np.ones(nb_features)
                std_2 = std_1
            else:
                std_1 = np.random.uniform(low=0.0, high=2.0, size=nb_features)
                std_2 = std_1 + np.sqrt(T_std)* np.ones(nb_features)
                
            diag_1 = np.diag(std_1**2)
            diag_2 = np.diag(std_2**2)
            
            A_1 = np.random.normal(0, 1, (nb_features, nb_features))
            corr_1 = A_1.T @ A_1
            corr_1 = corr_1 / np.outer(np.sqrt(np.diag(corr_1)), np.sqrt(np.diag(corr_1)))
            
            if t_correl == 1:
                corr_2 = corr_1
            else:
                A_2 = np.random.normal(0, 1, (nb_features, nb_features))
                corr_2 = A_2.T @ A_2
                corr_2 = corr_2 / np.outer(np.sqrt(np.diag(corr_2)), np.sqrt(np.diag(corr_2)))
            
            # Apply correlation strength
            corr_1 = (1 - t_correl) * np.eye(nb_features) + t_correl * corr_1
            corr_2 = (1 - t_correl) * np.eye(nb_features) + t_correl * corr_2
            
            cov_1 = np.sqrt(diag_1) @ corr_1 @ np.sqrt(diag_1)
            cov_2 = np.sqrt(diag_2) @ corr_2 @ np.sqrt(diag_2)
        
        # Generate X conditionally on S using the covariance matrices
        X_all = np.zeros((nb_obs, nb_features))
        indices_1 = data.index[data['S'] == 1].tolist()
        indices_2 = data.index[data['S'] == 2].tolist()
        
        param['X_1'] = np.random.multivariate_normal(mean=mu_1, cov=cov_1, size=len(indices_1))
        X_all[indices_1] = param['X_1']
        param['X_2'] = np.random.multivariate_normal(mean=mu_2, cov=cov_2, size=len(indices_2))
        X_all[indices_2] = param['X_2']

        # Add X columns to the dataframe
        for j in range(nb_features):
            data[f'X_{j}'] = X_all[:, j]

        # Calculate Y as sum of X features plus t*S
        Y = np.sum(X_all, axis=1) + t_unfair * data['S'].values
        data['Y'] = Y

        # Reset random seed to avoid affecting other code
        if random_seed is not None:
            np.random.seed(None)
        
        return data, param


    # def generate_synthetic_data(nb_features, nb_obs, tho, additional_s, t_unfair, T_mean, T_std, t_correl, random_seed):
    #     """Generate a syntetic dataset parametrized by (t_unfair, T_mean, T_std, t_correl)
    #        This function is a simpler implementation of the previous generate_synthetic_data function. 
    #        However, it does not exactly reproduced our results due to a slightly different simulation process.

    #     Args:
    #         nb_features (_type_): _description_
    #         nb_obs (_type_): _description_
    #         tho (_type_): _description_
    #         additional_s (_type_): _description_
    #         t_unfair (_type_): _description_
    #         T_mean (_type_): _description_
    #         T_std (_type_): _description_
    #         t_correl (_type_): _description_
    #         random_seed (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     # Set random seed if provided
    #     if random_seed is not None:
    #         np.random.seed(random_seed)

    #     # Simulation of S
    #     param = {}
    #     param['q'] = 1 - norm.pdf(tho, loc=0, scale=1)
    #     param['S'] = np.random.binomial(1, param['q'], (nb_obs)) + 1 + additional_s * np.random.binomial(1, param['q'], (nb_obs))
    #     data = pd.DataFrame({"S": param['S']})

    #     # Initialization of mean and std
    #     n_binomial, p_binomial = 3, np.random.uniform(low=0.0, high=1.0, size=nb_features)
    #     mu_1 = np.random.binomial(n_binomial, p_binomial, nb_features)
    #     std_1 = np.random.uniform(low=0.0, high=2.0) * np.ones(nb_features)

    #     if T_mean == 0: 
    #         mu_2 = mu_1
    #     else:
    #         # Group-dependant mean
    #         mu_2 = mu_1 + T_mean * np.ones(nb_features)
        
    #     if T_std == 0:  
    #             std_2 = std_1
    #     else:
    #         # Group-dependant std
    #         std_2 = std_1 + np.sqrt(T_std)* np.ones(nb_features)
        
    #     # Generate covariance matrices for each group
    #     if t_correl == 0:
    #         # Independant feature case
    #         cov_1 = np.diag(std_1**2)
    #         cov_2 = np.diag(std_2**2)
    #     else:
    #         diag_1 = np.diag(std_1**2)
    #         diag_2 = np.diag(std_2**2)
            
    #         A_1 = np.random.normal(0, 1, (nb_features, nb_features))
    #         corr_1 = A_1.T @ A_1
    #         corr_1 = corr_1 / np.outer(np.sqrt(np.diag(corr_1)), np.sqrt(np.diag(corr_1)))
            
    #         if t_correl == 1:
    #             corr_2 = corr_1
    #         else:
    #             # Group-dependant correlations/covariances
    #             A_2 = np.random.normal(0, 1, (nb_features, nb_features))
    #             corr_2 = A_2.T @ A_2
    #             corr_2 = corr_2 / np.outer(np.sqrt(np.diag(corr_2)), np.sqrt(np.diag(corr_2)))
            
    #         # t_correl drives the correlation strengh
    #         corr_1 = (1 - t_correl) * np.eye(nb_features) + t_correl * corr_1
    #         corr_2 = (1 - t_correl) * np.eye(nb_features) + t_correl * corr_2
            
    #         cov_1 = np.sqrt(diag_1) @ corr_1 @ np.sqrt(diag_1)
    #         cov_2 = np.sqrt(diag_2) @ corr_2 @ np.sqrt(diag_2)
        
    #     # Generate X conditionally on S using the covariance matrices
    #     X_all = np.zeros((nb_obs, nb_features))
    #     indices_1 = data.index[data['S'] == 1].tolist()
    #     indices_2 = data.index[data['S'] == 2].tolist()
        
    #     param['X_1'] = np.random.multivariate_normal(mean=mu_1, cov=cov_1, size=len(indices_1))
    #     X_all[indices_1] = param['X_1']
    #     param['X_2'] = np.random.multivariate_normal(mean=mu_2, cov=cov_2, size=len(indices_2))
    #     X_all[indices_2] = param['X_2']

    #     # Add X columns to the dataframe
    #     for j in range(nb_features):
    #         data[f'X_{j}'] = X_all[:, j]

    #     # Calculate Y as sum of X features plus t*S
    #     Y = np.sum(X_all, axis=1) + t_unfair * data['S'].values
    #     data['Y'] = Y

    #     # Reset random seed to avoid affecting other code
    #     if random_seed is not None:
    #         np.random.seed(None)
        
    #     return data, param


    def compute_unf_linear_model(param_dictionnary, test_dataset, S_variable): 
            
            # Computation of E[f^*|S] for first moment disparity
            cond_mean_score = 0
            for i in test_dataset[S_variable].unique():
                cond_mean_score += param_dictionnary[f'p_{int(i)}'] * test_dataset[test_dataset[S_variable]==i]['y_input_reg'].mean()

            # Computation of E[sqrt(Var(f|S))] for second moment disparity
            cond_std_score = 0
            for i in test_dataset[S_variable].unique():
                cond_std_score += param_dictionnary[f'p_{int(i)}'] * np.sqrt(test_dataset[test_dataset[S_variable]==i]['y_input_reg'].var())
                    
            # Computation of first_moment_disparity = Var[E[f^*|S]] and second_moment_disparity = Var[sqrt(Var[f^*|S])]
            first_moment_disparity = 0
            second_moment_disparity = 0
            for i in test_dataset[S_variable].unique():
                first_moment_disparity += param_dictionnary[f'p_{int(i)}'] * (test_dataset[test_dataset[S_variable]==i]['y_input_reg'].mean()-cond_mean_score)**2
                second_moment_disparity += param_dictionnary[f'p_{int(i)}'] * (np.sqrt(test_dataset[test_dataset[S_variable]==i]['y_input_reg'].var())-cond_std_score)**2
            
            total_unfairness = first_moment_disparity + second_moment_disparity
            return(total_unfairness, first_moment_disparity, second_moment_disparity)


    def run_experiment(varying_param, param_values, fixed_params, coefficients_analysis, bool_coef, bool_approximate_fairness):
        """
        Runs an experiment by varying a specific parameter.
        
        Args:
            varying_param: Name of the parameter to vary ('t_values', 'T_mean', or 'T_std')
            param_values: List of values for the varying parameter
            fixed_params: Dictionary of fixed parameters
            coefficients_analysis: Boolean to determine if coefficient analysis should be performed
            bool_coef: Boolean to include coefficient metrics if True
            bool_approximate_fairness: Boolean to include approximate fairness models if True
        
        Returns:
            all_results: Dictionary containing results for each parameter value
            results_df_exp: DataFrame with experiment results
            test_dataset: Test dataset from the last simulation
            pool_dataset: Pool dataset from the last simulation
            param_dictionary: Parameters dictionary from the last simulation
        """
        models = ['y_pred_fair', 'y_input_reg', 'y_score_equipy', 'y_pred_riken', 'y_pred_bias']
        metrics = ['r2', 'GWR2','mae', 'rmse', 'unfairness_W2', 'unfairness_W1', 'unfairness_computed','ks_stat', 
                'indirect_mean_bias', 'indirect_structural_bias', 'interaction', 'direct_mean_bias',
                'total_unfairness', 'first_moment_disparity', 'second_moment_disparity']
        coefs = ['beta_0_NoStd','beta_NoStd','gamma_NoStd',
                    'beta_0_1_Std','beta_0_2_Std','beta_1_Std','beta_2_Std','gamma_Std',
                    'fair_intercept_1_NoStd', 'fair_intercept_2_NoStd', 'beta_1_NoStd','beta_2_NoStd', 
                    'fair_intercept_Std','beta_Std',
                'riken_intercept_Std']
        # Dictionary to store all results
        all_results = {}
        if bool_coef is not None : 
            metrics = metrics + coefs
        if bool_approximate_fairness:
            for epsilon in np.arange(0.1, 1, 0.1):
                e = round((epsilon), 2)
                models.append(f'y_pred_fair_{e}')

        # Loop through each value of the varying parameter
        for param_value in tqdm(param_values, desc=f"Processing {varying_param}"):
            # Set parameters for this iteration
            current_params = fixed_params.copy()
            current_params[varying_param] = param_value
            
            # Generate synthetic data
            data_t, param_t = Data.generate_synthetic_data(
                current_params['nb_features'], 
                current_params['nb_obs'], 
                current_params['tho'], 
                current_params['additional_s'], 
                current_params['t_unfair'],
                current_params['T_mean'], 
                current_params['T_std'],  
                current_params['t_correl'],  
                current_params['random_seed'],  
            )
            
            X_features = data_t.drop(columns=[current_params['S_variable'], current_params['y'], 'X_4']).columns.to_list()
            
            # Initialize results for this parameter value
            t_results = {model: {metric: [] for metric in metrics} for model in models}
            
            # Perform n_simulations for this parameter value
            for bootstrap in tqdm(range(current_params['n_simulations']), desc=f"Simulations for {varying_param}={param_value}", leave=False):
                # Data preparation
                train_dataset, test_dataset = train_test_split(data_t, test_size=0.2, random_state=bootstrap)
                train_dataset, pool_dataset = train_test_split(train_dataset, test_size=0.2, random_state=bootstrap)
                
                # Extract sensitive feature for metrics calculation
                y_sensitive_feature = pd.DataFrame({f"{current_params['S_variable']}": test_dataset[f"{current_params['S_variable']}"].to_list()})
                unique_groups = test_dataset[current_params['S_variable']].unique()
                
                # Fair Linear Model
                coef_input_model, param_dictionnary, input_model, test_dataset = Fair_model.predict_fair_linear_score(
                    train_dataset, pool_dataset, test_dataset, current_params['S_variable'], current_params['y'], X_features, True, False, False)
                
                # Store coefficients and metrics for input regression model
                t_results['y_input_reg']['beta_0_NoStd'].append(param_dictionnary['beta_0'])
                t_results['y_input_reg']['beta_NoStd'].append(param_dictionnary['beta'])
                t_results['y_input_reg']['gamma_NoStd'].append(param_dictionnary['gamma'])
                t_results['y_input_reg']['beta_0_1_Std'].append(param_dictionnary['beta_0']+np.dot(param_dictionnary['empirical_mean_1'],param_dictionnary['beta']))
                t_results['y_input_reg']['beta_0_2_Std'].append(param_dictionnary['beta_0']+np.dot(param_dictionnary['empirical_mean_2'],param_dictionnary['beta']))
                t_results['y_input_reg']['beta_1_Std'].append(param_dictionnary['beta']*param_dictionnary['var_cov_product_1'])
                t_results['y_input_reg']['beta_2_Std'].append(param_dictionnary['beta']*param_dictionnary['var_cov_product_2'])
                t_results['y_input_reg']['gamma_Std'].append(param_dictionnary['gamma'])
                
                # Store bias metrics for input regression model
                t_results['y_input_reg']['interaction'].append(param_dictionnary['interaction'])
                t_results['y_input_reg']['direct_mean_bias'].append(param_dictionnary['direct_mean_bias'])
                t_results['y_input_reg']['indirect_structural_bias'].append(param_dictionnary['indirect_structural_bias'])
                t_results['y_input_reg']['indirect_mean_bias'].append(param_dictionnary['indirect_mean_bias'])
                t_results['y_input_reg']['unfairness_computed'].append(param_dictionnary['unfairness_input_model'])
                
                # Compute and store unfairness metrics
                total_unfairness, first_moment_disparity, second_moment_disparity = Data.compute_unf_linear_model(param_dictionnary, test_dataset, current_params['S_variable'])
                t_results['y_input_reg']['total_unfairness'].append(total_unfairness)
                t_results['y_input_reg']['first_moment_disparity'].append(first_moment_disparity)
                t_results['y_input_reg']['second_moment_disparity'].append(second_moment_disparity)

                # Store bias metrics for fair model
                # t_results['y_pred_fair']['indirect_structural_bias'].append(param_dictionnary['indirect_structural_bias_fair_model'])
                # t_results['y_pred_fair']['indirect_mean_bias'].append(param_dictionnary['indirect_mean_bias_fair_model'])
                # t_results['y_pred_fair']['unfairness_computed'].append(param_dictionnary['unfairness_our_model'])

                # Store coefficients for fair model
                t_results['y_pred_fair']['fair_intercept_1_NoStd'].append(param_dictionnary['fair_intercept']-param_dictionnary['invariant_var_cov_term']*np.dot(param_dictionnary['empirical_mean_1'],param_dictionnary['beta'])/param_dictionnary['var_cov_product_1'])
                t_results['y_pred_fair']['fair_intercept_2_NoStd'].append(param_dictionnary['fair_intercept']-param_dictionnary['invariant_var_cov_term']*np.dot(param_dictionnary['empirical_mean_2'],param_dictionnary['beta'])/param_dictionnary['var_cov_product_2'])
                t_results['y_pred_fair']['beta_1_NoStd'].append(param_dictionnary['invariant_var_cov_term']*param_dictionnary['beta']/param_dictionnary['var_cov_product_1'])
                t_results['y_pred_fair']['beta_2_NoStd'].append(param_dictionnary['invariant_var_cov_term']*param_dictionnary['beta']/param_dictionnary['var_cov_product_2'])
                t_results['y_pred_fair']['gamma_NoStd'].append(0)

                t_results['y_pred_fair']['fair_intercept_Std'].append(param_dictionnary['fair_intercept'])
                t_results['y_pred_fair']['beta_Std'].append(param_dictionnary['invariant_var_cov_term'])
                t_results['y_pred_fair']['gamma_Std'].append(0)

                # Create approximate fairness models if requested
                for epsilon in np.arange(0.1, 1, 0.1):
                    e = round((epsilon), 2)
                    test_dataset[f'y_pred_fair_{e}'] = (1-(e))*test_dataset['y_pred_fair'] + (e)*test_dataset['y_input_reg']
                    
                # EquiPy benchmark model
                Benchmark_model.benchmark_equipy(train_dataset, test_dataset, 'y_input_reg', current_params['S_variable'])
                
                # Riken benchmark model
                dictionnary_riken_raw = Benchmark_model.riken_prediction(train_dataset, test_dataset, current_params['S_variable'], X_features, current_params['y'])
                # t_results['y_pred_riken']['indirect_mean_bias'].append(dictionnary_riken_raw['indirect_mean_bias'])
                t_results['y_pred_riken']['indirect_structural_bias'].append(dictionnary_riken_raw['indirect_structural_bias'])
                t_results['y_pred_riken']['unfairness_computed'].append(dictionnary_riken_raw['total_unfairness'])

                # Bias model (Evgeni)
                beta_hat, intercept_hat, ps, total_unfairness, indirect_mean_bias, indirect_structural_bias = Benchmark_model.weighted_group_intercepts(train_dataset, test_dataset, X_features, current_params['y'], current_params['S_variable'], True)
                t_results['y_pred_bias']['indirect_mean_bias'].append(indirect_mean_bias)
                t_results['y_pred_bias']['indirect_structural_bias'].append(indirect_structural_bias)
                t_results['y_pred_bias']['unfairness_computed'].append(total_unfairness)
                
                # Calculate and store metrics for each model
                for prediction in models:
                    t_results[prediction]['r2'].append(r2_score(test_dataset[current_params['y']], test_dataset[prediction]))
                    t_results[prediction]['GWR2'].append(Metrics.group_weighted_r2(test_dataset, current_params['y'], prediction, current_params['S_variable']))
                    t_results[prediction]['mae'].append(mean_absolute_error(test_dataset[current_params['y']], test_dataset[prediction]))
                    t_results[prediction]['rmse'].append(np.sqrt(mean_squared_error(test_dataset[current_params['y']], test_dataset[prediction])))
                    t_results[prediction]['unfairness_W1'].append(unfairness(np.array(test_dataset[prediction].tolist()), y_sensitive_feature))
                    t_results[prediction]['unfairness_W2'].append(Metrics.unfairness_computation(prediction, current_params['S_variable'], test_dataset))                
                    
                    # KS test for distribution comparison between groups
                    if len(unique_groups) >= 2:
                        ks_stat = kstest(
                            rvs=test_dataset[test_dataset[current_params['S_variable']] == unique_groups[0]][prediction],
                            cdf=test_dataset[test_dataset[current_params['S_variable']] == unique_groups[1]][prediction],
                            alternative='two-sided'
                        ).statistic
                        t_results[prediction]['ks_stat'].append(ks_stat)
                        
            if coefficients_analysis:
                return t_results, test_dataset
            else: 
                # Calculate means for this parameter value
                summary_t = {}
                for model in models:
                    summary_t[model] = {}
                    for metric in metrics:
                        if t_results[model][metric]:  # Check if list is not empty
                            summary_t[model][f'{metric}_mean'] = round(np.mean(t_results[model][metric]), 5)
                            summary_t[model][f'{metric}_std'] = round(np.std(t_results[model][metric]), 5)
                        else:
                            summary_t[model][f'{metric}_mean'] = None
                            summary_t[model][f'{metric}_std'] = None
                
            # Store results for this parameter value
            all_results[param_value] = summary_t
            
        # Create a DataFrame from the results
        results_df_exp = []
        for t in all_results:
            for model in all_results[t]:
                row = {'t': t, 'model': model}
                row.update({k: v for k, v in all_results[t][model].items()})
                results_df_exp.append(row)

        results_df_exp = pd.DataFrame(results_df_exp)
            
        return all_results, results_df_exp, test_dataset, pool_dataset, param_dictionnary
    

    
        # def generate_synthetic_data(nb_features, nb_obs, tho, additional_s, t_unfair, T_mean, T_std, t_correl, random_seed):
        #     """Generate a syntetic dataset parametrized by (t_unfair, T_mean, T_std, t_correl)
        #        This function is a simpler implementation of the previous generate_synthetic_data function. 
        #        However, it does not exactly reproduced our results due to a slightly different simulation process.

        #     Args:
        #         nb_features (_type_): _description_
        #         nb_obs (_type_): _description_
        #         tho (_type_): _description_
        #         additional_s (_type_): _description_
        #         t_unfair (_type_): _description_
        #         T_mean (_type_): _description_
        #         T_std (_type_): _description_
        #         t_correl (_type_): _description_
        #         random_seed (_type_): _description_

        #     Returns:
        #         _type_: _description_
        #     """
        #     # Set random seed if provided
        #     if random_seed is not None:
        #         np.random.seed(random_seed)

        #     # Simulation of S
        #     param = {}
        #     param['q'] = 1 - norm.pdf(tho, loc=0, scale=1)
        #     param['S'] = np.random.binomial(1, param['q'], (nb_obs)) + 1 + additional_s * np.random.binomial(1, param['q'], (nb_obs))
        #     data = pd.DataFrame({"S": param['S']})

        #     # Initialization of mean and std
        #     n_binomial, p_binomial = 3, np.random.uniform(low=0.0, high=1.0, size=nb_features)
        #     mu_1 = np.random.binomial(n_binomial, p_binomial, nb_features)
        #     std_1 = np.random.uniform(low=0.0, high=2.0) * np.ones(nb_features)

        #     if T_mean == 0: 
        #         mu_2 = mu_1
        #     else:
        #         # Group-dependant mean
        #         mu_2 = mu_1 + T_mean * np.ones(nb_features)
            
        #     if T_std == 0:  
        #             std_2 = std_1
        #     else:
        #         # Group-dependant std
        #         std_2 = std_1 + np.sqrt(T_std)* np.ones(nb_features)
            
        #     # Generate covariance matrices for each group
        #     if t_correl == 0:
        #         # Independant feature case
        #         cov_1 = np.diag(std_1**2)
        #         cov_2 = np.diag(std_2**2)
        #     else:
        #         diag_1 = np.diag(std_1**2)
        #         diag_2 = np.diag(std_2**2)
                
        #         A_1 = np.random.normal(0, 1, (nb_features, nb_features))
        #         corr_1 = A_1.T @ A_1
        #         corr_1 = corr_1 / np.outer(np.sqrt(np.diag(corr_1)), np.sqrt(np.diag(corr_1)))
                
        #         if t_correl == 1:
        #             corr_2 = corr_1
        #         else:
        #             # Group-dependant correlations/covariances
        #             A_2 = np.random.normal(0, 1, (nb_features, nb_features))
        #             corr_2 = A_2.T @ A_2
        #             corr_2 = corr_2 / np.outer(np.sqrt(np.diag(corr_2)), np.sqrt(np.diag(corr_2)))
                
        #         # t_correl drives the correlation strengh
        #         corr_1 = (1 - t_correl) * np.eye(nb_features) + t_correl * corr_1
        #         corr_2 = (1 - t_correl) * np.eye(nb_features) + t_correl * corr_2
                
        #         cov_1 = np.sqrt(diag_1) @ corr_1 @ np.sqrt(diag_1)
        #         cov_2 = np.sqrt(diag_2) @ corr_2 @ np.sqrt(diag_2)
            
        #     # Generate X conditionally on S using the covariance matrices
        #     X_all = np.zeros((nb_obs, nb_features))
        #     indices_1 = data.index[data['S'] == 1].tolist()
        #     indices_2 = data.index[data['S'] == 2].tolist()
            
        #     param['X_1'] = np.random.multivariate_normal(mean=mu_1, cov=cov_1, size=len(indices_1))
        #     X_all[indices_1] = param['X_1']
        #     param['X_2'] = np.random.multivariate_normal(mean=mu_2, cov=cov_2, size=len(indices_2))
        #     X_all[indices_2] = param['X_2']

        #     # Add X columns to the dataframe
        #     for j in range(nb_features):
        #         data[f'X_{j}'] = X_all[:, j]

        #     # Calculate Y as sum of X features plus t*S
        #     Y = np.sum(X_all, axis=1) + t_unfair * data['S'].values
        #     data['Y'] = Y

        #     # Reset random seed to avoid affecting other code
        #     if random_seed is not None:
        #         np.random.seed(None)
            
        #     return data, param


class Visualization : 

    def plot_scores_densities(t_results, unfairness_measure, test_dataset, save_path=None, tick_fontsize=10):
        """Displays the density distribution of the main models.

        Args:
            t_results: Dictionary containing the results metrics
            unfairness_measure: The unfairness measure to display
            test_dataset: The dataset containing the predictions
            save_path: Path to save the figure (optional)
            tick_fontsize: Font size for the tick labels (default: 8)

        Returns:
            The figure object
        """
        # Select only the models we want to display
        display_models = ['y_input_reg', 'y_pred_bias', 'y_pred_riken', 'y_pred_fair']
        display_titles = ['Linear Model', 'CS22', 'FS23', 'Our model']
        
        # Create figure with 4 subplots (one for each selected model)
        fig, axes = plt.subplots(1, len(display_models), figsize=(12, 2))
        
        # For each model
        for i, model in enumerate(display_models):
            if model in test_dataset.columns:
                # Calculate metrics for the title
                r2 = Metrics.group_weighted_r2(test_dataset, 'Y', model, 'S')
                rmse = round(np.sqrt(mean_squared_error(test_dataset['Y'], test_dataset[model])), 2)
                unfairness = round(t_results[model][unfairness_measure], 2)
                
                # Plot histograms
                sns.histplot(
                    test_dataset[test_dataset['S']==1][model],
                    label='Group 1',
                    stat='density',
                    bins=15,
                    alpha=0.4,
                    ax=axes[i])
                sns.histplot(
                    test_dataset[test_dataset['S']==2][model],
                    label='Group 2',
                    stat='density',
                    bins=15,
                    alpha=0.4,
                    ax=axes[i])
                
                # Set title with metrics
                axes[i].set_title(f"{display_titles[i]}\nRMSE: {rmse}, GWR²: {r2:.3f}, U: {unfairness:.3f}", fontsize=12)
                
                # Clean up axes
                axes[i].set_xlabel("Prediction", fontsize=12)
                
                # Réduire la taille des chiffres sur les axes
                axes[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                if i == 0:
                    axes[i].set_ylabel("Density", fontsize=12)
                    axes[i].legend(fontsize=8)
                else:
                    axes[i].set_ylabel("")
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    

    def plot_approximate_fairness(results_dict, unfairness_measure):
        """
        Creates a GWR2 vs Unfairness plot for different models.
        
        Args:
            results_dict: Dictionary containing model results.
                        Each key is a model name and each value is a dictionary
                        containing 'GWR2' and 'unfairness_W2' metrics.
            unfairness_measure: The unfairness measure to use (e.g., 'unfairness_W2')
        """
        # Create the figure
        plt.figure(figsize=(5, 3))
        
        
        # Define special models with their labels
        special_models = {
            'y_pred_fair': 'Our Model',
            'y_input_reg': 'Linear Model',
            'y_pred_riken': 'FS23',
            'y_pred_bias': 'CS22'
        }
        
        # Colorblind-friendly colors with requested assignments
        # Our model (green), Base model (blue), FS_23 (purple), CS_22 (red)
        colors_dict = {
            'y_pred_fair': "#03C175",  # Green for Our Model
            'y_input_reg': "#2B92C2",  # Blue for Base model
            'y_pred_riken': "#8B7DAD",  # Purple for FS_23
            'y_pred_bias': "#D55E00"   # Red for CS_22
        }
        
        # Markers for special models
        markers_dict = {
            'y_pred_fair': '*',  # circle
            'y_input_reg': 'o',  # square
            'y_pred_riken': '^',  # triangle
            'y_pred_bias': 's'   # star
        }
        
        # Plot special points
        for model, label in special_models.items():
            if model in results_dict:
                plt.scatter(
                    results_dict[model][unfairness_measure][0], 
                    results_dict[model]['GWR2'][0],
                    marker=markers_dict[model], 
                    s=100, 
                    color=colors_dict[model],
                    label=label,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5
                )
        
        # Plot points for y_pred_fair_0.1 to y_pred_fair_0.9
        epsilon_color = "#5EC2CF"  # Light blue for epsilon models
        for i in range(1, 10):
            epsilon = i / 10
            key = f'y_pred_fair_{epsilon}'
            if key in results_dict:
                plt.scatter(
                    results_dict[key][unfairness_measure][0], 
                    results_dict[key]['GWR2'][0],
                    marker='o', 
                    s=50, 
                    color=epsilon_color,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                # Only show annotation for epsilon=0.1
                if epsilon == 0.1:
                    plt.annotate(
                        f'ε²={epsilon}', 
                        (results_dict[key][unfairness_measure][0], results_dict[key]['GWR2'][0]),
                        xytext=(5, -10), 
                        textcoords='offset points',
                        fontsize=11
                    )
        
        # Add labels and legend
        plt.xlabel('Unfairness', fontsize=12)
        plt.ylabel('GWR2', fontsize=12)
        plt.title(' ', fontsize=14)
        
        # Create custom legend that includes f_epsilon
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Add an element for f_epsilon
        epsilon_handle = Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=epsilon_color, 
                            markersize=8, 
                            markeredgecolor='black',
                            markeredgewidth=0.5,
                            label='f_ε²')
        
        handles.append(epsilon_handle)
        labels.append('f_ε²')
        
        plt.legend(handles=handles, labels=labels, title='Models', loc='best', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust axis limits if needed
        plt.tight_layout()
        
        return plt.gcf()



    def plot_fair_coefficient_evolution(t_results, nb_features_to_plot, name_figure):
        """generates a bar plot with the coefficient evolution between the base linear model and the fair linear model.

        Args:
            t_results (_type_): _description_
            nb_features_to_plot (_type_): _description_
            name_figure (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # Reset style to default and set clean theme
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.0)  # Reduced font scale
        
        # Create smaller figure
        fig, ax1 = plt.subplots(figsize=(3, 1.5), dpi=300)  # Reduced size
        
        # Prepare data structure for plotting
        variables = ['Intercept'] + [f'X_{i}' for i in range(nb_features_to_plot)] + ['S']
        n_vars = len(variables)
        
        # Extract coefficients
        # Group 1 base coefficients (input regression)
        base_g1 = [t_results['y_input_reg']['beta_0_NoStd'][0]]  # Intercept
        base_g1.extend([t_results['y_input_reg']['beta_NoStd'][0][i] for i in range(nb_features_to_plot)])  # Features
        base_g1.append(t_results['y_input_reg']['gamma_NoStd'][0])  # S feature
        base_g1 = np.array(base_g1)
        
        # Group 2 base coefficients (input regression)
        base_g2 = [t_results['y_input_reg']['beta_0_NoStd'][0]]  # Intercept
        base_g2.extend([t_results['y_input_reg']['beta_NoStd'][0][i] for i in range(nb_features_to_plot)])  # Features
        base_g2.append(t_results['y_input_reg']['gamma_NoStd'][0])  # S feature
        base_g2 = np.array(base_g2)
        
        # Group 1 fair coefficients
        fair_g1 = [t_results['y_pred_fair']['fair_intercept_1_NoStd'][0]]  # Intercept
        fair_g1.extend([t_results['y_pred_fair']['beta_1_NoStd'][0][i] for i in range(nb_features_to_plot)])  # Features
        fair_g1.append(t_results['y_pred_fair']['gamma_NoStd'][0])  # S feature
        fair_g1 = np.array(fair_g1)
        
        # Group 2 fair coefficients
        fair_g2 = [t_results['y_pred_fair']['fair_intercept_2_NoStd'][0]]  # Intercept
        fair_g2.extend([t_results['y_pred_fair']['beta_2_NoStd'][0][i] for i in range(nb_features_to_plot)])  # Features
        fair_g2.append(t_results['y_pred_fair']['gamma_NoStd'][0])  # S feature
        fair_g2 = np.array(fair_g2)
        
        # Calculate differences
        diff_g1 = fair_g1 - base_g1
        diff_g2 = fair_g2 - base_g2
        
        # Bar placement
        width = 0.3  # Reduced width
        x = np.arange(n_vars)
        offset = [-width / 2, width / 2]
        
        # Plot bars for each group
        groups = ["Group 1", "Group 2"]
        bases = [base_g1, base_g2]
        diffs = [diff_g1, diff_g2]
        
        for gi, group in enumerate(groups):
            xg_all = x + offset[gi]
            
            # Base bars (input regression)
            ax1.bar(xg_all, bases[gi], width,
                color="#F4F4F4", edgecolor="#222B36", lw=0.4,
                label=f"Input Reg ({group})" if gi == 0 else None, zorder=2)
            
            # Difference bars (to fair model)
            ax1.bar(xg_all, diffs[gi], width, bottom=bases[gi],
                color=np.where(diffs[gi] > 0, "#00A562", "#FFA100"),
                edgecolor="#222B36", lw=0.4,
                label=f"Δ to Fair Model" if gi == 0 else None, zorder=3)
            
            # Arrows showing direction of change - only for significant changes
            for vi, (b, d) in enumerate(zip(bases[gi], diffs[gi])):
                if abs(d) < 1e-10 or abs(d) < abs(b)*0.05:  # Skip if no significant change
                    continue
                x_c = xg_all[vi]
                y_start = b + (0.05 * d)  # Start 5% from base
                y_end = b + (0.95 * d)    # End 95% toward fair value
                ax1.annotate("",
                            xy=(x_c, y_end), xytext=(x_c, y_start),
                            arrowprops=dict(arrowstyle="->",
                                            color="#000000",
                                            lw=0.8),  # Thinner arrows
                            zorder=6)
        
        # X-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels(variables, rotation=0, ha='center', fontsize=7)  # Smaller font
        
        # Make sure y-axis tick labels are also consistent
        ax1.tick_params(axis='y', labelsize=8)  # Smaller font
        
        # Title and labels
        ax1.set_ylabel("Coefficient Value", fontsize=7)
        ax1.set_xlabel("Variable", fontsize=7)
        
        # Zero line
        ax1.axhline(0, color="#222B36", lw=0.5)
        
        # Create a compact legend
        # Create custom legend elements
        legend_elements = [
            # Group identification
            Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='#222B36', 
                markersize=7, label="Group 1 (left)"),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='#222B36', 
                markersize=7, label="Group 2 (right)"),
            # Color coding
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#F4F4F4', markeredgecolor='#222B36', 
                markersize=7, label='Input Reg.'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#00A562', markeredgecolor='#222B36', 
                markersize=7, label='Increase'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFA100', markeredgecolor='#222B36', 
                markersize=7, label='Decrease')
        ]
        
        # Place legend at the top of the figure
        fig.legend(handles=legend_elements, loc='upper center', 
                bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False, 
                fontsize=7, handletextpad=0.5, columnspacing=1.0)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(name_figure, bbox_inches='tight', dpi=300)
        
        return fig


    def plot_unfairness_drifts(all_results, t_values, perf_metric, unf_metric):
        """plots the GWR² of models'scores wrt. the unfairness when increasing a type of bias

        Args:
            all_results (_type_): _description_
            t_values (_type_): _description_
            perf_metric (_type_): _description_
            unf_metric (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Define models and their display names
        models = ['y_input_reg', 'y_pred_bias', 'y_pred_riken', 'y_pred_fair']
        display_names = ['Linear model', 'CS_22', 'FS_23', 'Our model']
        
        # Define colorblind-friendly colors
        # Colorblind-friendly palette: blue, orange, yellowish green, red-purple
        colors = ["#2B92C2", "#D55E00", "#8B7DAD", "#03C175"]
        #  'y_pred_fair': "#03C175",  # Green for Our Model
        #     'y_input_reg': "#2B92C2",  # Blue for Base model
        #     'y_pred_riken': "#8B7DAD",  # Purple for FS_23
        #     'y_pred_bias': "#D55E00"   # Red for CS_22
        markers = ['o', 's', '^', '*']  # circle, square, triangle, star
        
        # Create figure
        plt.figure(figsize=(5, 3))
        
        # For each model
        for i, (model, display_name) in enumerate(zip(models, display_names)):
            rmse_values = []
            unfairness_values = []
            
            # Collect values for each t
            for t in t_values:
                if t in all_results and model in all_results[t]:
                    rmse = all_results[t][model].get(f'{perf_metric}_mean')
                    unfairness = all_results[t][model].get(f'{unf_metric}_mean')
                    
                    if rmse is not None and unfairness is not None:
                        rmse_values.append(rmse)
                        unfairness_values.append(unfairness)
            
            # Plot points for this model
            plt.scatter(unfairness_values, rmse_values, 
                    label=display_name, 
                    color=colors[i], 
                    marker=markers[i], 
                    s=100,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5)
            
            # Add annotations for t values
            # Don't add annotations for "Our model" and "CS_22"
            if display_name == "Linear model":
                # For Linear model, annotate T=0, T=3, T=6 only
                for j, t in enumerate([0]):
                    if j < len(rmse_values):
                        plt.annotate(f"T={t}", 
                                (unfairness_values[j], rmse_values[j]),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha='center',
                                fontsize=10)
                for j, t in enumerate([3, 6]):  # Limited to 3 and 6 only
                    if j+1 < len(rmse_values):
                        plt.annotate(f"{t}", 
                                (unfairness_values[j+1], rmse_values[j+1]),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha='center',
                                fontsize=10)
            elif display_name == "FS_23":
                # For FS_23, shift annotations to the right
                for j, t in enumerate([0]):
                    if j < len(rmse_values):
                        plt.annotate(f"T={t}", 
                                (unfairness_values[j], rmse_values[j]),
                                textcoords="offset points",
                                xytext=(10, 0),  # Shift to the right (x=10)
                                ha='center',
                                fontsize=10)
                for j, t in enumerate([3, 6, 9, 12, 15]):
                    if j+1 < len(rmse_values):
                        plt.annotate(f"{t}", 
                                (unfairness_values[j+1], rmse_values[j+1]),
                                textcoords="offset points",
                                xytext=(10, -3),  # Shift to the right (x=10)
                                ha='center',
                                fontsize=10)
        
        # Add titles and labels
        # plt.title( , fontsize=14)
        plt.xlabel('Unfairness', fontsize=12)
        plt.ylabel(f'{perf_metric}', fontsize=12)
        
        # Add legend
        plt.legend(title='Models', loc='best', frameon=True)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust axis limits if needed
        # plt.xlim(0, max_unfairness * 1.1)
        # plt.ylim(min_r2 * 0.9, 1.0)
        
        plt.tight_layout()
        
        return plt.gcf()




    def plot_unfairness_decomposition(results_dict, param_name,param_name_legend, output_file, figsize=(6, 4), dpi=300):
        """
        Plot unfairness decomposition for a linear model, as stacked bars for varying parameter values.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with results for each parameter value
        param_name : str
            Name of the parameter that varies (e.g., 'T_correl')
        output_file : str
            Path to save the output figure
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Resolution for the saved figure
        """
        # Set style for scientific publication
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Use colorblind palette
        palette = sns.color_palette("colorblind")
        
        # Create figure with higher quality settings
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Extract parameter values and prepare data
        param_values = sorted(results_dict.keys())
        
        # Components to plot
        components = ['direct_mean_bias_mean', 'indirect_structural_bias_mean', 'indirect_mean_bias_mean', 'interaction_mean']
        component_labels = ['Direct Mean Bias', 'Indirect Structural Bias', 'Indirect Mean Bias', 'Interaction']
        
        # Create data structure for plotting
        data = []
        for param_val in param_values:
            result = results_dict[param_val]['y_input_reg']
            row = [param_val]
            for comp in components:
                row.append(result[comp])
            data.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[param_name] + components)
        
        # Use integer positions for bars, but label with actual parameter values
        x_positions = np.arange(len(param_values))
        
        # Set width of bars
        bar_width = 0.7
        
        # Plot stacked bars
        bottom = np.zeros(len(param_values))
        for i, comp in enumerate(components):
            values = df[comp].values
            ax.bar(x_positions, values, bottom=bottom, width=bar_width,
                label=component_labels[i], color=palette[i], 
                edgecolor='black', linewidth=0.5)
            bottom += values
        
        # Add total unfairness as a line
        df['total'] = df[components].sum(axis=1)
        ax.plot(x_positions, df['total'], 'k--', marker='o', markersize=5, 
                linewidth=1.5, label='Total Unfairness')
        
        # Set x-ticks to parameter values
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(val) for val in param_values])
        
        # Customize plot for scientific publication
        ax.set_xlabel(f'{param_name_legend}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Unfairness', fontsize=11, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        
        # Add grid only for y-axis
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend with clear formatting
        legend = ax.legend(fontsize=9, frameon=True, fancybox=False, 
                        edgecolor='black', ncol=1, loc='upper left', 
                        bbox_to_anchor=(1.02, 1))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, bbox_inches='tight')
        
        return fig
    



    def plot_multiple_unfairness_decompositions(results_dicts, param_names, param_legends, output_file, figsize=(17, 4), dpi=300):
        """
        Plot multiple unfairness decompositions as stacked bars in a single row with shared legend.
        
        Parameters:
        -----------
        results_dicts : list of dict
            List of dictionaries with results for each parameter
        param_names : list of str
            Names of the parameters that vary
        param_legends : list of str
            Display names for the parameters on x-axis
        output_file : str
            Path to save the output figure
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Resolution for the saved figure
        """
        # Set style for scientific publication
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Disable LaTeX and use matplotlib's built-in math renderer
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'
        
        # Use colorblind palette
        palette = sns.color_palette("colorblind")
        
        # Components to plot
        components = ['direct_mean_bias_mean', 'indirect_structural_bias_mean', 'indirect_mean_bias_mean', 'interaction_mean']
        component_labels = ['Direct Mean Bias', 'Indirect Structural Bias', 'Indirect Mean Bias', 'Interaction']
        
        # Create figure with subplots - add space at top for legend
        fig, axes = plt.subplots(1, len(results_dicts), figsize=figsize, dpi=dpi, sharey=True)
        
        # Store handles for legend
        legend_handles = []
        legend_labels = []
        
        # Process each subplot
        for idx, (results_dict, param_name, param_legend, ax) in enumerate(zip(results_dicts, param_names, param_legends, axes)):
            # Extract parameter values and prepare data
            param_values = sorted(results_dict.keys())
            
            # Create data structure for plotting
            data = []
            for param_val in param_values:
                result = results_dict[param_val]['y_input_reg']
                row = [param_val]
                for comp in components:
                    row.append(result[comp])
                data.append(row)
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[param_name] + components)
            
            # Use integer positions for bars, but label with actual parameter values
            x_positions = np.arange(len(param_values))
            
            # Set width of bars
            bar_width = 0.7
            
            # Plot stacked bars
            bottom = np.zeros(len(param_values))
            for i, comp in enumerate(components):
                values = df[comp].values
                bars = ax.bar(x_positions, values, bottom=bottom, width=bar_width,
                    label=component_labels[i] if idx == 0 else "", color=palette[i], 
                    edgecolor='black', linewidth=0.5)
                bottom += values
                
                # Store handles for first subplot only (for legend)
                if idx == 0:
                    legend_handles.append(bars)
                    legend_labels.append(component_labels[i])
            
            # Add total unfairness as a line
            df['total'] = df[components].sum(axis=1)
            line = ax.plot(x_positions, df['total'], 'k--', marker='o', markersize=5, 
                    linewidth=1.5, label='Total Unfairness' if idx == 0 else "")
            
            if idx == 0:
                legend_handles.append(line[0])
                legend_labels.append('Total Unfairness')
            
            # Set x-ticks to parameter values with increased font size
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(val) for val in param_values], fontsize=20)  # Increased from 15 to 20
            
            # Customize plot for scientific publication with increased font sizes
            ax.set_xlabel(f'{param_legend}', fontsize=22)  # Increased from 18 to 22
            if idx == 0:
                ax.set_ylabel('Unfairness', fontsize=22, fontweight='bold')  # Increased from 18 to 22
            ax.tick_params(axis='both', labelsize=20)  # Increased from 18 to 20
            
            # Add grid only for y-axis
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Add a single legend at the top with increased font size
        fig.legend(legend_handles, legend_labels, 
                loc='upper center', bbox_to_anchor=(0.5, 1.05),
                ncol=5, fontsize=20, frameon=True, fancybox=False,  # Increased from 18 to 20
                edgecolor='black')
        
        # Adjust layout with minimal space between legend and plots
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust this value to control space between legend and plots
        
        # Save figure
        plt.savefig(output_file, bbox_inches='tight')
        
        return fig
