import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import statsmodels.api as sm
import math
from scipy.stats import norm

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)
from sklearn.preprocessing import MinMaxScaler
from equipy.fairness import FairWasserstein

class Fair_model:
    """
    A class implementing fair linear regression methods to mitigate bias in predictive models.
    
    This implementation follows the methodology for creating fairness-aware linear regression
    models that account for sensitive attributes while maintaining predictive performance.
    The approach involves fitting an initial regression model and then computing fair
    predictions based on statistical properties of different sensitive groups.
    
    Methods:
        input_linear_regression: Fits initial linear/ridge regression on features and sensitive attribute
        estimation_of_parameters: Estimates group-specific statistical parameters
        compute_fair_model_terms: Computes fairness-aware model coefficients
        fair_regression_function: Generates fair predictions for individual samples
        predict_fair_linear_score: Main pipeline method for fair prediction
    """

    @staticmethod
    def input_linear_regression(train_data, test_data, 
                                S_variable, X_features, Y_target, 
                                param_dictionnary, bool_aware,
                                bool_intercept, bool_ridge, return_model_bool):
        """
        Performs linear or ridge regression of Y on X and S (sensitive attribute).
        
        This method fits an initial regression model that includes both the feature set X
        and the sensitive attribute S. The fitted model serves as the foundation for
        subsequent fairness adjustments.

        Args:
            train_data (pandas.DataFrame): Training dataset containing features, target, and sensitive attribute
            test_data (pandas.DataFrame): Test dataset for predictions
            S_variable (str): Name of the sensitive attribute column
            X_features (list of str): List of feature column names
            Y_target (str): Name of the target variable column
            param_dictionnary (dict): Dictionary to store model parameters (modified in-place)
            bool_intercept (bool): Whether to fit an intercept term
            bool_ridge (bool): If True, uses Ridge regression with cross-validation; otherwise LinearRegression
            return_model_bool (bool): Whether to return the fitted model object

        Returns:
            tuple: A tuple containing:
                - coef_input_model (list): Model coefficients including intercept
                - param_dictionnary (dict): Updated parameter dictionary with model coefficients
                - model (sklearn model or None): Fitted regression model if return_model_bool is True
                
        Note:
            The parameter dictionary is updated with:
            - 'gamma': coefficient of the sensitive attribute
            - 'beta': coefficients of the feature variables (excluding sensitive attribute)
            - 'beta_0': intercept term
            - 'reg_regularization': regularization parameter (alpha for Ridge, 0 for LinearRegression)
        """
        train_dataset, test_dataset = train_data, test_data
        if bool_aware : 
            X_train, X_test = train_dataset[X_features+[S_variable]],test_dataset[X_features+[S_variable]]
        else: 
            X_train, X_test = train_dataset[X_features],test_dataset[X_features]

        y_s = train_dataset[str(Y_target)]
        
        if bool_ridge == True: 
            reg = RidgeCV(fit_intercept=bool_intercept).fit(X_train, y_s)
            param_dictionnary['reg_regularization'] = reg.alpha_
        else:
            reg = LinearRegression(fit_intercept=bool_intercept).fit(X_train, y_s)
            param_dictionnary['reg_regularization'] = 0
    
        coef_input_model = reg.coef_.tolist()
        
        if bool_aware:
            param_dictionnary['gamma'] = reg.coef_[len(X_features)]
            param_dictionnary['beta'] = reg.coef_[:len(X_features)]
            train_dataset['y_input_reg'] = reg.predict(X_train)
            test_dataset['y_input_reg'] = reg.predict(X_test) 
        else: 
            param_dictionnary['beta'] = reg.coef_
            train_dataset['y_unaware'] = reg.predict(X_train)
            test_dataset['y_unaware'] = reg.predict(X_test)
        # Beta corresponds to coefficients of explanatory variables (excluding S coefficient and intercept)
        
        param_dictionnary['beta_0'] = reg.intercept_
        coef_input_model.insert(0, reg.intercept_) 
        # Intercept is added to the coefficient list

        if return_model_bool: 
            return(coef_input_model, param_dictionnary, reg)
        else:
            return(coef_input_model, param_dictionnary, None)

    @staticmethod
    def estimation_of_parameters(pool_data, S_variable, X_features, param_dictionnary):
        """
        Estimates group-specific statistical parameters conditional on sensitive attribute values.
        
        For each unique value of the sensitive attribute S, this method computes:
        - Group probability P(S=s)
        - Empirical mean of features μ_X|S=s
        - Covariance matrix Σ_X|S=s
        
        These parameters are essential for computing fairness-aware predictions that
        account for distributional differences across sensitive groups.

        Args:
            pool_data (pandas.DataFrame): Dataset used for parameter estimation (typically larger than train)
            S_variable (str): Name of the sensitive attribute column
            X_features (list of str): List of feature column names
            param_dictionnary (dict): Dictionary to store estimated parameters (modified in-place)

        Returns:
            dict: Updated parameter dictionary containing group-specific statistics:
                - 'p_{i}': probability of sensitive group i
                - 'empirical_mean_{i}': mean vector of features for group i
                - 'Sigma_{i}': covariance matrix of features for group i
                
        Note:
            The pool_data should be representative of the population to ensure
            accurate parameter estimation across all sensitive groups.
        """
        for i in (pool_data[str(S_variable)].unique()):
            param_dictionnary[f'p_{int(i)}'] = pool_data[pool_data[str(S_variable)] == i][S_variable].count() / pool_data[S_variable].count()
            param_dictionnary[f'empirical_mean_{int(i)}'] = [pool_data[pool_data[str(S_variable)] == i][str(j)].mean() for j in X_features]
            param_dictionnary[f'Sigma_{int(i)}'] = (pool_data[pool_data[str(S_variable)] == i][X_features].cov())
        return(param_dictionnary)

    @staticmethod
    def compute_fair_model_terms(train_data, S_variable, param_dictionnary):
        """
        Computes the invariant terms required for fair prediction.
        
        This method calculates two key components of the fair regression model:
        1. Invariant variance-covariance term: a weighted average of group-specific
           variance projections that ensures fairness across groups
        2. Fair intercept: an adjusted intercept that maintains calibration while
           promoting equalized outcomes across sensitive groups

        Args:
            train_data (pandas.DataFrame): Training dataset
            S_variable (str): Name of the sensitive attribute column
            param_dictionnary (dict): Dictionary containing model parameters (modified in-place)

        Returns:
            dict: Updated parameter dictionary with fair model terms:
                - 'invariant_var_cov_term': weighted average of variance projections across groups
                - 'fair_intercept': adjusted intercept for fair predictions
                - 'var_cov_product_{i}': variance projection for each group i
                
        Note:
            The invariant terms are computed as weighted averages using group probabilities,
            ensuring that the fair model accounts for the relative sizes of different groups.
        """
        param_dictionnary['invariant_var_cov_term'] = 0
        param_dictionnary['fair_intercept'] = 0

        for i in train_data[str(S_variable)].unique():  
            param_dictionnary[f'var_cov_product_{int(i)}']  = np.sqrt(param_dictionnary['beta'].T @ param_dictionnary[f'Sigma_{int(i)}'] @ param_dictionnary['beta'])
            param_dictionnary['invariant_var_cov_term'] += param_dictionnary[f'p_{int(i)}']*param_dictionnary[f'var_cov_product_{int(i)}']
            param_dictionnary['fair_intercept'] += param_dictionnary[f'p_{int(i)}']*(np.dot(param_dictionnary[f'empirical_mean_{int(i)}'],param_dictionnary['beta'])+i*param_dictionnary['gamma']) 
        param_dictionnary['fair_intercept'] += param_dictionnary['beta_0']
        
        #Computation of coefficients of the fair model for graphs
        for i in train_data[str(S_variable)].unique():  
            param_dictionnary[f'fair_intercept_{int(i)}_NoStd']= param_dictionnary['fair_intercept']-param_dictionnary['invariant_var_cov_term']*np.dot(param_dictionnary[f'empirical_mean_{int(i)}'],param_dictionnary['beta'])/param_dictionnary[f'var_cov_product_{int(i)}']
            param_dictionnary[f'beta_{int(i)}_NoStd']=(param_dictionnary['invariant_var_cov_term']*param_dictionnary['beta']/param_dictionnary[f'var_cov_product_{int(i)}'])
        param_dictionnary['gamma_NoStd']=(0)

        #Computation of the unfairness of the input model 
        param_dictionnary['mean_S']=0
        mu_beta=0
        mu_beta_fair=0
        mean_var_cov_product= 0
        mean_var_cov_product_fair = 0
        cov_s_mu_beta=0
        for i in train_data[S_variable].unique():
            param_dictionnary['mean_S'] += param_dictionnary[f'p_{int(i)}'] *i
            #moyenne du produit scalaire mu,beta
            mu_beta += param_dictionnary[f'p_{int(i)}']*np.dot(param_dictionnary[f'empirical_mean_{int(i)}'],param_dictionnary['beta'])
            mu_beta_fair += param_dictionnary[f'p_{int(i)}']*np.dot(param_dictionnary[f'empirical_mean_{int(i)}'],param_dictionnary[f'beta_{int(i)}_NoStd'])
            #moyenne de ||\beta||_\Sigma
            mean_var_cov_product += param_dictionnary[f'p_{int(i)}']*param_dictionnary[f'var_cov_product_{int(i)}']
            param_dictionnary[f'var_cov_product_{int(i)}_fair']  = np.sqrt(param_dictionnary[f'beta_{int(i)}_NoStd'].T @ param_dictionnary[f'Sigma_{int(i)}'] @ param_dictionnary[f'beta_{int(i)}_NoStd'])
            mean_var_cov_product_fair += param_dictionnary[f'p_{int(i)}']*param_dictionnary[f'var_cov_product_{int(i)}_fair']
        
        param_dictionnary['var_S']=0
        param_dictionnary['indirect_mean_bias'] =0
        param_dictionnary['indirect_mean_bias_fair_model'] =0
        param_dictionnary['indirect_structural_bias']=0
        param_dictionnary['indirect_structural_bias_fair_model']=0

        for i in train_data[S_variable].unique():
            param_dictionnary['var_S'] += param_dictionnary[f'p_{int(i)}']*(i-param_dictionnary['mean_S'] )**2
            param_dictionnary['indirect_mean_bias'] += param_dictionnary[f'p_{int(i)}']*(np.dot(param_dictionnary[f'empirical_mean_{int(i)}'],param_dictionnary['beta'])-mu_beta)**2
            param_dictionnary['indirect_mean_bias_fair_model'] += param_dictionnary[f'p_{int(i)}']*(np.dot(param_dictionnary[f'empirical_mean_{int(i)}'],param_dictionnary[f'beta_{int(i)}_NoStd'])-mu_beta_fair)**2
            param_dictionnary['indirect_structural_bias'] += param_dictionnary[f'p_{int(i)}']*( param_dictionnary[f'var_cov_product_{int(i)}'] - mean_var_cov_product )**2
            param_dictionnary['indirect_structural_bias_fair_model'] += param_dictionnary[f'p_{int(i)}']*( param_dictionnary[f'var_cov_product_{int(i)}_fair'] - mean_var_cov_product_fair )**2
            cov_s_mu_beta += param_dictionnary[f'p_{int(i)}']*(i-param_dictionnary['mean_S'])*(np.dot(param_dictionnary[f'empirical_mean_{int(i)}'],param_dictionnary['beta'])-mu_beta)

        param_dictionnary['direct_mean_bias']=param_dictionnary['gamma']**2 *param_dictionnary['var_S'] 
        param_dictionnary['interaction']=2*param_dictionnary['gamma']*cov_s_mu_beta

        param_dictionnary['unfairness_input_model']=param_dictionnary['direct_mean_bias']+ param_dictionnary['interaction']+param_dictionnary['indirect_mean_bias']+param_dictionnary['indirect_structural_bias']
        param_dictionnary['unfairness_our_model']=param_dictionnary['indirect_structural_bias_fair_model']+param_dictionnary['indirect_mean_bias_fair_model']

        return(param_dictionnary)

    @staticmethod
    def fair_regression_function(param_dictionnary, row_X_S, S_variable, X_features):
        """
        Generates a fair prediction for a single sample.
        
        This function implements the core fair regression formula that adjusts
        predictions based on group-specific statistics to ensure fairness across
        sensitive groups. The prediction is computed by normalizing the feature
        deviation from the group mean and scaling it by invariant terms.

        Args:
            param_dictionnary (dict): Dictionary containing all model parameters
            row_X_S (pandas.Series): Single row containing features and sensitive attribute
            S_variable (str): Name of the sensitive attribute column
            X_features (list of str): List of feature column names

        Returns:
            float: Fair prediction for the input sample
            
        Note:
            The fair prediction formula ensures that individuals with similar
            feature profiles receive similar predictions regardless of their
            sensitive attribute value, promoting equalized odds across groups.
            
        Formula:
            ŷ_fair = (invariant_var_cov_term / var_cov_product_s) * β^T(x - μ_s) + fair_intercept
            where s is the sensitive group of the individual.
        """
        # TODO: Replace hardcoded 0 and 1 with actual sensitive attribute values for generalization
        gender = int(row_X_S[str(S_variable)])
        prediction = param_dictionnary['invariant_var_cov_term'] * np.dot(row_X_S[X_features] - param_dictionnary[f'empirical_mean_{gender}'], param_dictionnary['beta']) / param_dictionnary[f'var_cov_product_{gender}'] + param_dictionnary['fair_intercept']
        return(prediction)

    @staticmethod
    def predict_fair_linear_score(train_dataset, pool_dataset, test_dataset,
                                  S_variable, Y_target, X_features, 
                                  bool_intercept, bool_ridge, return_input_model_bool):
        """
        Main pipeline for generating fair linear regression predictions.
        
        This method orchestrates the complete fair regression pipeline:
        1. Fits initial regression model on training data
        2. Estimates group-specific parameters from pool data
        3. Computes fair model coefficients
        4. Generates fair predictions for test data
        
        The method provides a complete end-to-end solution for fair machine learning
        that can be easily integrated into existing ML workflows.

        Args:
            train_dataset (pandas.DataFrame): Dataset for model training
            pool_dataset (pandas.DataFrame): Dataset for parameter estimation (can be same as train)
            test_dataset (pandas.DataFrame): Dataset for fair predictions
            S_variable (str): Name of the sensitive attribute column
            Y_target (str): Name of the target variable column
            X_features (list of str): List of feature column names
            bool_intercept (bool): Whether to include intercept in regression
            bool_ridge (bool): Whether to use Ridge regression (True) or LinearRegression (False)
            return_input_model_bool (bool): Whether to return the initial fitted model

        Returns:
            tuple: A tuple containing:
                - coef_input_model (list): Coefficients of the initial regression model
                - param_dictionnary (dict): Complete dictionary of model parameters
                - input_model (sklearn model or None): Initial fitted model if requested
                - test_dataset (pandas.DataFrame): Test data with added 'y_pred_fair' column
                
        Note:
            The test_dataset is modified in-place with a new column 'y_pred_fair'
            containing the fair predictions. Consider copying the dataset if the
            original needs to be preserved.
            
        Example:
            >>> coefs, params, model, results = Fair_model.predict_fair_linear_score(
            ...     train_df, pool_df, test_df, 'gender', 'income', 
            ...     ['age', 'education'], True, False, False
            ... )
        """
        param_dict = {}    
        param_dict['sensitive_variable'] = str(S_variable)                                           
        coef_input_model, param_dictionnary, input_model = Fair_model.input_linear_regression(
            train_dataset, test_dataset, S_variable, X_features, Y_target, 
            param_dict,True, bool_intercept, bool_ridge, return_input_model_bool
        )
        param_dictionnary = Fair_model.estimation_of_parameters(
            pool_dataset, S_variable, X_features, param_dictionnary
        )
        param_dictionnary = Fair_model.compute_fair_model_terms(
            train_dataset, S_variable, param_dictionnary
        )
        test_dataset['y_pred_fair'] = test_dataset.apply(
                lambda row: float(Fair_model.fair_regression_function(
                    param_dictionnary=param_dictionnary,
                    row_X_S=row,
                    S_variable=S_variable,  
                    X_features=X_features
                )),
                axis=1
            )

        return(coef_input_model, param_dictionnary, input_model, test_dataset)
    


class Benchmark_model:
        """
        Collection of benchmark models for comparison with fair regression approaches.
        
        This class implements various baseline and state-of-the-art methods for
        fairness-aware machine learning, providing comprehensive comparison capabilities.
        """
        def benchmark_equipy(X_train_results, X_test_results, score_input_linear_regression, sensitive_var):
                '''
                Post‑process linear regression scores using the Wasserstein fairness
                adjustment from EquipY (FairWasserstein).

                Parameters
                ----------
                X_train_results : pandas.DataFrame
                    Training results containing at least the columns
                    score_input_linear_regression and sensitive_var.
                X_test_results : pandas.DataFrame
                    Test results with the same columns as X_train_results.
                score_input_linear_regression : str
                    Name of the column that stores the raw linear‑regression scores.
                sensitive_var : str
                    Name of the sensitive attribute column.

                Side Effects
                ------------
                Adds the column 'y_score_equipy' to X_test_results containing the debiased predictions.
                '''
                y_train_skl = np.array(X_train_results[str(score_input_linear_regression)].to_list())
                y_test_skl = np.array(X_test_results[str(score_input_linear_regression)].to_list())
                sensitive_feature_y_test = pd.DataFrame({str(sensitive_var): X_test_results[str(sensitive_var)].to_list()})
                sensitive_feature_y_train = pd.DataFrame({str(sensitive_var): X_train_results[str(sensitive_var)].to_list()})

                wasserstein_fairModel = FairWasserstein(sigma=0)
                y_score_equipy = wasserstein_fairModel.fit_transform(y_train_skl, sensitive_feature_y_train, y_test_skl, sensitive_feature_y_test)
                X_test_results['y_score_equipy'] = y_score_equipy


        def weighted_group_intercepts(train_dataset,test_dataset,
                                    feature_cols,target_col,group_col,
                                    weight_by_freq):
            """
            Résout min_{β, b_s} ∑_s p_s ||Y_s - X_s β - b_s*1||^2.
            Pour plus d'informations sur cette approche : 
            "A Minimax Framework For Quantifying Risk-Fairness Trade-Off In Regression"
            by Evgenii Chzhen And Nicolas Schreuder

            Returns
            -------
            beta:     np.ndarray of shape (d,)
            intercepts: dict mapping each group s to b_s
            """

            # 1) calcul des poids p_s
            grouped = train_dataset.groupby(group_col)
            total_n = len(train_dataset)
            if weight_by_freq:
                ps = grouped.size() / total_n
            else:
                ps = pd.Series(1.0 / grouped.ngroups, index=grouped.groups.keys())

            d = len(feature_cols)
            A = np.zeros((d, d)) # matrice de taille d*d pour contenir ensuite X.T*X par groupe
            c = np.zeros(d) # vecteur de taille d pour contenir X.T*Y par groupe
            # stocker moyennes pour calcul de b_s
            Xbar = {}
            Ybar = {}

            for s, group in grouped:
                X = group[feature_cols].to_numpy()
                Y = group[target_col].to_numpy()
                p = ps[s]

                # moyennes
                xbar = X.mean(axis=0)
                ybar = Y.mean()
                Xbar[s] = xbar
                Ybar[s] = ybar

                # centrer X et Y
                Xc = X - xbar  # (n_s, d)
                Yc = Y - ybar  # (n_s,)

                # accumulation pour A, c
                A += p * (Xc.T @ Xc)
                c += p * (Xc.T @ Yc)

            # 3) calcul de β
            beta_hat = np.linalg.solve(A, c)

            # 4) calcul des b_s
            intercept_hat=0
            for s, group in grouped: 
                intercept_hat += ps[s]*(Ybar[s] - Xbar[s] @ beta_hat)

            # 5) Fit
            test_dataset['y_pred_bias'] = test_dataset.apply(
                        lambda row: np.dot(row[feature_cols],beta_hat)+intercept_hat,
                        axis=1
                    )

            # 6) Unfairness
            Mean_mu_beta=0
            indirect_mean_bias=0
            for s, group in grouped: 
                Mean_mu_beta += ps[s]*(Xbar[s] @ beta_hat)
            for s, group in grouped: 
                indirect_mean_bias += ps[s]*(Xbar[s] @ beta_hat - Mean_mu_beta)**2
            
            Mean_cov_product=0
            indirect_structural_bias=0
            unique_groups = train_dataset[group_col].unique()
            for i in unique_groups:
                sigma = train_dataset[train_dataset[group_col]==i][feature_cols].cov()
                Mean_cov_product += ps[s] * np.sqrt(beta_hat.T @ sigma @ beta_hat)
            for i in unique_groups:
                indirect_structural_bias += ps[s] * (np.sqrt(beta_hat.T @ sigma @ beta_hat) - Mean_cov_product)**2

            total_unfairness = indirect_structural_bias + indirect_mean_bias
            return beta_hat, intercept_hat, ps, total_unfairness, indirect_mean_bias, indirect_structural_bias
        
        
        
        def riken_prediction(train_data, test_data, S_variable, X_features, Y_target):
            """
            Implements the demographic parity constrained minimax optimal regression method.
            
            Based on the methodology described in:
            Fukuchi, K., & Sakuma, J. (2023). Demographic parity constrained minimax optimal 
            regression under linear model. Advances in Neural Information Processing Systems, 
            36, 8653-8689.
            
            This implementation follows the sample-splitting approach for cross-fitting to avoid
            overfitting in parameter estimation while maintaining statistical guarantees.
            
            Args:
                train_data: Training dataset
                test_data: Test dataset  
                S_variable: Sensitive attribute (protected variable)
                X_features: List of feature column names
                Y_target: Target variable name
            
            Returns:
                param: Dictionary containing estimated parameters for the minimax method
            """
            dictionnary = {}
            param = {}
            
            # Pre-compute unique values of the sensitive attribute
            unique_s_values = train_data[str(S_variable)].unique()
            total_count = len(train_data)
            
            for i in unique_s_values:
                # Filter data for current sensitive attribute value s=i
                subset = train_data[train_data[str(S_variable)] == i]
                n_i = len(subset)
                dictionnary[f'n_{i}'] = n_i
                param[f'p_{i}'] = n_i / total_count  # Empirical probability P(S=i)

                # Randomly shuffle and partition subset into three parts for cross-fitting
                # This sample-splitting prevents overfitting as described in Fukuchi & Sakuma (2023)
                dictionnary[f'D_{i}'] = subset.sample(frac=1, random_state=42).reset_index(drop=True)
                subset_size = n_i // 3
                
                # Create three non-overlapping datasets for parameter estimation
                dictionnary[f'D_{1},{i}'] = dictionnary[f'D_{i}'].iloc[:subset_size]
                dictionnary[f'D_{2},{i}'] = dictionnary[f'D_{i}'].iloc[subset_size:2*subset_size]
                dictionnary[f'D_{3},{i}'] = dictionnary[f'D_{i}'].iloc[2*subset_size:]
                
                # Assign remaining samples to the third subset
                reste = n_i % 3
                if reste > 0:
                    dictionnary[f'D_{3},{i}'] = pd.concat([dictionnary[f'D_{3},{i}'], dictionnary[f'D_{i}'].iloc[-reste:]])
                
                # Extract feature matrix and target vector for first subset
                X1 = dictionnary[f'D_{1},{i}'][X_features].values
                y1 = dictionnary[f'D_{1},{i}'][str(Y_target)].values
                
                # Estimate ||β_i|| using first subset D_{1,i}
                # Condition ensures sufficient sample size for reliable estimation
                if n_i > 18 * len(X_features):
                    reg_norm = LinearRegression(fit_intercept=False).fit(X1, y1)
                    param[f'norm_beta_{i}'] = np.linalg.norm(reg_norm.coef_)
                else:
                    param[f'norm_beta_{i}'] = 0

                # Estimate normalized direction β̂_i/||β̂_i|| using second subset D_{2,i}
                X2 = dictionnary[f'D_{2},{i}'][X_features].values
                y2 = dictionnary[f'D_{2},{i}'][str(Y_target)].values
                
                if n_i > 18 * len(X_features):
                    reg_normalized = LinearRegression(fit_intercept=False).fit(X2, y2)
                    coef_norm = np.linalg.norm(reg_normalized.coef_)
                    if coef_norm > 0:
                        param[f'normalized_beta_{i}'] = reg_normalized.coef_ / coef_norm
                    else:
                        param[f'normalized_beta_{i}'] = np.zeros(len(X_features))
                else:
                    param[f'normalized_beta_{i}'] = np.zeros(len(X_features))

                # Estimate empirical mean μ_i = E[X|S=i] using third subset D_{3,i}
                param[f"empirical_mean_{i}"] = dictionnary[f'D_3,{i}'][X_features].mean().values
                param[f'sigma_{i}'] = dictionnary[f'D_3,{i}'][X_features].cov()
                
                # Alternative partition: split data in half for independent estimation
                # This provides robustness through sample splitting
                dictionnary[f'D_{i}_bis'] = dictionnary[f'D_{i}'].copy()
                subset_size_bis = n_i // 2
                
                dictionnary[f'D_1,{i}_bis'] = dictionnary[f'D_{i}_bis'].iloc[:subset_size_bis]
                dictionnary[f'D_2,{i}_bis'] = dictionnary[f'D_{i}_bis'].iloc[subset_size_bis:]
                
                # Estimate β_i using first half of alternative partition
                X1_bis = dictionnary[f'D_1,{i}_bis'][X_features].values
                y1_bis = dictionnary[f'D_1,{i}_bis'][str(Y_target)].values
                
                if n_i > 12 * len(X_features):
                    reg_beta_bis = LinearRegression(fit_intercept=False).fit(X1_bis, y1_bis)
                    param[f'beta_{i}_bis'] = reg_beta_bis.coef_
                else:
                    param[f'beta_{i}_bis'] = np.zeros(len(X_features))
                
                # Estimate empirical mean μ_i using second half of alternative partition
                param[f"empirical_mean_{i}_bis"] = dictionnary[f'D_2,{i}_bis'][X_features].mean().values


            # Compute the invariant term for demographic parity constraint
            # This ensures E[f(X,S)] is constant across sensitive groups
            param['invariant_term'] = 0
            for i in unique_s_values:
                param['invariant_term'] += param[f'p_{i}'] * np.dot(param[f'beta_{i}_bis'], param[f'empirical_mean_{i}_bis'])
            
            # Compute weighted norm sum ||β||_w = Σ p_i ||β_i|| across all sensitive attribute values
            # This is the key component of the minimax optimal estimator
            param['norm_beta_sum'] = 0
            for i in unique_s_values:
                param['norm_beta_sum'] += param[f'p_{i}'] * param[f'norm_beta_{i}']

            # Compute unfairness
            param['indirect_mean_bias'] = 0
            param['indirect_structural_bias'] = 0
            mu_beta_mean=0
            Mean_cov_product = 0

          
            for i in unique_s_values:
                # Store the beta vector for each group with proper key formatting
                param[f'beta_{i}_fair_riken'] = param['norm_beta_sum'] * param[f'normalized_beta_{i}']

            #computation of scalar product (mu, beta)
                mu_beta_mean += param[f'p_{i}']* np.dot(param[f"empirical_mean_{i}"],param[f'beta_{i}_fair_riken'])
            #Computation of mean cov product
                # Calculate the covariance product for this group
                beta_i = param[f'beta_{i}_fair_riken']
                sigma_i = param[f'sigma_{i}']
                
                # Ensure proper matrix multiplication
                cov_product = np.sqrt(beta_i @ sigma_i @ beta_i)
                
                # Add to weighted average
                Mean_cov_product += param[f'p_{i}'] * cov_product

            # Second loop to calculate indirect mean bias (supposed to be 0) and structural bias
            for i in unique_s_values:
                beta_i = param[f'beta_{i}_fair_riken']
                sigma_i = param[f'sigma_{i}']
                
                param['indirect_mean_bias'] += param[f'p_{i}']* (np.dot(param[f"empirical_mean_{i}"],param[f'beta_{i}_fair_riken'])-mu_beta_mean)**2

                # Calculate group-specific covariance product
                cov_product_i = np.sqrt(beta_i @ sigma_i @ beta_i)
                # Add squared difference to indirect structural bias
                param['indirect_structural_bias'] += param[f'p_{i}'] * (cov_product_i - Mean_cov_product)**2

            param['total_unfairness'] = param['indirect_mean_bias'] + param['indirect_structural_bias']
            # Vectorized prediction function implementing the minimax estimator
            # Prediction formula: f(x,s) = ||β||_w * ⟨x - μ_s, β̂_s/||β̂_s||⟩ + invariant_term
            def predict_batch(df):
                results = np.zeros(len(df))
                
                for gender in unique_s_values:
                    mask = df[str(S_variable)] == gender
                    if not any(mask):
                        continue
                        
                    X_subset = df.loc[mask, X_features].values
                    mean_vector = param[f'empirical_mean_{gender}']
                    normalized_beta = param[f'normalized_beta_{gender}']
                    
                    # Vectorized computation: ⟨x - μ_s, β̂_s/||β̂_s||⟩
                    produit_scalaire = np.sum((X_subset - mean_vector) * normalized_beta, axis=1)
                    # Apply minimax formula with demographic parity constraint
                    results[mask] = param['norm_beta_sum'] * produit_scalaire + param['invariant_term']
                    
                return results
            
            # Apply minimax predictions to both training and test datasets
            train_data['y_pred_riken'] = predict_batch(train_data)
            test_data['y_pred_riken'] = predict_batch(test_data)
            
            return param
            

class Metrics:
        """Collection of performance and fairness metrics for regression models.

        The methods provided here do **not** depend on any instance state; they can
        therefore be declared as :pycode:`@staticmethod`s or moved to a utility
        module. They cover three main use‑cases:

        * **Fairness** – :pycode:`unfairness_computation` implements a simple
        Earth‑Mover‑Distance‑style group unfairness measure based on empirical
        quantile functions (EQFs).
        * **Group performance** – :pycode:`group_weighted_r2` computes the R²
        averaged across groups, weighted by group prevalence.
        * **Overall performance** – :pycode:`mse_loss` / :pycode:`mean_squared_error`
        return the classical Mean‑Squared‑Error (MSE).

        Notes
        -----
        * The class currently contains two identical MSE functions; keeping one
        alias is sufficient.
        * All functions assume *pandas* and *NumPy* are already imported as
        :pycode:`pd` and :pycode:`np`, and that :pycode:`math` is imported.
        """

        ###########################################################################
        # Fairness metrics
        ###########################################################################

        def unfairness_computation(pred_col, S_variable, test_dataset):
            """Compute the *unfairness* as the weighted L² distance between group
            empirical quantile functions.

            The statistic implemented is

            .. math::
                \sum_s p_s \left( \sum_{q \in Q} \bigl[ F^{-1}_{s}(q) - F^{-1}(q) \bigr]^2 \right)^{1/2},

            where :math:`p_s` is the sample proportion of sensitive group *s*,
            :math:`F^{-1}_{s}` the group‑specific quantile function of the
            predictions, and :math:`F^{-1}` the overall quantile function.

            Parameters
            ----------
            pred_col : str
                Name of the column containing predicted values.
            S_variable : str
                Sensitive attribute used to partition the data.
            test_dataset : pandas.DataFrame
                Dataset that includes *pred_col* and *S_variable*.

            Returns
            -------
            float
                The unfairness score (lower is better; 0 = perfectly fair).
            """
            probs = np.linspace(0.01, 0.99, num=100)
            eqf = np.quantile(test_dataset[pred_col], probs)
            unfairness = 0.0
            for j in test_dataset[S_variable].unique():
                data = test_dataset[test_dataset[S_variable] == j][pred_col]
                p = (
                    test_dataset[test_dataset[S_variable] == j].count().iloc[0]
                    / test_dataset.count().iloc[0]
                )
                eqf_j = np.quantile(data, probs)
                unfairness += p *(((eqf_j - eqf) ** 2).sum())
            return unfairness
        

        def calculate_marginal_contributions(df, feature_cols, s_col, param_our_model):
            """
            Calcule les contributions marginales en utilisant les moyennes et variances conditionnelles pré-calculées.
            
            Paramètres:
            df: pandas DataFrame contenant les features et la variable de stratification
            feature_cols: liste des noms de colonnes des features X
            s_col: nom de la colonne pour la variable de stratification S
            param_our_model: dictionnaire contenant les coefficients et statistiques conditionnelles
            
            Retourne:
            DataFrame avec les contributions marginales pour chaque feature
            """
            # Récupérer les coefficients du modèle
            beta_j_star = param_our_model['beta']
            gamma_star = param_our_model['gamma']
            
            n_features = len(feature_cols)
            unique_groups = df[s_col].unique()
            
            # Initialiser les résultats
            results = {
                'Feature': feature_cols,
                'Mean': [],
                'Std':[],
                'Beta_j_star': beta_j_star,
                'Var_mu_j': [],
                'Var_sigma_j': [],
                'Cov_S_mu_j': [],
                'Indirect_Mean_Bias': [],
                'Indirect_Structural_Bias': [],
                'Interaction_Effect': [],
                'Total_Marginal_Contribution': []
            }
            
            for j in range(n_features):
                
                # Calculer Var(μ_j^(S))
                mean_mu_j = 0
                mean_sigma_j=0
                mean_S = 0

                Mean=[]
                Std=[]
                for i in unique_groups:
                    Mean.append(param_our_model[f'empirical_mean_{i}'][j])
                    Std.append(param_our_model[f'Sigma_{i}'].iloc[j, j])
                    mean_mu_j += param_our_model[f'p_{i}']*param_our_model[f'empirical_mean_{i}'][j]
                    mean_sigma_j += param_our_model[f'p_{i}']*param_our_model[f'Sigma_{i}'].iloc[j, j]
                    mean_S = param_our_model[f'p_{i}']*i

                results['Mean'].append(Mean)
                results['Std'].append(Std)

                var_mu_j=0
                for i in unique_groups:
                    var_mu_j += param_our_model[f'p_{i}']*(param_our_model[f'empirical_mean_{i}'][j]-mean_mu_j)**2
                results['Var_mu_j'].append(var_mu_j)
                
                indirect_mean_bias = (beta_j_star[j]**2) * var_mu_j

                cov_S_mu_j=0
                for i in unique_groups:
                    cov_S_mu_j += param_our_model[f'p_{i}']*(param_our_model[f'empirical_mean_{i}'][j]-mean_mu_j)*(i-mean_S)

                results['Cov_S_mu_j'].append(cov_S_mu_j)
                interaction_effect = 2 * gamma_star * beta_j_star[j] * cov_S_mu_j

                var_sigma_j =0
                for i in unique_groups:
                    var_sigma_j += param_our_model[f'p_{i}']*(param_our_model[f'Sigma_{i}'].iloc[j, j]-mean_sigma_j)**2
                results['Var_sigma_j'].append(var_sigma_j)
                indirect_structural_bias = (beta_j_star[j]**4) * var_sigma_j
                
                results['Indirect_Mean_Bias'].append(indirect_mean_bias)
                results['Indirect_Structural_Bias'].append(indirect_structural_bias)
                results['Interaction_Effect'].append(interaction_effect)
                
                # Contribution marginale totale
                total_contribution = indirect_mean_bias + indirect_structural_bias + interaction_effect
                results['Total_Marginal_Contribution'].append(total_contribution)
            
            return results

        ###########################################################################
        # Group‑weighted performance metrics
        ###########################################################################

        def group_weighted_r2(df, y_col, y_pred_col, group_col):
            """Compute the group‑weighted R².

            The metric is defined as

            .. math::
                R^2_W = \sum_s p_s \left[ 1 - \frac{\operatorname{Var}(Y - \hat{Y} \mid S=s)}{\operatorname{Var}(Y \mid S=s)} \right],

            where :math:`p_s` is the prevalence of group *s*.

            Parameters
            ----------
            df : pandas.DataFrame
                Dataset that includes the true and predicted targets and the group label.
            y_col : str
                Column name of the true target variable.
            y_pred_col : str
                Column name of the predicted target variable.
            group_col : str
                Column name of the sensitive attribute.

            Returns
            -------
            float
                The group‑weighted coefficient of determination (higher is better; 1 = perfect).
            """
            grouped = df.groupby(group_col)
            n_tot = len(df)

            r2_weighted = 0.0
            for s, grp in grouped:
                p_s = len(grp) / n_tot  # prevalence of group s
                var_y = grp[y_col].var(ddof=1)  # sample variance of Y within group
                var_res = (grp[y_col] - grp[y_pred_col]).var(ddof=1)

                # if var_y > 0:
                r2_s = 1 - (var_res / var_y)
                # else:
                    # Undefined variance ⇒ no variation to explain.
                    # r2_s = 1.0

                r2_weighted += p_s * r2_s

            return r2_weighted

        ###########################################################################
        # Generic loss functions
        ###########################################################################

        def mse_loss(y_true, y_pred):
            """Mean‑Squared‑Error (MSE).

            Parameters
            ----------
            y_true : numpy.ndarray | pandas.Series
                Ground‑truth target values.
            y_pred : numpy.ndarray | pandas.Series
                Predicted target values.

            Returns
            -------
            float
                Average of squared residuals.
            """
            return np.mean((y_true - y_pred) ** 2)

        # Duplicate maintained for backward compatibility; consider deprecating.
        def mean_squared_error(y_true, y_pred):
            """Alias for :pycode:`mse_loss`. Retained for API compatibility.

            See Also
            --------
            mse_loss : Preferred, shorter name.
            """
            return np.mean((y_true - y_pred) ** 2)
