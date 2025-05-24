import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import seaborn as sns

from sklearn.preprocessing import scale, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.metrics import calinski_harabasz_score

from recs_preprocessing import Codebook, df_computed

# ML

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss

# Get the Set1 colormap
cmap = get_cmap('Set1')

# TODO add a fuzzy search utility
# this reques fuzzywuzzy
# def fuzzy_string_search(key: str, df: pd.DataFrame, col: str = 'label', num_results: int = -1, min_score: int = 75):
#     """
#     Perform a fuzzy string search on a DataFrame column.
    
#     Args:
#         key (str): The search string.
#         df (pd.DataFrame): The DataFrame to search.
#         col (str): The column to search in. Default is 'label'.
#         num_results (int): The number of results to return. Default is -1, which returns all results.
#         min_score (int): The minimum score for a match (0-100). Default is 75.
    
#     """
#     from fuzzywuzzy import fuzz

#     df_copy = df.copy()
#     df_copy['ratio'] = df_copy[col].apply(lambda x: fuzz.partial_ratio(key.lower(), str(x).lower()))
#     df_copy = df_copy[df_copy['ratio'] >= min_score]
#     if num_results < 0:
#         num_results = len(df_copy)
#     df_copy.sort_values(by=['ratio', col], ascending=False, inplace=True)
#     return df_copy.head(num_results).loc[:, df.columns]
cb = Codebook()
cols_discarded = cb.codebook[cb.codebook['Preserved'] <= .01]['Variable'].values.tolist()
cols_continuous = cb.codebook[(cb.codebook['Preserved'] >= 0.99) & ((cb.codebook['Notes'] == 'Numerical') & (cb.codebook['NaiveScale'] != 1))]['Variable'].values.tolist()
cols_scaled = cb.codebook[(cb.codebook['Preserved'] >= 0.99) & ((cb.codebook['Notes'] == 'Numerical') & (cb.codebook['NaiveScale'] == 1))]['Variable'].values.tolist() + ['climate_code_heat']
cols_categorical = cb.codebook[(cb.codebook['Preserved'] >= 0.99) & (cb.codebook['Notes'] == 'Categorical')]['Variable'].values.tolist() + ['climate_code_humidity']

# These are derived/computed columns
cols_computed = ['total_sqm_en', 'total_kwh',
    'total_kwh_sph', 'total_kwh_appliances', 'total_kwh_dhw',
    'total_kwh_lighting', 'total_kwh_electronics', 'total_kwh_vent',
    'total_kwh_col', 'eui_kwh_sqm', 'heating_eui_kwh_sqm',
    'appliances_eui_kwh_sqm', 'dhw_eui_kwh_sqm', 'lighting_eui_kwh_sqm',
    'electronics_eui_kwh_sqm', 'vent_eui_kwh_sqm', 'cooling_eui_kwh_sqm']

def separate_columns(data):
    """
    Separates columns of a dataset into predefined categories based on their membership 
    in specific column lists: `cols_continuous`, `cols_computed`, `cols_scaled`, 
    `cols_categorical`, and `cols_discarded`.
    Args:
        data (pd.DataFrame or pd.Series or iterable): The input dataset. It can be a 
            pandas DataFrame, Series, or any iterable containing column names.
    Returns:
        tuple: A tuple containing four arrays:
            - Columns that belong to `cols_continuous` or `cols_computed`.
            - Columns that belong to `cols_scaled`.
            - Columns that belong to `cols_categorical`.
            - Columns that belong to `cols_discarded`.
    Raises:
        NameError: If any of the required column lists (`cols_continuous`, `cols_computed`, 
            `cols_scaled`, `cols_categorical`, `cols_discarded`) are not defined in the 
            current scope.
    """

    if type(data) is pd.DataFrame:
        columns = data.columns
    elif type(data) is pd.Series:
        columns = data
    else:
        columns = pd.Series(data)
    return columns[
    columns.isin(cols_continuous + cols_computed)].values, columns[
    columns.isin(cols_scaled)].values, columns[
    columns.isin(cols_categorical)].values, columns[
    columns.isin(cols_discarded)].values

def select_subset(data, by={}):
    """
    Filters a DataFrame based on specified criteria.
    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame to be filtered.
    by : dict, optional
        A dictionary where keys are column names and values are the filtering criteria.
        The values can either be a single value or a list of values. Rows in the DataFrame
        are selected if the values in the specified columns match the criteria.
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame containing only the rows that match the filtering criteria.
    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': ['x', 'y', 'z', 'x']
    ... })
    >>> select_subset(data, by={'A': [1, 3], 'B': 'x'})
       A  B
    0  1  x
    """

    df_selected = data.copy()
    for key, value in by.items():
        if type(value) == list:
            df_selected = df_selected[df_selected[key].isin(value)]
        else:
            df_selected = df_selected[df_selected[key].isin([value])]
    return df_selected


def preprocess_columns(
        df,
        categorical_columns=np.array([]),
        continuous_columns=np.array([]),
        scaled_columns=np.array([]),
        one_hot=True,
        continuous_feature_scaler='standard',
        scale_feature_scaler='minmax'):
    """
    Preprocesses the columns of a DataFrame by applying scaling, one-hot encoding, 
    and other transformations based on the specified parameters.
    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        categorical_columns (np.array, optional): Array of column names to treat as categorical features. 
            Defaults to an empty array.
        continuous_columns (np.array, optional): Array of column names to treat as continuous features. 
            Defaults to an empty array.
        scaled_columns (np.array, optional): Array of column names to apply additional scaling. 
            Defaults to an empty array.
        one_hot (bool, optional): Whether to apply one-hot encoding to categorical features. 
            Defaults to True.
        continuous_feature_scaler (str, optional): The scaler to use for continuous features. 
            Options are 'standard', 'minmax', or 'robust'. Defaults to 'standard'.
        scale_feature_scaler (str, optional): The scaler to use for scaled features. 
            Options are 'standard', 'minmax', or 'robust'. Defaults to 'minmax'.
    Returns:
        tuple: A tuple containing:
            - preprocessed_data (pd.DataFrame): The transformed DataFrame after preprocessing.
            - categorical_indices (list): List of indices corresponding to the categorical columns.
            - preprocessor (ColumnTransformer): The fitted ColumnTransformer object used for preprocessing.
    """

    data=df.copy()

    categorical_indices = [data.columns.get_loc(col) for col in categorical_columns]
    scaler_features = scaled_columns
    if one_hot:
        categorical_features = categorical_columns
    else:
        categorical_features = np.array([])
        scaler_features = np.concatenate([scaled_columns, categorical_columns])

    scalers = {'standard': lambda: StandardScaler(), 'minmax': lambda: MinMaxScaler(), 'robust': lambda: RobustScaler() }
    if continuous_feature_scaler in scalers.keys():
        continuous_feature_scaler = scalers[continuous_feature_scaler]()
    else:
        continuous_feature_scaler = StandardScaler()
    if scale_feature_scaler in scalers.keys():
        scale_feature_scaler = scalers[scale_feature_scaler]()
    else:
        scale_feature_scaler = MinMaxScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_feature_scaler, continuous_columns),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('scale', scale_feature_scaler, scaler_features)
        ], verbose_feature_names_out=False
    ).set_output(transform='pandas')
    preprocessed_data = preprocessor.fit_transform(data)

    return preprocessed_data, categorical_indices, preprocessor



def automatic_preprocess_columns(df, OneHot=True, con_scaler='standard', scl_scaler='minmax', **kwargs):
    ccon, cscl, ccat, cdsc = separate_columns(df)
    if 'cols_cat' in kwargs:
        ccat = kwargs['cols_cat']
    if 'cols_con' in kwargs:
        ccon = kwargs['cols_con']
    if 'cols_scl' in kwargs:
        cscl = kwargs['cols_scl']

    return preprocess_columns(df, categorical_columns=ccat, continuous_columns=ccon, scaled_columns=cscl,
                      one_hot=OneHot, continuous_feature_scaler=con_scaler, scale_feature_scaler=scl_scaler)


class Metrics:
    def __init__(self, **kwargs):
        self.predictors = None
        self.predictors_performance = None
        self.pipelines = None
        self.model = None
        self.labels = None
        self.data = None
        self.AIC = None #lower
        self.BIC = None #lower
        self.silhouette = None #higher
        self.davies_bouldin = None #lower
        #self.homogeneity = None #higher
        #self.completeness = None #higher
        #self.v_measure = None #higher
        self.calinski_harabasz = None #higher
        self.RoNoE = None # Ratio of the number of elements (Tardioli et al 2018) #lower
        # self.explained_variance = None
        # self.total_energy_represented = 0
        # self.total_energy_actual = 1
        # self.energy_validator_column = []
        self.update = False
        self.__dict__.update(kwargs)
        if self.update:
            self.update_metrics()
    def update_metrics(self):
        self.silhouette =  silhouette_score(self.data, self.labels)
        self.davies_bouldin = davies_bouldin_score(self.data, self.labels)
        self.calinski_harabasz = calinski_harabasz_score(self.data, self.labels)
        counts = [np.sum(self.labels == label) for label in np.unique(self.labels)]
        self.RoNoE = np.min(counts) / np.max(counts)
        # self.explained_variance = calculate_explained_variance(self.data, self.labels)


def perform_gmm(data, n=2, seed=0, covariance_type='full'):
    np.random.seed(seed)
    gmm = GaussianMixture(n_components=n, covariance_type=covariance_type)
    gmm.fit(data)
    return gmm


def evaluate_gmm(data, n_min=2, n_max=20):
    n_max = min(n_max, data.shape[0] - 1)
    n_min = max(2, n_min)
    metrics = []
    range_n = range(n_min, n_max + 1)
    for n in range_n:
        gmm = perform_gmm(data, n=n)
        metric = Metrics(model=gmm, data=data, labels=gmm.predict(data), update=True)
        metric.update_metrics()
        metric.AIC = gmm.aic(data)
        metric.BIC = gmm.bic(data)
        metrics.append(metric)
    return range_n, metrics






def plot_evaluation(range_n, metrics_dict):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), layout='tight')
    for key, value in metrics_dict.items():
        axes[0].plot(range_n, [m.silhouette for m in value], label=key, marker='o', markersize=3)
        axes[1].plot(range_n, [m.davies_bouldin for m in value], label=key, marker='o', markersize=3)
        axes[2].plot(range_n, [m.calinski_harabasz for m in value], label=key, marker='o', markersize=3)
        axes[3].plot(range_n, [m.RoNoE for m in value], label=key, marker='o', markersize=3)
        #axes[4].plot(range_n, [m.explained_variance for m in value], label=key, marker='o', markersize=3)
    axes[0].set_ylabel('Silhouette Score')
    axes[1].set_ylabel('Davies-Bouldin Score')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[3].set_ylabel('RoNoE')
    for ax in axes:
        ax.set_xlabel('# of Clusters')
        ax.legend(prop={'size': 8})
        ax.set_xticks(range_n)
        ax.tick_params(axis='x', labelsize=10, rotation=90)
        ax.set_xlim(range_n[0] - 1, range_n[-1]+1)
    return fig, axes



def cluster_subset(ccat, ccon, subset, cluster_method, fig, axes, axpos=0):
    attributes = ccat + ccon
    subset = select_subset(df_computed, by=subset)
    indices = subset.index
    da = df_computed.loc[indices][attributes]

    interval, metrics = cluster_method(automatic_preprocess_columns(da, cols_cat=ccat, cols_con=ccon, cols_scl=[])[0])
    #fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    bics = [m.BIC for m in metrics]
    bmax = np.max(bics)
    bmin = np.min(bics)

    axes[axpos].plot(interval, bics, marker='o', markersize=5, label='BIC')
    axes[axpos].set_xticks(interval, interval)
    axes[axpos].set_xticklabels(interval, rotation=90, fontsize=8)
    axes[axpos].set_ylabel('BIC')
    axes[axpos].set_xlabel('Number of Clusters')
    axes[axpos].grid(True)

    axtwin = axes[axpos].twinx()
    axtwin.bar(interval, [evaluate_mls(m, da.index, plot=False)[0]['Accuracy'].max() for m in metrics], .4, alpha=0.5, label='Assignment Accuracy')
    axtwin.set_ylim(0, 1)
    handles, labels = axes[axpos].get_legend_handles_labels()
    handles2, labels2 = axtwin.get_legend_handles_labels()
    axes[axpos].legend(handles + handles2, labels + labels2, loc='upper right', prop={'size': 8})
    return da, metrics



excol = ['TOTSQFT_EN', 'TYPEHUQ', 'urban_grouped', 'acequipm_pub_grouped', 'FUELHEAT', 'EQUIPM', 'YEARMADERANGE', 'CELLAR', 'WALLTYPE', 'BASEFIN', 'num_u65', 'NUMADULT2',]
def evaluate_mls(metric, indices, plot=True): # determine optimal cluster count to perform evaluation
    # Define target and features
    y = metric.labels
    X = df_computed[excol].loc[indices]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define columns for preprocessing
    numerical_cols = ['TOTSQFT_EN', 'YEARMADERANGE', 'num_u65', 'NUMADULT2']
    categorical_cols = ['TYPEHUQ', 'urban_grouped', 'acequipm_pub_grouped', 'FUELHEAT', 'EQUIPM', 'CELLAR', 'WALLTYPE', 'BASEFIN']
    
    # Preprocessor for numerical and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    
    # Define models to evaluate
    if plot:
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Support Vector Machine': SVC(probability=True, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
    else:
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Support Vector Machine': SVC(probability=True, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
    
    # Evaluate each model
    results = []
    metric.pipelines = {}
    for model_name, model in models.items():
        # Create a pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Get predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        log_loss_val = log_loss(y_test, y_pred_proba, labels=np.arange(y_pred_proba.shape[1]))
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Log Loss': log_loss_val
        })
        metric.pipelines[model_name] = pipeline
    
    # Sort results by accuracy
    results_sorted = pd.DataFrame(sorted(results, key=lambda x: x['Accuracy'], reverse=True))
    metric.predictors = models
    
    x = np.arange(results_sorted.shape[0])
    accuracies = results_sorted['Accuracy']
    models = results_sorted['Model']
    log_losses = results_sorted['Log Loss']
    width = 0.5
    if plot:
        # Create a dual-axis plot
        fig, ax1 = plt.subplots(figsize=(6, 6))
        
        # Bar plot for Accuracy
        bars = ax1.bar(x, accuracies, width, label='Accuracy',)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.tick_params(axis='y', )
        
        # Add annotations for Accuracy bars
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                     ha='center', va='bottom', fontsize=9)
        
        # Line plot for Log Loss (second y-axis)
        ax2 = ax1.twinx()
        ax2.plot(x, log_losses, color=cmap(4/9), marker='o', label='Log Loss', linewidth=2)
        ax2.set_ylabel('Log Loss', )
        ax2.tick_params(axis='y', )
        
        # Add annotations for Log Loss points
        for i, loss in enumerate(log_losses):
            ax2.text(x[i], loss, f'{loss:.2f}',  ha='center', va='bottom', fontsize=9)
        
        # Add title and legend
        fig.suptitle('Cluster Prediction Model Accuracy and Log Loss')
        fig.tight_layout()  # Adjust layout to fit all labels
        
        # Show the plot
        plt.show()

    metric.predictors_performance = results_sorted
    return results_sorted, models


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

def evaluate_mls_kfold(metric, indices, plot=True, n_splits=5):  # use k-fold instead of split
    # Define target and features
    y = metric.labels
    X = df_computed[excol].loc[indices]

    # Define columns for preprocessing
    numerical_cols = ['TOTSQFT_EN', 'YEARMADERANGE', 'num_u65', 'NUMADULT2']
    categorical_cols = ['TYPEHUQ', 'urban_grouped', 'acequipm_pub_grouped', 'FUELHEAT', 'EQUIPM', 'CELLAR', 'WALLTYPE', 'BASEFIN']

    # Preprocessor for numerical and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Define models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'log_loss': make_scorer(log_loss, needs_proba=True, greater_is_better=False)
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    metric.pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False)
        
        results.append({
            'Model': model_name,
            'Accuracy': np.mean(scores['test_accuracy']),
            'Log Loss': -np.mean(scores['test_log_loss'])  # negate because we set greater_is_better=False
        })
        
        # Fit final model to all data for reuse
        pipeline.fit(X, y)
        metric.pipelines[model_name] = pipeline

    results_sorted = pd.DataFrame(sorted(results, key=lambda x: x['Accuracy'], reverse=True))
    metric.predictors = models

    # Plotting
    if plot:
        x = np.arange(results_sorted.shape[0])
        accuracies = results_sorted['Accuracy']
        models = results_sorted['Model']
        log_losses = results_sorted['Log Loss']
        width = 0.5

        fig, ax1 = plt.subplots(figsize=(6, 6))
        bars = ax1.bar(x, accuracies, width, label='Accuracy')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.tick_params(axis='y')

        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                     ha='center', va='bottom', fontsize=9)

        ax2 = ax1.twinx()
        ax2.plot(x, log_losses, color=cmap(4/9), marker='o', label='Log Loss', linewidth=2)
        ax2.set_ylabel('Log Loss')
        ax2.tick_params(axis='y')

        for i, loss in enumerate(log_losses):
            ax2.text(x[i], loss, f'{loss:.2f}', ha='center', va='bottom', fontsize=9)

        fig.suptitle('Cluster Prediction Model Accuracy and Log Loss (Stratified K-Fold)')
        fig.tight_layout()
        plt.show()

    metric.predictors_performance = results_sorted
    return results_sorted, models


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

def evaluate_catboost(metric, indices, plot=True):
    y = metric.labels
    X = df_computed[excol].loc[indices]

    # Categorical feature indices
    cat_features = [X.columns.get_loc(col) for col in [
        'TYPEHUQ', 'urban_grouped', 'acequipm_pub_grouped', 
        'FUELHEAT', 'EQUIPM', 'CELLAR', 'WALLTYPE', 'BASEFIN'
    ]]

    param_variants = [
        {'iterations': 100, 'learning_rate': 0.03, 'depth': 4},
        {'iterations': 200, 'learning_rate': 0.05, 'depth': 6},
        {'iterations': 300, 'learning_rate': 0.1,  'depth': 8}
    ]

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for params in param_variants:
        accs = []
        losses = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = CatBoostClassifier(**params, cat_features=cat_features, verbose=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            losses.append(log_loss(y_test, y_pred_proba, labels=np.unique(y)))

        results.append({
            'Params': params,
            'Accuracy': np.mean(accs),
            'Log Loss': np.mean(losses)
        })

    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

    if plot:
        fig, ax1 = plt.subplots(figsize=(6, 6))
        x = np.arange(len(results_df))
        accs = results_df['Accuracy']
        losses = results_df['Log Loss']
        labels = [str(p) for p in results_df['Params']]

        bars = ax1.bar(x, accs, width=0.5, label='Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1)

        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}',
                     ha='center', va='bottom', fontsize=8)

        ax2 = ax1.twinx()
        ax2.plot(x, losses, color=cmap(4/9), marker='o', label='Log Loss')
        ax2.set_ylabel('Log Loss')
        for i, loss in enumerate(losses):
            ax2.text(x[i], loss, f'{loss:.2f}', ha='center', va='bottom', fontsize=8)

        fig.suptitle('CatBoost Performance Across Param Variants')
        fig.tight_layout()
        plt.show()

    metric.predictors_performance = results_df
    return results_df


from sklearn.model_selection import GridSearchCV

def tune_catboost(metric, indices):
    y = metric.labels
    X = df_computed[excol].loc[indices]

    cat_features = [X.columns.get_loc(col) for col in [
        'TYPEHUQ', 'urban_grouped', 'acequipm_pub_grouped',
        'FUELHEAT', 'EQUIPM', 'CELLAR', 'WALLTYPE', 'BASEFIN'
    ]]

    model = CatBoostClassifier(cat_features=cat_features, verbose=0)

    param_grid = {
        'iterations': [100, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.03, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=1)
    clf.fit(X, y)

    best_model = clf.best_estimator_
    best_score = clf.best_score_
    best_params = clf.best_params_

    metric.pipelines = {'CatBoost': best_model}
    metric.predictors_performance = pd.DataFrame([{
        'Model': 'CatBoost (Tuned)',
        'Accuracy': best_score,
        'Params': best_params
    }])

    return best_model, best_score, best_params


def categorical_numeric_bar(df_plot, categorical, numeric):
    # Filter numeric data to remove outliers (values > 97th quantile)
    q97 = df_plot[numeric].quantile(0.97)
    filtered_df = df_plot[df_plot[numeric] <= q97]
    
    # Define custom bin intervals
    custom_bins = np.arange(0, q97, q97 / 10)  # Bins from 0 to 100 with intervals of 5
    filtered_df['bin'] = pd.cut(filtered_df[numeric], bins=custom_bins, include_lowest=True)
    
    # Group by bins and categorical feature
    grouped = filtered_df.groupby(['bin', categorical]).size().unstack(fill_value=0)
    
    # Plot the stacked bar chart
    plt.figure(figsize=(5, 5), layout='tight')
    grouped.plot(kind='bar', stacked=True, figsize=(6, 6),)
    
    plt.title('Stacked Bar Chart of Cluster Labels vs {}'.format(numeric))
    plt.xlabel(numeric)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
def categorical_categorical_bar(df_plot, categorical1: str, categorical2: str):
    """
    Plots a stacked bar chart for the relationship between two categorical features in the dataframe.
    
    Parameters:
        df_plot (pd.DataFrame): The dataframe containing the data.
        categorical1 (str): The first categorical feature for grouping (x-axis categories).
        categorical2 (str): The second categorical feature for stacking (y-axis categories).
    """
    # Group by the two categorical features
    grouped = df_plot.groupby([categorical1, categorical2]).size().unstack(fill_value=0)
    
    # Plot the stacked bar chart
    plt.figure(figsize=(5, 5), layout='tight')
    grouped.plot(kind='bar', stacked=True, figsize=(6, 6))
    plt.title(f'Stacked Bar Chart: {categorical1} vs. {categorical2}')
    plt.xlabel(categorical1)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(cb.get_legend(categorical2).set_index('code').loc[[str(c) for c in grouped.columns]]['description'], loc='best')


def auto_detect_cat(df):
    cols = df.columns
    ccat = []
    ccon = []
    for c in cols:
        if c in cols_categorical:
            ccat.append(c)
        else:
            if df[c].dtype == 'int64':
                if 99 in df[c].values:
                    ccat.append(c)
                elif -1 in df[c].values:
                    ccat.append(c)
                elif -2 in df[c].values:
                    ccat.append(c)
                elif df[c].unique().shape[0] < 4:
                    ccat.append(c)
                else:
                    ccon.append(c)
            else:
                ccon.append(c)
    return ccat, ccon


def plot_cluster_result_bars(da, metric, ccon=None, ccat=None):
    cccat, cccon = auto_detect_cat(da)
    if ccon is None:
        ccon = cccon
    if ccat is None:
        ccat = cccat
        
    da_clustered = df_computed.loc[da.index]
    da_clustered['label'] = metric.labels
    for c in ccon:
        categorical_numeric_bar(da_clustered, 'label', c)
    for c in ccat:
        categorical_categorical_bar(da_clustered, 'label', c)




def continuous_feature_kde(df_plot, feature: str, cluster_metric, Q: float = 0.99):
    """
    Plots KDEs for a continuous feature, segmented by clusters, with mean and variance displayed.
    
    Parameters:
        df_plot (pd.DataFrame): The dataframe containing the data.
        feature (str): The continuous feature for the X-axis.
        cluster_label (str): The categorical feature representing cluster labels.
        Q (float): The quantile to cap the data (default is 0.99).
    """
    df_plot = df_plot.copy()
    df_plot['label'] = cluster_metric.labels
    # Filter the data to cap at the specified quantile
    cap_value = df_plot[feature].quantile(Q)
    df_plot = df_plot[df_plot[feature] <= cap_value]
    cluster_palette = sns.color_palette("husl", n_colors=df_plot['label'].nunique())
    # Create a KDE plot for each cluster
    plt.figure(figsize=(5, 5))
    for i, cluster in enumerate(np.sort(df_plot['label'].unique())):
        cluster_data = df_plot.loc[df_plot['label']  == cluster][feature]
        mean = cluster_data.mean()
        variance = cluster_data.var()
        #sns.kdeplot(cluster_data, label=f'{cluster} (μ={mean:.2f}, σ²={variance:.2f})', fill=False, alpha=0.6)
        # Plot KDE
        kde = sns.kdeplot(
            cluster_data, 
            label=f'{cluster} (μ={mean:.2f}, σ²={variance:.2f})', 
            fill=True, 
            alpha=0.05, 
            color=cluster_palette[i]
        )
        
        # Add vertical line for the mean
        plt.axvline(mean, linestyle='--', color=cluster_palette[i], alpha=0.8)
    
    plt.title(f'{feature} by cluster label')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend(title=f'cluster_label')
    plt.tight_layout()
    
    plt.xlim(0, df_plot[feature].max())

    return plt.gcf(), plt.gca()


def compute_cluster_statistics(df, cluster_col, categorical_cols, numeric_cols):
    """
    Computes the medoid for categorical features and mean for numeric features for each cluster.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing data and cluster labels.
        cluster_col (str): The column name containing cluster labels.
        categorical_cols (list): List of column names labeled as categorical features.
        numeric_cols (list): List of column names labeled as numeric features.
        
    Returns:
        pd.DataFrame: A DataFrame where each row represents a cluster and the columns contain medoids
                      for categorical features and means for numeric features.
    """
    cluster_statistics = []
    
    for cluster in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cluster]
        cluster_stats = {'Cluster': cluster}
        
        # Compute medoid for categorical columns
        for col in categorical_cols:
            # Find the medoid (minimizing distance to other rows in the cluster)
            unique_vals = cluster_data[col].value_counts()
            cluster_stats[col] = unique_vals.idxmax()  # Medoid = most frequent category
        
        # Compute mean for numeric columns
        for col in numeric_cols:
            cluster_stats[col] = cluster_data[col].mean()
        
        cluster_statistics.append(cluster_stats)
    
    # Convert the results into a DataFrame
    stats_df = pd.DataFrame(cluster_statistics)
    return stats_df


def regular_display_df(df):
    df = df.copy()
    def round_nbr(n):
        if (n < 0.01) | (n > 1000):
            return np.format_float_scientific(n, precision=3)
        else:
            return np.round(n, decimals=3)
    def params_round(p):
        p_new = list(p)
        for i in range(len(p_new)):
            p_new[i] = round_nbr(p_new[i])
        return p_new
    for c in df.columns:
        if c in categorical_columns:
            pass
        else:
            if c == "Parameters":
                df[c] = df[c].apply(params_round)
            if df[c].dtype != object:
                df[c] = df[c].apply(round_nbr)
    return df