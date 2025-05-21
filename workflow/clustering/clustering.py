import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from recs_preprocessing import Codebook, df_computed


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
cols_discarded = cb.__codebook[cb.__codebook['Preserved'] <= .01]['Variable'].values.tolist()
cols_continuous = cb.__codebook[(cb.__codebook['Preserved'] >= 0.99) & ((cb.__codebook['Notes'] == 'Numerical') & (cb.__codebook['NaiveScale'] != 1))]['Variable'].values.tolist()
cols_scaled = cb.__codebook[(cb.__codebook['Preserved'] >= 0.99) & ((cb.__codebook['Notes'] == 'Numerical') & (cb.__codebook['NaiveScale'] == 1))]['Variable'].values.tolist() + ['climate_code_heat']
cols_categorical = cb.__codebook[(cb.__codebook['Preserved'] >= 0.99) & (cb.__codebook['Notes'] == 'Categorical')]['Variable'].values.tolist() + ['climate_code_humidity']

# These are derived/computed columns
cols_computed = ['total_sqm_en', 'total_kwh',
    'total_kwh_sph', 'total_kwh_appliances', 'total_kwh_dhw',
    'total_kwh_lighting', 'total_kwh_electronics', 'total_kwh_vent',
    'total_kwh_col', 'eui_kwh_sqm', 'heating_eui_kwh_sqm',
    'appliances_eui_kwh_sqm', 'dhw_eui_kwh_sqm', 'lighting_eui_kwh_sqm',
    'electronics_eui_kwh_sqm', 'vent_eui_kwh_sqm', 'cooling_eui_kwh_sqm']

def sep_cols(data):
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

def select(data, by={}):
    df_selected = data.copy()
    for key, value in by.items():
        if type(value) == list:
            df_selected = df_selected[df_selected[key].isin(value)]
        else:
            df_selected = df_selected[df_selected[key].isin([value])]
    return df_selected

from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import scale, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN

from sklearn.metrics import silhouette_score


from sklearn.manifold import TSNE

from sklearn.mixture import GaussianMixture

def preprocess(df, cols_cat=np.array([]), cols_con=np.array([]), cols_scl=np.array([]), OneHot=True, con_scaler='standard', scl_scaler='minmax'):
    
    data=df.copy()

    categorical_indices = [data.columns.get_loc(col) for col in cols_cat]
    scaler_features = cols_scl
    if OneHot:
        categorical_features = cols_cat
    else:
        categorical_features = np.array([])
        scaler_features = np.concatenate([cols_scl, cols_cat])

    scalers = {'standard': lambda: StandardScaler(), 'minmax': lambda: MinMaxScaler(), 'robust': lambda: RobustScaler() }
    if con_scaler in scalers.keys():
        con_scaler = scalers[con_scaler]()
    else:
        con_scaler = StandardScaler()
    if scl_scaler in scalers.keys():
        scl_scaler = scalers[scl_scaler]()
    else:
        scl_scaler = MinMaxScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', con_scaler, cols_con),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('scale', scl_scaler, scaler_features)
        ], verbose_feature_names_out=False
    ).set_output(transform='pandas')
    preprocessed_data = preprocessor.fit_transform(data)
    #preprocessed_data = preprocessed_data[data.columns]
    return preprocessed_data, categorical_indices, preprocessor

def sep_cols_and_preprocess(df, OneHot=True, con_scaler='standard', scl_scaler='minmax', **kwargs):
    ccon, cscl, ccat, cdsc = sep_cols(df)
    if 'cols_cat' in kwargs:
        ccat = kwargs['cols_cat']
    if 'cols_con' in kwargs:
        ccon = kwargs['cols_con']
    if 'cols_scl' in kwargs:
        cscl = kwargs['cols_scl']

    return preprocess(df, cols_cat=ccat, cols_con=ccon, cols_scl=cscl,
                      OneHot=OneHot, con_scaler=con_scaler, scl_scaler=scl_scaler)
# Just a basic class skeleton to work on
# Each clustering methods shall have their own way of
# constructing a Metrics class instance
from sklearn.metrics import silhouette_score, davies_bouldin_score
#from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import calinski_harabasz_score
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
        self.explained_variance = None
        self.total_energy_represented = 0
        self.total_energy_actual = 1
        self.energy_validator_column = []
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
        self.explained_variance = calculate_explained_variance(self.data, self.labels)

def calculate_explained_variance(data, labels):
    """
    Calculate the Explained Variance for clustered data.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features)
        The input data.
    - labels: np.ndarray, shape (n_samples,)
        Cluster labels for each point.

    Returns:
    - explained_variance: float
        The proportion of total variance explained by the clustering.
    """
    # Compute the overall mean of the data
    overall_mean = np.mean(data, axis=0)
    
    # Total Variance: Sum of squared distances from the overall mean
    total_variance = np.sum((data - overall_mean) ** 2)
    
    # Initialize Within-Cluster Variance
    within_cluster_variance = 0.0
    
    # Iterate through each unique cluster
    for cluster in np.unique(labels):
        # Select data points belonging to the current cluster
        cluster_data = data[labels == cluster]
        
        # Compute the centroid of the current cluster
        cluster_centroid = np.mean(cluster_data, axis=0)
        
        # Sum of squared distances from the cluster centroid
        cluster_variance = np.sum((cluster_data - cluster_centroid) ** 2)
        
        # Accumulate within-cluster variance
        within_cluster_variance += cluster_variance
    
    # Calculate Explained Variance
    explained_variance = (total_variance - within_cluster_variance) / total_variance
    
    return np.mean(explained_variance)


def GMM(data, n=2, seed=0):
    np.random.seed(seed)
    gmm = GaussianMixture(n_components=n, covariance_type='full')
    gmm.fit(data)
    return gmm


def kmeans(data, weights=None, n=2, random_state=42):
    kmeans = KMeans(n_clusters=n, random_state=random_state)
    if weights is None:
        kmeans.fit(data)
    else:
        kmeans.fit(data, sample_weight=weights)
    return kmeans
def kmeans_bic_aic(kmeans, X):
    """
    Calculate BIC and AIC for a fitted KMeans model.

    Parameters:
    - kmeans: Fitted KMeans model.
    - X: Data array of shape (n_samples, n_features).

    Returns:
    - bic: Bayesian Information Criterion.
    - aic: Akaike Information Criterion.
    """
    # Number of clusters and parameters
    n_clusters = kmeans.n_clusters
    n_features = X.shape[1]
    n_samples = X.shape[0]

    # Calculate the log-likelihood
    # Assuming spherical Gaussian distribution
    # Sum of squared distances to cluster centers
    sse = kmeans.inertia_
    
    # Estimate variance
    variance = sse / (n_samples - n_clusters)
    
    # Number of parameters: cluster centers and variance
    # Each center has n_features parameters
    # Plus one parameter for variance
    n_parameters = n_clusters * n_features + 1

    # Calculate log-likelihood
    log_likelihood = -0.5 * n_samples * (n_features * np.log(2 * np.pi * variance) + 1)

    # Calculate AIC and BIC
    aic = 2 * n_parameters - 2 * log_likelihood
    bic = np.log(n_samples) * n_parameters - 2 * log_likelihood

    return bic, aic


def kmeans_wcss(model):
    return model.inertia_
def kmeans_sscore(data, model):
    return silhouette_score(data, model.labels_)


from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
def hierarchical_clustering(data):
    Z = linkage(data, method='ward', metric='euclidean')
    return Z
# dendrogram(Z)
def hierarchical_labels(link, n=3):
    return cut_tree(link, n).flatten()
def hierarchical_kn(data, n=2, n_neighbors=10):
    connectivity = kneighbors_graph(data, n_neighbors=n_neighbors, include_self=False)
    ward = AgglomerativeClustering(
        n_clusters=n, connectivity=connectivity, linkage="ward"
    ).fit(data)
    return ward
def evaluate_kmeans_simple(data, n_min=2, n_max=20):
    n_max = min(n_max, data.shape[0] - 1)
    n_min = max(2, n_min)
    metrics = []
    range_n = range(n_min, n_max + 1)

    for n in range_n:
        km = kmeans(data, weights=recs_raw['NWEIGHT'].loc[data.index], n=n, random_state=42)
        metric = Metrics(model=km, data=data, labels=km.labels_, update=True)
        bic, aic = kmeans_bic_aic(km, data)
        metric.BIC = bic
        metric.AIC = aic
        
        centroids = km.cluster_centers_
        counts = np.unique_counts(km.labels_).counts

        metrics.append(metric)
        
    return range_n, metrics
def evaluate_kmeans(data, n_min=2, n_max=20, energy_validator_column='total_kwh'):
    n_max = min(n_max, data.shape[0] - 1)
    n_min = max(2, n_min)
    metrics = []
    range_n = range(n_min, n_max + 1)
    if type(energy_validator_column) == str:
        energy_validator_column = df_computed[energy_validator_column]
        total_energy_actual = np.sum(df_computed['NWEIGHT'].loc[data.index] * energy_validator_column.loc[data.index])
    else:
        total_energy_actual = np.sum(energy_validator_column.loc[data.index] * df_computed['NWEIGHT'].loc[data.index])
    for n in range_n:
        km = kmeans(data, weights=recs_raw['NWEIGHT'].loc[data.index], n=n, random_state=42)
        metric = Metrics(model=km, data=data, labels=km.labels_, update=True)
        bic, aic = kmeans_bic_aic(km, data)
        metric.BIC = bic
        metric.AIC = aic
        
        centroids = km.cluster_centers_
        counts = np.unique_counts(km.labels_).counts

        total_energy = 0
        for i in range(len(centroids)):
            centroid = centroids[i]
            distances = np.linalg.norm(data - centroid, axis=1)
            closest_sample_index = data.index[np.argmin(distances)]
            total_energy = total_energy + (energy_validator_column.loc[closest_sample_index] * df_computed['NWEIGHT'].loc[closest_sample_index]) * counts[i]
        
        metric.total_energy_represented = total_energy  / total_energy_actual
        metric.energy_validator_column = energy_validator_column
        metric.total_energy_actual = total_energy_actual
        metrics.append(metric)
        
    return range_n, metrics
def evaluate_gmm(data, n_min=2, n_max=20):
    n_max = min(n_max, data.shape[0] - 1)
    n_min = max(2, n_min)
    metrics = []
    range_n = range(n_min, n_max + 1)
    for n in range_n:
        gmm = GMM(data, n=n)
        metric = Metrics(model=gmm, data=data, labels=gmm.predict(data), update=True)
        metric.update_metrics()
        metric.AIC = gmm.aic(data)
        metric.BIC = gmm.bic(data)
        metrics.append(metric)
    return range_n, metrics
def evaluate_hierarchical(data, n_min=2, n_max=20):
    n_max = min(n_max, data.shape[0] - 1)
    n_min = max(2, n_min)
    metrics = []
    range_n = range(n_min, n_max + 1)
    for n in range_n:
        hc = hierarchical_clustering(data)
        metric = Metrics(model=hc, data=data, labels=hierarchical_labels(hc, n=n), update=True)
        metrics.append(metric)
    return range_n, metrics
def evaluate_hierarchical_kn(data, k=10, n_min=2, n_max=20):
    n_max = min(n_max, data.shape[0] - 1)
    n_min = max(2, n_min)
    metrics = []
    range_n = range(n_min, n_max + 1)
    for n in range_n:
        hc = hierarchical_kn(data, n=n, n_neighbors=k)
        metric = Metrics(model=hc, data=data, labels=hc.labels_, update=True)
        metrics.append(metric)
    return range_n, metrics


def get_total_energy_represented(data, km, actual, vali):
    centroids = km.cluster_centers_
    counts = np.unique_counts(km.labels_).counts

    total_energy = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        distances = np.linalg.norm(data - centroid, axis=1)
        closest_sample_index = data.index[np.argmin(distances)]
        total_energy = total_energy + (vali.loc[closest_sample_index] * df_computed['NWEIGHT'].loc[closest_sample_index]) * counts[i]
    return total_energy  / actual


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
def plot_evaluation_extended(range_n, metrics_dict):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), layout='tight')
    for key, value in metrics_dict.items():
        axes[0][0].plot(range_n, [m.silhouette for m in value], label=key, marker='o', markersize=3)
        axes[0][1].plot(range_n, [m.davies_bouldin for m in value], label=key, marker='o', markersize=3)
        axes[0][2].plot(range_n, [m.calinski_harabasz for m in value], label=key, marker='o', markersize=3)
        axes[1][0].plot(range_n, [m.RoNoE for m in value], label=key, marker='o', markersize=3)
        axes[1][1].plot(
            np.concat([[0], range_n]),
            np.concat(
                [
                    [get_total_energy_represented(
                        value[0].data,
                        kmeans(
                            value[0].data,
                            weights=recs_raw['NWEIGHT'].loc[value[0].data.index],
                            n=1,
                            random_state=42),
                        value[0].total_energy_actual,
                        value[0].energy_validator_column
                    ) - 1],
                [m.total_energy_represented - 1 for m in value]
                ]
            ),
                        label=key, marker='o', markersize=3)
        bic = np.array([m.BIC for m in value])
        #bic = (bic - np.min(bic)) / (np.max(bic) - np.min(bic)) # normalized BIC
        bic = bic / value[0].data.shape[0]
        #axes[1][2].plot(range_n, bic, label=key, marker='o', markersize=3)
        axes[1][2].plot(
            np.concat([[0], range_n]),
            np.concat(
                [
                    [np.where(np.abs(get_total_energy_represented(
                    value[0].data,
                        kmeans(
                            value[0].data,
                            weights=recs_raw['NWEIGHT'].loc[value[0].data.index],
                            n=1,
                            random_state=42),
                        value[0].total_energy_actual,
                        value[0].energy_validator_column
                    ) - 1)<0.4, 1, 0.1) * 0.6],
                [np.where(np.abs(m.total_energy_represented - 1)<0.4,1, 0.1) * ((m.silhouette + 1) / 2) for m in value]
                ]
            ),
                        label=key, marker='o', markersize=3)
    axes[0][0].set_ylabel('Silhouette Score')
    axes[0][1].set_ylabel('Davies-Bouldin Score')
    axes[0][2].set_ylabel('Calinski-Harabasz Score')
    axes[1][0].set_ylabel('RoNoE')
    axes[1][1].set_ylabel('Energy Consumption Represented')
    axes[1][2].set_ylabel('Normalized BIC')
    for ax in axes.flatten():
        ax.set_xlabel('# of Clusters')
        ax.legend(prop={'size': 8})
        ax.set_xticks(range_n)
        ax.tick_params(axis='x', labelsize=10, rotation=90)
        ax.set_xlim(range_n[0] - 1, range_n[-1]+1)
    return fig, axes
