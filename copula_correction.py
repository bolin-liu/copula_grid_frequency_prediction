from loss_functions import gaussian_loss
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import pickle
import tensorflow as tf
from keras.models import load_model
from loss_functions import correlated_gaussian_loss
from utilities import prepare_covariance_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import openturns as ot

seed = 42
np.random.seed(seed)
ot.RandomGenerator.SetSeed(seed)

# Laden der Daten
day_ahead_features_train = pd.read_pickle(r"..\data\day_ahead_features_train.pkl")
day_ahead_features_test = pd.read_pickle(r"..\data\day_ahead_features_test.pkl")
frequency_train = pd.read_pickle(r"..\data\frequency_train.pkl")
frequency_test = pd.read_pickle(r"..\data\frequency_test.pkl")

day_ahead_features_test = day_ahead_features_test.sort_index()
frequency_test = frequency_test.sort_index()

day_ahead_features_train_np = day_ahead_features_train.to_numpy()
day_ahead_features_test_np = day_ahead_features_test.to_numpy()
frequency_train_np = frequency_train.to_numpy()
frequency_test_np = frequency_test.to_numpy()

inputs_train = day_ahead_features_train_np
inputs_test = day_ahead_features_test_np

scaler = joblib.load(f"../trained_models/scaler.gz")
inputs_train = scaler.transform(inputs_train)
inputs_test = scaler.transform(inputs_test)

outputs_train = frequency_train_np - 50.0
outputs_test = frequency_test_np - 50.0

# Time window for evaluation
time_eval = 3600  # the whole hour

# load models
# Gaussian Based Model
loss = {"loss": gaussian_loss}
model_transformer_gaussian = load_model(f"../trained_models/transformer_gaussian")
model_gru_gaussian = load_model(f"../trained_models/gru_gaussian")
outputs_train_angular_frequency = outputs_train.astype(np.float32) * (2 * np.pi)
predictions_train_gru = model_gru_gaussian.predict(inputs_train)
predictions_train_transformer = model_transformer_gaussian.predict(inputs_train)

# gru independent
mu_train_gru = 2 * np.pi * predictions_train_gru[:, :time_eval]
sigma_train_gru = 2 * np.pi * tf.math.softplus(predictions_train_gru[:, 3600:3600 + time_eval])

# attention independent
mu_train_transformer = 2 * np.pi * predictions_train_transformer[:, :time_eval]
sigma_train_transformer = 2 * np.pi * tf.math.softplus(predictions_train_transformer[:, 3600:3600 + time_eval])

# Point Predictions

outputs_test_angular_frequency = outputs_test.astype(np.float32) * (2 * np.pi)
predictions_test_gru = model_gru_gaussian.predict(inputs_test)
predictions_test_transformer = model_transformer_gaussian.predict(inputs_test)

# GRU independent
mu_test_gru = 2 * np.pi * predictions_test_gru[:, :time_eval]
sigma_test_gru = 2 * np.pi * tf.math.softplus(predictions_test_gru[:, 3600:3600 + time_eval])

# Transformer independent
mu_test_transformer = 2 * np.pi * predictions_test_transformer[:, :time_eval]
sigma_test_transformer = 2 * np.pi * tf.math.softplus(predictions_test_transformer[:, 3600:3600 + time_eval])

# PIML Model independent
piml_path = f"../data/piml.pkl"
with open(save_path, "rb") as file:
    PIML_data = pickle.load(file)

mu_train_PIML = PIML_data["ml_pred_mean_train"]
sigma_train_PIML = PIML_data["ml_pred_stddev_train"]
mu_val_PIML = PIML_data["ml_pred_mean_val"]
sigma_val_PIML = PIML_data["ml_pred_stddev_val"]
mu_train_PIML = np.concatenate([mu_train_PIML, mu_val_PIML])
sigma_train_PIML = np.concatenate([sigma_train_PIML, sigma_val_PIML])
mu_test_PIML = PIML_data["ml_pred_mean_test"]
sigma_test_PIML = PIML_data["ml_pred_stddev_test"]

# KNN Model
with open("knn_predictions.pkl", "rb") as f:
    KNN_data = pickle.load(f)

mu_train_knn = KNN_data["train_predictions"]
mu_test_knn = KNN_data["test_predictions"]

# Daily Profile
save_path_dp = r"dp_predictions.pkl"
with open(save_path_dp, "rb") as file:
    Daily_Profile_data = pickle.load(file)

mu_train_DP = Daily_Profile_data["train_mean"] - 50
sigma_train_DP = Daily_Profile_data["train_std"]

mu_test_DP = Daily_Profile_data["test_mean"] - 50
sigma_test_DP = Daily_Profile_data["test_std"]

# error time series
errors = {}

errors['gru'] = outputs_train_angular_frequency - mu_train_gru
errors['piml'] = np.concatenate([PIML_data["y_train"], PIML_data["y_val"]]) - mu_train_PIML
errors['knn'] = outputs_train_angular_frequency - 2 * np.pi * mu_train_knn
errors['dp'] = outputs_train_angular_frequency - 2 * np.pi * mu_train_DP

# Minutes of precise observation in an hourly interval
errors['gru'] = errors['gru'][:, ::60]
errors['piml'] = errors['piml'][:, ::60]
errors['knn'] = errors['knn'][:, ::60]
errors['dp'] = errors['dp'][:, ::60]


# -----------------------------------------------------------------------------
def compute_characteristics(ts, max_lag=15):
    acf_values = np.array([acf(ts[i], nlags=max_lag, fft=True) for i in range(ts.shape[0])])
    acf_mean = np.mean(acf_values, axis=1)  # Durchschnittliche ACF über die Lags

    stats = {
        "mean": np.mean(ts, axis=1),
        "std": np.std(ts, axis=1),
        "acf_mean": acf_mean
    }

    return pd.DataFrame(stats)


error_statistics = {method: compute_characteristics(errors[method]) for method in errors}

feature_names = list(day_ahead_features_train.columns)

inputs_train_df = pd.DataFrame(inputs_train, columns=feature_names)

# mutual information
mi_results = {}

for method, df in error_statistics.items():
    mi_results[method] = {}

    for target_column in df.columns:
        mi_scores = mutual_info_regression(inputs_train_df, df[target_column])
        normalized_mi_scores = mi_scores / mi_scores.sum()

        mi_results[method][target_column] = pd.DataFrame({
            'Feature': inputs_train_df.columns,
            'Mutual Information': normalized_mi_scores
        })

avg_importance = {}

for method_name, method_results in mi_results.items():

    first_target = list(method_results.keys())[0]
    features = method_results[first_target]['Feature'].values
    all_mi_values = np.zeros((len(features), len(method_results)))
    for col_idx, (target_column, df) in enumerate(method_results.items()):
        for feat_idx, feature in enumerate(df['Feature']):
            all_mi_values[feat_idx, col_idx] = df['Mutual Information'].iloc[feat_idx]

    avg_mi_values = np.mean(all_mi_values, axis=1)

    avg_importance[method_name] = pd.DataFrame({
        'Feature': features,
        'Average Mutual Information': avg_mi_values
    })


def find_optimal_clusters(data, max_k=10):
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    optimal_k = range(2, max_k + 1)[np.argmax(silhouette_scores)]
    return optimal_k


cluster_results = {}

for method, importance_df in avg_importance.items():
    feature_importance = importance_df["Average Mutual Information"].values
    sqrt_weights = np.sqrt(feature_importance)
    weighted_data = inputs_train * sqrt_weights

    optimal_clusters = find_optimal_clusters(weighted_data)
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(weighted_data)

    cluster_results[method] = {
        "optimal_clusters": optimal_clusters,
        "cluster_labels": cluster_labels,
        "kmeans_model": kmeans
    }

    print(f"Methode: {method}, Optimal cluster number: {optimal_clusters}")

# PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(inputs_train)

# Cluster Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (method, importance_df) in enumerate(avg_importance.items()):
    feature_importance = importance_df["Average Mutual Information"].values
    sqrt_weights = np.sqrt(feature_importance)
    weighted_data = inputs_train * sqrt_weights

    optimal_clusters = cluster_results[method]["optimal_clusters"]
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(weighted_data)

    ax = axes[i]
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
    ax.set_title(f"Cluster für Methode: {method}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(*scatter.legend_elements(), title="Cluster")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------------
import openturns as ot

clustered_data = {}
copulas = {}

for method in ["gru", "knn", "dp", "piml"]:

    clustered_data[method] = {}
    copulas[method] = {}
    cluster_labels = cluster_results[method]["cluster_labels"]
    error_data_for_method = errors[method]
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        print(f"Methode={method}, Cluster={cluster_id} ...")

        idx = np.where(cluster_labels == cluster_id)[0]
        cluster_errors = error_data_for_method[idx, :]

        df = pd.DataFrame(cluster_errors)
        df_ranked = df.rank(method="average") / (len(df) + 1)

        clustered_data[method][cluster_id] = {
            "outputs": cluster_errors,
            "df": df,
            "df_ranked": df_ranked,
        }

        sample_data = df.to_numpy().tolist()
        sample = ot.Sample(sample_data)

        # Empirische Copula
        # m_copula = len(sample)
        m_copula = int(np.sqrt(len(sample)))
        copula = ot.EmpiricalBernsteinCopula(sample, m_copula)

        copulas[method][cluster_id] = copula

    print(f"Copulas für Methode '{method}' erfolgreich erstellt.\n")


def generate_sample_from_copula(method, cluster_id, num_samples):
    new_samples = copulas[method][cluster_id].getSample(num_samples)
    return np.array(new_samples)


def get_quantile(method, cluster_id, feature_index, p):
    df = clustered_data[method][cluster_id]['df']
    return df.iloc[:, feature_index].quantile(p)


def predict_from_copula(method, kmeans_model, new_point):
    feature_importance = avg_importance[method]["Average Mutual Information"].values
    sqrt_weights = np.sqrt(feature_importance)
    weighted_new_point = np.array(new_point) * sqrt_weights
    cluster = kmeans_model.predict([weighted_new_point])[0]
    sampled_p = np.array(copulas[method][cluster].getSample(1))[0]

    num_features = len(sampled_p)
    predicted_values = []

    for feature_index in range(num_features):
        quantile_result = get_quantile(method, cluster, feature_index, sampled_p[feature_index])
        predicted_values.append(quantile_result)

    return np.array(predicted_values)


mu_test_methods = {
    "gru": mu_test_gru,
    "knn": mu_test_knn,
    "piml": mu_test_PIML.numpy(),
    "dp": mu_test_DP
}

sigma_test_methods = {
    "gru": sigma_test_gru,
    "piml": sigma_test_PIML.numpy(),
    "dp": sigma_test_DP
}
#
# #--------------------------------------------------------------------------------------
# load the gaussian correlation model
aggregation = 15
N = 3600 // aggregation
covariance_matrix_rq = prepare_covariance_matrix(N, kernel_type='RationalQuadratic')
loss_rq = {"loss": correlated_gaussian_loss(N, covariance_matrix_rq)}
model_rational_quadratic = load_model(f"../trained_models/correlated_gaussian_rational_quadratic_GRU",
                                      custom_objects=loss_rq)
outputs_test_angular_frequency = outputs_test.astype(np.float32) * (2 * np.pi)
outputs_test_angular_frequency_aggregated = outputs_test_angular_frequency[:, ::aggregation]
predictions_rational_quadratic = model_rational_quadratic.predict(inputs_test)
mu_rational_quadratic = 2 * np.pi * predictions_rational_quadratic[:, :N]
sigma2_rational_quadratic = (2 * np.pi) ** 2 * tf.math.sigmoid(predictions_rational_quadratic[:, N:])
sigma2_rational_quadratic = np.array(sigma2_rational_quadratic).reshape(-1, 1, 1)
sigma2_rational_quadratic = np.array(sigma2_rational_quadratic) * covariance_matrix_rq

mu_gaussian_correlated = mu_rational_quadratic[:, ::4]
sigma2_gaussian_correlated = sigma2_rational_quadratic[:, ::4, ::4]


def extract_submatrix(matrix, indices):
    return matrix[np.ix_(indices, indices)]


def generate_simulations(start_pos, end_pos, stride=60, num_repetitions=1000):
    results_dict = {}

    true_means = outputs_test_angular_frequency[start_pos:end_pos, ::stride].mean(axis=1)

    for method in ["gru", "knn", "dp", "piml"]:

        inputs_test_select = inputs_test[start_pos:end_pos, :]

        sim_results = []

        for idx, x in enumerate(inputs_test_select):

            avg_values = []

            for _ in range(num_repetitions):
                predicted_values = predict_from_copula(method, cluster_results[method]["kmeans_model"], x)+ mu_test_methods[method][start_pos+idx, ::60]
                avg_values.append(np.mean(predicted_values))

            median_val = np.median(avg_values)
            q_l = np.quantile(avg_values, 0.02275)  # Lower Quantile
            q_u = np.quantile(avg_values, 0.97725)  # Upper Quantile

            sim_results.append([median_val, q_l, q_u])

        results_dict[method] = pd.DataFrame(sim_results, columns=["Median", "Q_l", "Q_u"])

    for method in ["gru", "piml", "dp"]:

        mu_selected = mu_test_methods[method][start_pos:end_pos, ::stride]
        sigma_selected = sigma_test_methods[method][start_pos:end_pos, ::stride]

        mean_values = mu_selected.mean(axis=1)
        std_values = np.sqrt(np.sum(sigma_selected ** 2, axis=1) / sigma_selected.shape[1] ** 2)

        results_dict[f"{method}_gaussian"] = pd.DataFrame(
            {"Mean": mean_values, "Q_l": mean_values - 2 * std_values, "Q_u": mean_values + 2 * std_values}
        )


    mu_selected = mu_gaussian_correlated[start_pos:end_pos, :]
    sigma2_selected = sigma2_gaussian_correlated[start_pos:end_pos, :]

    sim_results = []

    for i in range(mu_selected.shape[0]):
        mu_vec = mu_selected[i]
        sigma_mat = sigma2_selected[i]
        mvn_samples = np.random.multivariate_normal(mu_vec, sigma_mat, size=num_repetitions)
        mean_values = np.mean(mvn_samples, axis=1)
        q_l = np.quantile(mean_values, 0.02275)
        q_u = np.quantile(mean_values, 0.97725)

        sim_results.append([mu_vec.mean(), q_l, q_u])

    results_dict["gaussian_correlated"] = pd.DataFrame(
        sim_results, columns=["Mean", "Q_l", "Q_u"]
    )

    return results_dict, true_means


start_pos, end_pos = 998, 1022
results_dict, true_means = generate_simulations(start_pos, end_pos)

models = ["gru", "piml", "knn", "dp"]
gaussian_models = ["gru_gaussian", "piml_gaussian", "knn_gaussian", "dp_gaussian"]
correlated_gaussian = "gaussian_correlated"

colors = {
    "gru": "red", "knn": "blue", "dp": "limegreen", "piml": "purple",
    "gru_gaussian": "darkred", "piml_gaussian": "darkblue",
    "knn_gaussian": "darkcyan", "dp_gaussian": "darkgreen",
    "gaussian_correlated": "black"
}

display_names = {
    "gru": "GRU", "piml": "PIML", "knn": "KNN", "dp": "DP",
    "gru_gaussian": "GRU GAUSSIAN", "piml_gaussian": "PIML GAUSSIAN",
    "knn_gaussian": "KNN GAUSSIAN", "dp_gaussian": "DP GAUSSIAN",
    "gaussian_correlated": "CORRELATED GAUSSIAN"
}


fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, model in enumerate(models):
    ax = axes[i]
    x_range = range(len(true_means))

    # Wahre Werte plotten
    ax.plot(x_range, true_means, 'ro-', label='True Means', markersize=5)

    # Modell CI plotten
    if model in results_dict:
        df = results_dict[model]
        ax.fill_between(x_range, df["Q_l"], df["Q_u"], color=colors[model], alpha=0.2,
                        label=f"CI {display_names[model]}")

    # Gaussian Version ±2σ
    gaussian_model = gaussian_models[i]
    if gaussian_model in results_dict:
        df = results_dict[gaussian_model]
        ax.fill_between(x_range, df["Q_l"], df["Q_u"], color=colors[gaussian_model], alpha=0.2,
                        label=f"CI {display_names[gaussian_model]}")

    # Correlated Gaussian
    if correlated_gaussian in results_dict:
        df = results_dict[correlated_gaussian]
        ax.fill_between(x_range, df["Q_l"], df["Q_u"], color=colors[correlated_gaussian], alpha=0.3,
                        label=f"CI {display_names[correlated_gaussian]}")

    ax.set_title(display_names[model])
    ax.set_xlabel("Hour Index")
    ax.set_ylabel("Mean Frequency Deviation")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()
#---------------------------------------------------------------------------------------------------
# Generate data for evaluation with energy score and mean crps


with open("outputs_test_angular_frequency.pkl", "wb") as file:
    pickle.dump(outputs_test_angular_frequency, file)


# probabilistic models
# downsample

mu_test_methods = {
    "gru": mu_test_gru[:,::60],
    "knn": mu_test_knn[:,::60]*2 * np.pi,
    "piml": mu_test_PIML.numpy()[:,::60],
    "dp": mu_test_DP[:,::60]*2 * np.pi
}


sigma_test_methods = {
    "gru": sigma_test_gru.numpy()[:,::60],
    "piml": sigma_test_PIML.numpy()[:,::60],
    "dp": sigma_test_DP[:,::60]*2 * np.pi
}

with open("sigma_test_methods.pkl", "wb") as file:
    pickle.dump(sigma_test_methods, file)

# gaussian with correlation
with open("mu_rational_quadratic.pkl", "wb") as file:
    pickle.dump(mu_gaussian_correlated, file)

with open("sigma2_rational_quadratic.pkl", "wb") as file:
    pickle.dump(sigma2_gaussian_correlated.numpy(), file)


#---------------------------------------------------------------------------------------


# Parameters
m = 1000

# Storage for copula results
y_new_samples = {}

# Precompute sqrt of feature importances
method_sqrt_weights = {
    method: np.sqrt(avg_importance[method]["Average Mutual Information"].values)
    for method in mu_test_methods
}


numpy_data_cache = {}

for method_idx, method in enumerate(mu_test_methods):
    print(f"[{method_idx + 1}/{len(mu_test_methods)}] Processing method: {method}")

    mu_test = mu_test_methods[method]
    inputs_array = np.asarray(inputs_test)
    sqrt_weights = method_sqrt_weights[method]
    weighted_inputs = inputs_array * sqrt_weights

    kmeans_model = cluster_results[method]["kmeans_model"]
    cluster_assignments = kmeans_model.predict(weighted_inputs)

    num_test_samples, num_features = mu_test.shape
    y_new_samples[method] = np.empty((num_test_samples, m, num_features), dtype=np.float64)

    # Prepare cluster tasks
    unique_clusters = np.unique(cluster_assignments)
    cluster_tasks = [(cluster, np.where(cluster_assignments == cluster)[0]) for cluster in unique_clusters]
    cluster_tasks.sort(key=lambda x: len(x[1]), reverse=True)

    # Cache cluster data
    for cluster in unique_clusters:
        if (method, cluster) not in numpy_data_cache and 'df' in clustered_data[method][cluster]:
            numpy_data_cache[(method, cluster)] = clustered_data[method][cluster]['df'].values

    for cluster_idx, (cluster, indices) in enumerate(cluster_tasks):
        if cluster not in copulas[method]:
            print(f"Skipping cluster {cluster}: No copula.")
            continue

        cluster_data = numpy_data_cache.get((method, cluster))
        if cluster_data is None:
            print(f"No data for cluster {cluster}")
            continue

        n = len(indices)
        samples = np.array(copulas[method][cluster].getSample(n * m)).reshape(n, m, num_features)
        preds = np.empty_like(samples)

        for f in range(num_features):
            feature_data = np.sort(cluster_data[:, f])
            n_data = len(feature_data)

            feature_samples = samples[:, :, f].reshape(-1)
            idx_float = feature_samples * (n_data - 1)
            idx_low = np.floor(idx_float).astype(int)
            idx_high = np.ceil(idx_float).astype(int)
            weight_high = idx_float - idx_low

            idx_low = np.clip(idx_low, 0, n_data - 1)
            idx_high = np.clip(idx_high, 0, n_data - 1)

            interpolated = (1 - weight_high) * feature_data[idx_low] + weight_high * feature_data[idx_high]
            preds[:, :, f] = interpolated.reshape(n, m)

        y_new_samples[method][indices] = preds + mu_test[indices][:, None, :]

        print(f"Cluster {cluster} done.")

    print(f"Method {method} completed.")
    if method_idx < len(mu_test_methods) - 1:
        keys_to_delete = [k for k in numpy_data_cache if k[0] == method]
        for k in keys_to_delete:
            del numpy_data_cache[k]
        gc.collect()

print("All methods finished.")



with open("y_new_samples11.pkl", "wb") as f:
    pickle.dump(y_new_samples, f)



