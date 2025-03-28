import gc
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm
import scoringrules as sr
import pickle
import numpy as np
import os

seed = 42
np.random.seed(seed)

data_dir = os.path.join("..", "evaluation_data")

with open(os.path.join(data_dir, "outputs_test_angular_frequency.pkl"), "rb") as file:
    y_true = pickle.load(file)
y_true_array = y_true[:, ::60]

with open(os.path.join(data_dir, "mu_test_methods.pkl"), "rb") as f:
    mu_test_methods = pickle.load(f)

with open(os.path.join(data_dir, "sigma_test_methods.pkl"), "rb") as f:
    sigma_test_methods = pickle.load(f)

# Copula predictions
with open(os.path.join(data_dir, "y_new_samples.pkl"), "rb") as f:
    y_new_samples = pickle.load(f)

with open(os.path.join(data_dir, "mu_rational_quadratic.pkl"), "rb") as f:
    mu_rational_quadratic = pickle.load(f)

with open(os.path.join(data_dir, "sigma2_rational_quadratic.pkl"), "rb") as f:
    sigma2_rational_quadratic = pickle.load(f)

n_samples = y_true_array.shape[0]
dim = y_true_array.shape[1]

energy_score_results = {}  # Copula results
energy_score_results_gaussian = {}  # Independent Gaussian results
energy_score_results_gaussian_correlation = {"rational_quadratic": {}}  # Correlated Gaussian results


def compute_batched_scores(y_true, y_pred, score_func, batch_size=100):
    num_samples = y_true.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    scores = []

    batch_iterator = tqdm(range(num_batches), desc="Processing Batches")

    for i in batch_iterator:
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)
        batch_scores = score_func(y_true[start:end], y_pred[start:end])
        scores.append(batch_scores)
        gc.collect()

    return np.concatenate(scores)


def calculate_statistics(scores):
    return {
        "median": np.median(scores),
        "mean": np.mean(scores),
        "std": np.std(scores),
        "25%": np.percentile(scores, 25),
        "75%": np.percentile(scores, 75),
        "min": np.min(scores),
        "max": np.max(scores)
    }


def process_scores(y_true, y_samples, score_func, batch_size=100):
    scores = compute_batched_scores(y_true, y_samples, score_func, batch_size)
    return {"scores": scores, **calculate_statistics(scores)}


ensemble_size = 1000
batch_size = 10

# Evaluation Copula
for method, fct in y_new_samples.items():
    fct = fct.astype(np.float32)
    y_true_array_float32 = y_true_array.astype(np.float32)
    energy_score_results[method] = process_scores(y_true_array_float32, fct, sr.energy_score, batch_size)

# Generate Gaussian samples
y_new_sample_gaussian = {}
for method in mu_test_methods.keys():
    if method in sigma_test_methods:
        mu = mu_test_methods[method]
        sigma = sigma_test_methods[method]
        samples = np.random.normal(loc=mu[:, None, :], scale=sigma[:, None, :], size=(n_samples, ensemble_size, dim))
        y_new_sample_gaussian[method] = samples

# Evaluate Gaussian (independent)
for method, fct in y_new_sample_gaussian.items():
    energy_score_results_gaussian[method] = process_scores(y_true_array, fct, sr.energy_score, batch_size)

# Generate correlated Gaussian samples
y_new_sample_gaussian_correlation = np.zeros((n_samples, ensemble_size, dim))
for i in range(n_samples):
    mu = mu_rational_quadratic[i]
    sigma2 = sigma2_rational_quadratic[i]
    y_new_sample_gaussian_correlation[i] = np.random.multivariate_normal(mean=mu, cov=sigma2, size=ensemble_size)

energy_score_results_gaussian_correlation["rational_quadratic"] = process_scores(
    y_true_array, y_new_sample_gaussian_correlation, sr.energy_score, batch_size)


def print_results(title, results):
    print(f"\n{'=' * 50}")
    print(title)
    print(f"{'=' * 50}")
    for method, stats in results.items():
        print(
            f"{method}: "
            f"Median={stats['median']:.4f}, Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
            f"25%={stats['25%']:.4f}, 75%={stats['75%']:.4f}, "
            f"Min={stats['min']:.4f}, Max={stats['max']:.4f}"
        )


print_results("Energy Scores (Copula)", energy_score_results)
print_results("Energy Scores (Gaussian Samples)", energy_score_results_gaussian)
print_results("Energy Scores (Gaussian Correlation Samples)", energy_score_results_gaussian_correlation)


def mean_marginal_crps(y_true, y_pred):
    n_samples, ensemble_size, dim = y_pred.shape
    crps_values = np.zeros(n_samples)

    for d in range(dim):
        crps_values += sr.crps_ensemble(y_true[:, d], y_pred[:, :, d])

    crps_values /= dim
    return crps_values


crps_results = {}
crps_results_gaussian = {}
crps_results_gaussian_correlation = {"rational_quadratic": {}}

for method, fct in y_new_samples.items():
    fct = fct.astype(np.float32)
    y_true_array_float32 = y_true_array.astype(np.float32)
    crps_results[method] = process_scores(y_true_array_float32, fct, mean_marginal_crps, batch_size)

for method, fct in y_new_sample_gaussian.items():
    crps_results_gaussian[method] = process_scores(y_true_array, fct, mean_marginal_crps, batch_size)

crps_results_gaussian_correlation["rational_quadratic"] = process_scores(
    y_true_array, y_new_sample_gaussian_correlation, mean_marginal_crps, batch_size
)


print_results("CRPS (Copula)", crps_results)
print_results("CRPS (Gaussian Samples)", crps_results_gaussian)
print_results("CRPS (Gaussian Correlation Samples)", crps_results_gaussian_correlation)


# --------------------------- Test -------------------------------------------------
energy_corr_scores = energy_score_results_gaussian_correlation["rational_quadratic"]["scores"]
energy_corr_median = energy_score_results_gaussian_correlation["rational_quadratic"]["median"]

crps_corr_scores = crps_results_gaussian_correlation["rational_quadratic"]["scores"]
crps_corr_median = crps_results_gaussian_correlation["rational_quadratic"]["median"]

energy_rows = []
crps_rows = []
gauss_vs_corr_energy = []
gauss_vs_corr_crps = []

for method in energy_score_results:
    model_scores_e = energy_score_results[method]["scores"]
    median_model_e = energy_score_results[method]["median"]

    if method in energy_score_results_gaussian:
        gauss_scores_e = energy_score_results_gaussian[method]["scores"]
        median_gauss_e = energy_score_results_gaussian[method]["median"]
        p_gauss_e = wilcoxon(model_scores_e[:min(len(model_scores_e), len(gauss_scores_e))],
                             gauss_scores_e[:min(len(model_scores_e), len(gauss_scores_e))])[1]

        p_gauss_corr_e = wilcoxon(gauss_scores_e[:min(len(gauss_scores_e), len(energy_corr_scores))],
                                  energy_corr_scores[:min(len(gauss_scores_e), len(energy_corr_scores))])[1]

        gauss_vs_corr_energy.append({
            "Method": method,
            "Median Gaussian": median_gauss_e,
            "Median Gauss-Corr": energy_corr_median,
            "p-Value": p_gauss_corr_e,
            "Significant (p<0.05)": p_gauss_corr_e < 0.05
        })
    else:
        median_gauss_e = None
        p_gauss_e = None

    p_corr_e = wilcoxon(model_scores_e[:min(len(model_scores_e), len(energy_corr_scores))],
                        energy_corr_scores[:min(len(model_scores_e), len(energy_corr_scores))])[1]

    energy_rows.append({
        "Method": method,
        "Median Model": median_model_e,
        "Median Gaussian": median_gauss_e,
        "Median Gauss-Corr": energy_corr_median,
        "Model < Gauss?": median_gauss_e is not None and median_model_e < median_gauss_e,
        "Model < Gauss-Corr?": median_model_e < energy_corr_median,
        "p-Value Gauss": p_gauss_e,
        "Significant Gauss": p_gauss_e is not None and p_gauss_e < 0.05,
        "p-Value Gauss-Corr": p_corr_e,
        "Significant Gauss-Corr": p_corr_e < 0.05
    })

    if method not in crps_results:
        continue

    model_scores_c = crps_results[method]["scores"]
    median_model_c = crps_results[method]["median"]

    if method in crps_results_gaussian:
        gauss_scores_c = crps_results_gaussian[method]["scores"]
        median_gauss_c = crps_results_gaussian[method]["median"]
        p_gauss_c = wilcoxon(model_scores_c[:min(len(model_scores_c), len(gauss_scores_c))],
                             gauss_scores_c[:min(len(model_scores_c), len(gauss_scores_c))])[1]

        p_gauss_corr_c = wilcoxon(gauss_scores_c[:min(len(gauss_scores_c), len(crps_corr_scores))],
                                  crps_corr_scores[:min(len(gauss_scores_c), len(crps_corr_scores))])[1]

        gauss_vs_corr_crps.append({
            "Method": method,
            "Median Gaussian": median_gauss_c,
            "Median Gauss-Corr": crps_corr_median,
            "p-Value": p_gauss_corr_c,
            "Significant (p<0.05)": p_gauss_corr_c < 0.05
        })
    else:
        median_gauss_c = None
        p_gauss_c = None

    p_corr_c = wilcoxon(model_scores_c[:min(len(model_scores_c), len(crps_corr_scores))],
                        crps_corr_scores[:min(len(model_scores_c), len(crps_corr_scores))])[1]

    crps_rows.append({
        "Method": method,
        "Median Model": median_model_c,
        "Median Gaussian": median_gauss_c,
        "Median Gauss-Corr": crps_corr_median,
        "Model < Gauss?": median_gauss_c is not None and median_model_c < median_gauss_c,
        "Model < Gauss-Corr?": median_model_c < crps_corr_median,
        "p-Value Gauss": p_gauss_c,
        "Significant Gauss": p_gauss_c is not None and p_gauss_c < 0.05,
        "p-Value Gauss-Corr": p_corr_c,
        "Significant Gauss-Corr": p_corr_c < 0.05
    })


df_energy = pd.DataFrame(energy_rows)
df_crps = pd.DataFrame(crps_rows)
df_gauss_vs_corr_energy = pd.DataFrame(gauss_vs_corr_energy)
df_gauss_vs_corr_crps = pd.DataFrame(gauss_vs_corr_crps)

print("\nTable: Energy Score – Models vs. Gaussian & Gauss-Corr")
print(df_energy.to_string(index=False))

print("\nTable: CRPS – Models vs. Gaussian & Gauss-Corr")
print(df_crps.to_string(index=False))

print("\nTable: Gaussian vs. Gaussian-Correlation (Energy Score)")
print(df_gauss_vs_corr_energy.to_string(index=False))

print("\nTable: Gaussian vs. Gaussian-Correlation (CRPS)")
print(df_gauss_vs_corr_crps.to_string(index=False))
