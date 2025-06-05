from scipy.stats import ks_2samp
from scipy.stats import entropy, gaussian_kde
import numpy as np 
import re
from scipy.stats import wasserstein_distance

def calc_kl(gt, model_scores, country_code="USA"):

    kl_divergences = [] 
    traits = ["O", "C", "E", "A", "N"]
    preds_ = model_scores 
    grid_size = 100

    kds = [] 

    for trait in traits:
        # Extract data
        model_trait_data = preds_[trait]
        gt = gt[gt['country'] == country_code] 
        gt_trait_data = gt[gt["trait"] == trait].dropna()

        if len(model_trait_data) < 2 or len(gt_trait_data) < 2:
            print(f"Warning: Not enough data for trait {trait}. Skipping...")
            kl_divergences[trait] = np.nan  # Skip traits with insufficient data
            continue

        # Define a common grid
        common_grid = np.linspace(min(gt_trait_data.min(), model_trait_data.min()), 
                                max(gt_trait_data.max(), model_trait_data.max()), 
                                grid_size)

        # Compute KDEs using SciPy
        kde_gt = gaussian_kde(gt_trait_data, bw_method=0.3)  # Adjust bw_method as needed
        kde_model = gaussian_kde(model_trait_data, bw_method=0.3)

        # Evaluate KDEs on the common grid
        gt_density = kde_gt(common_grid)
        model_density = kde_model(common_grid)

        # Normalize densities to sum to 1
        gt_density /= np.trapz(gt_density, common_grid)
        model_density /= np.trapz(model_density, common_grid)

        # Avoid log(0) errors by adding a small epsilon
        eps = 1e-10
        kl_div = entropy(gt_density + eps, model_density + eps)

        kds.append(kl_div)

        # ks_stat, p_value = ks_2samp(gt_trait_data, model_trait_data)
        # kss.append(p_value) 

    kl_divergences.extend(kds) 

    return kl_divergences 

def calc_ks(gt, model_scores, country_code="USA"):

    kss = [] 
    ksp = [] 

    traits = ["O", "C", "E", "A", "N"]

    for trait in traits:
        # Extract data
        model_trait_data = model_scores[trait]
        gt = gt[gt['country'] == country_code] 
        gt_trait_data = gt[gt["trait"] == trait].dropna()

        # compute ks_stat between the gt_trait_data + model_trait_data
        ks_stat, p_value = ks_2samp(gt_trait_data, model_trait_data) 
        kss.append(ks_stat)
        ksp.append(p_value)

    return kss, ksp 

def calc_ttr(eog_results): 
    # combine all results for oeg 
    text = '\n'.join(eog_results)

    # Normalize text: lowercase and remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    return ttr

def calc_w(gt, model_scores, country_code="USA"):
    wasser = [] 

    traits = ["O", "C", "E", "A", "N"]

    for trait in traits:
        # Extract data
        model_trait_data = model_scores[trait]
        gt = gt[gt['country'] == country_code] 
        gt_trait_data = gt[gt["trait"] == trait].dropna()
        w = wasserstein_distance(gt_trait_data, model_trait_data)
        wasser.append(w) 
    
    return wasser 

    