import numpy as np
import pandas as pd

# Load original precipitation data
# id = '01096000'
# file_name = f"data/hbv_input_{id}.csv"
# P_original = pd.read_csv(file_name)["precip"].values

def synthetic_precip(P_original): #P_original is array of original precipitation data
    # Bias factor for systematic error
    bias_factor = 0.95

    # Different error scales corresponding to forecasting error levels
    error_scales = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])

    # Temporal correlation coefficient for AR(1)
    temporal_corr = 0.1

    # Initialize dictionary for synthetic realizations
    synthetic_realizations = {}
    for iter in range(3):
        for error_scale in error_scales:
            # Initialize synthetic precipitation and errors
            synthetic_precip = np.zeros_like(P_original)
            synthetic_errors = np.zeros_like(P_original)

            # Generate synthetic precipitation
            synthetic_precip[0] = P_original[0] * bias_factor + np.random.normal(scale=error_scale * P_original[0])
            synthetic_precip[0] = max(synthetic_precip[0], 0)
            synthetic_precip[0] = round(synthetic_precip[0], 2)

            for t in range(1, len(P_original)):
                synthetic_precip[t] = P_original[t] * bias_factor
                synthetic_errors[t] = (temporal_corr * synthetic_errors[t-1] +
                                    np.random.normal(scale=error_scale * P_original[t]))
                synthetic_precip[t] += synthetic_errors[t]
                synthetic_precip[t] = max(synthetic_precip[t], 0)
                synthetic_precip[t] = round(synthetic_precip[t], 2)

            # Store the realization in the dictionary
            synthetic_realizations[f"iter_{iter}_error{error_scale}"] = synthetic_precip

    # Create a DataFrame with all realizations
    synthetic_df = pd.DataFrame(synthetic_realizations)
    synthetic_df["original"] = P_original

    # Calculate the RMSE for each realization
    rmse = {col: np.round(np.sqrt(np.mean((synthetic_df[col] - synthetic_df["original"])**2)), 2)
            for col in synthetic_df.columns if col != "original"}

    # Find the realization with RMSE closest to 1, 2, 3, 4
    best_realizations = {f"rmse_{target_rmse}": synthetic_df[min(rmse, key=lambda x: abs(rmse[x] - target_rmse))]
                        for target_rmse in [1, 2, 3, 4]}

    # Make dataframe with the best realizations
    best_realizations_df = pd.DataFrame(best_realizations)

    # # Find RMSE for the best realizations
    # rmse_best = {col: np.round(np.sqrt(np.mean((best_realizations_df[col] - synthetic_df["original"])**2)), 2)
    #             for col in best_realizations_df.columns if col != "original"}
    # print(rmse_best)
    return best_realizations['rmse_1'], best_realizations['rmse_2'], best_realizations['rmse_3'], best_realizations['rmse_4']
