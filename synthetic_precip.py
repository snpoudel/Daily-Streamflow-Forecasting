import numpy as np
import pandas as pd

# Load original precipitation data
P_original = pd.read_csv("og_file.csv")["precip"].values

# Bias factor for systematic error
bias_factor = 0.95 #0.95 means synthetic data will be 5% lower than the original data

# Different error scales corresponding to forecasting error levels
error_scales = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]  # Proportional error levels (10%, 20%, 30%, 40%,.. of precipitation)
#0.1 means that the error is 10% of the precipitation value

# Temporal correlation coefficient for AR(1)
temporal_corr = 0.1 #0.1 means 10% of the error at time t-1 is carried over to time t

# Initialize dictionary for synthetic realizations
synthetic_realizations = {}
for iter in range(10): 
    for idx, error_scale in enumerate(error_scales):
        # Initialize synthetic precipitation and errors
        synthetic_precip = np.zeros_like(P_original)
        synthetic_errors = np.zeros_like(P_original)
        
        # Generate synthetic precipitation
        for t in range(len(P_original)):
            # Systematic bias
            synthetic_precip[t] = P_original[t] * bias_factor
            
            # Random error with temporal correlation
            if t > 0:
                synthetic_errors[t] = (temporal_corr * synthetic_errors[t-1] +
                                    np.random.normal(scale=error_scale * P_original[t]))
            else:
                synthetic_errors[t] = np.random.normal(scale=error_scale * P_original[t])
            
            # Add errors and ensure non-negative values
            synthetic_precip[t] += synthetic_errors[t]
            synthetic_precip[t] = max(synthetic_precip[t], 0)

            #round to 2 decimal places
            synthetic_precip[t] = round(synthetic_precip[t], 2)
        
        # Store the realization in the dictionary
        synthetic_realizations[f"iter_{iter}_error{error_scale}"] = synthetic_precip

# Create a DataFrame with all realizations
synthetic_df = pd.DataFrame(synthetic_realizations)
synthetic_df["original"] = P_original

#calculate the RMSE for each realization and keep realization close to [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
rmse = {}
for col in synthetic_df.columns:
    if col != "original":
        rmse[col] = np.round(np.sqrt(np.mean((synthetic_df[col] - synthetic_df["original"])**2)), 2)

#find the realization with RMSE closest to 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4
best_realizations = {}
for target_rmse in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
    best_realization = min(rmse, key=lambda x: abs(rmse[x] - target_rmse))
    best_realizations[f"rmse_{target_rmse}"] = synthetic_df[best_realization]

#make dataframe with the best realizations
best_realizations_df = pd.DataFrame(best_realizations)
#rmse of best realizations
rmse_best_realizations = {}
for col in best_realizations_df.columns:
    rmse_best_realizations[col] = np.round(np.sqrt(np.mean((best_realizations_df[col] - synthetic_df["original"])**2)), 2)
print(rmse_best_realizations)

#plot the original and synthetic realizations
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3))
plt.plot(synthetic_df["original"][0:60], label="Original")
plt.plot(best_realizations_df["rmse_2"][0:60], label="Error")
plt.legend()
plt.show()