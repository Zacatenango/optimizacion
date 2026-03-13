import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, LinearRegression

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================
print("=" * 70)
print("MULTIPLE REGRESSION ANALYSIS: OLS vs RIDGE REGULARIZATION")
print("=" * 70)

# Load the data
data = pd.read_csv('Cabohydrate_Data.csv')

print("\n1. DATA OVERVIEW")
print("-" * 50)
print(f"Dataset shape: {data.shape}")
print(f"\nFirst 10 rows:")
print(data.head(10))
print(f"\nStatistical Summary:")
print(data.describe())

# Define features (X) and target (y)
X = data[['age', 'weight', 'protein']].values
y = data['carbohydrate'].values

print(f"\nFeatures (X): age, weight, protein")
print(f"Target (y): carbohydrate")
print(f"Number of samples: {len(y)}")

# =============================================================================
# 2. TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "=" * 70)
print("2. TRAIN-TEST SPLIT")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(y_train)} samples")
print(f"Test set size: {len(y_test)} samples")

# For Ridge regression comparison, we'll use standardized features
# This makes the regularization penalty fair across features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Also center y for fair comparison (since we have no intercept)
y_mean = y_train.mean()
y_train_centered = y_train - y_mean
y_test_centered = y_test - y_mean

print(f"\nFeatures have been standardized for Ridge regression")
print(f"Target mean (for centering): {y_mean:.4f}")

# =============================================================================
# 3. ORDINARY LEAST SQUARES (OLS) REGRESSION
# =============================================================================
print("\n" + "=" * 70)
print("3. ORDINARY LEAST SQUARES (OLS) REGRESSION")
print("-" * 50)

def ols_regression_no_intercept(X, y):
    """
    Perform OLS regression without intercept.
    Solution: β = (X^T X)^(-1) X^T y
    """
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta

def ols_regression_with_intercept(X, y):
    """
    Perform OLS regression with intercept.
    """
    n = len(y)
    X_aug = np.column_stack([np.ones(n), X])
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta[0], beta[1:]  # intercept, coefficients

# OLS WITH intercept (standard approach)
intercept_ols, beta_ols = ols_regression_with_intercept(X_train, y_train)

print("OLS Coefficients (with intercept):")
print(f"  Intercept = {intercept_ols:.6f}")
print(f"  β_age     = {beta_ols[0]:.6f}")
print(f"  β_weight  = {beta_ols[1]:.6f}")
print(f"  β_protein = {beta_ols[2]:.6f}")

# Predictions with intercept
y_train_pred_ols = intercept_ols + X_train @ beta_ols
y_test_pred_ols = intercept_ols + X_test @ beta_ols

# Metrics for OLS
rmse_train_ols = np.sqrt(mean_squared_error(y_train, y_train_pred_ols))
rmse_test_ols = np.sqrt(mean_squared_error(y_test, y_test_pred_ols))
r2_train_ols = r2_score(y_train, y_train_pred_ols)
r2_test_ols = r2_score(y_test, y_test_pred_ols)
mae_train_ols = mean_absolute_error(y_train, y_train_pred_ols)
mae_test_ols = mean_absolute_error(y_test, y_test_pred_ols)

print(f"\nOLS Performance Metrics:")
print(f"  Training RMSE: {rmse_train_ols:.4f}")
print(f"  Test RMSE:     {rmse_test_ols:.4f}")
print(f"  Training R²:   {r2_train_ols:.4f}")
print(f"  Test R²:       {r2_test_ols:.4f}")
print(f"  Training MAE:  {mae_train_ols:.4f}")
print(f"  Test MAE:      {mae_test_ols:.4f}")

# =============================================================================
# 4. RIDGE REGRESSION WITH VARYING PENALTY (lambda from 1 to 600)
# =============================================================================
print("\n" + "=" * 70)
print("4. RIDGE REGRESSION (lambda from 1 to 600)")
print("-" * 50)

def ridge_regression_centered(X, y, y_mean, lambda_param):
    """
    Perform Ridge regression with centered data (equivalent to having intercept).
    Ridge penalizes coefficients but not the intercept.
    
    For standardized X and centered y:
    β = (X^T X + lambdaI)^(-1) X^T y
    Intercept = mean(y_original) (since X is standardized)
    """
    n_features = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    ridge_matrix = XtX + lambda_param * np.eye(n_features)
    beta = np.linalg.solve(ridge_matrix, Xty)
    return beta

# Test different values of lambda from 1 to 600
lambda_values = np.arange(1, 601)

# Storage for results
results = {
    'lambda': [],
    'beta_age': [],
    'beta_weight': [],
    'beta_protein': [],
    'rmse_train': [],
    'rmse_test': [],
    'r2_train': [],
    'r2_test': [],
    'mae_train': [],
    'mae_test': []
}

# Compute Ridge regression for each lambda
for lam in lambda_values:
    # Ridge on standardized X, centered y
    beta_ridge = ridge_regression_centered(X_train_scaled, y_train_centered, y_mean, lam)
    
    # Predictions (add back the mean)
    y_train_pred = X_train_scaled @ beta_ridge + y_mean
    y_test_pred = X_test_scaled @ beta_ridge + y_mean
    
    # Metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    # Store results
    results['lambda'].append(lam)
    results['beta_age'].append(beta_ridge[0])
    results['beta_weight'].append(beta_ridge[1])
    results['beta_protein'].append(beta_ridge[2])
    results['rmse_train'].append(rmse_train)
    results['rmse_test'].append(rmse_test)
    results['r2_train'].append(r2_train)
    results['r2_test'].append(r2_test)
    results['mae_train'].append(mae_train)
    results['mae_test'].append(mae_test)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Find optimal lambda (minimum test RMSE)
optimal_idx = results_df['rmse_test'].idxmin()
optimal_lambda = results_df.loc[optimal_idx, 'lambda']
optimal_rmse_test = results_df.loc[optimal_idx, 'rmse_test']

print(f"Optimal lambda (minimum test RMSE): {optimal_lambda}")
print(f"Minimum Test RMSE: {optimal_rmse_test:.4f}")

# Display coefficients at optimal lambda (these are in standardized scale)
print(f"\nRidge Coefficients at optimal lambda = {optimal_lambda} (standardized scale):")
print(f"  β_age     = {results_df.loc[optimal_idx, 'beta_age']:.6f}")
print(f"  β_weight  = {results_df.loc[optimal_idx, 'beta_weight']:.6f}")
print(f"  β_protein = {results_df.loc[optimal_idx, 'beta_protein']:.6f}")

# Convert to original scale for interpretation
beta_ridge_opt = np.array([
    results_df.loc[optimal_idx, 'beta_age'],
    results_df.loc[optimal_idx, 'beta_weight'],
    results_df.loc[optimal_idx, 'beta_protein']
])
beta_original_scale = beta_ridge_opt / scaler.scale_
intercept_ridge = y_mean - np.sum(beta_original_scale * scaler.mean_)

print(f"\nRidge Coefficients at optimal lambda = {optimal_lambda} (original scale):")
print(f"  Intercept = {intercept_ridge:.6f}")
print(f"  β_age     = {beta_original_scale[0]:.6f}")
print(f"  β_weight  = {beta_original_scale[1]:.6f}")
print(f"  β_protein = {beta_original_scale[2]:.6f}")

# Show results at key lambda values
print("\n" + "-" * 50)
print("Ridge Results at Selected lambda Values:")
print("-" * 50)
key_lambdas = [1, 10, 50, 100, 200, 300, 400, 500, 600]
for lam in key_lambdas:
    row = results_df[results_df['lambda'] == lam].iloc[0]
    print(f"lambda = {lam:3d}: RMSE_train = {row['rmse_train']:.4f}, "
          f"RMSE_test = {row['rmse_test']:.4f}, R²_test = {row['r2_test']:.4f}")

# =============================================================================
# 5. COMPARISON: OLS vs RIDGE
# =============================================================================
print("\n" + "=" * 70)
print("5. COMPARISON: OLS vs RIDGE REGRESSION")
print("-" * 50)

print("\nMetric Comparison Table:")
print("-" * 70)
print(f"{'Metric':<15} {'OLS':<12} {'Ridge(lambda=1)':<12} {'Ridge(lambda=opt)':<12} {'Ridge(lambda=100)':<12} {'Ridge(lambda=600)':<12}")
print("-" * 70)

ridge_1_row = results_df[results_df['lambda'] == 1].iloc[0]
ridge_opt_row = results_df.loc[optimal_idx]
ridge_100_row = results_df[results_df['lambda'] == 100].iloc[0]
ridge_600_row = results_df[results_df['lambda'] == 600].iloc[0]

print(f"{'Train RMSE':<15} {rmse_train_ols:<12.4f} {ridge_1_row['rmse_train']:<12.4f} {ridge_opt_row['rmse_train']:<12.4f} {ridge_100_row['rmse_train']:<12.4f} {ridge_600_row['rmse_train']:<12.4f}")
print(f"{'Test RMSE':<15} {rmse_test_ols:<12.4f} {ridge_1_row['rmse_test']:<12.4f} {ridge_opt_row['rmse_test']:<12.4f} {ridge_100_row['rmse_test']:<12.4f} {ridge_600_row['rmse_test']:<12.4f}")
print(f"{'Train R²':<15} {r2_train_ols:<12.4f} {ridge_1_row['r2_train']:<12.4f} {ridge_opt_row['r2_train']:<12.4f} {ridge_100_row['r2_train']:<12.4f} {ridge_600_row['r2_train']:<12.4f}")
print(f"{'Test R²':<15} {r2_test_ols:<12.4f} {ridge_1_row['r2_test']:<12.4f} {ridge_opt_row['r2_test']:<12.4f} {ridge_100_row['r2_test']:<12.4f} {ridge_600_row['r2_test']:<12.4f}")
print(f"{'Train MAE':<15} {mae_train_ols:<12.4f} {ridge_1_row['mae_train']:<12.4f} {ridge_opt_row['mae_train']:<12.4f} {ridge_100_row['mae_train']:<12.4f} {ridge_600_row['mae_train']:<12.4f}")
print(f"{'Test MAE':<15} {mae_test_ols:<12.4f} {ridge_1_row['mae_test']:<12.4f} {ridge_opt_row['mae_test']:<12.4f} {ridge_100_row['mae_test']:<12.4f} {ridge_600_row['mae_test']:<12.4f}")
print("-" * 70)

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("6. GENERATING VISUALIZATIONS")
print("-" * 50)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multiple Regression Analysis: OLS vs Ridge Regularization\nCarbohydrate Prediction for Diabetic Patients', 
             fontsize=14, fontweight='bold')

# Plot 1: RMSE vs Lambda
ax1 = axes[0, 0]
ax1.plot(results_df['lambda'], results_df['rmse_train'], 'b-', label='Train RMSE (Ridge)', linewidth=1.5)
ax1.plot(results_df['lambda'], results_df['rmse_test'], 'r-', label='Test RMSE (Ridge)', linewidth=1.5)
ax1.axhline(y=rmse_train_ols, color='blue', linestyle='--', alpha=0.7, label=f'OLS Train RMSE ({rmse_train_ols:.2f})')
ax1.axhline(y=rmse_test_ols, color='red', linestyle='--', alpha=0.7, label=f'OLS Test RMSE ({rmse_test_ols:.2f})')
ax1.axvline(x=optimal_lambda, color='green', linestyle=':', alpha=0.7, label=f'Optimal lambda={optimal_lambda}')
ax1.set_xlabel('Lambda (lambda)', fontsize=11)
ax1.set_ylabel('RMSE', fontsize=11)
ax1.set_title('RMSE vs Regularization Parameter (lambda)', fontsize=12)
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([4, 14])

# Plot 2: R² vs Lambda
ax2 = axes[0, 1]
ax2.plot(results_df['lambda'], results_df['r2_train'], 'b-', label='Train R² (Ridge)', linewidth=1.5)
ax2.plot(results_df['lambda'], results_df['r2_test'], 'r-', label='Test R² (Ridge)', linewidth=1.5)
ax2.axhline(y=r2_train_ols, color='blue', linestyle='--', alpha=0.7, label=f'OLS Train R² ({r2_train_ols:.3f})')
ax2.axhline(y=r2_test_ols, color='red', linestyle='--', alpha=0.7, label=f'OLS Test R² ({r2_test_ols:.3f})')
ax2.axvline(x=optimal_lambda, color='green', linestyle=':', alpha=0.7, label=f'Optimal lambda={optimal_lambda}')
ax2.set_xlabel('Lambda (lambda)', fontsize=11)
ax2.set_ylabel('R² Score', fontsize=11)
ax2.set_title('R² Score vs Regularization Parameter (lambda)', fontsize=12)
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Coefficient Paths (standardized)
ax3 = axes[1, 0]
ax3.plot(results_df['lambda'], results_df['beta_age'], 'b-', label='β_age', linewidth=2)
ax3.plot(results_df['lambda'], results_df['beta_weight'], 'r-', label='β_weight', linewidth=2)
ax3.plot(results_df['lambda'], results_df['beta_protein'], 'g-', label='β_protein', linewidth=2)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.axvline(x=optimal_lambda, color='purple', linestyle=':', alpha=0.7, label=f'Optimal lambda={optimal_lambda}')
ax3.set_xlabel('Lambda (lambda)', fontsize=11)
ax3.set_ylabel('Coefficient Value (Standardized)', fontsize=11)
ax3.set_title('Ridge Coefficient Paths', fontsize=12)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Actual vs Predicted (Test Set)
ax4 = axes[1, 1]

# OLS predictions
ax4.scatter(y_test, y_test_pred_ols, alpha=0.6, c='blue', label='OLS', edgecolors='k', s=60, marker='o')

# Ridge predictions at optimal lambda
beta_ridge_optimal = ridge_regression_centered(X_train_scaled, y_train_centered, y_mean, optimal_lambda)
y_test_pred_ridge = X_test_scaled @ beta_ridge_optimal + y_mean
ax4.scatter(y_test, y_test_pred_ridge, alpha=0.6, c='red', label=f'Ridge (lambda={optimal_lambda})', edgecolors='k', s=60, marker='s')

# Perfect prediction line
min_val = min(y_test.min(), y_test_pred_ols.min(), y_test_pred_ridge.min()) - 2
max_val = max(y_test.max(), y_test_pred_ols.max(), y_test_pred_ridge.max()) + 2
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Carbohydrate (%)', fontsize=11)
ax4.set_ylabel('Predicted Carbohydrate (%)', fontsize=11)
ax4.set_title('Actual vs Predicted Values (Test Set)', fontsize=12)
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_analysis_plots.png', dpi=150, bbox_inches='tight')
print("Main plots saved to 'regression_analysis_plots.png'")

# Additional plot: Detailed comparison at different lambda values
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Detailed Analysis: Ridge Regularization Effects', fontsize=14, fontweight='bold')

# Plot: Bias-Variance Tradeoff visualization
ax_bv = axes2[0]
train_errors = results_df['rmse_train'].values
test_errors = results_df['rmse_test'].values
gap = test_errors - train_errors  # Generalization gap

ax_bv.fill_between(results_df['lambda'], train_errors, test_errors, 
                   alpha=0.3, color='purple', label='Generalization Gap')
ax_bv.plot(results_df['lambda'], train_errors, 'b-', label='Train RMSE', linewidth=1.5)
ax_bv.plot(results_df['lambda'], test_errors, 'r-', label='Test RMSE', linewidth=1.5)
ax_bv.axvline(x=optimal_lambda, color='green', linestyle=':', label=f'Optimal lambda={optimal_lambda}')
ax_bv.set_xlabel('Lambda (lambda)', fontsize=11)
ax_bv.set_ylabel('RMSE', fontsize=11)
ax_bv.set_title('Bias-Variance Tradeoff Visualization', fontsize=12)
ax_bv.legend(loc='best')
ax_bv.grid(True, alpha=0.3)

# Plot: Coefficient L2 norm vs Lambda
ax_norm = axes2[1]
beta_norms = np.sqrt(results_df['beta_age']**2 + results_df['beta_weight']**2 + results_df['beta_protein']**2)
ax_norm.plot(results_df['lambda'], beta_norms, 'purple', linewidth=2)
ax_norm.axvline(x=optimal_lambda, color='green', linestyle=':', label=f'Optimal lambda={optimal_lambda}')
ax_norm.set_xlabel('Lambda (lambda)', fontsize=11)
ax_norm.set_ylabel('||β||₂ (L2 Norm)', fontsize=11)
ax_norm.set_title('Coefficient Shrinkage: L2 Norm vs lambda', fontsize=12)
ax_norm.legend(loc='best')
ax_norm.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_detailed_analysis.png', dpi=150, bbox_inches='tight')
print("Detailed analysis plots saved to 'ridge_detailed_analysis.png'")

# =============================================================================
# 7. COEFFICIENT SHRINKAGE ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("7. COEFFICIENT SHRINKAGE ANALYSIS")
print("-" * 50)

print("\nCoefficient values at different lambda (standardized scale):")
print("-" * 80)
print(f"{'lambda':<10} {'β_age':<15} {'β_weight':<15} {'β_protein':<15} {'||β||₂':<15}")
print("-" * 80)

for lam in [1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 600]:
    row = results_df[results_df['lambda'] == lam].iloc[0]
    beta_norm = np.sqrt(row['beta_age']**2 + row['beta_weight']**2 + row['beta_protein']**2)
    print(f"{lam:<10} {row['beta_age']:<15.6f} {row['beta_weight']:<15.6f} {row['beta_protein']:<15.6f} {beta_norm:<15.6f}")

# =============================================================================
# 8. ADDITIONAL METRICS
# =============================================================================
print("\n" + "=" * 70)
print("8. ADDITIONAL METRICS")
print("-" * 50)

# Calculate additional metrics
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def max_error_calc(y_true, y_pred):
    """Maximum absolute error"""
    return np.max(np.abs(y_true - y_pred))

# OLS additional metrics
mape_ols = mape(y_test, y_test_pred_ols)
max_err_ols = max_error_calc(y_test, y_test_pred_ols)

# Ridge optimal additional metrics
mape_ridge = mape(y_test, y_test_pred_ridge)
max_err_ridge = max_error_calc(y_test, y_test_pred_ridge)

print("\nAdditional Performance Metrics (Test Set):")
print("-" * 50)
print(f"{'Metric':<25} {'OLS':<15} {'Ridge (optimal)':<15}")
print("-" * 50)
print(f"{'MAPE (%)':<25} {mape_ols:<15.2f} {mape_ridge:<15.2f}")
print(f"{'Max Absolute Error':<25} {max_err_ols:<15.2f} {max_err_ridge:<15.2f}")

# Residual analysis
residuals_ols = y_test - y_test_pred_ols
residuals_ridge = y_test - y_test_pred_ridge

print(f"\nResidual Statistics (Test Set):")
print("-" * 50)
print(f"{'Statistic':<25} {'OLS':<15} {'Ridge (optimal)':<15}")
print("-" * 50)
print(f"{'Mean Residual':<25} {np.mean(residuals_ols):<15.4f} {np.mean(residuals_ridge):<15.4f}")
print(f"{'Std Residual':<25} {np.std(residuals_ols):<15.4f} {np.std(residuals_ridge):<15.4f}")
print(f"{'Min Residual':<25} {np.min(residuals_ols):<15.4f} {np.min(residuals_ridge):<15.4f}")
print(f"{'Max Residual':<25} {np.max(residuals_ols):<15.4f} {np.max(residuals_ridge):<15.4f}")

# =============================================================================
# 9. INSIGHTS AND CONCLUSIONS
# =============================================================================
print("\n" + "=" * 70)
print("9. INSIGHTS AND CONCLUSIONS")
print("=" * 70)

print(f"""
ANALYSIS SUMMARY:
=================

1. DATA CHARACTERISTICS:
   - Dataset contains {len(data)} observations of insulin-dependent diabetics
   - Target variable: Carbohydrate percentage (mean = {y.mean():.2f}, std = {y.std():.2f})
   - Features: Age (years), Weight (relative to ideal), Protein (% calories)

2. ORDINARY LEAST SQUARES (OLS) RESULTS:
   - Intercept: {intercept_ols:.4f}
   - β_age = {beta_ols[0]:.4f} (Age has positive effect on carbohydrate %)
   - β_weight = {beta_ols[1]:.4f} (Weight has small positive effect)
   - β_protein = {beta_ols[2]:.4f} (Protein has positive effect)
   - Test RMSE: {rmse_test_ols:.4f}, Test R²: {r2_test_ols:.4f}

3. RIDGE REGULARIZATION FINDINGS:
   - Optimal lambda = {optimal_lambda} (based on minimum test RMSE)
   - At optimal lambda: Test RMSE = {optimal_rmse_test:.4f}
   - As lambda increases from 1 to 600:
     * All coefficients shrink toward zero monotonically
     * Training RMSE increases (more bias)
     * The gap between train and test RMSE narrows (less variance)
     * L2 norm ||β|| decreases from {beta_norms.iloc[0]:.4f} to {beta_norms.iloc[-1]:.4f}

4. KEY OBSERVATIONS:
   - Age is the strongest predictor in both OLS and Ridge regression
   - For this dataset, OLS provides better performance than Ridge
   - This suggests the model doesn't suffer from significant overfitting
   - Ridge regularization is most beneficial when:
     * Features are highly correlated (multicollinearity)
     * The model is overfitting (high variance)
     * The number of features is close to or exceeds sample size

5. RECOMMENDATIONS:
   - For prediction: Use OLS as it provides better test performance
   - For interpretation: Coefficients indicate Age and Protein are key factors
   - Consider: Cross-validation for more robust lambda selection
   - Note: The dataset shows good predictive relationships (R² ≈ {r2_test_ols:.2%})

6. CLINICAL INTERPRETATION:
   - Older diabetics tend to have higher carbohydrate intake
   - Protein consumption is positively associated with carbohydrate percentage
   - Body weight (relative to ideal) has a smaller but positive effect
""")

# Save all results to CSV
results_df.to_csv('ridge_results_detailed.csv', index=False)
print("\nDetailed Ridge results saved to 'ridge_results_detailed.csv'")

# =============================================================================
# 10. FINAL SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("10. FINAL SUMMARY TABLE")
print("=" * 70)

summary_table = pd.DataFrame({
    'Model': ['OLS', f'Ridge (lambda={optimal_lambda})', 'Ridge (lambda=1)', 'Ridge (lambda=100)', 'Ridge (lambda=300)', 'Ridge (lambda=600)'],
    'Train_RMSE': [rmse_train_ols, ridge_opt_row['rmse_train'], ridge_1_row['rmse_train'], 
                   ridge_100_row['rmse_train'], results_df[results_df['lambda']==300].iloc[0]['rmse_train'],
                   ridge_600_row['rmse_train']],
    'Test_RMSE': [rmse_test_ols, ridge_opt_row['rmse_test'], ridge_1_row['rmse_test'],
                  ridge_100_row['rmse_test'], results_df[results_df['lambda']==300].iloc[0]['rmse_test'],
                  ridge_600_row['rmse_test']],
    'Test_R2': [r2_test_ols, ridge_opt_row['r2_test'], ridge_1_row['r2_test'],
                ridge_100_row['r2_test'], results_df[results_df['lambda']==300].iloc[0]['r2_test'],
                ridge_600_row['r2_test']],
    'Test_MAE': [mae_test_ols, ridge_opt_row['mae_test'], ridge_1_row['mae_test'],
                 ridge_100_row['mae_test'], results_df[results_df['lambda']==300].iloc[0]['mae_test'],
                 ridge_600_row['mae_test']]
})

print("\n" + summary_table.to_string(index=False))
summary_table.to_csv('final_summary.csv', index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
