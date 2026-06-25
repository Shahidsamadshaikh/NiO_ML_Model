import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist

# ============================================================================
# 1. LOAD SAVED MODEL AND SCALER
# ============================================================================
with open('rbf_krr_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

scaler = model_data['scaler']
gamma = model_data['gamma']
alpha = model_data['lambda']          # regularization
coeffs = model_data['coefficients']
X_train = model_data['X_train']       # scaled training features

# Retrieve original training data for plotting (you may also reload from CSV)
# Here we recreate the original DataFrame (same as in training code)
data = {
    'U(eV)': [5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4, 6.6, 6.8],
    'magnetic-moment': [1.94, 1.94, 1.94, 1.945, 1.945, 1.945, 1.945, 1.95, 1.95, 1.95],
    'band-gap': [3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.9, 4.0]
}
df_train = pd.DataFrame(data)

# ============================================================================
# 2. DEFINE PREDICTION FUNCTION (USING SAVED MODEL)
# ============================================================================
def rbf_kernel(X1, X2, gamma):
    pairwise_dists = cdist(X1, X2, metric='sqeuclidean')
    return np.exp(-gamma * pairwise_dists)

def predict(new_X_raw):
    """new_X_raw: numpy array of shape (n_samples, 2) with columns [U(eV), magnetic-moment]"""
    new_X_scaled = scaler.transform(new_X_raw)
    K = rbf_kernel(new_X_scaled, X_train, gamma)
    return K.dot(coeffs)

# ============================================================================
# 3. GENERATE PREDICTIONS FOR A RANGE OF U VALUES (FIXED MAGNETIC MOMENT)
# ============================================================================
# Choose a magnetic moment value. Here we use the mean of training moments.
fixed_moment = df_train['magnetic-moment'].mean()   # ~1.9455
print(f"Fixed magnetic moment = {fixed_moment:.4f}")

# Define U range (extend slightly beyond training range)
U_min, U_max = 4.8, 7.0
U_values = np.linspace(U_min, U_max, 200)

# Create input array: each row = [U, fixed_moment]
X_pred_raw = np.column_stack([U_values, np.full_like(U_values, fixed_moment)])
y_pred = predict(X_pred_raw)

# ============================================================================
# 4. BEAUTIFUL PLOT
# ============================================================================
plt.figure(figsize=(10, 6))

# Prediction curve
plt.plot(U_values, y_pred, 'b-', linewidth=2.5, label=f'KRR prediction (μ = {fixed_moment:.4f})')

# Training points
plt.scatter(df_train['U(eV)'], df_train['band-gap'], 
            s=120, c='darkred', edgecolors='black', linewidth=1.5, 
            alpha=0.9, zorder=5, label='Training data')

# Optional: uncertainty band using bootstrap (load saved coefficients from multiple models?)
# Since we only have one model, we skip confidence band here, but can add a simple
# error bar based on residuals from LOOCV (just for illustration).
# For a more honest uncertainty, you would need to save an ensemble. We'll keep it clean.

# Formatting
plt.xlabel('U (eV)', fontsize=14)
plt.ylabel('Band Gap (eV)', fontsize=14)
plt.title(f'RBF Kernel Ridge Regression Prediction\nγ={gamma}, λ={alpha} | Fixed magnetic moment = {fixed_moment:.4f}', 
          fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.xlim(U_min, U_max)
plt.ylim(3.0, 4.2)

# Add a secondary x-axis for annotated points? Not needed.

# Optionally, mark a specific unknown material (example: U=6.5 eV, same fixed moment)
unknown_U = 6.5
unknown_X = np.array([[unknown_U, fixed_moment]])
unknown_pred = predict(unknown_X)[0]
plt.scatter(unknown_U, unknown_pred, s=200, c='gold', edgecolors='black', 
            linewidth=2, zorder=10, label=f'Prediction for U={unknown_U} eV\n({unknown_pred:.3f} eV)')

plt.tight_layout()
plt.show()

# ============================================================================
# 5. PRINT PREDICTIONS FOR A FEW SPECIFIC U VALUES
# ============================================================================
print("\nPredictions for selected U values (with fixed μ = {:.4f}):".format(fixed_moment))
for u in [5.0, 5.5, 6.0, 6.5, 6.8]:
    pred = predict(np.array([[u, fixed_moment]]))[0]
    print(f"  U = {u:3.1f} eV  →  Band gap = {pred:.4f} eV")