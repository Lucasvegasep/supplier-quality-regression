import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def perform_pre_model_analysis(df):
    """
    Step 1: Exploratory Data Analysis & Hypothesis Testing.
    Project Reference: STRAT-MOD-001
    """
    print("--- STRAT-MOD-001: Pre-Model Analysis ---")
    
    # 1. Descriptive Statistics
    print("\n[1] Descriptive Statistics by Supplier:")
    print(df.groupby('supplier')['tic_temp'].describe())

    # 2. Visualization: Boxplot for Variance Analysis
    # Essential to visualize why Supplier B is more stable.
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='supplier', y='tic_temp', data=df)
    plt.axhline(140, color='red', linestyle='--', label='Target Temp (140°C)')
    plt.title('Curing Temperature Variance by Supplier')
    plt.legend()
    # plt.show() # In GitHub, we document this logic.

    # 3. Hypothesis Testing: t-student (Supplier A vs Supplier B)
    # Testing if Supplier B is significantly closer to target.
    supp_a = df[df['supplier'] == 'Supplier_A']['tic_temp']
    supp_b = df[df['supplier'] == 'Supplier_B']['tic_temp']
    
    t_stat, p_val = stats.ttest_ind(supp_a, supp_b)
    
    print(f"\n[2] Hypothesis Test (t-student):")
    print(f"P-Value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Conclusion: Significant quality difference detected. Proceeding with Supplier B.")
    
    return p_val

if __name__ == "__main__":
    # Simulated data reflecting the study's findings
    data = {
        'supplier': ['Supplier_A']*10 + ['Supplier_B']*10,
        'tic_temp': [132, 135, 131, 138, 133, 134, 130, 136, 132, 133,  # Supp A: High Variance
                     140, 139, 141, 140, 140, 139, 141, 140, 140, 140]   # Supp B: Stable
    }
    sample_df = pd.DataFrame(data)
    perform_pre_model_analysis(sample_df)
