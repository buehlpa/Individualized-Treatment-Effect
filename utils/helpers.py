
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def split_train_val_calib_test(df, val_size=0.2, calib_size=0.2, test_size=0.2, seed=12345):
    """
    Splits the data into train, validation, calibration, and test sets.
    Sizes are fractions of the full dataset.
    """
    assert val_size + calib_size + test_size < 1.0, "Combined val+calib+test size must be < 1"

    df_temp, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    df_temp, df_calib = train_test_split(df_temp, test_size=calib_size / (1 - test_size), random_state=seed)
    df_train, df_val = train_test_split(df_temp, test_size=val_size / (1 - test_size - calib_size), random_state=seed)

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_calib.reset_index(drop=True),
        df_test.reset_index(drop=True)
    )
    
    
    
    
def plot_estimated_vs_true_ite_colored(df_result, title_suffix=''):
    if 'ITE_true' not in df_result.columns:
        print(f"No 'ITE_true' column found in dataset. Skipping plot {title_suffix}.")
        return

    plt.figure(figsize=(8, 6))

    # Scatter: untreated
    untreated = df_result[df_result['Treatment'] == 0]
    plt.scatter(untreated['ITE_true'], untreated['ITE_hat'], alpha=0.2, label='Untreated (T=0)', color='blue')

    # Scatter: treated
    treated = df_result[df_result['Treatment'] == 1]
    plt.scatter(treated['ITE_true'], treated['ITE_hat'], alpha=0.2, label='Treated (T=1)', color='orange')

    # Diagonal reference line
    min_val = min(df_result['ITE_true'].min(), df_result['ITE_hat'].min())
    max_val = max(df_result['ITE_true'].max(), df_result['ITE_hat'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal: Estimated = True')

    plt.xlabel('True ITE')
    plt.ylabel('Estimated ITE')
    plt.title(f'Estimated vs. True ITE {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_estimated_vs_true_probabilities(df_result, title_suffix=''):
    if 'Y_prob' not in df_result.columns:
        print(f"No 'Y_prob' column found in dataset. Skipping plot {title_suffix}.")
        return

    plt.figure(figsize=(8, 6))
    
    # Plot for untreated
    untreated = df_result[df_result['Treatment'] == 0]
    plt.scatter(untreated['Y_prob'], untreated['mu0_hat'], alpha=0.2, label='Untreated (T=0)', color='blue')

    # Plot for treated
    treated = df_result[df_result['Treatment'] == 1]
    plt.scatter(treated['Y_prob'], treated['mu1_hat'], alpha=0.2, label='Treated (T=1)', color='orange')

    # Diagonal reference line
    min_val = min(df_result['Y_prob'].min(), df_result[['mu0_hat', 'mu1_hat']].min().min())
    max_val = max(df_result['Y_prob'].max(), df_result[['mu0_hat', 'mu1_hat']].max().max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal: Estimated = True')

    plt.xlabel('True Probability (Y_prob)')
    plt.ylabel('Estimated Probability')
    plt.title(f'Estimated vs. True Probabilities {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
