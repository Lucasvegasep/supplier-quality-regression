import pandas as pd
import statsmodels.api as sm

def run_predictive_model(df_b):
    """
    Step 2: Multiple Linear Regression for Supplier B.
    Predicts Curing Initiation Temperature (TIC) based on Catalyst dosage and Purity.
    """
    # Definición de variables
    X = df_b[['catalyst_pct', 'purity']]
    y = df_b['tic_temp']
    
    # Agregar constante para el intercepto (Indispensable en OLS)
    X = sm.add_constant(X)
    
    # Ajuste del modelo de Regresión Lineal Múltiple
    model = sm.OLS(y, X).fit()
    
    print("\n--- STRAT-MOD-001: Final Predictive Model Results ---")
    print(model.summary())
    return model

if __name__ == "__main__":
    # Ejemplo de datos para el Proveedor B
    data_b = {
        'catalyst_pct': [1.2, 1.5, 1.8, 2.0, 1.4, 1.7],
        'purity': [98, 98.5, 99, 98.2, 98.8, 99.1],
        'tic_temp': [136, 140, 143, 145, 138, 142]
    }
    df_b = pd.DataFrame(data_b)
    run_predictive_model(df_b)
