from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


# logistic Regression 

def t_learner_logistic(df_train, df_test, feature_cols, outcome_col='Y', treatment_col='Treatment'):
    # Split the training set into treated and untreated
    treated = df_train[df_train[treatment_col] == 1]
    untreated = df_train[df_train[treatment_col] == 0]

    # Train models separately for treated and untreated groups
    model_treated = LogisticRegression(max_iter=1000)
    model_treated.fit(treated[feature_cols], treated[outcome_col])

    model_untreated = LogisticRegression(max_iter=1000)
    model_untreated.fit(untreated[feature_cols], untreated[outcome_col])

    # Predict on test set
    mu1_test = model_treated.predict_proba(df_test[feature_cols])[:, 1]
    mu0_test = model_untreated.predict_proba(df_test[feature_cols])[:, 1]
    ite_test = mu1_test - mu0_test

    df_result_test = df_test.copy()
    df_result_test['mu1_hat'] = mu1_test
    df_result_test['mu0_hat'] = mu0_test
    df_result_test['ITE_hat'] = ite_test

    # Predict on training set
    mu1_train = model_treated.predict_proba(df_train[feature_cols])[:, 1]
    mu0_train = model_untreated.predict_proba(df_train[feature_cols])[:, 1]
    ite_train = mu1_train - mu0_train

    df_result_train = df_train.copy()
    df_result_train['mu1_hat'] = mu1_train
    df_result_train['mu0_hat'] = mu0_train
    df_result_train['ITE_hat'] = ite_train

    return df_result_train, df_result_test, model_treated, model_untreated



## RF

def t_learner_random_forest(df_train, df_test, feature_cols, outcome_col='Y', treatment_col='Treatment'):
    # Split training data into treated and untreated
    treated = df_train[df_train[treatment_col] == 1]
    untreated = df_train[df_train[treatment_col] == 0]

    # Train separate random forest classifiers
    model_treated = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=0)
    model_treated.fit(treated[feature_cols], treated[outcome_col])

    model_untreated = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=0)
    model_untreated.fit(untreated[feature_cols], untreated[outcome_col])

    # Predict potential outcomes on test set
    mu1_test = model_treated.predict_proba(df_test[feature_cols])[:, 1]
    mu0_test = model_untreated.predict_proba(df_test[feature_cols])[:, 1]
    ite_test = mu1_test - mu0_test

    df_result_test = df_test.copy()
    df_result_test['mu1_hat'] = mu1_test
    df_result_test['mu0_hat'] = mu0_test
    df_result_test['ITE_hat'] = ite_test

    # Predict potential outcomes on train set
    mu1_train = model_treated.predict_proba(df_train[feature_cols])[:, 1]
    mu0_train = model_untreated.predict_proba(df_train[feature_cols])[:, 1]
    ite_train = mu1_train - mu0_train

    df_result_train = df_train.copy()
    df_result_train['mu1_hat'] = mu1_train
    df_result_train['mu0_hat'] = mu0_train
    df_result_train['ITE_hat'] = ite_train

    return df_result_train, df_result_test, model_treated, model_untreated


# Tuned RF

def t_learner_random_forest_tuned(df_train, df_test, feature_cols, outcome_col='Y', treatment_col='Treatment'):
    # Split training data into treated and untreated
    treated = df_train[df_train[treatment_col] == 1]
    untreated = df_train[df_train[treatment_col] == 0]

    # Use tuned hyperparameters
    model_treated = RandomForestClassifier(
        n_estimators=700,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=5,
        oob_score=True,
        random_state=0
    )
    model_treated.fit(treated[feature_cols], treated[outcome_col])

    model_untreated = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=10,
        oob_score=True,
        random_state=0
    )
    model_untreated.fit(untreated[feature_cols], untreated[outcome_col])

    # Predict potential outcomes on test set
    mu1_test = model_treated.predict_proba(df_test[feature_cols])[:, 1]
    mu0_test = model_untreated.predict_proba(df_test[feature_cols])[:, 1]
    ite_test = mu1_test - mu0_test

    df_result_test = df_test.copy()
    df_result_test['mu1_hat'] = mu1_test
    df_result_test['mu0_hat'] = mu0_test
    df_result_test['ITE_hat'] = ite_test

    # Predict potential outcomes on train set
    mu1_train = model_treated.predict_proba(df_train[feature_cols])[:, 1]
    mu0_train = model_untreated.predict_proba(df_train[feature_cols])[:, 1]
    ite_train = mu1_train - mu0_train

    df_result_train = df_train.copy()
    df_result_train['mu1_hat'] = mu1_train
    df_result_train['mu0_hat'] = mu0_train
    df_result_train['ITE_hat'] = ite_train

    return df_result_train, df_result_test, model_treated, model_untreated


## NN

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_nn_model(X, y, input_dim, X_val=None, y_val=None,
                   epochs=100, batch_size=128, lr=1e-3, seed=0,
                   patience=10, min_delta=1e-4):
    torch.manual_seed(seed)

    # Prepare datasets
    train_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        val_data = (torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    else:
        val_data = None

    model = SimpleMLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation step
        if val_data:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_data[0])
                val_loss = criterion(val_pred, val_data[1]).item()
            history['val_loss'].append(val_loss)

            # Early stopping check
            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            history['val_loss'].append(None)

    return model, history


def plot_nn_training_history(history, title='Training History'):
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    if any(history['val_loss']):
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def predict_nn_model(model, X):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    return preds

def t_learner_nn(df_train, df_test, feature_cols, outcome_col='Y', treatment_col='Treatment', val_frac=0.2, **kwargs):
    from sklearn.model_selection import train_test_split

    X_train = df_train[feature_cols].values
    y_train = df_train[outcome_col].values
    t_train = df_train[treatment_col].values

    X_test = df_test[feature_cols].values

    # Treated model
    X_treat = X_train[t_train == 1]
    y_treat = y_train[t_train == 1]
    X_treat_train, X_treat_val, y_treat_train, y_treat_val = train_test_split(X_treat, y_treat, test_size=val_frac)
    model_treat, history_treat = train_nn_model(X_treat_train, y_treat_train, input_dim=X_train.shape[1],
                                                X_val=X_treat_val, y_val=y_treat_val, **kwargs)

    # Untreated model
    X_control = X_train[t_train == 0]
    y_control = y_train[t_train == 0]
    X_control_train, X_control_val, y_control_train, y_control_val = train_test_split(X_control, y_control, test_size=val_frac)
    model_control, history_control = train_nn_model(X_control_train, y_control_train, input_dim=X_train.shape[1],
                                                    X_val=X_control_val, y_val=y_control_val, **kwargs)

    # Predictions
    mu1_test = predict_nn_model(model_treat, X_test)
    mu0_test = predict_nn_model(model_control, X_test)
    ite_test = mu1_test - mu0_test

    mu1_train = predict_nn_model(model_treat, X_train)
    mu0_train = predict_nn_model(model_control, X_train)
    ite_train = mu1_train - mu0_train

    df_result_test = df_test.copy()
    df_result_test['mu1_hat'] = mu1_test
    df_result_test['mu0_hat'] = mu0_test
    df_result_test['ITE_hat'] = ite_test

    df_result_train = df_train.copy()
    df_result_train['mu1_hat'] = mu1_train
    df_result_train['mu0_hat'] = mu0_train
    df_result_train['ITE_hat'] = ite_train

    return df_result_train, df_result_test, model_treat, model_control, history_treat, history_control





## RF CV Tuning



def t_learner_rf_with_tuning(df_train, df_test, feature_cols, outcome_col='Y', treatment_col='Treatment', cv=3):
    # Split training data into treated and untreated
    treated = df_train[df_train[treatment_col] == 1]
    untreated = df_train[df_train[treatment_col] == 0]

    # Expanded hyperparameter grid
    param_grid = {
        'n_estimators': [100, 300, 500, 700],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }

    # Grid search for treated
    rf_treated = RandomForestClassifier(random_state=0)
    grid_treated = GridSearchCV(rf_treated, param_grid, cv=cv, n_jobs=-1, scoring='neg_log_loss', verbose=1)
    grid_treated.fit(treated[feature_cols], treated[outcome_col])
    best_model_treated = grid_treated.best_estimator_

    # Grid search for untreated
    rf_untreated = RandomForestClassifier(random_state=0)
    grid_untreated = GridSearchCV(rf_untreated, param_grid, cv=cv, n_jobs=-1, scoring='neg_log_loss', verbose=1)
    grid_untreated.fit(untreated[feature_cols], untreated[outcome_col])
    best_model_untreated = grid_untreated.best_estimator_

    # Predict on test set
    mu1_test = best_model_treated.predict_proba(df_test[feature_cols])[:, 1]
    mu0_test = best_model_untreated.predict_proba(df_test[feature_cols])[:, 1]
    ite_test = mu1_test - mu0_test

    df_result_test = df_test.copy()
    df_result_test['mu1_hat'] = mu1_test
    df_result_test['mu0_hat'] = mu0_test
    df_result_test['ITE_hat'] = ite_test

    # Predict on train set
    mu1_train = best_model_treated.predict_proba(df_train[feature_cols])[:, 1]
    mu0_train = best_model_untreated.predict_proba(df_train[feature_cols])[:, 1]
    ite_train = mu1_train - mu0_train

    df_result_train = df_train.copy()
    df_result_train['mu1_hat'] = mu1_train
    df_result_train['mu0_hat'] = mu0_train
    df_result_train['ITE_hat'] = ite_train

    return df_result_train, df_result_test, best_model_treated, best_model_untreated, grid_treated, grid_untreated
