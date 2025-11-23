# src/train_model.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from data_preprocess import load_data, clean_data, split_xy, build_preprocessor
import os

def train(path='data/Telco-Customer-Churn.csv', model_out='models/churn_model.pkl'):
    # 1. load and clean
    df = load_data(path)
    df = clean_data(df)

    # 2. split X, y
    X, y = split_xy(df, target_col='Churn')

    # 3. build preprocessor
    preprocessor = build_preprocessor(X)

    # 4. train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. build model pipeline (preprocessing + classifier)
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # 6. simple hyperparameter tuning (small grid — fast)
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [8, 12, None],
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best_model = grid.best_estimator_

    # 7. evaluate
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:,1]
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))

    # 8. save model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(best_model, model_out)
    print(f"Saved model to {model_out}")

if __name__ == "__main__":
    train()