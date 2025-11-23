import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def make_demo_data(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    age = rng.randint(20, 90, size=n)
    num_prev_adm = rng.poisson(1.2, size=n)
    last_hr = rng.normal(80, 12, size=n)
    lab_a = rng.normal(5.0, 1.2, size=n)
    comorbidity_score = rng.randint(0, 5, size=n)

    logits = -3 + 0.02*age + 0.6*num_prev_adm + 0.01*(last_hr-70) + 0.3*comorbidity_score + 0.1*(lab_a-5)
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.rand(n) < probs).astype(int)

    df = pd.DataFrame({
        'age': age,
        'num_prev_adm': num_prev_adm,
        'last_hr': last_hr,
        'lab_a': lab_a,
        'comorbidity_score': comorbidity_score,
        'readmit_30d': y
    })
    return df


def train_and_save(json_path='model/model_params.json'):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    df = make_demo_data()
    X = df.drop(columns=['readmit_30d'])
    y = df['readmit_30d']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    preds = clf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")

    model_json = {
        "feature_names": list(X.columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_.flatten().tolist(),
        "intercept": float(clf.intercept_[0])
    }

    with open(json_path, "w") as f:
        json.dump(model_json, f)

    print(f"Saved model parameters to {json_path}")


if __name__ == "__main__":
    train_and_save()
