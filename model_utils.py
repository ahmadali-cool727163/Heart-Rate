import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
FEATURES = COLS[:-1]


def load_data(path="processed.cleveland.data"):
    df = pd.read_csv(path, header=None, names=COLS, na_values='?').dropna()
    df['target'] = (df['target'] > 0).astype(int)
    return df


def train(path="processed.cleveland.data"):
    df = load_data(path)
    X, y = df[FEATURES], df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_tr, y_tr)
    return model, model.score(X_te, y_te)


if __name__ == "__main__":
    _, acc = train()
    print(f"Test accuracy: {acc:.3f}")
