# scripts/train_lightgbm.py
from __future__ import annotations
import argparse, os
import joblib
import pandas as pd
from collections import Counter

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--model_out", default="models/gbm.pkl")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--estimators", type=int, default=800)  # um pouco mais profundo
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    df = pd.read_csv(args.features)

    y = df["y"].astype(int)
    X = df.drop(columns=["y"]).astype(float)

    # distribuição de classes
    cnt = Counter(y.tolist())
    print("Label distribution:", dict(cnt))
    total = sum(cnt.values())
    class_weight = {cls: total / (3 * max(1, cnt.get(cls, 0))) for cls in [0, 1, 2]}
    print("Class weight:", class_weight)

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    reports = []

    for i, (tr, va) in enumerate(tscv.split(X)):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            learning_rate=args.learning_rate,
            n_estimators=args.estimators,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1.0,
            reg_lambda=2.0,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X.iloc[tr], y.iloc[tr])
        yp = model.predict(X.iloc[va])

        # métricas por fold
        cm = confusion_matrix(y.iloc[va], yp, labels=[0, 1, 2])
        rep = classification_report(y.iloc[va], yp, digits=3, output_dict=True)
        reports.append(rep)

        print(f"\nFold {i+1} accuracy={rep['accuracy']:.4f}")
        print("Confusion matrix (rows=true, cols=pred) [HOLD,BUY,SELL]:")
        print(cm)
        print("Report:", {k: (v if isinstance(v, float) else v.get('f1-score', None)) for k, v in rep.items() if k in ['accuracy', '0', '1', '2']})

    # treino final em TODO o dataset (usa os mesmos hiperparams e class_weight)
    final = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=args.learning_rate,
        n_estimators=args.estimators,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=2.0,
        class_weight=class_weight,  # <-- agora também no final
        random_state=42,
        n_jobs=-1,
    )
    final.fit(X, y)

    # guardar importâncias das features (útil para debugging)
    try:
        fi = pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False)
        os.makedirs("models", exist_ok=True)
        fi.to_csv("models/feature_importances.csv")
        print("Saved feature importances to models/feature_importances.csv")
    except Exception:
        pass

    bundle = {"model": final, "columns": list(X.columns)}
    joblib.dump(bundle, args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()