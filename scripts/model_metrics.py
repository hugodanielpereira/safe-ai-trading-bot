# scripts/model_metrics.py
from __future__ import annotations
import joblib
import pandas as pd

def main():
    b = joblib.load("models/gbm.pkl")
    mdl = b.get("model", b)
    cols = b.get("columns")
    print("classes_:", getattr(mdl, "classes_", None))
    try:
        fi = pd.Series(mdl.feature_importances_, index=cols).sort_values(ascending=False)
        print(fi.head(20))
    except Exception as e:
        print("no importances:", e)

if __name__ == "__main__":
    main()