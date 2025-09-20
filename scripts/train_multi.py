from __future__ import annotations
import os, argparse, yaml, json
import pandas as pd
import numpy as np
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib

# --- same lightweight features as antes ---
def rsi(s: pd.Series, n=14):
    d = s.diff()
    up = pd.Series(np.where(d>0, d, 0.0), index=s.index).ewm(alpha=1/n, adjust=False).mean()
    dn = pd.Series(np.where(d<0, -d,0.0), index=s.index).ewm(alpha=1/n, adjust=False).mean()
    rs = up/(dn+1e-12); return 100 - (100/(1+rs))

def atr(df: pd.DataFrame, n=14):
    h,l,c1 = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h-l),(h-c1).abs(),(l-c1).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret1"]  = df["close"].pct_change()
    out["ret5"]  = df["close"].pct_change(5)
    out["ret15"] = df["close"].pct_change(15)
    out["vol"]   = df.get("volume", pd.Series(0, index=df.index))
    out["rsi14"] = rsi(df["close"],14)
    out["atr14"] = atr(df,14)/(df["close"]+1e-12)
    out["sma9"]  = df["close"].rolling(9).mean() /(df["close"]+1e-12)
    out["sma21"] = df["close"].rolling(21).mean()/(df["close"]+1e-12)
    m20 = df["close"].rolling(20).mean(); s20 = df["close"].rolling(20).std()
    out["bb_width"]  = (2*s20)/(m20+1e-12)
    out["zscore_20"] = (df["close"]-m20)/(s20+1e-12)
    out["ret1_abs"]  = out["ret1"].abs()
    return out.replace([np.inf,-np.inf],np.nan).fillna(method="ffill").fillna(0.0)

def labels_quantile(df: pd.DataFrame, horizon=10, q=0.62) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    up = fwd.quantile(q); dn = fwd.quantile(1-q)
    y = pd.Series(0, index=df.index)
    y[fwd>=up]=1; y[fwd<=dn]=2
    return y

def train_one(features: pd.DataFrame, y: pd.Series, estimators=800, lr=0.03):
    cnt = Counter(y.tolist()); total = sum(cnt.values())
    cw = {cls: total/(3*max(1,cnt.get(cls,0))) for cls in [0,1,2]}
    tscv = TimeSeriesSplit(n_splits=5)
    accs=[]
    for tr, va in tscv.split(features):
        mdl = LGBMClassifier(objective="multiclass", num_class=3,
                             n_estimators=estimators, learning_rate=lr,
                             subsample=0.9, colsample_bytree=0.9,
                             min_child_samples=50, reg_alpha=1.0, reg_lambda=2.0,
                             class_weight=cw, random_state=42, n_jobs=-1)
        mdl.fit(features.iloc[tr], y.iloc[tr])
        yp = mdl.predict(features.iloc[va])
        rep = classification_report(y.iloc[va], yp, output_dict=True)
        accs.append(rep["accuracy"])
    # final fit
    final = LGBMClassifier(objective="multiclass", num_class=3,
                           n_estimators=estimators, learning_rate=lr,
                           subsample=0.9, colsample_bytree=0.9,
                           min_child_samples=50, reg_alpha=1.0, reg_lambda=2.0,
                           class_weight=cw, random_state=42, n_jobs=-1)
    final.fit(features, y)
    return final, float(np.mean(accs)), cw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/symbols.yaml")
    ap.add_argument("--out_csv", default="models/metrics_summary.csv")
    args = ap.parse_args()

    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)
    symbols = cfg["symbols"]; interval = cfg.get("interval","1m")
    H = int(cfg.get("feature_horizon",10)); q = float(cfg.get("quantile_q",0.62))

    os.makedirs("models", exist_ok=True)
    rows=[]
    for sym in symbols:
        klp = f"data/{sym}/klines.csv"
        if not os.path.exists(klp):
            print(f"! Sem data para {sym}, ignora"); continue
        print(f">> Features+treino {sym}")
        df = pd.read_csv(klp, parse_dates=["open_time","close_time"]).sort_values("open_time")
        feats = make_features(df)
        y = labels_quantile(df, H, q)
        out = feats.copy(); out["y"]=y
        out = out.iloc[21:-H] if H>0 else out.iloc[21:]
        out.dropna(inplace=True)
        if len(out)<1000:
            print(f"! Poucos dados pós-processamento ({len(out)}) em {sym}, skip")
            continue
        X = out.drop(columns=["y"]).astype(float); Y = out["y"].astype(int)
        mdl, acc, cw = train_one(X, Y)
        # salvar
        bundle={"model": mdl, "columns": list(X.columns)}
        mp = f"models/gbm_{sym}.pkl"
        joblib.dump(bundle, mp)
        # métricas simples
        try:
            fi = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
            fi.head(25).to_csv(f"models/feature_importances_{sym}.csv")
        except Exception: pass
        rows.append({"symbol": sym, "interval": interval, "rows": len(out), "cv_accuracy": acc, "model_path": mp})

        # guarda também json individual
        with open(f"models/metrics_{sym}.json","w") as f:
            json.dump({"symbol": sym, "cv_accuracy": acc, "class_weight": cw}, f, indent=2)

    if rows:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f">> Summary -> {args.out_csv}")
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("! Nada treinado")

if __name__ == "__main__":
    main()