import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import shap, warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 120
RND = 42

# ---------------- LOAD & CLEAN (real UCI Taiwan data) ----------------
df = pd.read_csv("data/credit_default_taiwan.csv")
df = df.rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"})
df = df.drop(columns=["ID"])
PAY  = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]   # repayment status, most-recent first
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYA = [f"PAY_AMT{i}"  for i in range(1, 7)]

base_rate = df["default"].mean()
print(f"Rows: {len(df):,} | features: {df.shape[1]-1} | default rate: {base_rate:.1%}")
print(f"Trivial 'approve everyone' accuracy = {1-base_rate:.1%}  <-- why accuracy is the wrong metric here")

# ---------------- FEATURE ENGINEERING (credit-relevant) ----------------
df["utilisation"]   = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(-1, 5)        # balance / credit limit
df["avg_bill"]      = df[BILL].mean(axis=1)
df["avg_payment"]   = df[PAYA].mean(axis=1)
df["payment_ratio"] = (df["avg_payment"] / (df["avg_bill"].abs() + 1)).clip(0, 2)  # how much of the bill is repaid
df["months_delinquent"] = (df[PAY] > 0).sum(axis=1)                          # # of months behind
df["max_delinquency"]   = df[PAY].max(axis=1)
ENG = ["utilisation", "avg_bill", "avg_payment", "payment_ratio", "months_delinquent", "max_delinquency"]

# Deliberately EXCLUDE demographics (SEX, AGE, MARRIAGE, EDUCATION) from the decision model -> fair-lending
FEATURES = ["LIMIT_BAL"] + PAY + BILL + PAYA + ENG
X, y = df[FEATURES], df["default"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RND)
print(f"Train {len(Xtr):,} | Test {len(Xte):,}  (stratified)")

# ---------------- FIG 1: class imbalance + the dominant signal ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
df["default"].value_counts().sort_index().plot(kind="bar", ax=ax[0], color=["#4c72b0", "#c44e52"])
ax[0].set_xticklabels(["Repaid (0)", "Default (1)"], rotation=0)
ax[0].set_title(f"The target is imbalanced: {base_rate:.0%} default"); ax[0].set_ylabel("clients")
rate_by_pay = df.groupby("PAY_1")["default"].mean()
rate_by_pay.plot(kind="bar", ax=ax[1], color="#c44e52")
ax[1].axhline(base_rate, ls="--", color="black", label=f"overall {base_rate:.0%}")
ax[1].set_title("Default rate by most-recent repayment status (PAY_1)")
ax[1].set_xlabel("months delinquent last month  (-2..-1 = paid/no use)"); ax[1].set_ylabel("default rate"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/01_imbalance_and_signal.png"); plt.close()
print("\nDefault rate by recent delinquency (PAY_1):")
print((rate_by_pay.round(3)).to_string())

# ---------------- MODELS ----------------
# 1) Logistic regression (scaled, class-weighted) - the interpretable baseline
scaler = StandardScaler().fit(Xtr)
logit = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
p_logit = logit.predict_proba(scaler.transform(Xte))[:, 1]

# 2) XGBoost (imbalance handled via scale_pos_weight) - the strong model
spw = (ytr == 0).sum() / (ytr == 1).sum()
xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                    colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                    random_state=RND, n_jobs=4)
xgb.fit(Xtr, ytr)
p_xgb = xgb.predict_proba(Xte)[:, 1]

def scores(name, p):
    print(f"  {name:20s} ROC-AUC {roc_auc_score(yte,p):.3f} | PR-AUC {average_precision_score(yte,p):.3f}")
print("\nModel performance (test):")
scores("Logistic Regression", p_logit); scores("XGBoost", p_xgb)
print(f"  {'No-skill (PR-AUC)':20s} = {base_rate:.3f}  (a random model's precision)")

# ---------------- FIG 2: ROC & PR curves ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for name, p, c in [("Logistic", p_logit, "#4c72b0"), ("XGBoost", p_xgb, "#55a868")]:
    fpr, tpr, _ = roc_curve(yte, p); ax[0].plot(fpr, tpr, c=c, label=f"{name} (AUC {roc_auc_score(yte,p):.3f})")
    pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr, c=c, label=f"{name} (AP {average_precision_score(yte,p):.3f})")
ax[0].plot([0,1],[0,1],"k--",lw=1); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("ROC curve"); ax[0].legend()
ax[1].axhline(base_rate,ls="--",color="black",lw=1,label=f"no-skill ({base_rate:.2f})"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall curve (the one that matters under imbalance)"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/02_roc_pr_curves.png"); plt.close()

# ---------------- COST-BASED THRESHOLD (the commercial decision) ----------------
# A missed default (false negative) costs more than a wrongly-declined good customer (false positive).
def optimal_threshold(p, ratio):
    ths = np.linspace(0.01, 0.99, 99); costs = []
    for t in ths:
        pred = (p >= t).astype(int)
        fp = ((pred == 1) & (yte == 0)).sum(); fn = ((pred == 0) & (yte == 1)).sum()
        costs.append(ratio * fn + fp)            # FN is 'ratio' times as costly as FP
    costs = np.array(costs); return ths[costs.argmin()], costs
print("\nCost-optimal decision threshold (XGBoost), by cost ratio FN:FP:")
for r in [2, 5, 10]:
    t, _ = optimal_threshold(p_xgb, r)
    pred = (p_xgb >= t).astype(int)
    rec = ((pred==1)&(yte==1)).sum()/(yte==1).sum(); appr = (pred==0).mean()
    print(f"  FN={r}x FP -> threshold {t:.2f} | catches {rec:.0%} of defaulters | approves {appr:.0%} of applicants")

RATIO = 5
t_star, costs = optimal_threshold(p_xgb, RATIO)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ths = np.linspace(0.01, 0.99, 99)
ax[0].plot(ths, costs, color="#8172b3"); ax[0].axvline(t_star, ls="--", color="#c44e52", label=f"optimal = {t_star:.2f}")
ax[0].axvline(0.5, ls=":", color="grey", label="default 0.5")
ax[0].set_xlabel("decision threshold"); ax[0].set_ylabel(f"expected cost (FN = {RATIO}x FP)")
ax[0].set_title("A business-optimal threshold is well below 0.5"); ax[0].legend()
cm = confusion_matrix(yte, (p_xgb >= t_star).astype(int))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax[1],
            xticklabels=["Approve","Decline"], yticklabels=["Repaid","Defaulted"])
ax[1].set_title(f"Decisions at the cost-optimal threshold ({t_star:.2f})"); ax[1].set_ylabel("actual"); ax[1].set_xlabel("model decision")
plt.tight_layout(); plt.savefig("figures/03_cost_threshold.png"); plt.close()
print(f"\nAt the cost-optimal threshold ({t_star:.2f}, FN={RATIO}x FP):")
print(classification_report(yte, (p_xgb >= t_star).astype(int), target_names=["Repaid","Default"], digits=3))

# ---------------- FIG 4: acceptance curve (the commercial picture) ----------------
order = np.argsort(p_xgb)                     # approve the lowest-risk applicants first
y_ord = yte.values[order]
appr = np.linspace(0.05, 1.0, 60)
bad  = np.array([y_ord[:max(1, int(r*len(y_ord)))].mean() for r in appr])
plt.figure(figsize=(8, 5))
plt.plot(appr*100, bad*100, color="#55a868", lw=2)
plt.axhline(base_rate*100, ls="--", color="black", label=f"approve-everyone default rate ({base_rate:.0%})")
plt.xlabel("Approval rate (%) - approving the safest applicants first")
plt.ylabel("Default rate among approved (%)")
plt.title("Risk-based approval cuts losses: default rate among the approved")
plt.legend(); plt.tight_layout(); plt.savefig("figures/04_acceptance_curve.png"); plt.close()
print("\nAcceptance curve (default rate among approved):")
for r in [0.5, 0.7, 0.9]:
    print(f"  approve safest {r:.0%} -> default rate among approved = {y_ord[:int(r*len(y_ord))].mean():.1%}")

# ---------------- FIG 5: probability calibration ----------------
frac_pos, mean_pred = calibration_curve(yte, p_xgb, n_bins=10, strategy="quantile")
plt.figure(figsize=(7, 5.5))
plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
plt.plot(mean_pred, frac_pos, "o-", color="#8172b3", label="XGBoost (raw scores)")
plt.xlabel("Mean predicted risk"); plt.ylabel("Observed default rate")
plt.title("Calibration: the model ranks risk well, but raw scores need calibrating")
plt.legend(); plt.tight_layout(); plt.savefig("figures/05_calibration.png"); plt.close()

# ---------------- FIG 6: SHAP explainability (why the model decides) ----------------
expl = shap.TreeExplainer(xgb)
samp = Xte.sample(min(2000, len(Xte)), random_state=RND)
sv = expl.shap_values(samp)
plt.figure(); shap.summary_plot(sv, samp, show=False, max_display=12)
plt.title("What drives the model's risk score (SHAP)", fontsize=11); plt.tight_layout()
plt.savefig("figures/06_shap_summary.png", bbox_inches="tight"); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=samp.columns).sort_values(ascending=False)
print("\nTop drivers of predicted default risk (mean |SHAP|):")
print(imp.head(8).round(3).to_string())

# ---------------- FIG 7: score separation + KS statistic (classic credit metric) ----------------
from scipy.stats import ks_2samp
ks = ks_2samp(p_xgb[yte == 1], p_xgb[yte == 0]).statistic
plt.figure(figsize=(8, 5))
sns.kdeplot(p_xgb[yte == 0], fill=True, color="#4c72b0", label="Repaid", clip=(0, 1))
sns.kdeplot(p_xgb[yte == 1], fill=True, color="#c44e52", label="Defaulted", clip=(0, 1))
plt.xlabel("predicted risk score"); plt.ylabel("density")
plt.title(f"The score separates defaulters from non-defaulters (KS = {ks:.2f})")
plt.legend(); plt.tight_layout(); plt.savefig("figures/07_score_separation.png"); plt.close()
print(f"\nKS statistic (rank separation, a standard credit metric) = {ks:.3f}")

# ---------------- FIG 8: explaining a single decision (SHAP waterfall) ----------------
risk_samp = xgb.predict_proba(samp)[:, 1]
idx = int(np.argmax(risk_samp))                       # the highest-risk applicant in the sample
ex = shap.Explanation(values=sv[idx], base_values=float(np.ravel(expl.expected_value)[0]),
                      data=samp.iloc[idx].values, feature_names=list(samp.columns))
plt.figure()
shap.plots.waterfall(ex, max_display=10, show=False)
plt.title(f"Why one applicant scored high-risk (predicted {risk_samp[idx]:.0%})", fontsize=10)
plt.tight_layout(); plt.savefig("figures/08_shap_waterfall.png", bbox_inches="tight"); plt.close()
print(f"Single-decision explanation saved for an applicant scored {risk_samp[idx]:.0%} risk")

# ---------------- SAVE ----------------
out = Xte.copy(); out["actual"] = yte.values; out["risk_score"] = p_xgb
out.to_csv("data/scored_test_set.csv", index=False)
print("\nAll figures + scored test set saved. DONE.")
