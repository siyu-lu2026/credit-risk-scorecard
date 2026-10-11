"""
model_real.py — 信用风险评分卡进阶版：逻辑回归 + AUC / KS
数据集：data/loan.csv（Kaggle yasserh/loan-default-dataset）
目标：用真实数据训练逻辑回归预测违约(status)，评估区分能力(AUC/KS)，
      并量化每个特征对违约的影响（标准化系数），验证"信用分几乎没用"的反直觉结论。
运行：py model_real.py
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

TARGET = 'status'
DATA = 'data/loan.csv'
RANDOM_STATE = 42

# ---------- 1. 读数据 ----------
df = pd.read_csv(DATA)
df.columns = [c.strip().lower() for c in df.columns]
df = df.drop(columns=[c for c in ['id'] if c in df.columns], errors='ignore')
y = df[TARGET].astype(int)

# ---------- 2. 特征工程 ----------
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
# pandas 3.x 里字符串列的 dtype 是 'str'（不是 object），必须单独纳入
cat_cols = [c for c in df.select_dtypes(include=['object', 'string']).columns if c != TARGET]

X = df[num_cols + cat_cols].copy()
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')
    X[c] = X[c].fillna(X[c].median()).fillna(0)

if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 去掉零方差列，避免标准化除零
keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1 and X[c].std() > 0]
X = X[keep]
feature_names = list(X.columns)

# ---------- 3. 标准化 + 切分 ----------
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    Xs, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

# ---------- 4. 训练逻辑回归 ----------
model = LogisticRegression(max_iter=3000, C=1.0)
model.fit(X_train, y_train)

# ---------- 5. 评估 ----------
prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, prob)


def ks_stat(y_true, y_score):
    order = np.argsort(y_score)
    yt = np.asarray(y_true)[order]
    cum_bad = np.cumsum(yt) / (yt.sum() if yt.sum() else 1)
    cum_good = np.cumsum(1 - yt) / ((1 - yt).sum() if (1 - yt).sum() else 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


ks = ks_stat(y_test, prob)

# ---------- 6. 特征重要性（标准化系数绝对值） ----------
coef = model.coef_[0]
imp = pd.DataFrame({'feature': feature_names, 'coef': coef,
                    'abs_coef': np.abs(coef)}).sort_values('abs_coef', ascending=False)
imp.to_csv('output/模型_特征系数.csv', index=False, encoding='utf-8-sig')

# ---------- 7. 绘图 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fpr, tpr, _ = roc_curve(y_test, prob)
axes[0].plot(fpr, tpr, color='#c0392b', lw=2, label=f'ROC (AUC={auc:.3f})')
axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(loc='lower right')

top = imp.head(15).iloc[::-1]
colors = ['#2980b9' if v >= 0 else '#c0392b' for v in top['coef']]
axes[1].barh(top['feature'], top['coef'], color=colors)
axes[1].axvline(0, color='black', lw=0.8)
axes[1].set_title('Top 15 Feature Coefficients\n(blue=raises default prob, red=lowers)')
axes[1].set_xlabel('Logistic Regression Standardized Coef')

plt.tight_layout()
plt.savefig('output/模型评估.png', dpi=110)
plt.close()

# ---------- 8. 输出结论 ----------
print(f"样本量: {len(df):,}  | 整体违约率: {y.mean()*100:.2f}%")
print(f"训练集: {len(y_train):,}   测试集: {len(y_test):,}")
print(f"AUC = {auc:.4f}    KS = {ks:.4f}")
print("\nTop 10 特征（按 |系数|）:")
print(imp.head(10).to_string(index=False))

if 'credit_score' in imp['feature'].values:
    rank = int(imp.reset_index(drop=True).index[imp['feature'] == 'credit_score'][0]) + 1
    total = len(imp)
    cs = float(imp.loc[imp['feature'] == 'credit_score', 'coef'].values[0])
    print(f"\n信用分 credit_score 在 {total} 个特征中排第 {rank} 位（系数={cs:.4f}）")
