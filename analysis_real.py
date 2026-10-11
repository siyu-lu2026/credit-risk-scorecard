# -*- coding: utf-8 -*-
"""
项目：credit-risk-scorecard（信贷违约分析 · 真实数据版）
适配数据集：Kaggle 的 yasserh/loan-default-dataset（文件 Loan_Default.csv）

【怎么运行（Windows）】
  1. 把 Kaggle 下载的 Loan_Default.csv 放进 data 文件夹，改名成 loan.csv
  2. 打开 cmd，粘贴下面两行，回车：
        cd /d D:\WorkBuddy-Output\credit-risk-scorecard
        py analysis_real.py

【跑完得到什么】
  output\违约率_by_purpose.csv     按贷款用途的违约率
  output\违约率_by_credit.csv      按信用分档的违约率
  output\违约率_by_income.csv      按收入档的违约率
  output\违约率_by_amount.csv      按贷款金额档的违约率
  output\违约率分析.png            汇总对比图（可放 GitHub）

【它会自动做的事】
  - 读入真实数据（列名大小写自动统一）
  - 把数值列转成数字、处理缺失值
  - 按 用途 / 信用分 / 收入 / 金额 分组看违约率
  - 出图 + 存表
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # 不弹窗，直接存图
import matplotlib.pyplot as plt

# 中文显示（Windows 自带微软雅黑 / 黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "loan.csv"
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)


# ============================================================
# 第 1 步：读数据（真实数据）
# ============================================================
def load_data():
    if not DATA.exists():
        raise SystemExit(
            f"[错误] 找不到 {DATA}\n"
            "请把 Kaggle 下载的 Loan_Default.csv 放进 data 文件夹，并改名为 loan.csv\n"
            "（改完名再运行本脚本）"
        )
    # low_memory=False：数据大、列多，避免类型推断告警
    df = pd.read_csv(DATA, low_memory=False)
    # 统一小写列名，避免大小写不一致导致找不到列
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


# ============================================================
# 第 2 步：清洗（转数字、处理缺失）
# ============================================================
def clean(df):
    if "status" not in df.columns:
        raise SystemExit(
            "[错误] 数据里没有 'status' 列（违约标签）。\n"
            "请确认下载的是 yasserh/loan-default-dataset，并已改名 loan.csv"
        )
    # 目标列：status 已经是 0/1，保险起见强制转 int
    df["status"] = pd.to_numeric(df["status"], errors="coerce")

    # 常见数值列：统一转数字（带 '%'、'-' 等非数字会变成 NaN，后面分组会跳过）
    num_cols = ["loan_amount", "income", "credit_score", "rate_of_interest",
                "term", "ltv", "dtir1", "property_value",
                "interest_rate_spread", "upfront_charges"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# 第 3 步：分析
# ============================================================
def analyze(df):
    line = "=" * 56
    print(line)
    print("[1] 数据概览")
    print(line)
    print(f"行 x 列: {df.shape[0]} 行 x {df.shape[1]} 列")
    print("列名:", list(df.columns))
    print("\n前 5 行：")
    print(df.head().to_string(index=False))
    print("\n缺失值（前 15 多的列）：")
    miss = df.isna().sum().sort_values(ascending=False).head(15)
    print(miss.to_string())

    print("\n" + line)
    print("[2] 整体违约率（status=1 表示违约）")
    print(line)
    overall = df["status"].mean()
    print(f"样本数: {len(df)}    违约数: {int(df['status'].sum())}")
    print(f"整体违约率: {overall:.2%}")

    results = {}

    # [3] 按贷款用途
    if "loan_purpose" in df.columns:
        print("\n" + line)
        print("[3] 按贷款用途(loan_purpose)看违约率")
        print(line)
        g = (df.groupby("loan_purpose")["status"]
               .agg(笔数="count", 违约率="mean")
               .sort_values("违约率", ascending=False))
        g["违约率%"] = (g["违约率"] * 100).round(2)
        print(g.to_string())
        results["purpose"] = g

    # [4] 按信用分分箱
    if "credit_score" in df.columns:
        print("\n" + line)
        print("[4] 按信用分(credit_score)分箱看违约率")
        print(line)
        d = df.dropna(subset=["credit_score"]).copy()
        d["信用分档"] = pd.cut(d["credit_score"],
                             bins=[0, 600, 700, 800, 1000],
                             labels=["<600", "600-700", "700-800", ">800"])
        g = d.groupby("信用分档", observed=True)["status"].agg(笔数="count", 违约率="mean")
        g["违约率%"] = (g["违约率"] * 100).round(2)
        print(g.to_string())
        results["credit"] = g

    # [5] 按收入分箱（本数据集 income 值域约 0–58万，中位数约 5760）
    if "income" in df.columns:
        print("\n" + line)
        print("[5] 按收入(income)分箱看违约率")
        print(line)
        d = df.dropna(subset=["income"]).copy()
        d["收入档"] = pd.cut(d["income"],
                           bins=[0, 3000, 5000, 7000, 10000, 1e9],
                           labels=["<3000", "3000-5000", "5000-7000",
                                   "7000-10000", ">10000"])
        g = d.groupby("收入档", observed=True)["status"].agg(笔数="count", 违约率="mean")
        g["违约率%"] = (g["违约率"] * 100).round(2)
        print(g.to_string())
        results["income"] = g

    # [6] 按贷款金额分箱（本数据集 loan_amount 值域约 1.65万–357万，中位数约 29.6万）
    if "loan_amount" in df.columns:
        print("\n" + line)
        print("[6] 按贷款金额(loan_amount)分箱看违约率")
        print(line)
        d = df.dropna(subset=["loan_amount"]).copy()
        d["金额档"] = pd.cut(d["loan_amount"],
                           bins=[0, 150000, 250000, 350000, 500000, 1e9],
                           labels=["<15万", "15-25万", "25-35万", "35-50万", ">50万"])
        g = d.groupby("金额档", observed=True)["status"].agg(笔数="count", 违约率="mean")
        g["违约率%"] = (g["违约率"] * 100).round(2)
        print(g.to_string())
        results["amount"] = g

    return results


# ============================================================
# 第 4 步：存表 + 画图
# ============================================================
def save_and_plot(results):
    # 存 CSV（每张表单独存，方便查看）
    for name, g in results.items():
        g.drop(columns=["违约率"] if "违约率" in g.columns else []).to_csv(
            OUT / f"违约率_by_{name}.csv", encoding="utf-8-sig")
    print("\n[已保存] output 下的 违约率_by_*.csv")

    # 画图：最多 4 个子图（用途/信用分/收入/金额）
    items = list(results.items())[:4]
    n = len(items)
    rows = (n + 1) // 2
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    titles = {"purpose": "各贷款用途的违约率", "credit": "各信用分档的违约率",
              "income": "各收入档的违约率", "amount": "各贷款金额档的违约率"}
    colors = {"purpose": "#2c7fb8", "credit": "#c0392b",
              "income": "#27ae60", "amount": "#e67e22"}

    for i, (name, g) in enumerate(items):
        ax = axes[i]
        labels = [str(x) for x in g.index]
        vals = (g["违约率"] * 100).values
        ax.bar(range(len(labels)), vals, color=colors.get(name, "#34495e"))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30 if name == "purpose" else 0,
                           ha="right" if name == "purpose" else "center")
        ax.set_title(titles.get(name, name))
        ax.set_ylabel("违约率 (%)")
        ax.grid(axis="y", alpha=0.3)
        for j, v in enumerate(vals):
            ax.text(j, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # 隐藏多余的子图
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("信贷违约率分析（真实数据）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "违约率分析.png", dpi=150)
    print(f"[已保存] {OUT / '违约率分析.png'}")


# ============================================================
# 主流程
# ============================================================
def main():
    df = load_data()
    df = clean(df)
    results = analyze(df)
    save_and_plot(results)
    print("\n全部完成。打开 output 文件夹看结果吧。")


if __name__ == "__main__":
    main()
