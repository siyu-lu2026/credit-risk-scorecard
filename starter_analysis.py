# -*- coding: utf-8 -*-
"""
项目：credit-risk-scorecard（信贷违约分析入门）
文件：starter_analysis.py

【这份脚本干什么】
  1. 生成一份"模拟贷款数据" —— 不用联网、不用下载任何东西，双击就能跑
  2. 用 pandas 做最基础的分析：
       看数据长什么样  ->  算整体违约率  ->  按等级/用途/收入分组对比
  3. 结果自动存成 output 文件夹里的 CSV + 一张图

【怎么运行（Windows）】
  打开"命令提示符"(cmd)，粘贴下面两行，回车：
      cd /d D:\WorkBuddy-Output\credit-risk-scorecard
      py starter_analysis.py

【跑完之后得到什么】
  output\分析结果汇总.csv        <- 分析表格
  output\违约率对比.png          <- 一张对比图（可放进 GitHub）
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # 不弹窗，直接存图
import matplotlib.pyplot as plt

# 中文显示设置（Windows 自带微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 输出目录
BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)


# ============================================================
# 第 1 步：生成模拟数据（真实项目里这步是"读入 CSV"）
# ============================================================
def make_fake_loan_data(n=5000, seed=42):
    """造一份假的贷款数据，字段模仿真实信贷数据集。"""
    rng = np.random.default_rng(seed)

    grades = ["A", "B", "C", "D", "E", "F", "G"]
    # 信用等级越低，违约率越高
    grade_rate = {"A": 0.02, "B": 0.04, "C": 0.07, "D": 0.12, "E": 0.18, "F": 0.25, "G": 0.33}
    purposes = ["debt_consolidation", "credit_card", "home_improvement",
                "major_purchase", "small_business", "medical"]

    grade = rng.choice(grades, size=n, p=[0.20, 0.26, 0.22, 0.14, 0.09, 0.05, 0.04])
    purpose = rng.choice(purposes, size=n)
    annual_inc = np.round(rng.lognormal(mean=11.0, sigma=0.5, size=n), 0)   # 年收入
    loan_amnt = np.round(rng.uniform(1000, 40000, size=n), 0)               # 贷款金额
    gidx = np.array([grades.index(g) for g in grade])
    int_rate = np.round(5.5 + gidx * 3.2 + rng.normal(0, 0.8, size=n), 2)   # 利率

    # 违约概率 = 等级基准率 * 收入影响（高收入更不易违约）
    inc_factor = np.clip(120000 / annual_inc, 0.5, 2.0)
    p_default = np.clip(np.array([grade_rate[g] for g in grade]) * inc_factor, 0.005, 0.9)
    default = (rng.random(n) < p_default).astype(int)

    return pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "grade": grade,
        "purpose": purpose,
        "int_rate": int_rate,
        "default": default,
    })


# ============================================================
# 第 2 步：读数据（想换真实数据，就改这一行）
# ============================================================
def load_data():
    """默认用模拟数据。想用 Kaggle 真实数据见文件末尾注释。"""
    return make_fake_loan_data()


# ============================================================
# 第 3 步：分析
# ============================================================
def analyze(df: pd.DataFrame):
    line = "=" * 56

    # --- 3.1 看数据长什么样 ---
    print(line)
    print("[1] 数据长什么样")
    print(line)
    print(f"行数 x 列数: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}")
    print("\n前 5 行：")
    print(df.head().to_string(index=False))
    print("\n每列的缺失值个数：")
    print(df.isna().sum().to_string())

    # --- 3.2 整体违约率 ---
    print("\n" + line)
    print("[2] 整体违约率")
    print(line)
    overall = df["default"].mean()
    print(f"违约笔数: {df['default'].sum()} / {len(df)}")
    print(f"整体违约率: {overall:.2%}")

    # --- 3.3 按信用等级 ---
    print("\n" + line)
    print("[3] 按信用等级(grade)看违约率  —— 等级越低应越高")
    print(line)
    by_grade = (df.groupby("grade")["default"]
                  .agg(笔数="count", 违约率="mean")
                  .sort_index())
    by_grade["违约率"] = by_grade["违约率"].map(lambda x: f"{x:.2%}")
    print(by_grade.to_string())

    # --- 3.4 按贷款用途 ---
    print("\n" + line)
    print("[4] 按贷款用途(purpose)看违约率")
    print(line)
    by_purpose = (df.groupby("purpose")["default"]
                    .agg(笔数="count", 违约率="mean")
                    .sort_values("违约率", ascending=False))
    by_purpose["违约率"] = by_purpose["违约率"].map(lambda x: f"{x:.2%}")
    print(by_purpose.to_string())

    # --- 3.5 按收入分组 ---
    print("\n" + line)
    print("[5] 按年收入分组看违约率  —— 收入越高应越低")
    print(line)
    df = df.copy()
    df["收入分组"] = pd.cut(df["annual_inc"],
                         bins=[0, 60000, 120000, 260000, np.inf],
                         labels=["<6万", "6-12万", "12-26万", ">26万"])
    by_inc = (df.groupby("收入分组", observed=True)["default"]
                .agg(笔数="count", 违约率="mean"))
    by_inc["违约率"] = by_inc["违约率"].map(lambda x: f"{x:.2%}")
    print(by_inc.to_string())

    return by_grade, by_purpose, by_inc


# ============================================================
# 第 4 步：存结果 + 画图
# ============================================================
def save_and_plot(df, by_grade, by_purpose):
    # 存 CSV（把三张表拼成一份汇总）
    summary = by_grade.copy()
    summary.to_csv(OUT / "分析结果汇总.csv", encoding="utf-8-sig")
    print(f"\n[已保存] {OUT / '分析结果汇总.csv'}")

    # 画图：左=按等级，右=按用途
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    gr = df.groupby("grade")["default"].mean()
    axes[0].bar(gr.index, gr.values, color="#c0392b")
    axes[0].set_title("各信用等级的违约率")
    axes[0].set_xlabel("信用等级")
    axes[0].set_ylabel("违约率")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    pr = df.groupby("purpose")["default"].mean().sort_values()
    axes[1].barh(pr.index, pr.values, color="#2c7fb8")
    axes[1].set_title("各贷款用途的违约率")
    axes[1].set_xlabel("违约率")
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    for ax in axes:
        ax.grid(axis="x" if ax is axes[1] else "y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "违约率对比.png", dpi=150)
    print(f"[已保存] {OUT / '违约率对比.png'}")


# ============================================================
# 主流程
# ============================================================
def main():
    df = load_data()
    by_grade, by_purpose, by_inc = analyze(df)
    save_and_plot(df, by_grade, by_purpose)
    print("\n全部完成。打开 output 文件夹看结果吧。")


if __name__ == "__main__":
    main()


# ============================================================
# 【换成真实数据】—— 等你从 Kaggle 下载了贷款数据集之后
# ============================================================
# 1) 把下载的 csv 放到本项目 data 文件夹下，改名成 loan.csv
# 2) 把上面 load_data() 函数改成：
#
#       def load_data():
#           path = BASE / "data" / "loan.csv"
#           df = pd.read_csv(path)
#           # 列名要和下面代码用的一致，真实数据列名不同时在这里改名：
#           # df = df.rename(columns={"loan_amnt": "loan_amnt",
#           #                         "grade": "grade",
#           #                         "purpose": "purpose",
#           #                         "annual_inc": "annual_inc",
#           #                         "loan_status": "default"})
#           # 真实数据里违约通常是文本，需要转成 0/1，例如：
#           # df["default"] = (df["default"] == "Charged Off").astype(int)
#           return df
#
# 3) 再运行一次 py starter_analysis.py 即可。
