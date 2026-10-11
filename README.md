# credit-risk-scorecard（信贷违约分析入门项目）

金融工程 / 风控方向第一个练手项目。目标：用一个真实贷款数据集，做出"不同客群违约率对比"的分析，作为 GitHub 简历项目。

## 三个脚本的区别

| 脚本 | 用什么数据 | 什么时候跑 | 用途 |
|---|---|---|---|
| `starter_analysis.py` | 内置**模拟数据**（不用下载） | 想先看效果、环境还没配好时 | 入门演示，验证 pandas 装好没 |
| `analysis_real.py` | **Kaggle 真实数据** `data/loan.csv` | 已下载好真实数据后 | EDA：违约率分箱对比（描述性分析） |
| `model_real.py` | **Kaggle 真实数据** `data/loan.csv` | EDA 跑通后 | 进阶：逻辑回归预测违约 + AUC / KS + 特征重要性 |

## 运行方式（Windows）

打开 cmd（按 Win 键 → 输入 `cmd` → 回车），粘贴：

```
cd /d D:\WorkBuddy-Output\credit-risk-scorecard
py starter_analysis.py        # 模拟版，零数据即可跑
py analysis_real.py           # 真实版，需先放好 data/loan.csv
```

## 怎么拿到真实数据（Kaggle：yasserh/loan-default-dataset）

1. 打开 https://www.kaggle.com/datasets/yasserh/loan-default-dataset
2. 点 **Download dataset as zip (5 MB)** 下载，解压得到 `Loan_Default.csv`
3. 把 `Loan_Default.csv` 复制到本项目的 `data\` 文件夹，并**改名为 `loan.csv`**（一字不差）
4. 运行 `py analysis_real.py`

## 跑完得到什么（真实版）

**EDA（analysis_real.py）**
- `output/违约率_by_purpose.csv` 按贷款用途
- `output/违约率_by_credit.csv`  按信用分档
- `output/违约率_by_income.csv`  按收入档
- `output/违约率_by_amount.csv`  按贷款金额档
- `output/违约率分析.png`        汇总对比图（可放 GitHub）

**建模（model_real.py，需先 `pip install scikit-learn`）**
- `output/模型评估.png`         ROC 曲线 + Top15 特征系数图
- `output/模型_特征系数.csv`      全部 48 个特征的标准化系数（按重要性排序）

## 数据集说明（真实列名）

- 目标列 `status`：0=未违约，1=违约
- 特征：`loan_amount`(金额)、`income`(年收入)、`credit_score`(信用分)、`loan_purpose`(用途)、`term`、`rate_of_interest`、`ltv`、`dtir1` 等
- 数据约 28MB、有缺失值；脚本会自动转数字、跳过缺失、处理大小写

## 分析结论（真实数据，148,670 条）

| 维度 | 结论 |
|---|---|
| 整体违约率 | **24.64%** |
| 收入 income | **强相关**：<3000 档违约率 36.2%，>7000 后降到 ~20%（收入越低越容易违约） |
| 贷款金额 loan_amount | <15万 档最高 31.8%，随金额上升缓降 |
| 贷款用途 loan_purpose | p2 最高 33.1%，p4 最低 23.0% |
| 信用分 credit_score | **几乎无差异（24.3%–25.2%）——一个反直觉发现** ⚠️ |

> **关键洞察**：这份数据里"信用分"几乎测不出区分度，"收入"和"金额"反而更能区分违约人群。
> 这正说明——**不能凭直觉假设某个变量有用，要用数据验证**。这也是风控建模里做"变量筛选"（IV/WOE）的起点，
> 面试时讲这个"反直觉发现 + 怎么处理"，比单纯说"我算了违约率"更能体现数据思维。

## 建模结果（model_real.py，148,670 条，48 个特征含类别变量）

- **AUC = 0.8628，KS = 0.5896** —— 区分度很强（随机为 0.5，0.8+ 即为强模型）
- **Top 驱动因子**：`credit_type_EQUI`（征信机构类型，最强）、`ltv`（贷款价值比）、`upfront_charges`、`interest_rate_spread`、`income`、`loan_type`
- **信用分 `credit_score` 排第 43 / 48 位**（系数仅 0.0147）——**在完整预测模型里依旧垫底**，再次印证"信用分近乎无用"的反直觉结论
- 这说明：该数据集下违约风险主要由**贷款结构（LTV / 费率 / 首付 / 征信渠道）与收入**决定，而非信用分本身

## 下一步（写进简历前）

1. 把上面的结论 + 图放进一段项目说明（**重点讲"信用分反直觉"这个发现**，面试官爱看数据洞察）
2. 在 GitHub 仓库 `credit-risk-scorecard` 上传新增文件：`model_real.py`、`output/模型评估.png`、`output/模型_特征系数.csv`
   （**不要传 28MB 的 `loan.csv`**，仓库里只放代码、说明和结果图）
3. 可继续：换成 XGBoost / LightGBM 提升 AUC，或做 IV/WOE 变量筛选、评分卡打分（scorecardpy）
