# MacroBounce: Predicting Economic Downturn Recovery Speed with TimeNet

## 1. The Problem & Who It Helps
When a country enters a recession, the most important question isn't just how deep the contraction is—it's how fast the economy will bounce back. Predicting early whether a downturn will be **Fast (V-shaped, 1–2 years)** or **Slow (U/L-shaped, 3+ years)** helps:
* **Policymakers & Development Banks:** Calibrating stimulus and debt relief early.
* **Emerging Market Investors & Trade Credit Insurers:** Pricing sovereign risk and identifying post-shock recovery opportunities.
* **Multinational Supply Chain Leaders:** Deciding whether to stay invested or pull back capital during local downturns.

## 2. Dataset in TimeNet Format (`MacroBounce_WDI_Recovery`)
We onboarded 65 years (1960–2024) of official **World Bank WDI** data into TimeNet format (`manifest.json`, `values.parquet`, `control.duckdb`):
* **Scale:** **1,126** labeled recession episodes across **208 countries** (432 Fast / 694 Slow).
* **Multivariate Channels (6 signals, 5-year window `t-4` to `t`):** GDP growth (%), GDP per capita % change, CPI inflation (%), unemployment (%), trade (% of GDP), and gross investment (% of GDP).
* **Target:** Whether GDP per capita returned to its pre-crisis peak within 1–2 years (`fast_v_shaped`) or took 3+ years (`slow_ul_shaped`).

## 3. Key Signal Discovery
*(Insert `chart2_recovery_trajectories.png` here)*

Across 65 years of global recessions, economies that recover quickly look structurally different **years before** the downturn hits:
* **Lower pre-crisis inflation:** ~7–8% vs. ~10–12% for slow recoveries.
* **Higher gross investment:** ~23.5–24% of GDP vs. ~22.5% for slow recoveries.
* **Shallower onset shock:** Milder initial GDP contraction at `t=0`.

## 4. Modeling & Strict Held-Out Evaluation
To prevent data leakage, we evaluated on **42 completely unseen countries** (223 test episodes; trained on 903 episodes from 166 countries).

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: |
| Snapshot Baseline (Logistic Reg, `t=0`) | 68.2% | 59.9% | **74.2%** |
| **Temporal 1D-CNN (5-yr Multivariate)** | 67.7% | **61.7%** | 73.6% |

*(Insert `chart1_model_comparison.png` here)*

**Why Native Temporal AI (OpenTSLM) Matters:** We also tested a zero-shot text LLM (`Qwen2.5-1.5B-Instruct`) on raw serialized trajectories. While fluent, the text-only LLM misread missing values and over-predicted slow recoveries—demonstrating why dedicated multimodal time-series models like OpenTSLM are essential for reliable numeric reasoning.

## 5. Limitations & Next Steps
* **Limitations:** Annual WDI data misses intra-year shocks; smaller economies have missing historical series; exogenous shocks (pandemics, conflicts) cannot be predicted from prior macro data alone.
* **Next Steps:** Expand to quarterly/monthly IMF & central bank series and fine-tune an OpenTSLM soft-prompt checkpoint end-to-end on `MacroBounce_WDI_Recovery`.
<img width="828" height="458" alt="download (11)" src="https://github.com/user-attachments/assets/41aef961-219f-415f-bb96-54da9d4d244b" />
<img width="1378" height="462" alt="download (12)" src="https://github.com/user-attachments/assets/3be11d0c-c3b8-48f7-9d3d-a4a86889d5a8" />
