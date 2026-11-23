!pip install -q duckdb pyarrow scikit-learn matplotlib seaborn torch requests

import os, json, requests
import numpy as np
import pandas as pd
import duckdb

# 1. دریافت لیست کشورهای واقعی (حذف تجمیع‌های منطقه‌ای مثل World یا High Income)
c_resp = requests.get("https://api.worldbank.org/v2/country?format=json&per_page=400", timeout=30).json()
valid_countries = {
    c["id"]: {"name": c["name"], "region": c["region"]["value"], "income": c["incomeLevel"]["value"]}
    for c in c_resp[1] if c["region"]["id"] != "NA"
}
print(f"Total valid countries found: {len(valid_countries)}")

# 2. شاخص‌های کلیدی سری‌زمانی
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "NY.GDP.PCAP.KD":    "gdp_per_capita",
    "FP.CPI.TOTL.ZG":    "inflation",
    "SL.UEM.TOTL.ZS":    "unemployment",
    "NE.TRD.GNFS.ZS":    "trade_pct_gdp",
    "NE.GDI.TOTL.ZS":    "investment_pct_gdp",
}

def fetch_wb_indicator(code, col_name):
    rows = []
    page = 1
    while True:
        url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000&date=1960:2024&page={page}"
        resp = requests.get(url, timeout=60).json()
        if len(resp) < 2 or not resp[1]:
            break
        for item in resp[1]:
            iso3 = item["countryiso3code"]
            if iso3 in valid_countries and item["value"] is not None:
                rows.append((iso3, int(item["date"]), float(item["value"])))
        if page >= resp[0]["pages"]:
            break
        page += 1
    print(f"Fetched {col_name}: {len(rows):,} observations")
    return pd.DataFrame(rows, columns=["iso3", "year", col_name])

# دانلود و ادغام همه شاخص‌ها
df = None
for code, name in INDICATORS.items():
    sub = fetch_wb_indicator(code, name)
    df = sub if df is None else pd.merge(df, sub, on=["iso3", "year"], how="outer")

df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
df["country"] = df["iso3"].map(lambda x: valid_countries[x]["name"])
df["region"]  = df["iso3"].map(lambda x: valid_countries[x]["region"])
df["income"]  = df["iso3"].map(lambda x: valid_countries[x]["income"])
print("Merged panel shape:", df.shape)
df.head()

WINDOW = 5  # پنجره ۵ ساله منتهی به سال شروع بحران (t-4 تا t)
FEATURES = ["gdp_growth", "inflation", "unemployment", "trade_pct_gdp", "investment_pct_gdp", "gdp_pc_pct_change"]

episodes = []
ts_records = []
ep_id = 0

for iso3, g in df.groupby("iso3"):
    g = g.sort_values("year").set_index("year")
    # ایجاد ایندکس سالانه پیوسته
    full_years = range(int(g.index.min()), int(g.index.max()) + 1)
    g = g.reindex(full_years)
    g["gdp_pc_pct_change"] = g["gdp_per_capita"].pct_change() * 100.0
    
    last_ep_year = -999
    years = list(g.index)
    
    for yr in years:
        # شرط شروع بحران: رشد منفی GDP و حداقل ۳ سال فاصله از بحران قبلی
        if yr - last_ep_year < 3:
            continue
        gdp_g = g.at[yr, "gdp_growth"]
        if pd.isna(gdp_g) or gdp_g >= 0:
            continue
        
        # بررسی وجود سقف قبل از بحران (t-1)
        if (yr - 1) not in g.index or pd.isna(g.at[yr - 1, "gdp_per_capita"]):
            continue
        peak_pc = g.at[yr - 1, "gdp_per_capita"]
        
        # بررسی پنجره ۵ ساله ورودی
        win_years = list(range(yr - WINDOW + 1, yr + 1))
        if not all(y in g.index for y in win_years):
            continue
        win_df = g.loc[win_years, FEATURES].copy()
        # اگر کل رشد GDP یا سرانه در پنجره ناقص باشد، رد کن
        if win_df["gdp_growth"].isna().sum() > 1:
            continue
        
        # محاسبه سرعت بازگشت (Recovery Years)
        future_years = [y for y in range(yr + 1, yr + 11) if y in g.index and not pd.isna(g.at[y, "gdp_per_capita"])]
        if len(future_years) < 3:
            continue  # برای جلوگیری از برچسب‌گذاری اشتباه بحران‌های خیلی جدید (مثل ۲۰۲۳)
            
        recovery_k = None
        for fy in future_years:
            if g.at[fy, "gdp_per_capita"] >= peak_pc:
                recovery_k = fy - yr
                break
        
        # برچسب: 1 = Fast (V-shaped, 1-2 yrs), 0 = Slow (U/L-shaped, 3+ yrs)
        if recovery_k is not None and recovery_k <= 2:
            label = 1
            label_str = "fast_v_shaped"
        else:
            label = 0
            label_str = "slow_ul_shaped"
            
        ep_id += 1
        series_id = f"ep_{ep_id:04d}_{iso3}_{yr}"
        last_ep_year = yr
        
        episodes.append({
            "series_id": series_id,
            "iso3": iso3,
            "country": valid_countries[iso3]["name"],
            "region": valid_countries[iso3]["region"],
            "income": valid_countries[iso3]["income"],
            "onset_year": yr,
            "onset_gdp_growth": float(gdp_g),
            "recovery_years": recovery_k if recovery_k is not None else 10,
            "label": label,
            "label_name": label_str
        })
        
        # ذخیره داده‌های سری‌زمانی برای فرمت TimeNet
        for step_idx, wy in enumerate(win_years):
            row = {"series_id": series_id, "step": step_idx - (WINDOW - 1), "year": wy}
            for f in FEATURES:
                val = win_df.at[wy, f]
                row[f] = float(val) if not pd.isna(val) else np.nan
            ts_records.append(row)

ep_df = pd.DataFrame(episodes)
ts_df = pd.DataFrame(ts_records)

print(f"Total labeled downturn episodes: {len(ep_df)}")
print(ep_df["label_name"].value_counts())

# --- ساخت بسته استاندارد TimeNet ---
os.makedirs("timenet_macrobounce", exist_ok=True)
ts_df.to_parquet("timenet_macrobounce/values.parquet", index=False)

manifest = {
    "dataset_name": "MacroBounce_WDI_Recovery",
    "version": "1.0.0",
    "description": "Multivariate macroeconomic downturn episodes (1960-2024) labeled by recovery speed (Fast 1-2y vs Slow 3+y).",
    "source": "World Bank World Development Indicators (WDI), CC-BY 4.0",
    "sampling_frequency": "annual",
    "window_length": WINDOW,
    "channels": FEATURES,
    "task": "multivariate_time_series_classification_and_reasoning",
    "num_episodes": int(len(ep_df)),
    "num_countries": int(ep_df["iso3"].nunique()),
    "class_balance": ep_df["label_name"].value_counts().to_dict()
}
with open("timenet_macrobounce/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

con = duckdb.connect("timenet_macrobounce/control.duckdb")
con.execute("CREATE OR REPLACE TABLE episodes AS SELECT * FROM ep_df")
con.execute("CREATE OR REPLACE TABLE values_ts AS SELECT * FROM ts_df")
con.close()

print("Saved TimeNet dataset to ./timenet_macrobounce/ (manifest.json, values.parquet, control.duckdb)")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# تبدیل سری‌زمانی به تنسور (N, Channels, Timesteps)
series_ids = ep_df["series_id"].tolist()
pivoted = ts_df.set_index(["series_id", "step"])[FEATURES]

X_raw = np.zeros((len(series_ids), len(FEATURES), WINDOW), dtype=np.float32)
for i, sid in enumerate(series_ids):
    sub = pivoted.loc[sid].sort_index()
    # پر کردن مقادیر گمشده داخل پنجره (ffill/bfill)
    sub = sub.ffill().bfill()
    X_raw[i] = sub.values.T  # shape: (C, T)

y = ep_df["label"].values
groups = ep_df["iso3"].values

# تقسیم سخت‌گیرانه بر اساس کشور (Held-out Countries)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X_raw, y, groups=groups))

X_train, X_test = X_raw[train_idx].copy(), X_raw[test_idx].copy()
y_train, y_test = y[train_idx], y[test_idx]

# جایگزینی NaNهای باقی‌مانده فقط با میانه Train + نرمال‌سازی (بدون Data Leakage)
for c in range(len(FEATURES)):
    med = np.nanmedian(X_train[:, c, :])
    X_train[:, c, :] = np.nan_to_num(X_train[:, c, :], nan=med)
    X_test[:, c, :]  = np.nan_to_num(X_test[:, c, :],  nan=med)
    
    # کلیپ کردن داده‌های پرت شدید (مثل ابرتورم‌های چند هزار درصدی)
    p1, p99 = np.percentile(X_train[:, c, :], [1, 99])
    X_train[:, c, :] = np.clip(X_train[:, c, :], p1, p99)
    X_test[:, c, :]  = np.clip(X_test[:, c, :],  p1, p99)
    
    mean, std = X_train[:, c, :].mean(), X_train[:, c, :].std() + 1e-6
    X_train[:, c, :] = (X_train[:, c, :] - mean) / std
    X_test[:, c, :]  = (X_test[:, c, :]  - mean) / std

print(f"Train episodes: {len(y_train)} ({len(set(groups[train_idx]))} countries)")
print(f"Test  episodes: {len(y_test)}  ({len(set(groups[test_idx]))} unseen countries)")

# --- مدل ۱: Baseline (فقط اسنپ‌شات سال بحران t بدون روند زمانی) ---
clf_base = LogisticRegression(class_weight="balanced", max_iter=1000)
clf_base.fit(X_train[:, :, -1], y_train)
base_prob = clf_base.predict_proba(X_test[:, :, -1])[:, 1]
base_pred = (base_prob >= 0.5).astype(int)

# --- مدل ۲: Temporal 1D-CNN (روند ۵ ساله چندکاناله) ---
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

class TemporalCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = TemporalCNN(len(FEATURES)).to(device)
pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
model.train()
for epoch in range(45):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    cnn_logits = model(torch.tensor(X_test).to(device)).cpu().numpy()
    cnn_prob = 1 / (1 + np.exp(-cnn_logits))
    cnn_pred = (cnn_prob >= 0.5).astype(int)

# محاسبه و چاپ متریک‌های نهایی برای کپی کردن در فرم Tally
results = {
    "Snapshot Baseline\n(Logistic Reg, t=0)": {
        "Accuracy": accuracy_score(y_test, base_pred),
        "F1-Score": f1_score(y_test, base_pred),
        "ROC-AUC":  roc_auc_score(y_test, base_prob),
    },
    "Temporal 1D-CNN\n(5-yr Multivariate)": {
        "Accuracy": accuracy_score(y_test, cnn_pred),
        "F1-Score": f1_score(y_test, cnn_pred),
        "ROC-AUC":  roc_auc_score(y_test, cnn_prob),
    }
}
res_df = pd.DataFrame(results).T
print("\n=== HELD-OUT TEST RESULTS (UNSEEN COUNTRIES) ===")
print((res_df * 100).round(2))

# --- نمودار ۱: مقایسه عملکرد مدل‌ها ---
sns.set_theme(style="whitegrid", font_scale=1.15)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
res_plot = res_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
res_plot["Score"] *= 100
sns.barplot(data=res_plot, x="Metric", y="Score", hue="index", palette=["#94A3B8", "#2563EB"], ax=ax)
ax.set_ylim(0, 100)
ax.set_ylabel("Score (%) on Held-out Countries")
ax.set_xlabel("")
ax.set_title("MacroBounce: Predicting Economic Downturn Recovery Speed", fontweight="bold", pad=14)
ax.legend(title="", frameon=True, loc="lower right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("chart1_model_comparison.png", dpi=300)
plt.show()

# --- نمودار ۲: تفاوت روند ۵ ساله کشورهای با بازگشت سریع در برابر بازگشت کند ---
plot_ts = ts_df.merge(ep_df[["series_id", "label_name"]], on="series_id")
plot_ts["Recovery"] = plot_ts["label_name"].map({
    "fast_v_shaped": "Fast Recovery (1–2 yrs)",
    "slow_ul_shaped": "Slow Recovery (3+ yrs)"
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
panels = [
    ("gdp_growth", "GDP Growth (%)", (-12, 12)),
    ("inflation", "Inflation CPI (%)", (0, 25)),
    ("investment_pct_gdp", "Gross Investment (% of GDP)", (15, 32)),
]
for ax, (col, title, ylim) in zip(axes, panels):
    clean = plot_ts[(plot_ts[col] >= plot_ts[col].quantile(0.02)) & (plot_ts[col] <= plot_ts[col].quantile(0.98))]
    sns.lineplot(data=clean, x="step", y=col, hue="Recovery",
                 palette={"Fast Recovery (1–2 yrs)": "#10B981", "Slow Recovery (3+ yrs)": "#EF4444"},
                 linewidth=2.6, marker="o", errorbar=("se", 1.5), ax=ax)
    ax.axvline(0, color="#334155", linestyle="--", alpha=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Years Relative to Downturn (0 = Onset)")
    ax.set_ylabel("")
    ax.set_ylim(ylim)
    if ax != axes[0]:
        ax.get_legend().remove()
    else:
        ax.legend(title="", frameon=True, loc="lower left")

fig.suptitle("Pre-Crisis 5-Year Trajectories: Fast vs. Slow Economic Recoveries (1960–2024)", fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("chart2_recovery_trajectories.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: chart1_model_comparison.png & chart2_recovery_trajectories.png")

!pip install -q transformers accelerate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

# انتخاب ۳ نمونه از مجموعه Test برای نمایش استدلال مدل
sample_eps = ep_df.iloc[test_idx].head(3)

for _, row in sample_eps.iterrows():
    sid = row["series_id"]
    sub = ts_df[ts_df["series_id"] == sid].sort_values("step")
    series_txt = "\n".join(
        f"Year {int(r.year)} (t{int(r.step):+d}): GDP_growth={r.gdp_growth:.1f}%, Inflation={r.inflation:.1f}%, Investment={r.investment_pct_gdp:.1f}%, Trade={r.trade_pct_gdp:.1f}%"
        for _, r in sub.iterrows()
    )
    prompt = f"""You are a macroeconomic temporal reasoning AI.
Country: {row['country']} ({row['region']}, {row['income']})
5-year multivariate time-series ending in recession onset ({int(row['onset_year'])}):
{series_txt}

Task: Predict whether GDP per capita will recover FAST (1-2 years, V-shaped) or SLOW (3+ years, U/L-shaped), and give a 2-sentence explanation based strictly on the trajectory."""

    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(llm.device)
    out = llm.generate(**inputs, max_new_tokens=120, do_sample=False)
    ans = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print("=" * 70)
    print(f"CASE: {row['country']} ({int(row['onset_year'])}) | TRUE LABEL: {row['label_name']} ({int(row['recovery_years'])} yrs)")
    print("AI REASONING:\n", ans.strip())
