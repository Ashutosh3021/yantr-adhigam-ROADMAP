import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ─── Load & Clean Data ───────────────────────────────────────────────────────
CSV_PATH = "stock_data.csv"   # change if needed

df = pd.read_csv(CSV_PATH, header=[0, 1, 2], index_col=0)

# Flatten multi-level columns
df.columns = ['_'.join([c for c in col if not c.startswith('Unnamed')]).strip('_')
              for col in df.columns]
df.index = pd.to_datetime(df.index, errors='coerce')
df = df[df.index.notna()].sort_index()
df = df.apply(pd.to_numeric, errors='coerce')

# Extract close prices for each asset
assets = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMZN', 'META', 'BTC', 'ETH', 'GLD', 'SLV']
close_cols = {a: f'{a}_Close' for a in assets}

close = pd.DataFrame({a: df[c] for a, c in close_cols.items() if c in df.columns}).dropna(how='all')

# Normalise to 100 at start
norm = close.divide(close.bfill().iloc[0]) * 100

# Daily returns
returns = close.pct_change().dropna()

# ─── Color palette ────────────────────────────────────────────────────────────
palette = sns.color_palette("tab10", n_colors=len(close.columns))
asset_colors = dict(zip(close.columns, palette))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Normalised Price Performance
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(16, 7))
for asset in norm.columns:
    ax.plot(norm.index, norm[asset], label=asset, color=asset_colors[asset], linewidth=1.8)
ax.set_title("Normalised Price Performance (Base = 100)", fontsize=16, fontweight='bold')
ax.set_ylabel("Indexed Price (Base 100)")
ax.set_xlabel("Date")
ax.legend(ncol=5, loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig1_normalised_performance.png", dpi=150)
plt.close()
print("Saved fig1")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Individual Asset OHLCV (Close + Volume)
# ══════════════════════════════════════════════════════════════════════════════
vol_cols = {a: f'{a}_Volume' for a in assets}
n = len(close.columns)
fig2, axes = plt.subplots(n, 2, figsize=(18, n * 2.5), sharex=False)
fig2.suptitle("Individual Asset: Close Price & Volume", fontsize=16, fontweight='bold', y=1.01)

for i, asset in enumerate(close.columns):
    c = close[asset].dropna()
    ax_p, ax_v = axes[i, 0], axes[i, 1]

    ax_p.plot(c.index, c.values, color=asset_colors[asset], linewidth=1.5)
    ax_p.set_title(f"{asset} – Close Price", fontsize=10, fontweight='bold')
    ax_p.set_ylabel("Price (USD)")
    ax_p.grid(alpha=0.3)

    vcol = vol_cols[asset]
    if vcol in df.columns:
        v = df[vcol].dropna()
        ax_v.bar(v.index, v.values, color=asset_colors[asset], alpha=0.6, width=1)
        ax_v.set_title(f"{asset} – Volume", fontsize=10, fontweight='bold')
        ax_v.set_ylabel("Volume")
        ax_v.grid(alpha=0.3)
    else:
        ax_v.set_visible(False)

    sns.despine(ax=ax_p); sns.despine(ax=ax_v)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig2_individual_ohlcv.png", dpi=130, bbox_inches='tight')
plt.close()
print("Saved fig2")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Correlation Heatmap (Closing Prices)
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(18, 7))
fig3.suptitle("Correlation Analysis", fontsize=16, fontweight='bold')

corr_price = close.corr()
sns.heatmap(corr_price, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, center=0, ax=axes[0],
            linewidths=0.5, square=True, annot_kws={"size": 9})
axes[0].set_title("Price Correlation", fontsize=12)

corr_ret = returns.corr()
sns.heatmap(corr_ret, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, center=0, ax=axes[1],
            linewidths=0.5, square=True, annot_kws={"size": 9})
axes[1].set_title("Daily Returns Correlation", fontsize=12)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig3_correlation_heatmaps.png", dpi=150)
plt.close()
print("Saved fig3")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Rolling 90-day Correlation vs AAPL
# ══════════════════════════════════════════════════════════════════════════════
fig4, ax = plt.subplots(figsize=(16, 6))
ref = 'AAPL'
if ref in returns.columns:
    for asset in returns.columns:
        if asset != ref:
            roll_corr = returns[ref].rolling(90).corr(returns[asset])
            ax.plot(roll_corr.index, roll_corr, label=asset,
                    color=asset_colors[asset], linewidth=1.5, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"Rolling 90-Day Returns Correlation vs {ref}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Pearson Correlation")
    ax.set_xlabel("Date")
    ax.legend(ncol=5, fontsize=9)
    ax.grid(alpha=0.3)
    sns.despine()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig4_rolling_correlation.png", dpi=150)
plt.close()
print("Saved fig4")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Volatility (30-day Rolling Std of Returns)
# ══════════════════════════════════════════════════════════════════════════════
fig5, ax = plt.subplots(figsize=(16, 6))
for asset in returns.columns:
    vol = returns[asset].rolling(30).std() * np.sqrt(252)
    ax.plot(vol.index, vol, label=asset, color=asset_colors[asset], linewidth=1.5)
ax.set_title("Annualised 30-Day Rolling Volatility", fontsize=14, fontweight='bold')
ax.set_ylabel("Volatility (Annualised)")
ax.set_xlabel("Date")
ax.legend(ncol=5, fontsize=9)
ax.grid(alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig5_rolling_volatility.png", dpi=150)
plt.close()
print("Saved fig5")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Distribution of Daily Returns (KDE + Hist)
# ══════════════════════════════════════════════════════════════════════════════
n_assets = len(returns.columns)
cols_g = 5
rows_g = (n_assets + cols_g - 1) // cols_g
fig6, axes6 = plt.subplots(rows_g, cols_g, figsize=(20, rows_g * 3.5))
fig6.suptitle("Daily Returns Distribution", fontsize=16, fontweight='bold')
axes6 = axes6.flatten()

for i, asset in enumerate(returns.columns):
    r = returns[asset].dropna()
    sns.histplot(r, kde=True, ax=axes6[i], color=asset_colors[asset],
                 bins=60, alpha=0.6, line_kws={'linewidth': 2})
    axes6[i].axvline(r.mean(), color='red', linestyle='--', linewidth=1.2, label='Mean')
    axes6[i].set_title(f"{asset}", fontsize=11, fontweight='bold')
    axes6[i].set_xlabel("Daily Return")
    axes6[i].grid(alpha=0.3)
    sns.despine(ax=axes6[i])

for j in range(i + 1, len(axes6)):
    axes6[j].set_visible(False)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig6_return_distributions.png", dpi=150)
plt.close()
print("Saved fig6")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Pairplot (Returns) – Stocks only to keep it readable
# ══════════════════════════════════════════════════════════════════════════════
stocks = [a for a in ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMZN', 'META'] if a in returns.columns]
ret_stocks = returns[stocks].dropna()

g = sns.pairplot(ret_stocks, plot_kws=dict(alpha=0.3, s=8),
                 diag_kind='kde', diag_kws=dict(fill=True),
                 corner=True)
g.figure.suptitle("Pairplot of Daily Returns – Equities", fontsize=14, fontweight='bold', y=1.01)
g.figure.set_size_inches(14, 14)
plt.savefig("/mnt/user-data/outputs/fig7_pairplot_equities.png", dpi=130, bbox_inches='tight')
plt.close()
print("Saved fig7")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Cumulative Returns Bar Chart (Total %)
# ══════════════════════════════════════════════════════════════════════════════
total_ret = ((close.iloc[-1] / close.bfill().iloc[0]) - 1) * 100
total_ret = total_ret.dropna().sort_values(ascending=False)

fig8, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(total_ret.index, total_ret.values,
              color=[asset_colors.get(a, 'steelblue') for a in total_ret.index], edgecolor='white')
ax.bar_label(bars, fmt='%.0f%%', fontsize=9, padding=3)
ax.set_title("Total Cumulative Return (Full Period)", fontsize=14, fontweight='bold')
ax.set_ylabel("Return (%)")
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig8_cumulative_returns_bar.png", dpi=150)
plt.close()
print("Saved fig8")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Clustermap of Returns Correlation
# ══════════════════════════════════════════════════════════════════════════════
g2 = sns.clustermap(returns.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=-1, vmax=1, center=0,
                    figsize=(10, 9), linewidths=0.5,
                    annot_kws={"size": 9})
g2.figure.suptitle("Clustered Correlation Map – Daily Returns", fontsize=13, fontweight='bold', y=1.02)
plt.savefig("/mnt/user-data/outputs/fig9_clustermap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9")

print("\n✅ All 9 figures saved to /mnt/user-data/outputs/")