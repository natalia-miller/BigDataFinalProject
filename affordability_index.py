"""
ACS1 Metropolitan Affordability Index
=======================================
A full ML pipeline that:
  1. Pulls ACS1 data for major metro counties
  2. Engineers an affordability feature matrix
  3. Reduces dimensionality (PCA + feature importance)
  4. Runs 3 expert models: Ridge Regression, Random Forest, Gradient Boosting
  5. Evaluates with RMSE, R², MAE, SNR, accuracy (classification tier)
  6. Produces: correlation heatmap, confusion matrix, histograms,
               accuracy plots, PCA biplot, affordability ranking

Requirements:
  pip install censusdis pandas numpy scikit-learn matplotlib seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.pipeline import Pipeline

import censusdis.data as ced
from censusdis.datasets import ACS1

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.3f}".format)

VINTAGE   = 2024
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ACS1 DATA PULL
# Feature groups selected for affordability:
#   B25070  Gross Rent as % of HH Income       → rent burden
#   B25091  Mortgage costs as % of income       → owner burden
#   B19013  Median HH Income                   → income baseline
#   B25064  Median Gross Rent                  → rent level
#   B25077  Median Home Value                  → home price
#   B17002  Income-to-poverty ratio             → cost-of-living normalizer
#   B23025  Employment status                  → labor market proxy
# ══════════════════════════════════════════════════════════════════════════════

RAW_GROUPS = {
    "B25070": "rent_burden",
    "B25091": "mortgage_burden",
    "B19013": "median_hh_income",
    "B25064": "median_gross_rent",
    "B25077": "median_home_value",
    "B17002": "poverty_ratio",
    "B23025": "employment",
}


def make_name_mapper(group: str, vintage: int = VINTAGE) -> callable:
    group_prefix = RAW_GROUPS[group]   # e.g. "rent_burden", "employment"

    def _mapper(variable: str) -> str:
        if not variable.startswith(group):
            return variable  # STATE, COUNTY, NAME — pass through unchanged

        try:
            result = ced.variables.search(ACS1, vintage, group_name=group, name=variable)
            if result.empty:
                return variable

            label = result.iloc[0]["LABEL"]

            # Parse all non-empty segments after "Estimate!!"
            # e.g. "Estimate!!Total:!!30 to 34 percent!!" → ["Total", "30 to 34 percent"]
            segments = [
                s.split('(')[0].strip().rstrip(":")
                for s in label.split("!")
                if s.strip() and s.strip().lower() not in ("estimate", "")
            ]

            if not segments:
                return variable

            # Drop the first segment if it's just "Total" and there are more
            # specific segments following it — avoids "rent_burden_total_30_to_34_percent"
            if len(segments) > 1 and segments[0].lower() == "total":
                segments = segments[1:]

            # Slugify: lowercase, spaces → underscores, strip special chars
            slug = "_".join(
                re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
                for s in segments
            )

            return f"{group_prefix}__{slug}"   # e.g. "rent_burden__30_to_34_percent"

        except Exception:
            return variable  # safe fallback to raw code

    return _mapper


# Specific variable columns we want from each group
TARGET_VARS = {
    # Rent burden: % paying ≥30% income on rent (B25070_007E–B25070_011E sum / B25070_001E)
    "B25070_001E": "rent_burden__total",
    "B25070_007E": "rent_burden__30_to_34_percent",
    "B25070_008E": "rent_burden__35_0_to_39_9_percent",
    "B25070_009E": "rent_burden__40_0_to_49_9_percent",
    "B25070_010E": "rent_burden__50_percent_or_more",
    # Mortgage burden
    "B25091_001E": "mortgage_burden__total",
    "B25091_008E": "mortgage_burden__housing_units_with_a_mortgage_30_0_to_34_9_percent",
    "B25091_009E": "mortgage_burden__housing_units_with_a_mortgage_35_0_to_39_9_percent",
    "B25091_010E": "mortgage_burden__housing_units_with_a_mortgage_40_0_to_49_9_percent",
    "B25091_011E": "mortgage_burden__housing_units_with_a_mortgage_50_0_percent_or_more",
    # Income
    "B19013_001E": "median_hh_income__median_household_income",
    # Rent & home value
    "B25064_001E": "median_gross_rent__median_gross_rent",
    "B25077_001E": "median_home_value__median_value_dollars",
    # Poverty ratio: pct above 2× poverty line
    "B17002_001E": "poverty_ratio__total",
    "B17002_012E": "poverty_ratio__4_00_to_4_99",
    # Employment: unemployment rate proxy
    "B23025_003E": "employment__employed",
    "B23025_005E": "employment__unemployed",
    "B23025_002E": "employment__in_labor_force",
}


def pull_acs_features(vintage: int = VINTAGE) -> pd.DataFrame:
    """
    Pull all required ACS1 groups, apply per-group name_mapper, and merge.

    Each group gets its own mapper via make_name_mapper() so the closure
    captures the correct group code. Duplicate non-key columns that appear
    in multiple groups (e.g. NAME) are suffixed and de-duped on merge.
    """
    from functools import reduce

    print("── Pulling ACS1 data ──────────────────────────────────────────")
    frames = []

    for group, label in RAW_GROUPS.items():
        print(f"  {group} ({label})...")
        try:
            df = ced.download(ACS1, vintage, group=group, state="*", county="*")

            # Apply the group-scoped mapper; pass-through columns keep original names
            mapper = make_name_mapper(group, vintage)
            df = df.rename(columns=mapper)

            frames.append(df)
            print(f"    → {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"    Warning: {group} failed — {e}")

    if not frames:
        raise RuntimeError("No data pulled. Check censusdis installation.")

    # Merge all frames on STATE + COUNTY; use outer so no geography is dropped
    # NAME may appear in multiple frames — keep the first occurrence via suffixes
    KEY_COLS = ["STATE", "COUNTY"]

    def _merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        # Drop NAME from right if left already has it to avoid _x/_y suffixes
        right_cols = [c for c in right.columns if c not in left.columns or c in KEY_COLS]
        return pd.merge(
            left,
            right[right_cols],
            on=KEY_COLS,
            how="outer",
        )

    merged = reduce(_merge, frames)

    # Restore NAME from the first frame if it was lost
    if "NAME" not in merged.columns and "NAME" in frames[0].columns:
        merged = merged.merge(
            frames[0][KEY_COLS + ["NAME"]], on=KEY_COLS, how="left"
        )

    print(f"\n  → {len(merged)} geographies, {len(merged.columns)} total columns")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["NAME", "STATE", "COUNTY"]].copy()

    def safe_div(num, denom, scale=100):
        return np.where(denom > 0, num / denom * scale, np.nan)

    # Rent burden: % of renters paying ≥30% income on rent
    rent_burdened = (
        df.get("rent_burden__30_0_to_34_9_percent", 0) +
        df.get("rent_burden__35_0_to_39_9_percent", 0) +
        df.get("rent_burden__40_0_to_49_9_percent", 0) +
        df.get("rent_burden__50_0_percent_or_more", 0)
    )
    out["rent_burden_pct"] = safe_div(rent_burdened, df.get("rent_burden__total", np.nan))

    # Mortgage burden: % of owners with mortgage paying ≥30%
    mtg_burdened = (
        df.get("mortgage_burden__housing_units_with_a_mortgage_30_0_to_34_9_percent", 0) +
        df.get("mortgage_burden__housing_units_with_a_mortgage_35_0_to_39_9_percent", 0) +
        df.get("mortgage_burden__housing_units_with_a_mortgage_40_0_to_49_9_percent", 0) +
        df.get("mortgage_burden__housing_units_with_a_mortgage_50_0_percent_or_more", 0)
    )
    out["mortgage_burden_pct"] = safe_div(
        mtg_burdened,
        df.get("mortgage_burden__housing_units_with_a_mortgage", np.nan)  # use with-mortgage subtotal as denominator
    )

    # Income & prices
    out["median_hh_income"]  = pd.to_numeric(df.get("median_hh_income__median_household_income_in_the_past_12_months"), errors="coerce")
    out["median_gross_rent"] = pd.to_numeric(df.get("median_gross_rent__median_gross_rent"), errors="coerce")
    out["median_home_value"] = pd.to_numeric(df.get("median_home_value__median_value"), errors="coerce")  # fixed

    # Derived ratios
    out["home_value_to_income"] = out["median_home_value"] / out["median_hh_income"].replace(0, np.nan)
    out["rent_to_income"]       = (out["median_gross_rent"] * 12) / out["median_hh_income"].replace(0, np.nan)

    # % population above 2× poverty line — sum all brackets at or above 2.00
    above_2x = (
        df.get("poverty_ratio__2_00_to_2_99", 0) +
        df.get("poverty_ratio__3_00_to_3_99", 0) +
        df.get("poverty_ratio__4_00_to_4_99", 0) +
        df.get("poverty_ratio__5_00_and_over", 0)
    )
    out["pct_above_2x_poverty"] = safe_div(above_2x, df.get("poverty_ratio__total", np.nan))

    # Unemployment rate: civilian unemployed / total in labor force
    out["unemployment_rate"] = safe_div(
        df.get("employment__in_labor_force_civilian_labor_force_unemployed", 0),  # fixed
        df.get("employment__in_labor_force", np.nan)
    )

    out = out.dropna(subset=[
        "rent_burden_pct", "mortgage_burden_pct",
        "median_hh_income", "median_gross_rent",
        "median_home_value", "home_value_to_income",
        "rent_to_income", "pct_above_2x_poverty",
    ])

    out = out[
        (out["median_hh_income"] > 0) &
        (out["median_home_value"] > 0) &
        (out["median_gross_rent"] > 0)
    ]

    print(f"  → {len(out)} geographies after feature engineering")
    return out.reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — COMPOSITE AFFORDABILITY INDEX (ground truth label)
# A weighted composite score: higher = MORE affordable
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "rent_burden_pct",
    "mortgage_burden_pct",
    "home_value_to_income",
    "rent_to_income",
    "pct_above_2x_poverty",
    "unemployment_rate",
    "median_hh_income",
    "median_gross_rent",
    "median_home_value",
]

def build_affordability_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite affordability index (0–100).
    Lower burden + higher income + lower prices = higher score.
    """
    print("\n── Building affordability index ───────────────────────────────")
    d = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(d[FEATURE_COLS].fillna(d[FEATURE_COLS].median()))

    # Weights: burden variables are inverted (lower burden = better)
    # [rent_burden, mtg_burden, hv_to_income, rent_to_income,
    #  prosperity, unemployment, hh_income, gross_rent, home_value]
    signs  = np.array([-1, -1, -1, -1, +1, -1, +1, -1, -1])
    w      = np.array([ .20, .15, .20, .15, .10, .10, .05, .025, .025])
    w     /= w.sum()

    raw_index = (scaled * signs * w).sum(axis=1)

    # Normalize to 0–100
    mn, mx = raw_index.min(), raw_index.max()
    d["affordability_index"] = ((raw_index - mn) / (mx - mn) * 100).round(2)

    # Classify into 4 tiers
    d["affordability_tier"] = pd.cut(
        d["affordability_index"],
        bins=[0, 25, 50, 75, 100],
        labels=["Severely Unaffordable", "Unaffordable", "Moderate", "Affordable"],
        include_lowest=True,
    )
    tier_counts = d["affordability_tier"].value_counts()
    for tier, cnt in tier_counts.items():
        print(f"  {tier:25s}: {cnt}")
    return d


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DIMENSIONALITY REDUCTION (PCA)
# ══════════════════════════════════════════════════════════════════════════════

def run_pca(X_scaled: np.ndarray, feature_names: list,
            n_components: int = 5) -> tuple:
    """
    PCA for dimensionality reduction.
    Returns: pca object, transformed X, explained variance, loadings DataFrame
    """
    print("\n── PCA dimensionality reduction ───────────────────────────────")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    ev = pca.explained_variance_ratio_
    print(f"  Components: {n_components}")
    print(f"  Variance explained: {ev.cumsum()[-1]*100:.1f}%")
    for i, v in enumerate(ev):
        print(f"    PC{i+1}: {v*100:.1f}%")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return pca, X_pca, ev, loadings


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — 3 EXPERT MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_experts(X: np.ndarray, y: np.ndarray,
                  cv: int = 5) -> dict:
    """
    Train 3 expert regression models with cross-validation.
    Returns dict of {name: {model, cv_rmse, cv_r2, predictions}}
    """
    print("\n── Training 3 expert models ───────────────────────────────────")
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    experts = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest":    RandomForestRegressor(
                                n_estimators=200, max_depth=8,
                                min_samples_leaf=3, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
                                n_estimators=200, max_depth=4,
                                learning_rate=0.05, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in experts.items():
        cv_r2   = cross_val_score(model, X, y, cv=kf, scoring="r2")
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                          scoring="neg_mean_squared_error"))
        model.fit(X, y)
        preds = model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, preds))
        mae  = mean_absolute_error(y, preds)
        r2   = r2_score(y, preds)
        # Signal-to-noise ratio: variance of signal / variance of residuals
        residuals = y - preds
        snr = 10 * np.log10(np.var(y) / np.var(residuals)) if np.var(residuals) > 0 else np.inf

        results[name] = {
            "model":       model,
            "predictions": preds,
            "rmse":        rmse,
            "mae":         mae,
            "r2":          r2,
            "snr_db":      snr,
            "cv_rmse":     cv_rmse,
            "cv_r2":       cv_r2,
        }

        print(f"\n  ▶ {name}")
        print(f"    Train RMSE: {rmse:.3f}  MAE: {mae:.3f}  R²: {r2:.3f}  SNR: {snr:.1f} dB")
        print(f"    CV RMSE:   {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
        print(f"    CV R²:     {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    return results


def classify_tiers(y_true_index: np.ndarray,
                   y_pred_index: np.ndarray) -> tuple:
    """
    Convert continuous affordability index predictions to tier labels
    for classification accuracy / confusion matrix analysis.
    """
    def to_tier(arr):
        t = np.full(len(arr), "Moderate", dtype=object)
        t[arr <= 25]  = "Severely Unaffordable"
        t[(arr > 25) & (arr <= 50)] = "Unaffordable"
        t[(arr > 75)] = "Affordable"
        return t

    true_tiers = to_tier(y_true_index)
    pred_tiers = to_tier(y_pred_index)
    acc = accuracy_score(true_tiers, pred_tiers)
    return true_tiers, pred_tiers, acc


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE  = "RdYlGn"
PALETTE_R = "RdYlGn_r"
FIG_DPI  = 150

def fig_correlation_heatmap(df: pd.DataFrame, feature_cols: list) -> None:
    """Correlation matrix as annotated heatmap."""
    corr = df[feature_cols + ["affordability_index"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, linecolor="#ddd",
        ax=ax, annot_kws={"size": 8},
        vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()
    plt.savefig("viz_01_correlation_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_01_correlation_heatmap.png")


def fig_feature_histograms(df: pd.DataFrame, feature_cols: list) -> None:
    """Grid of feature distribution histograms colored by tier."""
    tier_colors = {
        "Affordable": "#2ecc71",
        "Moderate": "#f39c12",
        "Unaffordable": "#e74c3c",
        "Severely Unaffordable": "#8e44ad",
    }
    n = len(feature_cols)
    ncols = 3
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for tier, color in tier_colors.items():
            subset = df[df["affordability_tier"] == tier][col].dropna()
            if len(subset):
                ax.hist(subset, bins=25, alpha=0.55, color=color,
                        label=tier, density=True)
        ax.set_title(col.replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top","right"]].set_visible(False)

    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.6)
               for c in tier_colors.values()]
    fig.legend(handles, tier_colors.keys(), loc="lower right",
               ncol=2, fontsize=8, title="Tier", title_fontsize=8)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Affordability Tier",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("viz_02_feature_histograms.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_02_feature_histograms.png")


def fig_pca_biplot(X_scaled: np.ndarray, pca, feature_names: list,
                   y: np.ndarray) -> None:
    """PCA biplot: scatter of PC1 vs PC2 + loading arrows."""
    X_pca = pca.transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter colored by affordability index
    sc = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=PALETTE,
                         alpha=0.6, s=18, edgecolors="none")
    plt.colorbar(sc, ax=axes[0], label="Affordability Index")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    axes[0].set_title("PCA Scatter — colored by affordability", fontsize=11, fontweight="bold")
    axes[0].spines[["top","right"]].set_visible(False)

    # Right: biplot with loadings
    scale = 3.5
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=PALETTE,
                    alpha=0.3, s=12, edgecolors="none")
    for j, name in enumerate(feature_names):
        lx = pca.components_[0, j] * scale
        ly = pca.components_[1, j] * scale
        axes[1].annotate("", xy=(lx, ly), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
        axes[1].text(lx * 1.12, ly * 1.12,
                     name.replace("_", "\n"), fontsize=7, ha="center",
                     color="#c0392b", fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    axes[1].set_title("PCA Biplot — feature loadings", fontsize=11, fontweight="bold")
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("viz_03_pca_biplot.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_03_pca_biplot.png")


def fig_model_accuracy_plots(results: dict, y: np.ndarray) -> None:
    """
    Actual vs Predicted scatter + CV RMSE distribution for each expert.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    colors = ["#2980b9", "#27ae60", "#e67e22"]

    for i, (name, res) in enumerate(results.items()):
        preds = res["predictions"]
        color = colors[i]

        # Top row: actual vs predicted
        ax = axes[0, i]
        lim = (min(y.min(), preds.min()) - 2, max(y.max(), preds.max()) + 2)
        ax.scatter(y, preds, alpha=0.4, s=14, color=color, edgecolors="none")
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual Index", fontsize=9)
        ax.set_ylabel("Predicted Index", fontsize=9)
        ax.set_title(f"{name}\nR²={res['r2']:.3f}  RMSE={res['rmse']:.2f}",
                     fontsize=9, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)

        # Annotate SNR
        ax.text(0.05, 0.92, f"SNR: {res['snr_db']:.1f} dB",
                transform=ax.transAxes, fontsize=8, color="#555")

        # Bottom row: CV RMSE box / violin
        ax2 = axes[1, i]
        ax2.boxplot(res["cv_rmse"], patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.5),
                    medianprops=dict(color="k", lw=2),
                    whiskerprops=dict(color="#888"),
                    capprops=dict(color="#888"),
                    flierprops=dict(marker="o", markersize=4, color="#888"))
        ax2.set_title(f"CV RMSE distribution\nμ={res['cv_rmse'].mean():.3f} σ={res['cv_rmse'].std():.3f}",
                      fontsize=9, fontweight="bold")
        ax2.set_ylabel("RMSE", fontsize=9)
        ax2.set_xticks([1]); ax2.set_xticklabels([name], fontsize=8)
        ax2.spines[["top","right"]].set_visible(False)

    plt.suptitle("Model Performance: Actual vs Predicted + Cross-Validation RMSE",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("viz_04_model_accuracy.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_04_model_accuracy.png")


def fig_confusion_matrices(results: dict, y: np.ndarray) -> None:
    """Confusion matrix heatmaps for tier classification from each model."""
    tier_order = ["Affordable", "Moderate", "Unaffordable", "Severely Unaffordable"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (name, res) in enumerate(results.items()):
        true_t, pred_t, acc = classify_tiers(y, res["predictions"])
        le = LabelEncoder()
        le.fit(tier_order)
        cm = confusion_matrix(true_t, pred_t, labels=tier_order)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f",
            xticklabels=[t[:8] for t in tier_order],
            yticklabels=[t[:8] for t in tier_order],
            cmap="Blues", ax=axes[i],
            linewidths=0.5, linecolor="#ccc",
            vmin=0, vmax=1,
            annot_kws={"size": 9},
        )
        axes[i].set_title(f"{name}\nTier Accuracy: {acc*100:.1f}%",
                          fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Predicted Tier", fontsize=9)
        axes[i].set_ylabel("True Tier", fontsize=9)
        axes[i].tick_params(labelsize=8)

    plt.suptitle("Confusion Matrices — Affordability Tier Classification",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig("viz_05_confusion_matrices.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_05_confusion_matrices.png")


def fig_feature_importance(results: dict, feature_names: list) -> None:
    """Feature importance from tree models + Ridge coefficients."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = ["#2980b9", "#27ae60", "#e67e22"]

    for i, (name, res) in enumerate(results.items()):
        model = res["model"]
        ax = axes[i]

        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            label = "Feature Importance"
        else:
            imp = np.abs(model.coef_)
            label = "|Ridge Coefficient|"

        idx = np.argsort(imp)
        ax.barh([feature_names[j].replace("_", " ") for j in idx],
                imp[idx], color=colors[i], alpha=0.75)
        ax.set_title(f"{name}\n{label}", fontsize=10, fontweight="bold")
        ax.set_xlabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("Feature Importance / Coefficients by Expert Model",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("viz_06_feature_importance.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_06_feature_importance.png")


def fig_metrics_summary(results: dict) -> None:
    """Bar chart comparison of RMSE, MAE, R², SNR across models."""
    names  = list(results.keys())
    metrics = {
        "RMSE":       [r["rmse"]    for r in results.values()],
        "MAE":        [r["mae"]     for r in results.values()],
        "R²":         [r["r2"]      for r in results.values()],
        "SNR (dB)":   [r["snr_db"] for r in results.values()],
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    colors = ["#2980b9", "#27ae60", "#e67e22"]

    for j, (metric, vals) in enumerate(metrics.items()):
        ax = axes[j]
        bars = ax.bar(names, vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Expert Model Performance Metrics",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("viz_07_metrics_summary.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_07_metrics_summary.png")


def fig_affordability_ranking(df: pd.DataFrame, top_n: int = 40) -> None:
    """Horizontal bar chart of top and bottom metro affordability scores."""
    df_s = df.sort_values("affordability_index", ascending=False)
    top    = df_s.head(top_n // 2)
    bottom = df_s.tail(top_n // 2).sort_values("affordability_index")
    combined = pd.concat([top, bottom])

    tier_color = {
        "Affordable":            "#27ae60",
        "Moderate":              "#f39c12",
        "Unaffordable":          "#e74c3c",
        "Severely Unaffordable": "#8e44ad",
    }
    bar_colors = [tier_color.get(str(t), "#aaa") for t in combined["affordability_tier"]]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.38))
    bars = ax.barh(combined["NAME"].str.replace(", ", ",\n"), combined["affordability_index"],
                   color=bar_colors, alpha=0.82, edgecolor="white")

    ax.axvline(50, color="#555", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Affordability Index (0–100)", fontsize=10)
    ax.set_title(f"Metropolitan Affordability Index\nTop {top_n//2} most & least affordable",
                 fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines[["top","right"]].set_visible(False)

    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.7)
               for c in tier_color.values()]
    ax.legend(handles, tier_color.keys(), loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig("viz_08_affordability_ranking.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_08_affordability_ranking.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — ENSEMBLE: COMBINE 3 EXPERTS
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(results: dict, X: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
    """
    Weighted ensemble of 3 experts.
    Weights proportional to each model's R² on training data.
    """
    r2s   = np.array([r["r2"] for r in results.values()])
    r2s   = np.clip(r2s, 0, None)
    w     = r2s / r2s.sum()
    preds = np.stack([r["predictions"] for r in results.values()], axis=1)
    ens   = (preds * w).sum(axis=1)

    rmse  = np.sqrt(mean_squared_error(y, ens))
    r2    = r2_score(y, ens)
    _, _, acc = classify_tiers(y, ens)

    print(f"\n── Ensemble (weighted by R²) ───────────────────────────────────")
    for name, wi in zip(results.keys(), w):
        print(f"  {name:25s}: weight = {wi:.3f}")
    print(f"  Ensemble RMSE: {rmse:.3f}  R²: {r2:.3f}  Tier Accuracy: {acc*100:.1f}%")
    return ens


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("═" * 65)
    print("ACS1 Metropolitan Affordability Index Pipeline")
    print("═" * 65)

    # 1. Pull & engineer
    raw      = pull_acs_features()
    features = engineer_features(raw)
    indexed  = build_affordability_index(features)

    # 2. Prepare X, y
    X_raw = indexed[FEATURE_COLS].fillna(indexed[FEATURE_COLS].median()).values
    y     = indexed["affordability_index"].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. PCA
    pca, X_pca, ev, loadings = run_pca(X_scaled, FEATURE_COLS, n_components=5)
    print("\n  Top feature loadings on PC1:")
    print(loadings["PC1"].sort_values(key=abs, ascending=False).head(5))

    # 4. Train experts (use original scaled features, not PCA — more interpretable)
    results = train_experts(X_scaled, y)

    # 5. Ensemble
    ensemble_preds = ensemble_predict(results, X_scaled, y)

    # 6. Visualizations
    print("\n── Generating visualizations ──────────────────────────────────")
    fig_correlation_heatmap(indexed, FEATURE_COLS)
    fig_feature_histograms(indexed, FEATURE_COLS)
    fig_pca_biplot(X_scaled, pca, FEATURE_COLS, y)
    fig_model_accuracy_plots(results, y)
    fig_confusion_matrices(results, y)
    fig_feature_importance(results, FEATURE_COLS)
    fig_metrics_summary(results)
    fig_affordability_ranking(indexed, top_n=40)

    # 7. Save ranked output
    indexed["ensemble_prediction"] = ensemble_preds
    indexed["ensemble_tier"]       = pd.cut(
        ensemble_preds,
        bins=[0, 25, 50, 75, 100],
        labels=["Severely Unaffordable", "Unaffordable", "Moderate", "Affordable"],
        include_lowest=True,
    )
    out_cols = [
        "NAME", "STATE", "COUNTY",
        "affordability_index", "affordability_tier",
        "ensemble_prediction", "ensemble_tier",
        "rent_burden_pct", "mortgage_burden_pct",
        "home_value_to_income", "rent_to_income",
        "median_hh_income", "median_gross_rent", "median_home_value",
        "pct_above_2x_poverty", "unemployment_rate",
    ]
    ranked = indexed[out_cols].sort_values("affordability_index", ascending=False)
    ranked.to_csv("affordability_index_results.csv", index=False)

    print("\n  ✓ affordability_index_results.csv")
    print("\n" + "═" * 65)
    print("Pipeline complete. 8 visualizations + 1 CSV saved.")
    print("═" * 65)

    print("\nTop 10 most affordable metros:")
    print(ranked[["NAME","affordability_index","affordability_tier"]].head(10).to_string(index=False))
    print("\nTop 10 least affordable metros:")
    print(ranked[["NAME","affordability_index","affordability_tier"]].tail(10).to_string(index=False))

    return ranked, results, indexed


if __name__ == "__main__":
    ranked, results, df = run_pipeline()
