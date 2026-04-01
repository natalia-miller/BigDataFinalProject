"""
ACS1 Income Opportunity Analysis
==================================
"Given my demographic profile, which city will I earn the most in?"

Parameters:
  - Sex:            Male | Female
  - Race/Ethnicity: White (non-Hispanic) | Black | Hispanic/Latino |
                    Asian | American Indian | Native Hawaiian | Other | Two or More
  - Work Experience: Full-time year-round | Other (part-time/part-year)
  - Age Group:      <25 | 25-34 | 35-44 | 45-54 | 55-64 | 65+

Data Sources (ACS1):
  B19326  — Median Income by Sex × Work Experience          (primary signal)
  B19013* — Median Household Income by Race (suffixed)      (race-adjusted baseline)
  B07011  — Median Income by Geographic Mobility            (mover premium)
  B17002  — Ratio of Income to Poverty Level                (cost-of-living normalizer)

Usage:
  python income_opportunity_analysis.py
  → Edit the PROFILE dict at the bottom to change demographic inputs.
"""

import time
import pandas as pd
import censusdis.data as ced
from censusdis.datasets import ACS1

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.0f}".format)
pd.set_option("display.max_rows", 50)

VINTAGE = 2023  # Most recent ACS1 year


# ──────────────────────────────────────────────────────────────────────────────
# VARIABLE MAPS
# ──────────────────────────────────────────────────────────────────────────────

# B19013 race suffixes → ACS group codes
RACE_GROUP_MAP = {
    "white_non_hispanic": "B19013H",   # White alone, not Hispanic/Latino
    "black":              "B19013B",   # Black or African American alone
    "hispanic":           "B19013I",   # Hispanic or Latino
    "asian":              "B19013D",   # Asian alone
    "american_indian":    "B19013C",   # American Indian/Alaska Native alone
    "native_hawaiian":    "B19013E",   # Native Hawaiian/Pacific Islander alone
    "other":              "B19013F",   # Some other race alone
    "two_or_more":        "B19013G",   # Two or more races
}

# B19326 columns: Median income by sex × work experience
# Structure: Total | Male (FT | Other) | Female (FT | Other)
B19326_COLS = {
    "male_fulltime":    "B19326_002E",
    "male_other":       "B19326_003E",
    "female_fulltime":  "B19326_005E",
    "female_other":     "B19326_006E",
}

# B07011 columns: Median income by mobility status
# _001E=Total(stayed) | _002E=Moved within county | _003E=Moved within state
# _004E=Moved from diff state | _005E=Moved from abroad
B07011_COLS = {
    "stayed":             "B07011_001E",
    "moved_within_county":"B07011_002E",
    "moved_within_state": "B07011_003E",
    "moved_from_state":   "B07011_004E",
    "moved_from_abroad":  "B07011_005E",
}

# B17002 columns: Ratio of income to poverty level (above 2.0 = comfortable)
# _001E=Total | _010E=1.50-1.84 | _011E=1.85-1.99 | _012E=>=2.0
B17002_ABOVE_2X = "B17002_012E"  # count of people at 200%+ poverty level
B17002_TOTAL    = "B17002_001E"


# ──────────────────────────────────────────────────────────────────────────────
# DATA PULLS
# ──────────────────────────────────────────────────────────────────────────────

def pull_b19326(vintage: int = VINTAGE) -> pd.DataFrame:
    """Median income by sex × work experience — all counties."""
    print("  [1/4] Pulling B19326 (income by sex × work experience)...")
    df = ced.download(ACS1, vintage, group="B19326", state="*", county="*")
    df = df.rename(columns={v: k for k, v in B19326_COLS.items()})
    return df[["NAME", "STATE", "COUNTY",
               "male_fulltime", "male_other",
               "female_fulltime", "female_other"]]


def pull_b19013_race(race_key: str, vintage: int = VINTAGE) -> pd.DataFrame:
    """Median household income for the selected race group."""
    group = RACE_GROUP_MAP.get(race_key, "B19013I")
    print(f"  [2/4] Pulling {group} (race-adjusted median HH income)...")
    df = ced.download(ACS1, vintage, group=group, state="*", county="*")
    col = f"{group}_001E"
    df = df.rename(columns={col: "race_median_hh_income"})
    return df[["STATE", "COUNTY", "race_median_hh_income"]]


def pull_b07011(vintage: int = VINTAGE) -> pd.DataFrame:
    """Median income by geographic mobility — mover premium signal."""
    print("  [3/4] Pulling B07011 (median income by mobility)...")
    df = ced.download(ACS1, vintage, group="B07011", state="*", county="*")
    df = df.rename(columns={v: k for k, v in B07011_COLS.items()})
    available = [c for c in B07011_COLS.keys() if c in df.columns]
    return df[["STATE", "COUNTY"] + available]


def pull_b17002(vintage: int = VINTAGE) -> pd.DataFrame:
    """Poverty ratio — share of population above 200% poverty line."""
    print("  [4/4] Pulling B17002 (income-to-poverty ratio)...")
    df = ced.download(ACS1, vintage, group="B17002", state="*", county="*")
    df = df.rename(columns={
        B17002_ABOVE_2X: "above_2x_poverty_count",
        B17002_TOTAL:    "poverty_ratio_total",
    })
    df["pct_above_2x_poverty"] = (
        df["above_2x_poverty_count"] / df["poverty_ratio_total"] * 100
    ).round(1)
    return df[["STATE", "COUNTY", "pct_above_2x_poverty"]]


# ──────────────────────────────────────────────────────────────────────────────
# PROFILE → COLUMN SELECTOR
# ──────────────────────────────────────────────────────────────────────────────

def resolve_income_column(sex: str, work_experience: str) -> str:
    """
    Map sex + work_experience inputs to the correct B19326 column name.

    sex:             "male" | "female"
    work_experience: "fulltime" | "other"
    """
    sex = sex.lower().strip()
    work_experience = work_experience.lower().replace("-", "").replace(" ", "_")

    key = f"{sex}_{work_experience}"
    if key not in B19326_COLS:
        valid = list(B19326_COLS.keys())
        raise ValueError(f"Invalid combination '{key}'. Valid: {valid}")
    return key  # already renamed in pull_b19326()


def build_mover_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mover premium = (median income of people who moved from a different state)
                  - (median income of people who stayed).
    Positive = city attracts higher earners from out of state.
    """
    if "moved_from_state" in df.columns and "stayed" in df.columns:
        df["mover_premium"] = (df["moved_from_state"] - df["stayed"]).clip(lower=0)
    else:
        df["mover_premium"] = 0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def score_geographies(
    df: pd.DataFrame,
    income_col: str,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute an opportunity score for each geography.

    Scoring components (all normalized 0–100):
      - target_income:       Median income for your sex × work experience (50%)
      - race_income:         Race-adjusted median HH income               (25%)
      - mover_premium_norm:  Normalized mover income premium              (15%)
      - prosperity_pct:      % population above 200% poverty line         (10%)

    Weights can be overridden via the `weights` dict.
    """
    if weights is None:
        weights = {
            "target_income":      0.50,
            "race_income":        0.25,
            "mover_premium_norm": 0.15,
            "prosperity_pct":     0.10,
        }

    def normalize(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(50.0, index=series.index)
        return ((series - mn) / (mx - mn) * 100).round(2)

    df = df.copy()
    df["target_income_norm"]    = normalize(df[income_col].fillna(0))
    df["race_income_norm"]      = normalize(df["race_median_hh_income"].fillna(0))
    df["mover_premium_norm"]    = normalize(df["mover_premium"].fillna(0))
    df["prosperity_pct_norm"]   = normalize(df["pct_above_2x_poverty"].fillna(0))

    df["opportunity_score"] = (
        weights["target_income"]      * df["target_income_norm"]    +
        weights["race_income"]        * df["race_income_norm"]       +
        weights["mover_premium_norm"] * df["mover_premium_norm"]     +
        weights["prosperity_pct"]     * df["prosperity_pct_norm"]
    ).round(2)

    return df.sort_values("opportunity_score", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def analyze_income_opportunity(
    sex:             str = "female",
    race:            str = "hispanic",
    work_experience: str = "fulltime",
    top_n:           int = 25,
    vintage:         int = VINTAGE,
    custom_weights:  dict = None,
    save_csv:        bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: pull ACS1 data → merge → score → rank geographies.

    Parameters
    ----------
    sex             : "male" | "female"
    race            : "white_non_hispanic" | "black" | "hispanic" | "asian" |
                      "american_indian" | "native_hawaiian" | "other" | "two_or_more"
    work_experience : "fulltime" | "other"
    top_n           : number of top geographies to display
    vintage         : ACS1 year (default 2023)
    custom_weights  : override scoring weights (dict with keys:
                      target_income, race_income, mover_premium_norm, prosperity_pct)
    save_csv        : save full ranked results to CSV

    Returns
    -------
    pd.DataFrame    : ranked geographies with scores and income figures
    """
    print("\n" + "═" * 65)
    print("ACS1 Income Opportunity Analysis")
    print("═" * 65)
    print(f"  Profile:  {sex.title()} | {race.replace('_', ' ').title()} | "
          f"{work_experience.replace('_', ' ').title()} worker")
    print(f"  Vintage:  ACS1 {vintage}")
    print(f"  Scoring:  {'custom weights' if custom_weights else 'default weights'}")
    print("═" * 65 + "\n")

    # ── 1. Validate inputs ────────────────────────────────────────────────────
    income_col = resolve_income_column(sex, work_experience)

    if race not in RACE_GROUP_MAP:
        raise ValueError(
            f"Unknown race '{race}'. Valid options:\n  {list(RACE_GROUP_MAP.keys())}"
        )

    # ── 2. Pull data ──────────────────────────────────────────────────────────
    df_income  = pull_b19326(vintage)
    df_race    = pull_b19013_race(race, vintage)
    df_movers  = pull_b07011(vintage)
    df_poverty = pull_b17002(vintage)

    # ── 3. Merge all tables on STATE + COUNTY ─────────────────────────────────
    print("\n  Merging tables...")
    merged = df_income.copy()
    for df in [df_race, df_movers, df_poverty]:
        merged = merged.merge(df, on=["STATE", "COUNTY"], how="left")

    # ── 4. Compute mover premium ──────────────────────────────────────────────
    merged = build_mover_premium(merged)

    # ── 5. Drop rows with missing primary income signal ───────────────────────
    before = len(merged)
    merged = merged[merged[income_col].notna() & (merged[income_col] > 0)]
    print(f"  Geographies after filtering: {len(merged)} (dropped {before - len(merged)} missing)")

    # ── 6. Score ──────────────────────────────────────────────────────────────
    ranked = score_geographies(merged, income_col, weights=custom_weights)

    # ── 7. Build clean output table ───────────────────────────────────────────
    display_cols = [
        "NAME",
        income_col,              # your demographic income
        "race_median_hh_income", # race-adjusted HH income
        "mover_premium",         # out-of-state mover income premium
        "pct_above_2x_poverty",  # % comfortable earners
        "opportunity_score",     # final score
    ]
    display_cols = [c for c in display_cols if c in ranked.columns]

    result = ranked[display_cols].rename(columns={
        "NAME":                  "Geography",
        income_col:              "Your_Median_Income",
        "race_median_hh_income": "Race_Adj_HH_Income",
        "mover_premium":         "Mover_Premium",
        "pct_above_2x_poverty":  "Pct_2x_Poverty",
        "opportunity_score":     "Opportunity_Score",
    }).reset_index(drop=True)

    result.index = result.index + 1  # 1-based rank

    # ── 8. Print results ──────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"TOP {top_n} GEOGRAPHIES — Opportunity Score (higher = better)")
    print(f"{'═' * 65}")
    print(result.head(top_n).to_string())

    print(f"\n{'─' * 65}")
    print("BOTTOM 5 (for context)")
    print(f"{'─' * 65}")
    print(result.tail(5).to_string())

    # ── 9. Summary stats ──────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print("SUMMARY")
    print(f"{'═' * 65}")
    top1 = result.iloc[0]
    print(f"  Best geography:     {top1['Geography']}")
    print(f"  Your median income: ${top1['Your_Median_Income']:,.0f}")
    print(f"  Race-adj HH income: ${top1['Race_Adj_HH_Income']:,.0f}")
    print(f"  Opportunity score:  {top1['Opportunity_Score']:.1f} / 100")

    nat_median = result["Your_Median_Income"].median()
    print(f"\n  National median ({sex}/{work_experience}): ${nat_median:,.0f}")
    print(f"  Income premium of top city: "
          f"+${top1['Your_Median_Income'] - nat_median:,.0f} "
          f"({(top1['Your_Median_Income'] / nat_median - 1) * 100:.1f}%)")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    if save_csv:
        fname = f"income_opportunity_{sex}_{race}_{work_experience}_{vintage}.csv"
        result.to_csv(fname)
        print(f"\n  ✓ Saved full results to: {fname}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON HELPER — run multiple profiles side by side
# ──────────────────────────────────────────────────────────────────────────────

def compare_profiles(profiles: list[dict], top_n: int = 10) -> pd.DataFrame:
    """
    Run analyze_income_opportunity for multiple demographic profiles
    and return a side-by-side comparison of top geographies.

    profiles: list of dicts with keys: sex, race, work_experience, label
    """
    all_results = {}

    for p in profiles:
        label = p.get("label", f"{p['sex']}_{p['race']}_{p['work_experience']}")
        print(f"\n{'▶' * 3} Running profile: {label}")
        result = analyze_income_opportunity(
            sex=p.get("sex", "female"),
            race=p.get("race", "hispanic"),
            work_experience=p.get("work_experience", "fulltime"),
            top_n=top_n,
            save_csv=False,
        )
        all_results[label] = result[["Geography", "Opportunity_Score"]].head(top_n)
        all_results[label] = all_results[label].rename(
            columns={"Opportunity_Score": f"Score_{label}"}
        )

    # Merge all profiles on geography
    combined = list(all_results.values())[0][["Geography"]].copy()
    for label, df in all_results.items():
        combined = combined.merge(df, on="Geography", how="outer")

    combined = combined.fillna(0).sort_values(
        combined.columns[1], ascending=False
    ).reset_index(drop=True)

    print(f"\n{'═' * 65}")
    print("PROFILE COMPARISON — Top Geographies")
    print("═" * 65)
    print(combined.head(top_n).to_string())

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — Edit PROFILE to change inputs
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Single profile analysis ───────────────────────────────────────────────
    PROFILE = {
        "sex":             "female",      # "male" | "female"
        "race":            "hispanic",    # see RACE_GROUP_MAP keys above
        "work_experience": "fulltime",    # "fulltime" | "other"
        "top_n":           25,
        "vintage":         2023,
    }

    results = analyze_income_opportunity(**PROFILE)

    # ── Optional: compare multiple profiles ──────────────────────────────────
    # Uncomment to run a side-by-side comparison across demographics:

    # PROFILES_TO_COMPARE = [
    #     {"sex": "female", "race": "hispanic",         "work_experience": "fulltime", "label": "Hispanic_F"},
    #     {"sex": "female", "race": "black",            "work_experience": "fulltime", "label": "Black_F"},
    #     {"sex": "female", "race": "white_non_hispanic","work_experience": "fulltime", "label": "White_F"},
    #     {"sex": "male",   "race": "hispanic",         "work_experience": "fulltime", "label": "Hispanic_M"},
    # ]
    # compare_profiles(PROFILES_TO_COMPARE, top_n=15)
