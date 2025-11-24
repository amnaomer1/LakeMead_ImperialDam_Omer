#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCI 895.01 — lakemead_functions.py

Author: Amna Omer
Created: 2025-11-20
License: MIT

File Description
----------------

Utility functions for analyzing links between Lake Mead elevation
and California’s Colorado River water use.

This module provides a small toolkit that can be imported into a Jupyter
notebook or other scripts. It is organized into five main parts:

0. Imports & global plotting style
   - `set_default_plot_style()` sets a consistent Matplotlib theme for all
     figures in the project.

1. Data loading & basic diagnostic plots
   - `compute_mead_end_of_year_elevation()` extracts December (end-of-year)
     Lake Mead elevations from a monthly time series (ft).
   - `plot_mead_end_of_year_elevation()` plots the resulting annual
     elevation series.
   - `compute_california_annual_wateruse()` cleans the California water-use
     dataset and aggregates monthly diversions and consumptive use
     (acre-feet) to annual totals.
   - `plot_california_annual_wateruse()` plots annual diversions and
     consumptive use.

2. Shared helpers: seasonal adjustment & correlation tools
   - `zscore_by_month()` removes the mean seasonal cycle and returns
     month-specific z-scores.
   - `effective_n()`, `pearson_with_ci_ac()`, `holm_bonferroni()`, and
     `partial_corr_linear_time()` provide correlation, confidence interval,
     and multiple-testing utilities that account for autocorrelation and
     time trends.

3. Lagged correlation analysis
   - `compute_lag_correlation_mead_cu()` merges Lake Mead elevation and
     California consumptive-use series, builds seasonal anomalies and
     z-scores, and scans lagged correlations across ±max_lag months.
   - `plot_lag_correlation()` visualizes Pearson lag correlations with
     confidence intervals and Holm–Bonferroni significance markers.
   - `plot_mead_cu_zscores()` plots the standardized z-score time series
     for visual comparison.
   - `simple_lag_scan_raw()` and `plot_lag_correlation_sensitivity()` give
     a sensitivity check using raw monthly values (no seasonal adjustment).

4. Response elasticity analysis (monthly & annual)
   - `compute_response_elasticity()` estimates how sensitive California’s
     consumptive use is to changes in Lake Mead elevation at both monthly
     and annual scales, returning regression statistics for each.
   - `plot_response_elasticity()` produces side-by-side scatter plots with
     fitted lines for monthly and annual elasticity.

5. Policy milestones & before–after analysis
   - `compute_policy_impacts_ca_wateruse()` aggregates annual water-use
     (converted to million acre-feet, MAF), joins Lake Mead December
     elevation (ft), and computes before–after percent changes in mean
     use for a user-specified set of policy years.
   - `plot_policy_timeseries()` plots annual consumptive use, diversions,
     and Lake Mead elevation with shaded policy periods and vertical
     policy markers.
   - `plot_policy_delta_means()` creates a grouped bar chart summarizing
     the percent change in mean annual water use before vs. after each
     policy.

All functions are written to be reusable: they take tidy pandas DataFrames
as inputs and return DataFrames and/or Matplotlib Axes objects, so the same
workflow can be applied to other sites or time periods with similar data
structure.
"""

#%% 0. Imports & global plotting style

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, norm, t as tdist
from scipy import stats

def set_default_plot_style():
    """
    Set a consistent Matplotlib style for all project plots.
    """
    plt.rcParams.update({
        "figure.figsize": (9, 4.6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

#%% 1. Data loading & basic diagnostic plots

#  ---  Lake Mead  ---

# Data-processing function
def compute_mead_end_of_year_elevation(mead: pd.DataFrame) -> pd.DataFrame:
    """
    Extract end-of-year (December) Lake Mead elevation values from a
    monthly time series.

    Parameters
    ----------
    mead : pandas.DataFrame
        DataFrame with at least the columns:
        - 'Date' : datetime-like, monthly timestamps
        - 'Elevation_ft' : float, reservoir elevation in feet

    Returns
    -------
    annual_end : pandas.DataFrame
        DataFrame with one row per year containing:
        - 'Year'         : int, calendar year
        - 'Date'         : Timestamp for the December record
        - 'Elevation_ft' : float, December elevation for that year

    Notes
    -----
    This function assumes that the input DataFrame contains one row per
    month and that December (month == 12) is present for each year of
    interest. Any rows with missing 'Date' or 'Elevation_ft' values
    are dropped before processing.
    """
    # Defensive copy to avoid modifying the original dataframe
    mead = mead.copy()

    # Ensure Date is datetime and sorted
    mead["Date"] = pd.to_datetime(mead["Date"])
    mead = mead.dropna(subset=["Date", "Elevation_ft"]).sort_values("Date")

    # Identify December records and extract year
    mead["Month"] = mead["Date"].dt.month
    annual_end = mead.loc[mead["Month"] == 12].copy()
    annual_end["Year"] = annual_end["Date"].dt.year

    # Keep only the relevant columns
    annual_end = annual_end[["Year", "Date", "Elevation_ft"]].reset_index(drop=True)

    return annual_end

# Plotting function
def plot_mead_end_of_year_elevation(annual_end: pd.DataFrame, ax=None):
    """
    Plot end-of-year (December) Lake Mead elevation as a time series.

    Parameters
    ----------
    annual_end : pandas.DataFrame
        Output from `compute_mead_end_of_year_elevation`, with columns
        'Year' and 'Elevation_ft'.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure and axes
        are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        annual_end["Year"],
        annual_end["Elevation_ft"],
        lw=2,
        marker="o",
    )
    ax.set_title("Lake Mead Elevation (End-of-Year, 2000–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Elevation (ft)")
    ax.grid(True, alpha=0.3)

    return ax


# ---  California Water Use  ---

# Data-processing function
def compute_california_annual_wateruse(wateruse_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and aggregate California monthly water-use data to annual
    totals for diversions and consumptive use.

    Parameters
    ----------
    wateruse_raw : pandas.DataFrame
        Raw monthly water-use data with at least the columns:
        - 'Date'             : datetime-like or string
        - 'Diversion_af'     : str/float, total diversions (acre-feet)
        - 'ConsumptiveUse_af': str/float, consumptive use (acre-feet)
        Optional columns:
        - 'MeasuredReturns_af'
        - 'UnmeasuredReturns_af'

    Returns
    -------
    wateruse : pandas.DataFrame
        Cleaned monthly dataset with:
        - 'Date'  : Timestamp
        - 'Year'  : int, calendar year
        - *_af    : float columns converted from strings
    annual : pandas.DataFrame
        Annual totals for complete years only (12 months present), with:
        - 'Year'           : int
        - 'Diversion_af'   : float, annual total diversions
        - 'ConsumptiveUse_af' : float, annual total consumptive use

    Notes
    -----
    - Thousand separators (commas) in *_af columns are removed.
    - Non-numeric entries in *_af columns are coerced to NaN.
    - Only years with 12 monthly observations are retained in `annual`.
    """
    # Work on a copy
    df = wateruse_raw.copy()

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Identify numeric columns and clean them
    num_cols = [c for c in df.columns if c.endswith("_af")]
    for c in num_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"": None})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Add year column
    df["Year"] = df["Date"].dt.year

    # Keep only complete years (12 months)
    months_per_year = df.groupby("Year")["Date"].nunique()
    complete_years = months_per_year[months_per_year == 12].index

    annual = (
        df.loc[df["Year"].isin(complete_years)]
        .groupby("Year", as_index=False)[["Diversion_af", "ConsumptiveUse_af"]]
        .sum(min_count=1)
        .sort_values("Year")
        .reset_index(drop=True)
    )

    return df, annual

# Plotting function

def plot_california_annual_wateruse(annual: pd.DataFrame, ax=None):
    """
    Plot annual California water use (diversions and consumptive use).

    Parameters
    ----------
    annual : pandas.DataFrame
        Annual totals as returned by `compute_california_annual_wateruse`,
        with columns 'Year', 'Diversion_af', and 'ConsumptiveUse_af'.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure and axes
        are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        annual["Year"],
        annual["Diversion_af"],
        label="Total Diversions (acre-feet/year)",
        lw=2,
    )

    if "ConsumptiveUse_af" in annual.columns:
        ax.plot(
            annual["Year"],
            annual["ConsumptiveUse_af"],
            label="Consumptive Use (acre-feet/year)",
            lw=2,
        )

    ax.set_title("California Water Use (Annual Totals)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Annual Volume (acre-feet)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


#%% 2. Shared helpers: seasonal adjustment & correlation tools

# Computes monthly z-scores by removing mean seasonal cycle (seasonal adjustment)
def zscore_by_month(series: pd.Series) -> pd.Series:
    """
    Compute z-scores by calendar month to remove the mean seasonal cycle.

    Assumes the Series index is a DatetimeIndex or PeriodIndex that can be
    converted to month. For each calendar month (Jan, Feb, ...), subtract
    that month's long-term mean and divide by that month's standard deviation.
    """
    # Use the Series index month, not the Date column
    month = series.index.to_period("M").month
    g = series.groupby(month)

    mu = g.transform("mean")
    sd = g.transform(lambda x: x.std(ddof=1))

    return (series - mu) / sd

# Estimates effective sample size accounting for autocorrelation in both series
def effective_n(x, y):
    """
    Approximate effective sample size for correlation in both series.
    neff = n * (1 - rx*ry) / (1 + rx*ry)
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    n = min(len(x), len(y))
    if n < 4:
        return max(3, n)

    r1x = x.autocorr(lag=1)
    r1y = y.autocorr(lag=1)

    if r1x is None or np.isnan(r1x):
        r1x = 0.0
    if r1y is None or np.isnan(r1y):
        r1y = 0.0

    denom = (1 + r1x * r1y)
    if np.isclose(denom, 0.0):
        return max(3, n)

    neff = n * (1 - r1x * r1y) / denom
    return float(np.clip(neff, 3, n))

# Pearson correlation with confidence interval and p-value using autocorrelation-adjusted n
def pearson_with_ci_ac(x, y, alpha=0.05):
    """
    Pearson r with CI and p-value using Fisher z but with an effective n
    that accounts (crudely) for autocorrelation.
    """
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    n_raw = len(valid)
    if n_raw < 3:
        return np.nan, np.nan, n_raw, (np.nan, np.nan), np.nan

    r, _ = pearsonr(valid["x"], valid["y"])
    n_eff = effective_n(valid["x"], valid["y"])

    # Fisher z CI using n_eff
    r_clip = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r_clip)
    se = 1 / np.sqrt(n_eff - 3)
    zcrit = norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh([z - zcrit * se, z + zcrit * se])

    # two-sided p using t with df = n_eff - 2
    tval = r * np.sqrt((n_eff - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
    p = 2 * (1 - tdist.cdf(abs(tval), df=max(1, n_eff - 2)))
    return r, p, n_eff, (lo, hi), n_raw

# Holm-Bonferroni multiple-testing correction for p-values
def holm_bonferroni(pvals):
    """
    Holm–Bonferroni adjustment (no external deps).
    Returns adjusted p-values in original order.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0

    for i, idx in enumerate(order):
        adj_p = (m - i) * pvals[idx]   # Holm step-down
        adj[idx] = max(adj_p, prev)
        prev = adj[idx]

    adj = np.minimum(adj, 1.0)
    return adj

# Partial correlation between two variables while controlling for linear time trend
def partial_corr_linear_time(x, y):
    """
    Partial correlation of x and y controlling for a linear time trend.
    Regress x and y separately on [1, t], correlate residuals.
    """
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(tmp)
    if n < 5:
        return np.nan, np.nan, n

    t = np.arange(n)
    X = np.column_stack([np.ones(n), t])

    bx = np.linalg.lstsq(X, tmp["x"].values, rcond=None)[0]
    by = np.linalg.lstsq(X, tmp["y"].values, rcond=None)[0]
    xres = tmp["x"].values - X @ bx
    yres = tmp["y"].values - X @ by

    r, p = pearsonr(xres, yres)
    return r, p, n

#%% 3. Lagged correlation analysi

# Runs full lagged-correlation analysis between Lake Mead elevation and CA consumptive use (seasonally adjusted)
def compute_lag_correlation_mead_cu(
    mead: pd.DataFrame,
    cu: pd.DataFrame,
    max_lag: int = 12,
    alpha: float = 0.05,
    study_start: str = "2000-01-01",
    study_end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute lagged correlations between Lake Mead elevation and California
    consumptive use, after removing the mean seasonal cycle.

    The function:
      1) Merges the two monthly time series on Date.
      2) Keeps data within a user-defined study period.
      3) Builds monthly anomalies and z-scores for both variables.
      4) Scans lags from -max_lag to +max_lag and calculates:
         Pearson r (with autocorrelation-aware CI and p-value),
         Spearman rho, and partial correlation controlling for a linear
         time trend.
    """
    mead = mead.copy()
    cu = cu.copy()

    # Standardize dates to month-end timestamps
    mead["Date"] = pd.to_datetime(mead["Date"]).dt.to_period("M").dt.to_timestamp("M")
    cu["Date"]   = pd.to_datetime(cu["Date"]).dt.to_period("M").dt.to_timestamp("M")

    # Merge on Date
    df = (
        pd.merge(
            mead[["Date", "Elevation_ft"]],
            cu[["Date", "ConsumptiveUse_af"]],
            on="Date",
            how="inner",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Restrict to study period
    start = pd.to_datetime(study_start)
    end   = pd.to_datetime(study_end)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    # Drop missing values in the key variables
    df_clean = df.dropna(subset=["Elevation_ft", "ConsumptiveUse_af"]).copy()

    #  Make sure the key series are numeric 
    for col in ["Elevation_ft", "ConsumptiveUse_af"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Water year
    df_clean["WY"] = (df_clean["Date"] + pd.offsets.MonthBegin(3)).dt.year

    # Monthly anomalies and z-scores
    for col in ["Elevation_ft", "ConsumptiveUse_af"]:
        mo_grp = df_clean.groupby(df_clean["Date"].dt.month)[col]
        df_clean[col + "_anom"] = df_clean[col] - mo_grp.transform("mean")
        df_clean[col + "_z"] = (
            df_clean[col + "_anom"] - df_clean[col + "_anom"].mean()
        ) / df_clean[col + "_anom"].std(ddof=1)

    mead_series = df_clean["Elevation_ft_z"]
    cu_series   = df_clean["ConsumptiveUse_af_z"]

    # Lag scan
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        xs = mead_series.shift(lag)
        ys = cu_series

        rP, pP, n_eff, (loP, hiP), n_raw = pearson_with_ci_ac(xs, ys)

        valid = pd.DataFrame({"x": xs, "y": ys}).dropna()
        if len(valid) >= 3:
            rhoS, pS = spearmanr(valid["x"], valid["y"])
        else:
            rhoS, pS = np.nan, np.nan

        rPart, pPart, nPart = partial_corr_linear_time(xs, ys)

        rows.append({
            "lag_months": lag,
            "pearson_r": rP,
            "pearson_p": pP,
            "pearson_lo": loP,
            "pearson_hi": hiP,
            "n_eff": n_eff,
            "n_raw": n_raw,
            "spearman_rho": rhoS,
            "spearman_p": pS,
            "partial_r_time": rPart,
            "partial_p_time": pPart,
            "n_partial": nPart,
        })

    lagdf = pd.DataFrame(rows).sort_values("lag_months").reset_index(drop=True)

    # Holm–Bonferroni adjustment on Pearson p-values
    mask = lagdf["pearson_p"].notna()
    lagdf.loc[mask, "pearson_p_adj"] = holm_bonferroni(
        lagdf.loc[mask, "pearson_p"].values
    )

    return df_clean, lagdf

# Plots Pearson lag correlations with confidence intervals and adjusted significance markers
def plot_lag_correlation(lagdf):
    """Plot Pearson lag correlations with CIs and Holm–Bonferroni significance."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = lagdf["lag_months"]
    y = lagdf["pearson_r"]
    yerr = np.vstack([y - lagdf["pearson_lo"], lagdf["pearson_hi"] - y])

    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    ax.axhline(0, ls="--", lw=1, color="gray")
    ax.axvline(0, ls=":",  lw=1, color="gray")
    ax.set_xlabel("Lag (months) — positive: Lake Mead leads CU; negative: CU leads Mead")
    ax.set_ylabel("Pearson r (seasonally adjusted)")
    ax.set_title("Lagged correlation: Lake Mead Elevation vs California Consumptive Use")

    sig = (lagdf["pearson_p_adj"] <= 0.05) & lagdf["pearson_p_adj"].notna()
    ax.scatter(
        x[sig],
        y[sig],
        s=70,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Holm–Bonferroni p≤0.05"
    )
    ax.legend(loc="best")
    plt.show()

# Plots seasonally adjusted z-score time series for visual comparison
def plot_mead_cu_zscores(df_clean):
    """Plot seasonally adjusted z-scores for Lake Mead elevation and CA CU."""
    ax = df_clean.plot(
        x="Date",
        y=["Elevation_ft_z", "ConsumptiveUse_af_z"],
        lw=1.2
    )
    ax.set_title(
        "Seasonally-Adjusted Z-scores\n"
        "Lake Mead Elevation vs California Consumptive Use"
    )
    ax.set_ylabel("Z-score (monthly anomaly standardized)")
    ax.legend(["Mead (z)", "CA CU (z)"])
    plt.show()

# Computes simple lag correlations on raw monthly data (no seasonal adjustment) for sensitivity comparison
def simple_lag_scan_raw(
    df_clean: pd.DataFrame,
    x_col: str = "Elevation_ft",
    y_col: str = "ConsumptiveUse_af",
    max_lag: int = 12
) -> pd.DataFrame:
    """
    Compute simple Pearson lagged correlations between two raw monthly
    series (no seasonal adjustment, no autocorrelation correction).

    Parameters
    ----------
    df_clean : DataFrame
        Must contain columns x_col and y_col, aligned in time.
    x_col : str
        Upstream variable column (e.g., Lake Mead elevation).
    y_col : str
        Downstream variable column (e.g., CA consumptive use).
    max_lag : int
        Maximum lag (months) to scan in both directions.

    Returns
    -------
    lagdf_raw : DataFrame
        Columns: ['lag_months', 'pearson_r_raw', 'p_raw'].
        Positive lag means x leads y.
    """
    x = df_clean[x_col]
    y = df_clean[y_col]

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        xs = x.shift(lag)
        valid = pd.DataFrame({"x": xs, "y": y}).dropna()
        if len(valid) >= 3:
            r, p = pearsonr(valid["x"], valid["y"])
        else:
            r, p = np.nan, np.nan
        rows.append({
            "lag_months": lag,
            "pearson_r_raw": r,
            "p_raw": p
        })

    lagdf_raw = pd.DataFrame(rows).sort_values("lag_months").reset_index(drop=True)
    return lagdf_raw

# Compares lag correlations with vs. without seasonal adjustment (z-scores vs raw data)
def plot_lag_correlation_sensitivity(lagdf_z: pd.DataFrame,
                                     lagdf_raw: pd.DataFrame) -> None:
    """
    Plot a comparison of lagged Pearson correlations:
    - seasonally adjusted z-scores (lagdf_z)
    - raw monthly values (lagdf_raw).
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    # Seasonally adjusted (from scan_lag_correlation_mead_cu)
    ax.plot(
        lagdf_z["lag_months"],
        lagdf_z["pearson_r"],
        label="Seasonally adjusted (z-scores)"
    )

    # Raw monthly (from simple_lag_scan_raw)
    ax.plot(
        lagdf_raw["lag_months"],
        lagdf_raw["pearson_r_raw"],
        label="Raw monthly (no seasonal adjustment)",
        linestyle="--"
    )

    ax.axhline(0, color="gray", linestyle=":")
    ax.axvline(0, color="gray", linestyle=":")

    ax.set_xlabel("Lag (months) — positive: Lake Mead leads CU")
    ax.set_ylabel("Pearson r")
    ax.set_title("Lagged Correlation: With vs Without Seasonal Adjustment")
    ax.legend()
    plt.show()

#%% 4. Response elasticity analysis (monthly & annual)

def compute_response_elasticity(
    mead: pd.DataFrame,
    water: pd.DataFrame,
    study_start: str = "2000-01-01",
    study_end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Estimate how sensitive California's consumptive use is to changes in
    Lake Mead elevation at both monthly and annual time scales.

    The function:
      1. Aligns the two datasets on a common monthly time axis.
      2. Restricts the data to a user-defined study period.
      3. Builds standardized z-scores:
         - Monthly: z-scores are computed by calendar month to remove
           the mean seasonal cycle.
         - Annual: Lake Mead elevation is averaged by year and CU is
           summed by year, then both are standardized.
      4. Fits simple linear regressions:
         - Monthly: z(CU) ~ z(Mead)
         - Annual:  z(CU) ~ z(Mead)

    Parameters
    ----------
    mead : pandas.DataFrame
        Lake Mead data with at least:
        - 'Date'         : datetime-like
        - 'Elevation_ft' : float
    water : pandas.DataFrame
        California water-use data with at least:
        - 'Date'              : datetime-like
        - 'ConsumptiveUse_af' : float
    study_start : str, optional
        Start date for the analysis window (e.g., "2000-01-01").
    study_end : str, optional
        End date for the analysis window (e.g., "2024-12-31").

    Returns
    -------
    df_monthly : pandas.DataFrame
        Monthly merged dataframe within the study period with columns:
        - 'Elevation_ft', 'ConsumptiveUse_af'
        - 'z_elev_m', 'z_cu_m'  (monthly z-scores)
    df_annual : pandas.DataFrame
        Annual dataframe with columns:
        - 'Elev_ft', 'CU_af'
        - 'z_elev_a', 'z_cu_a'  (annual z-scores)
    stats_dict : dict
        Regression summary for each time scale:
        {
          "monthly": {"slope", "intercept", "r2", "p"},
          "annual":  {"slope", "intercept", "r2", "p"},
        }
    """
    # Work on copies
    mead = mead.copy()
    water = water.copy()

    # Standardize dates to monthly (month-end) timestamps
    to_month = lambda s: pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")
    mead["Date"] = to_month(mead["Date"])
    water["Date"] = to_month(water["Date"])

    # Keep only the columns we need
    mead = mead[["Date", "Elevation_ft"]]
    water = water[["Date", "ConsumptiveUse_af"]]

    # Ensure numeric types
    mead["Elevation_ft"] = pd.to_numeric(mead["Elevation_ft"], errors="coerce")
    water["ConsumptiveUse_af"] = pd.to_numeric(water["ConsumptiveUse_af"], errors="coerce")

    # Merge on Date and restrict to study window
    df = (
        mead.set_index("Date")[["Elevation_ft"]]
        .join(water.set_index("Date")[["ConsumptiveUse_af"]], how="inner")
        .dropna()
        .sort_index()
    )

    start = pd.to_datetime(study_start)
    end = pd.to_datetime(study_end)
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()

    # ---------- Monthly z-scores (by calendar month) ----------
    df["z_elev_m"] = zscore_by_month(df["Elevation_ft"])
    df["z_cu_m"] = zscore_by_month(df["ConsumptiveUse_af"])

    # Monthly regression: z(CU) ~ z(Mead)
    res_m = stats.linregress(df["z_elev_m"], df["z_cu_m"])
    b_m, a_m = res_m.slope, res_m.intercept
    r2_m, p_m = res_m.rvalue**2, res_m.pvalue

    # ---------- Annual aggregation & z-scores ----------
    annual_elev = (
        mead.groupby(mead["Date"].dt.year)["Elevation_ft"]
        .mean()
        .rename("Elev_ft")
    )
    annual_cu = (
        water.groupby(water["Date"].dt.year)["ConsumptiveUse_af"]
        .sum()
        .rename("CU_af")
    )

    df_ann = pd.concat([annual_elev, annual_cu], axis=1).dropna()

    df_ann["z_elev_a"] = (
        df_ann["Elev_ft"] - df_ann["Elev_ft"].mean()
    ) / df_ann["Elev_ft"].std(ddof=1)
    df_ann["z_cu_a"] = (
        df_ann["CU_af"] - df_ann["CU_af"].mean()
    ) / df_ann["CU_af"].std(ddof=1)

    # Annual regression: z(CU) ~ z(Mead)
    res_a = stats.linregress(df_ann["z_elev_a"], df_ann["z_cu_a"])
    b_a, a_a = res_a.slope, res_a.intercept
    r2_a, p_a = res_a.rvalue**2, res_a.pvalue

    stats_dict = {
        "monthly": {"slope": b_m, "intercept": a_m, "r2": r2_m, "p": p_m},
        "annual": {"slope": b_a, "intercept": a_a, "r2": r2_a, "p": p_a},
    }

    # Return with clear names
    df_monthly = df.copy()
    df_annual = df_ann.copy()

    return df_monthly, df_annual, stats_dict

# Plotting function for the elasticity figure
def plot_response_elasticity(
    df_monthly: pd.DataFrame,
    df_annual: pd.DataFrame,
    stats_dict: dict,
    title_suffix: str = "2000–2024",
    axes=None,
):
    """
    Create side-by-side scatter plots showing monthly and annual
    elasticity between Lake Mead elevation and California consumptive use.

    Parameters
    ----------
    df_monthly : pandas.DataFrame
        Monthly dataframe with z-score columns:
        - 'z_elev_m', 'z_cu_m'.
    df_annual : pandas.DataFrame
        Annual dataframe with z-score columns:
        - 'z_elev_a', 'z_cu_a'.
    stats_dict : dict
        Output from `compute_response_elasticity`, containing regression
        statistics for "monthly" and "annual".
    title_suffix : str, optional
        Text to append to the overall figure title (e.g., "2000–2024").
    axes : array-like of matplotlib.axes.Axes, optional
        Existing axes (length 2). If None, a new figure and axes
        are created.

    Returns
    -------
    axes : ndarray of matplotlib.axes.Axes
        The two axes containing the monthly and annual plots.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
        plt.subplots_adjust(wspace=0.32)

    # Unpack stats
    m = stats_dict["monthly"]
    a = stats_dict["annual"]

    # ---------- (a) Monthly ----------
    ax0 = axes[0]
    x_m = df_monthly["z_elev_m"].values
    y_m = df_monthly["z_cu_m"].values
    xx = np.linspace(x_m.min() - 1, x_m.max() + 1, 200)

    ax0.scatter(x_m, y_m, alpha=0.7, s=22)
    ax0.plot(xx, xx, "--", color="gray", lw=1)
    ax0.plot(xx, m["intercept"] + m["slope"] * xx, color="orange", lw=2)
    ax0.set_title(f"(a) Monthly Elasticity  b={m['slope']:.2f}, R²={m['r2']:.2f}")
    ax0.set_xlabel("Lake Mead Elevation (monthly z)")
    ax0.set_ylabel("Consumptive Use (monthly z)")
    ax0.grid(True)

    # ---------- (b) Annual ----------
    ax1 = axes[1]
    x_a = df_annual["z_elev_a"].values
    y_a = df_annual["z_cu_a"].values
    xxa = np.linspace(x_a.min() - 0.5, x_a.max() + 0.5, 150)

    ax1.scatter(x_a, y_a, s=44, color="dodgerblue", edgecolor="k")
    ax1.plot(xxa, xxa, "--", color="gray", lw=1)
    ax1.plot(xxa, a["intercept"] + a["slope"] * xxa, color="orange", lw=2)
    ax1.set_title(f"(b) Annual Elasticity  b={a['slope']:.2f}, R²={a['r2']:.2f}")
    ax1.set_xlabel("Lake Mead Elevation (annual z)")
    ax1.set_ylabel("Consumptive Use (annual z)")
    ax1.grid(True)

    # Overall title
    axes[0].figure.suptitle(
        f"Response Elasticity: CA Consumptive Use vs. Lake Mead Elevation ({title_suffix})",
        y=1.05,
        fontsize=13,
    )

    return axes

#%% 5. Policy milestones & before–after analysis

def compute_policy_impacts_ca_wateruse(
    water: pd.DataFrame,
    mead: pd.DataFrame,
    policies: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build annual California water-use series and quantify how mean annual
    use changes before and after major policy milestones.

    The function:
      1. Aggregates monthly CA water-use data to annual totals
         (only full years with 12 months are kept).
      2. Converts acre-feet to million acre-feet (MAF).
      3. Extracts Lake Mead December elevation by year and merges it into
         the annual table.
      4. For each policy year, compares mean annual values before vs.
         after the policy and reports the percent change in the mean.

    Parameters
    ----------
    water : pandas.DataFrame
        California water-use data with at least:
        - 'Date'              : datetime-like
        - 'ConsumptiveUse_af' : float
        - 'Diversion_af'      : float
    mead : pandas.DataFrame
        Lake Mead elevation data with at least:
        - 'Date'         : datetime-like
        - 'Elevation_ft' : float
    policies : dict[str, int], optional
        Mapping from policy label to policy year, e.g.:
        {
          "QSA (2003)": 2003,
          "Interim Guidelines (2007)": 2007,
          ...
        }
        If None, a default set of Colorado River policies is used.

    Returns
    -------
    annual : pandas.DataFrame
        Annual series with columns:
        - 'Year'
        - 'CU_MAF'       : mean annual consumptive use (MAF)
        - 'Div_MAF'      : mean annual diversions (MAF)
        - 'Mead_Elev_ft' : Lake Mead December elevation (ft)
    delta_pct_df : pandas.DataFrame
        Before–after summary table with columns:
        - 'Policy'       : policy label
        - 'Year'         : policy year
        - 'Metric'       : 'Consumptive Use' or 'Diversions'
        - 'ΔMean (%)'    : percent change in mean after vs. before
    """
    # Default policy set if user does not supply one
    if policies is None:
        policies = {
            "QSA (2003)":                       2003,
            "Interim Guidelines (2007)":        2007,
            "System Conservation Pilot (2014)": 2014,
            "Minute 323 (2017)":                2017,
            "DCP (2019)":                       2019,
            "Federal Shortage (2021)":          2021,
            "Post-2022 Incentive Programs":     2022,
        }

    water = water.copy()
    mead = mead.copy()

    # --- Annual water use (MAF) ---
    water["Date"] = pd.to_datetime(water["Date"])
    water["ConsumptiveUse_af"] = pd.to_numeric(water["ConsumptiveUse_af"], errors="coerce")
    water["Diversion_af"] = pd.to_numeric(water["Diversion_af"], errors="coerce")
    water["Year"] = water["Date"].dt.year

    annual = (
        water.groupby("Year")
             .agg(
                 n_months=("ConsumptiveUse_af", "count"),
                 CU_total_af=("ConsumptiveUse_af", "sum"),
                 Div_total_af=("Diversion_af", "sum"),
             )
             .reset_index()
    )

    # Keep only complete years (12 months)
    annual = annual[annual["n_months"] == 12].copy()

    # Convert to MAF
    annual["CU_MAF"] = annual["CU_total_af"] / 1e6
    annual["Div_MAF"] = annual["Div_total_af"] / 1e6

    # --- Lake Mead December elevation by year ---
    mead["Date"] = pd.to_datetime(mead["Date"])
    mead["Year"] = mead["Date"].dt.year
    mead["Month"] = mead["Date"].dt.month

    mead_dec = mead[mead["Month"] == 12].copy()

    annual_mead = (
        mead_dec[["Year", "Elevation_ft"]]
        .rename(columns={"Elevation_ft": "Mead_Elev_ft"})
        .sort_values("Year")
        .reset_index(drop=True)
    )

    annual = (
        annual.merge(annual_mead, on="Year", how="inner")
              .sort_values("Year")
              .reset_index(drop=True)
    )

    # --- Before–after ΔMean(%) ---
    def before_after_delta_pct(df_in, value_col, metric_label):
        rows = []
        for name, year in policies.items():
            pre = df_in.loc[df_in["Year"] < year, value_col].dropna()
            post = df_in.loc[df_in["Year"] >= year, value_col].dropna()

            if len(pre) == 0 or len(post) == 0:
                continue

            mean_pre, mean_post = pre.mean(), post.mean()
            d_pct = (mean_post / mean_pre - 1) * 100

            rows.append({
                "Policy": name,
                "Year": year,
                "Metric": metric_label,
                "ΔMean (%)": round(d_pct, 1),
            })

        return pd.DataFrame(rows)

    delta_cu  = before_after_delta_pct(annual, "CU_MAF",  "Consumptive Use")
    delta_div = before_after_delta_pct(annual, "Div_MAF", "Diversions")

    delta_pct_df = (
        pd.concat([delta_cu, delta_div], ignore_index=True)
          .sort_values(["Year", "Metric"])
          .reset_index(drop=True)
    )

    return annual, delta_pct_df

# Plot functions


def plot_policy_timeseries(
    annual: pd.DataFrame,
    policy_styles: dict[str, dict] | None = None,
    pilot_span: tuple[int, int] | None = (2014, 2017),
    incent_span_start: int | None = 2022,
    ax=None,
):
    """
    Plot annual CU and diversions (MAF) together with Lake Mead December
    elevation and user-defined policy milestones.

    Parameters
    ----------
    annual : pandas.DataFrame
        Annual series with columns:
        - 'Year', 'CU_MAF', 'Div_MAF', 'Mead_Elev_ft'.
    policy_styles : dict[str, dict], optional
        Mapping from policy label to plotting style, e.g.:
        {
          "QSA (2003)":  {"year": 2003, "color": "darkred", "ls": "-."},
          "Interim Guidelines (2007)": {"year": 2007, "color": "maroon", "ls": "-"},
          ...
        }
        If None, a default set of policies and styles is used.
    pilot_span : (int, int) or None, optional
        (start_year, end_year) for the System Conservation Pilot shading.
        If None, this shading is skipped.
    incent_span_start : int or None, optional
        First year of the post-2022 incentive program shading.
        The end year is taken as the last year in `annual`.
        If None, this shading is skipped.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the time-series plot.
    """
    if policy_styles is None:
        policy_styles = {
            "QSA (2003)":                      {"year": 2003, "color": "darkred",     "ls": "-."},
            "Interim Guidelines (2007)":       {"year": 2007, "color": "maroon",      "ls": "-"},
            "System Conservation Pilot start": {"year": 2014, "color": "indianred",   "ls": "--"},
            "Minute 323 (2017)":              {"year": 2017, "color": "saddlebrown", "ls": ":"},
            "DCP (2019)":                     {"year": 2019, "color": "firebrick",   "ls": "-."},
            "Federal Shortage (2021)":        {"year": 2021, "color": "purple",      "ls": "--"},
            "Tier 1 Shortage (2022)":         {"year": 2022, "color": "crimson",     "ls": "-"},
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    ax2 = ax.twinx()

    # Primary axis: CU & Diversions
    cu_line, = ax.plot(
        annual["Year"], annual["CU_MAF"],
        marker="o", lw=2, color="steelblue",
        label="Consumptive Use (MAF/year)",
    )
    div_line, = ax.plot(
        annual["Year"], annual["Div_MAF"],
        marker="s", lw=2, color="seagreen",
        label="Diversions (MAF/year)",
    )

    # Secondary axis: Lake Mead elevation
    mead_line, = ax2.plot(
        annual["Year"],
        annual["Mead_Elev_ft"],
        color="black",
        lw=2,
        ls="--",
        marker="D",
        label="Lake Mead Elevation (Dec, ft)",
    )
    ax2.set_ylabel("Lake Mead Elevation (Dec, ft)", fontsize=10)

    ax.set_xlim(annual["Year"].min() - 0.5, annual["Year"].max() + 0.5)
    ax2.set_xlim(ax.get_xlim())

    # Shaded bands
    last_year = int(annual["Year"].max())
    if pilot_span is not None:
        ps, pe = pilot_span
        ax.axvspan(ps, pe + 1, color="lightcoral", alpha=0.15,
                   label="System Conservation Pilot" if ps == pilot_span[0] else None)

    if incent_span_start is not None:
        ax.axvspan(incent_span_start, last_year + 1, color="lightgreen", alpha=0.15,
                   label="Post-2022 Incentive Programs" if incent_span_start == incent_span_start else None)

    # Policy lines + legend handles
    policy_handles = []
    for label, spec in policy_styles.items():
        year = spec["year"]
        color = spec.get("color", "gray")
        ls = spec.get("ls", "-")
        dummy_line = Line2D([0], [0], color=color, ls=ls, lw=2, label=label)
        policy_handles.append(dummy_line)
        ax.axvline(year, color=color, ls=ls, lw=1.5, alpha=0.7)

    # Labels & styling
    ax.set_title(
        "California Annual Diversions and Consumptive Use\n"
        "with Lake Mead End-of-Year Elevation and Policy Milestones",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Million Acre-Feet per Year", fontsize=10)
    ax.grid(True, alpha=0.3)

    years_unique = np.sort(annual["Year"].unique())
    ax.set_xticks(years_unique)
    ax.set_xticklabels(years_unique, rotation=45, ha="right")
    ax.set_ylim(bottom=0)

    # Combined legend
    legend_items = [
        cu_line,
        div_line,
        mead_line,
        Rectangle((0, 0), 1, 1, facecolor="lightcoral", alpha=0.15,
                  label="System Conservation Pilot (2014–2017)")
        if pilot_span is not None else None,
        Rectangle((0, 0), 1, 1, facecolor="lightgreen", alpha=0.15,
                  label="Post-2022 Incentive Programs")
        if incent_span_start is not None else None,
    ]
    # Filter out Nones
    legend_items = [h for h in legend_items if h is not None] + policy_handles

    ax.figure.legend(
        handles=legend_items,
        frameon=False,
        fontsize=9,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    ax.figure.tight_layout(rect=[0, 0, 1, 0.92])
    return ax

def plot_policy_delta_means(
    delta_pct_df: pd.DataFrame,
    policy_order: list[str] | None = None,
    ax=None
):
    """
    Plot grouped bar chart showing the percent change in mean annual
    water use before vs. after each policy, for both consumptive use
    and diversions.

    Parameters
    ----------
    delta_pct_df : pandas.DataFrame
        Output table from `compute_policy_impacts_ca_wateruse` containing:
        - 'Policy'       : policy label
        - 'Metric'       : 'Consumptive Use' or 'Diversions'
        - 'ΔMean (%)'    : percent change in mean after vs before
    policy_order : list of str, optional
        Order in which policies will appear on the x-axis.
        If None, the function preserves the order in delta_pct_df.
    ax : matplotlib.axes.Axes, optional
        Axis object to plot on. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis containing the plotted grouped bar chart.
    """
    # Determine policy order
    if policy_order is None:
        policy_order = delta_pct_df["Policy"].unique().tolist()

    # Pivot table for grouped bars
    pivot = (
        delta_pct_df
        .pivot(index="Policy", columns="Metric", values="ΔMean (%)")
        .reindex(policy_order)
    )

    # Create plot axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(policy_order))
    width = 0.35

    # Extract values for bars
    cu_vals = pivot["Consumptive Use"].values
    div_vals = pivot["Diversions"].values

    # Plot grouped bars
    ax.bar(x - width/2, cu_vals, width, label="Consumptive Use")
    ax.bar(x + width/2, div_vals, width, label="Diversions")

    # Formatting
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(policy_order, rotation=30, ha="right")
    ax.set_ylabel("ΔMean (%) (After vs. Before)")
    ax.set_title("Percent Change in Mean Annual Water Use\nBefore vs After Each Policy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)

    ax.figure.tight_layout()
    return ax


