"""
Statistical test wrappers: normality, independence, and stability tests.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Any, Dict

warnings.filterwarnings('ignore')


# =============================================================================
# LOG RETURNS
# =============================================================================

def calculate_log_returns(bars: pd.DataFrame, price_col: str = 'close') -> pd.Series:
    """
    Calculate log returns from bar data.

    Parameters
    ----------
    bars : pd.DataFrame
        Bar data containing at minimum a price column.
    price_col : str, default 'close'
        Column name to use for price.

    Returns
    -------
    pd.Series
        Log returns with leading NaN dropped.
    """
    prices = bars[price_col]
    return np.log(prices / prices.shift(1)).dropna()


# =============================================================================
# NORMALITY
# =============================================================================

def shapiro_wilk_test(
    log_returns: pd.Series,
    max_sample: int = 5000,
    random_state: int = 42
) -> dict:
    """
    Shapiro-Wilk test for normality.

    H0: Data is normally distributed.

    Parameters
    ----------
    log_returns : pd.Series
        Series of log returns.
    max_sample : int, default 5000
        Maximum sample size (test loses power above this).
    random_state : int, default 42
        Random seed for reproducible sampling.

    Returns
    -------
    dict
        Test statistic, p-value, sample info, and interpretation.
    """
    n = len(log_returns)
    sampled = n > max_sample

    if sampled:
        rng = np.random.default_rng(random_state)
        sample = rng.choice(log_returns.values, size=max_sample, replace=False)
        n_used = max_sample
    else:
        sample = log_returns.values
        n_used = n

    stat, p_value = stats.shapiro(sample)

    return {
        'test': 'Shapiro-Wilk',
        'statistic': stat,
        'p_value': p_value,
        'n_observations': n,
        'n_used': n_used,
        'sampled': sampled,
        'reject_normality': p_value < 0.05,
        'interpretation': (
            f"{'Sampled ' + str(n_used) + ' from ' + str(n) + ' observations. ' if sampled else f'n={n}. '}"
            f"{'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} normality at 5% level "
            f"(p={p_value:.6f})."
        )
    }


def print_shapiro_wilk_results(
    series_dict: dict,
    max_sw_sample: int = 5000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run Shapiro-Wilk normality test for each series and print results.

    Parameters
    ----------
    series_dict : dict
        {series_name: log_returns_series}, e.g.
        {'Time Bars': time_log_returns, 'Dollar Bars': dollar_log_returns}
    max_sw_sample : int, default 5000
        Maximum sample size for Shapiro-Wilk.
    random_state : int, default 42
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Summary table of Shapiro-Wilk results.
    """
    print("  SHAPIRO-WILK NORMALITY TEST RESULTS")

    rows = []
    for name, log_returns in series_dict.items():
        result = shapiro_wilk_test(log_returns, max_sw_sample, random_state)
        verdict = "❌ REJECT normality" if result['reject_normality'] else "✅ FAIL TO REJECT normality"

        print(f"\n  Series : {name}")
        print(
            f"  N      : {result['n_observations']:,}"
            + (f"  (sampled {result['n_used']:,})" if result['sampled'] else "")
        )
        print(f"  W stat : {result['statistic']:.6f}")
        print(f"  p-value: {result['p_value']:.6f}")
        print(f"  Verdict: {verdict}")

        rows.append({
            'Series': name,
            'N': result['n_observations'],
            'N_Used': result['n_used'],
            'Sampled': result['sampled'],
            'SW_Statistic': result['statistic'],
            'SW_PValue': result['p_value'],
            'SW_Reject': result['reject_normality'],
        })

    return pd.DataFrame(rows)


# =============================================================================
# INDEPENDENCE
# =============================================================================

def test_independence(
    log_returns: pd.Series,
    lags: int = 25,
    significance_level: float = 0.05
) -> dict:
    """
    Ljung-Box test for serial independence.

    H0: No autocorrelation up to lag k.

    Parameters
    ----------
    log_returns : pd.Series
        Log returns to test.
    lags : int, default 25
        Number of lags to test.
    significance_level : float, default 0.05
        Significance level for rejection.

    Returns
    -------
    dict
        results_df, independent_lags, dependent_lags, overall_independent.
    """
    results = acorr_ljungbox(log_returns, lags=lags, return_df=True)
    results.columns = ['test_statistic', 'p_value']
    results['independent'] = results['p_value'] > significance_level

    independent_lags = results['independent'].sum()
    dependent_lags = lags - independent_lags

    return {
        'results_df': results,
        'independent_lags': independent_lags,
        'dependent_lags': dependent_lags,
        'overall_independent': independent_lags > (lags / 2)
    }


def compare_independence(
    bar_returns: dict,
    lags: int = 25,
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Compare Ljung-Box independence across multiple bar types.

    Parameters
    ----------
    bar_returns : dict
        {bar_type_name: log_returns_series}
    lags : int, default 25
        Number of lags to test.
    significance_level : float, default 0.05
        Significance level for rejection.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by independence rank (most independent first).
    """
    comparison = {}

    for bar_type, returns in bar_returns.items():
        results = acorr_ljungbox(returns, lags=lags, return_df=True)
        results.columns = ['test_statistic', 'p_value']

        dependent_mask = results['p_value'] <= significance_level
        comparison[bar_type] = {
            'mean_p_value': results['p_value'].mean(),
            'median_p_value': results['p_value'].median(),
            'mean_test_statistic': results['test_statistic'].mean(),
            'dependent_lags': dependent_mask.sum(),
            'independent_lags': (~dependent_mask).sum(),
            'first_dependent_lag': (
                results[dependent_mask].index[0]
                if dependent_mask.any() else None
            )
        }

    df = pd.DataFrame(comparison).T
    df['independence_rank'] = df['mean_test_statistic'].rank()
    return df.sort_values('independence_rank')


# ===========================================================
# STABILITY
# ===========================================================

def test_log_return_stability(
    log_returns: pd.Series,
    significance_level: float = 0.05,
    rolling_window: int = 30
) -> Dict[str, Any]:
    """
    Test the stability of log returns using multiple statistical tests.
    
    Parameters:
    -----------
    log_returns : pd.Series
        Log returns series to test
    significance_level : float, default 0.05
        Significance level for hypothesis tests
    rolling_window : int, default 30
        Window size for rolling statistics
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing test results and interpretations
    """
    results = {}

    # -------------------------------------------------------------------------
    # 1. Basic Descriptive Statistics
    # -------------------------------------------------------------------------
    results["descriptive_stats"] = {
        "mean": log_returns.mean(),
        "std": log_returns.std(),
        "skewness": log_returns.skew(),
        "kurtosis": log_returns.kurtosis(),
        "min": log_returns.min(),
        "max": log_returns.max(),
        "observations": len(log_returns),
    }

    # -------------------------------------------------------------------------
    # 2. Stationarity Tests
    # -------------------------------------------------------------------------

    # Augmented Dickey-Fuller Test
    # H0: Unit root exists (non-stationary)
    # H1: No unit root (stationary)
    adf_stat, adf_p, adf_lags, adf_nobs, adf_critical, _ = adfuller(log_returns)
    results["adf_test"] = {
        "statistic": adf_stat,
        "p_value": adf_p,
        "lags_used": adf_lags,
        "critical_values": adf_critical,
        "is_stationary": adf_p < significance_level,
        "interpretation": (
            "Stationary (reject unit root)"
            if adf_p < significance_level
            else "Non-stationary (fail to reject unit root)"
        ),
    }

    # KPSS Test
    # H0: Series is stationary
    # H1: Series is non-stationary
    kpss_stat, kpss_p, kpss_lags, kpss_critical = kpss(log_returns, regression="c", nlags="auto")
    results["kpss_test"] = {
        "statistic": kpss_stat,
        "p_value": kpss_p,
        "lags_used": kpss_lags,
        "critical_values": kpss_critical,
        "is_stationary": kpss_p > significance_level,
        "interpretation": (
            "Stationary (fail to reject stationarity)"
            if kpss_p > significance_level
            else "Non-stationary (reject stationarity)"
        ),
    }

    # -------------------------------------------------------------------------
    # 3. Normality Tests
    # -------------------------------------------------------------------------

    # Jarque-Bera Test
    # H0: Returns are normally distributed
    # H1: Returns are not normally distributed
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(log_returns)
    results["jarque_bera_test"] = {
        "statistic": jb_stat,
        "p_value": jb_p,
        "skewness": jb_skew,
        "kurtosis": jb_kurt,
        "is_normal": jb_p > significance_level,
        "interpretation": (
            "Normal distribution (fail to reject normality)"
            if jb_p > significance_level
            else "Non-normal distribution (reject normality)"
        ),
    }

    # Shapiro-Wilk Test (best for smaller samples)
    if len(log_returns) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(log_returns)
        results["shapiro_wilk_test"] = {
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "is_normal": shapiro_p > significance_level,
            "interpretation": (
                "Normal distribution (fail to reject normality)"
                if shapiro_p > significance_level
                else "Non-normal distribution (reject normality)"
            ),
        }
    else:
        results["shapiro_wilk_test"] = {
            "interpretation": "Skipped: Sample size > 5000 (use Jarque-Bera instead)"
        }

    # -------------------------------------------------------------------------
    # 4. Volatility Stability (ARCH Effects)
    # -------------------------------------------------------------------------

    # ARCH LM Test
    # H0: No ARCH effects (constant variance)
    # H1: ARCH effects present (volatility clustering)
    arch_lm_stat, arch_lm_p, _, _ = het_arch(log_returns)
    results["arch_test"] = {
        "statistic": arch_lm_stat,
        "p_value": arch_lm_p,
        "has_arch_effects": arch_lm_p < significance_level,
        "interpretation": (
            "Volatility clustering present (ARCH effects detected)"
            if arch_lm_p < significance_level
            else "No significant volatility clustering"
        ),
    }

    # -------------------------------------------------------------------------
    # 5. Rolling Statistics Stability
    # -------------------------------------------------------------------------
    rolling_mean = log_returns.rolling(window=rolling_window).mean()
    rolling_std = log_returns.rolling(window=rolling_window).std()

    # Coefficient of Variation of rolling stats (measures how much they fluctuate)
    rolling_mean_cv = rolling_mean.std() / abs(rolling_mean.mean()) if rolling_mean.mean() != 0 else np.inf
    rolling_std_cv = rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else np.inf

    results["rolling_stability"] = {
        "window": rolling_window,
        "rolling_mean_cv": rolling_mean_cv,
        "rolling_std_cv": rolling_std_cv,
        "mean_is_stable": rolling_mean_cv < 1.0,  # Threshold: CV < 1
        "volatility_is_stable": rolling_std_cv < 0.5,  # Threshold: CV < 0.5
        "interpretation": (
            "Mean and volatility are stable"
            if rolling_mean_cv < 1.0 and rolling_std_cv < 0.5
            else "Mean or volatility shows instability"
        ),
    }

    # -------------------------------------------------------------------------
    # 6. Structural Break Detection (Chow-style using variance ratio)
    # -------------------------------------------------------------------------
    mid_point = len(log_returns) // 2
    first_half = log_returns.iloc[:mid_point]
    second_half = log_returns.iloc[mid_point:]

    # Levene's test for equality of variances between two halves
    levene_stat, levene_p = stats.levene(first_half, second_half)

    # T-test for equality of means between two halves
    ttest_stat, ttest_p = stats.ttest_ind(first_half, second_half)

    results["structural_break"] = {
        "split_point": log_returns.index[mid_point] if hasattr(log_returns.index, '__getitem__') else mid_point,
        "first_half_mean": first_half.mean(),
        "second_half_mean": second_half.mean(),
        "first_half_std": first_half.std(),
        "second_half_std": second_half.std(),
        "levene_test": {
            "statistic": levene_stat,
            "p_value": levene_p,
            "equal_variance": levene_p > significance_level,
        },
        "ttest": {
            "statistic": ttest_stat,
            "p_value": ttest_p,
            "equal_mean": ttest_p > significance_level,
        },
        "has_structural_break": (levene_p < significance_level) or (ttest_p < significance_level),
        "interpretation": (
            "Potential structural break detected (mean or variance shifted)"
            if (levene_p < significance_level) or (ttest_p < significance_level)
            else "No significant structural break detected"
        ),
    }

    # -------------------------------------------------------------------------
    # 7. Overall Stability Summary
    # -------------------------------------------------------------------------
    is_stationary = results["adf_test"]["is_stationary"] and results["kpss_test"]["is_stationary"]
    no_arch_effects = not results["arch_test"]["has_arch_effects"]
    no_structural_break = not results["structural_break"]["has_structural_break"]
    mean_stable = results["rolling_stability"]["mean_is_stable"]
    vol_stable = results["rolling_stability"]["volatility_is_stable"]

    stability_score = sum([is_stationary, no_arch_effects, no_structural_break, mean_stable, vol_stable])

    results["overall_stability"] = {
        "is_stationary": is_stationary,
        "no_arch_effects": no_arch_effects,
        "no_structural_break": no_structural_break,
        "mean_is_stable": mean_stable,
        "volatility_is_stable": vol_stable,
        "stability_score": f"{stability_score}/5",
        "verdict": (
            "Stable" if stability_score >= 4
            else "Moderately Stable" if stability_score >= 2
            else "Unstable"
        ),
    }

    return results

def compare_stability(bar_returns: dict, 
                      significance_level: float = 0.05,
                      rolling_window: int = 30) -> pd.DataFrame:
    rows = []
    
    for bar_type, returns in bar_returns.items():
        result = test_log_return_stability(
            returns,
            significance_level=significance_level,
            rolling_window=rolling_window
        )
        
        rows.append({
            "Bar Type"      : bar_type,
            "Stationary"    : "✓" if result["adf_test"]["is_stationary"] 
                              else "✗",
            "ADF p-value"   : round(result["adf_test"]["p_value"], 4),
            "ARCH Effects"  : "✗" if result["arch_test"]["has_arch_effects"] 
                              else "✓",
            "ARCH p-value"  : round(result["arch_test"]["p_value"], 4),
            "Mean CV"       : round(result["rolling_stability"]["rolling_mean_cv"], 3),
            "Vol CV"        : round(result["rolling_stability"]["rolling_std_cv"], 3),
            "Var Stable"    : "✓" if result["structural_break"]["levene_test"]["equal_variance"] 
                              else "✗",
            "Levene p-value": round(result["structural_break"]["levene_test"]["p_value"], 4),
            "Score"         : result["overall_stability"]["stability_score"],
            "Verdict"       : result["overall_stability"]["verdict"],
        })
    
    return pd.DataFrame(rows).set_index("Bar Type")

def print_stability_report(results: Dict[str, Any]) -> None:
    """
    Pretty print the stability test results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results dictionary from test_log_return_stability()
    """
    print("=" * 60)
    print("         LOG RETURNS STABILITY REPORT")
    print("=" * 60)

    print("\n📊 DESCRIPTIVE STATISTICS")
    print("-" * 40)
    stats_data = results["descriptive_stats"]
    for key, value in stats_data.items():
        print(f"  {key:<20}: {value:.6f}" if isinstance(value, float) else f"  {key:<20}: {value}")

    print("\n📈 STATIONARITY TESTS")
    print("-" * 40)
    print(f"  ADF Test   : {results['adf_test']['interpretation']}")
    print(f"               p-value = {results['adf_test']['p_value']:.4f}")
    print(f"  KPSS Test  : {results['kpss_test']['interpretation']}")
    print(f"               p-value = {results['kpss_test']['p_value']:.4f}")

    print("\n🔔 NORMALITY TESTS")
    print("-" * 40)
    print(f"  Jarque-Bera: {results['jarque_bera_test']['interpretation']}")
    print(f"               p-value = {results['jarque_bera_test']['p_value']:.4f}")
    print(f"  Shapiro-Wilk: {results['shapiro_wilk_test']['interpretation']}")
    if "p_value" in results["shapiro_wilk_test"]:
        print(f"               p-value = {results['shapiro_wilk_test']['p_value']:.4f}")

    print("\n⚡ VOLATILITY STABILITY (ARCH TEST)")
    print("-" * 40)
    print(f"  {results['arch_test']['interpretation']}")
    print(f"  p-value = {results['arch_test']['p_value']:.4f}")

    print("\n📉 ROLLING STATISTICS STABILITY")
    print("-" * 40)
    print(f"  Window     : {results['rolling_stability']['window']}")
    print(f"  {results['rolling_stability']['interpretation']}")
    print(f"  Mean CV    : {results['rolling_stability']['rolling_mean_cv']:.4f}")
    print(f"  Std CV     : {results['rolling_stability']['rolling_std_cv']:.4f}")

    print("\n🔍 STRUCTURAL BREAK TEST")
    print("-" * 40)
    print(f"  {results['structural_break']['interpretation']}")
    print(f"  Levene p-value : {results['structural_break']['levene_test']['p_value']:.4f}")
    print(f"  T-test p-value : {results['structural_break']['ttest']['p_value']:.4f}")

    print("\n✅ OVERALL STABILITY SUMMARY")
    print("-" * 40)
    summary = results["overall_stability"]
    print(f"  Stationary       : {'✓' if summary['is_stationary'] else '✗'}")
    print(f"  No ARCH Effects  : {'✓' if summary['no_arch_effects'] else '✗'}")
    print(f"  No Struct. Break : {'✓' if summary['no_structural_break'] else '✗'}")
    print(f"  Mean Stable      : {'✓' if summary['mean_is_stable'] else '✗'}")
    print(f"  Vol. Stable      : {'✓' if summary['volatility_is_stable'] else '✗'}")
    print(f"\n  Stability Score  : {summary['stability_score']}")
    print(f"  Verdict          : {summary['verdict']}")
    print("=" * 60)


