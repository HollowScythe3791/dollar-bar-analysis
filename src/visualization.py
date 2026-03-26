import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from .statistics import shapiro_wilk_test
from matplotlib.patches import Rectangle
from datetime import datetime, timezone
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from typing import Dict, Any

def plot_bars(
    df: pd.DataFrame,
    title: str = "Price Chart",
    figsize: tuple = (14, 8),
    convert_time: bool = True,
    show_volume: bool = True,
    bullish_color: str = '#26a69a',
    bearish_color: str = '#ef5350',
    volume_alpha: float = 0.5
) -> plt.Figure:
    """
    Plot candlestick chart with volume from aggregated bar data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Bar data with columns: 'time', 'open', 'high', 'low', 'close', 'volume'
    title : str, default "Price Chart"
        Chart title
    figsize : tuple, default (14, 8)
        Figure size (width, height)
    convert_time : bool, default True
        If True, converts unix timestamp to datetime for x-axis
    show_volume : bool, default True
        If True, shows volume subplot below price chart
    bullish_color : str, default '#26a69a' (green)
        Color for bullish (up) candles
    bearish_color : str, default '#ef5350' (red)
        Color for bearish (down) candles
    volume_alpha : float, default 0.5
        Transparency for volume bars
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object for further customization if needed
    """
    df = df.copy()
    
    # Convert unix timestamp to datetime if requested
    if convert_time:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Determine if each bar is bullish or bearish
    df['bullish'] = df['close'] >= df['open']
    
    # Create figure with subplots
    if show_volume:
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=figsize,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    # Use index for x-axis positioning
    x = np.arange(len(df))
    
    # Calculate candle width
    width = 0.8
    wick_width = 0.1
    
    # Plot candlesticks
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        color = bullish_color if row['bullish'] else bearish_color
        
        # Draw the wick (high-low line)
        ax1.plot(
            [idx, idx],
            [row['low'], row['high']],
            color=color,
            linewidth=wick_width * 10,
            solid_capstyle='round'
        )
        
        # Draw the body (open-close rectangle)
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        
        # Handle doji (open == close)
        if body_height == 0:
            body_height = (row['high'] - row['low']) * 0.01
            if body_height == 0:
                body_height = row['close'] * 0.001
        
        rect = Rectangle(
            (idx - width / 2, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor=color
        )
        ax1.add_patch(rect)
    
    # Style price chart
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-1, len(df))
    
    # Add some padding to y-axis
    price_range = df['high'].max() - df['low'].min()
    ax1.set_ylim(
        df['low'].min() - price_range * 0.05,
        df['high'].max() + price_range * 0.05
    )
    
    # Plot volume if requested
    if show_volume:
        colors = [bullish_color if b else bearish_color for b in df['bullish']]
        ax2.bar(x, df['volume'], width=width, color=colors, alpha=volume_alpha)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(-1, len(df))
    
    # Set x-axis labels
    bottom_ax = ax2 if show_volume else ax1
    
    # Create readable x-axis labels
    if convert_time and len(df) > 0:
        # Select subset of ticks for readability
        num_ticks = min(10, len(df))
        tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
        tick_labels = [df['time'].iloc[i].strftime('%Y-%m-%d\n%H:%M') for i in tick_indices]
        
        bottom_ax.set_xticks(tick_indices)
        bottom_ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    bottom_ax.set_xlabel('Time', fontsize=12)
    
    plt.tight_layout()
    
    return fig

def plot_comparison(
    time_bars: pd.DataFrame,
    dollar_bars: pd.DataFrame,
    start_date: str = None,
    end_date: str = None,
    figsize: tuple = (16, 12),
    bullish_color: str = '#26a69a',
    bearish_color: str = '#ef5350',
    volume_alpha: float = 0.6,
    time_bar_label: str = "Time Bars",
    dollar_bar_label: str = "Dollar Bars",
    show_volume: bool = True
) -> plt.Figure:
    """
    Plot time bars and dollar bars with aligned time axes.
    
    Parameters:
    -----------
    time_bars : pd.DataFrame
        Time bar data with columns: 'time', 'open', 'high', 'low', 'close', 'volume'
    dollar_bars : pd.DataFrame
        Dollar bar data with columns: 'time', 'open', 'high', 'low', 'close', 'volume'
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format (e.g., '2020-01-01')
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format (e.g., '2020-02-01')
    figsize : tuple, default (16, 12)
        Figure size (width, height)
    bullish_color : str
        Color for bullish candles
    bearish_color : str
        Color for bearish candles
    volume_alpha : float
        Transparency for volume bars
    time_bar_label : str
        Title label for time bars chart
    dollar_bar_label : str
        Title label for dollar bars chart
    show_volume : bool, default True
        If True, shows volume subplots below each price chart
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object for further customization
    """
    # -------------------------------------------------------------------------
    # Prepare Data
    # -------------------------------------------------------------------------
    time_bars = time_bars.copy()
    dollar_bars = dollar_bars.copy()
    
    # Convert unix timestamps to datetime
    time_bars['time'] = pd.to_datetime(time_bars['time'], unit='s', utc=True)
    dollar_bars['time'] = pd.to_datetime(dollar_bars['time'], unit='s', utc=True)
    
    # Parse date filters
    if start_date:
        start_dt = pd.Timestamp(start_date, tz='UTC')
    else:
        start_dt = min(time_bars['time'].min(), dollar_bars['time'].min())
    
    if end_date:
        end_dt = pd.Timestamp(end_date, tz='UTC')
    else:
        end_dt = start_dt + pd.DateOffset(months=1)
    
    # Filter both dataframes to the selected time window
    time_bars = time_bars[
        (time_bars['time'] >= start_dt) &
        (time_bars['time'] <= end_dt)
    ].reset_index(drop=True)
    
    dollar_bars = dollar_bars[
        (dollar_bars['time'] >= start_dt) &
        (dollar_bars['time'] <= end_dt)
    ].reset_index(drop=True)
    
    print(f"Plotting from {start_dt.date()} to {end_dt.date()}")
    print(f"Time bars in window:   {len(time_bars):,}")
    print(f"Dollar bars in window: {len(dollar_bars):,}")
    
    # -------------------------------------------------------------------------
    # Create Figure - layout depends on show_volume
    # -------------------------------------------------------------------------
    if show_volume:
        # 4 panels: price + volume for each bar type
        n_rows = 4
        height_ratios = [3, 1, 3, 1]
    else:
        # 2 panels: price only for each bar type
        n_rows = 2
        height_ratios = [1, 1]
    
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios},
    )
    
    # Assign axes based on layout
    if show_volume:
        ax_time_price   = axes[0]
        ax_time_vol     = axes[1]
        ax_dollar_price = axes[2]
        ax_dollar_vol   = axes[3]
        bottom_ax       = ax_dollar_vol
    else:
        ax_time_price   = axes[0]
        ax_dollar_price = axes[1]
        ax_time_vol     = None
        ax_dollar_vol   = None
        bottom_ax       = ax_dollar_price

    # -------------------------------------------------------------------------
    # Candlestick Drawing Helper
    # -------------------------------------------------------------------------
    def draw_candles(ax, df, color_up, color_down):
        """
        Efficient candlestick drawing using matplotlib vlines and bar.
        Uses real datetime x-axis.
        """
        if df.empty:
            return
        
        # Calculate candle width in days (matplotlib date units)
        if len(df) > 1:
            time_diffs = df['time'].diff().dropna()
            median_diff = time_diffs.median().total_seconds() / 86400
            candle_width = median_diff * 0.8
        else:
            candle_width = 1 / 24  # Default to 1 hour
        
        bull = df[df['close'] >= df['open']]
        bear = df[df['close'] < df['open']]
        
        if not bull.empty:
            ax.vlines(
                bull['time'],
                bull['low'], bull['high'],
                color=color_up,
                linewidth=0.8,
                zorder=2
            )
            ax.bar(
                bull['time'],
                (bull['close'] - bull['open']).abs(),
                bottom=bull['open'],
                width=pd.Timedelta(days=candle_width),
                color=color_up,
                edgecolor=color_up,
                linewidth=0.5,
                zorder=3
            )
        
        if not bear.empty:
            ax.vlines(
                bear['time'],
                bear['low'], bear['high'],
                color=color_down,
                linewidth=0.8,
                zorder=2
            )
            ax.bar(
                bear['time'],
                (bear['close'] - bear['open']).abs(),
                bottom=bear['close'],
                width=pd.Timedelta(days=candle_width),
                color=color_down,
                edgecolor=color_down,
                linewidth=0.5,
                zorder=3
            )
    
    # -------------------------------------------------------------------------
    # Volume Drawing Helper
    # -------------------------------------------------------------------------
    def draw_volume(ax, df, color_up, color_down):
        """Draw volume bars with candle colors on datetime x-axis."""
        if df.empty or ax is None:
            return
        
        if len(df) > 1:
            time_diffs = df['time'].diff().dropna()
            median_diff = time_diffs.median().total_seconds() / 86400
            bar_width = median_diff * 0.8
        else:
            bar_width = 1 / 24
        
        colors = [
            color_up if c >= o else color_down
            for c, o in zip(df['close'], df['open'])
        ]
        
        ax.bar(
            df['time'],
            df['volume'],
            width=pd.Timedelta(days=bar_width),
            color=colors,
            alpha=volume_alpha,
            zorder=2
        )
    
    # -------------------------------------------------------------------------
    # Plot Time Bars
    # -------------------------------------------------------------------------
    draw_candles(ax_time_price, time_bars, bullish_color, bearish_color)
    
    if show_volume:
        draw_volume(ax_time_vol, time_bars, bullish_color, bearish_color)
    
    # -------------------------------------------------------------------------
    # Plot Dollar Bars
    # -------------------------------------------------------------------------
    draw_candles(ax_dollar_price, dollar_bars, bullish_color, bearish_color)
    
    if show_volume:
        draw_volume(ax_dollar_vol, dollar_bars, bullish_color, bearish_color)
    
    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------
    ax_time_price.set_xlim(start_dt, end_dt)
    
    # Format x-axis date labels on bottom axis only
    date_format = mdates.DateFormatter('%b %d\n%H:%M')
    bottom_ax.xaxis.set_major_formatter(date_format)
    bottom_ax.tick_params(axis='x', labelsize=8)
    
    # Grid for all axes
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    
    # Hide x tick labels on all but the bottom axis
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    # Y-axis labels
    ax_time_price.set_ylabel('Price (USD)', fontsize=10)
    ax_dollar_price.set_ylabel('Price (USD)', fontsize=10)
    
    if show_volume:
        ax_time_vol.set_ylabel('Volume', fontsize=10)
        ax_dollar_vol.set_ylabel('Volume', fontsize=10)
    
    # Titles
    ax_time_price.set_title(
        f"{time_bar_label}  |  {start_dt.date()} to {end_dt.date()}  "
        f"| {len(time_bars):,} bars",
        fontsize=11,
        fontweight='bold',
        pad=8
    )
    ax_dollar_price.set_title(
        f"{dollar_bar_label}  |  {start_dt.date()} to {end_dt.date()}  "
        f"| {len(dollar_bars):,} bars",
        fontsize=11,
        fontweight='bold',
        pad=8
    )
    
    # Y-axis price padding
    for ax, df in [(ax_time_price, time_bars), (ax_dollar_price, dollar_bars)]:
        if not df.empty:
            price_range = df['high'].max() - df['low'].min()
            ax.set_ylim(
                df['low'].min() - price_range * 0.05,
                df['high'].max() + price_range * 0.05
            )
    
    # Separator line between the two chart sections
    fig.add_artist(plt.Line2D(
        [0.05, 0.95], [0.505, 0.505],
        transform=fig.transFigure,
        color='gray',
        linewidth=0.8,
        linestyle='--',
        alpha=0.5
    ))
    
    plt.suptitle(
        "BTC Bar Comparison",
        fontsize=14,
        fontweight='bold',
        y=1.01
    )
    
    plt.tight_layout()
    
    return fig

def plot_return_distributions(
    series_dict: dict,
    max_sw_sample: int = 5000,
    random_state: int = 42
) -> None:
    """
    Plot a distribution histogram with normal fit overlay for each series.
    Annotates each plot with the Shapiro-Wilk test result.

    Parameters:
    -----------
    series_dict : dict
        Dictionary with series name as key and log return pd.Series as value.
        e.g. {'Time Bars': time_log_returns, 'Dollar Bars': dollar_log_returns}
    max_sw_sample : int, default 5000
        Maximum sample size for Shapiro-Wilk test
    random_state : int, default 42
        Random seed for reproducible sampling

    Returns:
    --------
    None
    """

    for name, log_returns in series_dict.items():
        sw_result = shapiro_wilk_test(log_returns, max_sw_sample, random_state)
        mu_fit, sigma_fit = stats.norm.fit(log_returns)
        x_range = np.linspace(log_returns.min(), log_returns.max(), 300)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            log_returns,
            bins=min(100, len(log_returns) // 20),
            density=True,
            alpha=0.6,
            color='steelblue',
            edgecolor='white',
            linewidth=0.5,
            label='Empirical'
        )
        ax.plot(
            x_range,
            stats.norm.pdf(x_range, mu_fit, sigma_fit),
            'r-',
            linewidth=2,
            label=f'Normal fit  μ={mu_fit:.5f}  σ={sigma_fit:.5f}'
        )

        verdict_str = "REJECT ❌" if sw_result['reject_normality'] else "FAIL TO REJECT ✅"
        annotation = (
            f"Shapiro-Wilk\n"
            f"W = {sw_result['statistic']:.6f}\n"
            f"p = {sw_result['p_value']:.6f}\n"
            f"{verdict_str}"
        )
        ax.text(
            0.98, 0.95, annotation,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
        )

        ax.set_title(f'Return Distribution: {name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Log Return', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()



def plot_qq_plots(
    series_dict: dict,
) -> None:
    """
    Plot a Q-Q plot against the normal distribution for each series.
    Title includes tail behaviour description, skewness, and kurtosis.

    Parameters:
    -----------
    series_dict : dict
        Dictionary with series name as key and log return pd.Series as value.
        e.g. {'Time Bars': time_log_returns, 'Dollar Bars': dollar_log_returns}

    Returns: --------
    None
    """

    for name, log_returns in series_dict.items():
        fig, ax = plt.subplots(figsize=(7, 7))

        (osm, osr), (slope, intercept, r) = stats.probplot(log_returns, dist='norm')

        ax.scatter(
            osm, osr,
            alpha=0.3, s=8,
            color='steelblue',
            label='Data quantiles'
        )
        ax.plot(
            osm,
            slope * np.array(osm) + intercept,
            'r-',
            linewidth=2,
            label=f'Normal reference  (R={r:.4f})'
        )

        skew_val = stats.skew(log_returns)
        kurt_val = stats.kurtosis(log_returns)

        if abs(kurt_val) > 1:
            qq_note = "Fat tails (leptokurtic)" if kurt_val > 0 else "Thin tails (platykurtic)"
        elif abs(skew_val) > 0.5:
            qq_note = f"{'Right' if skew_val > 0 else 'Left'} skew"
        else:
            qq_note = "Approximately normal"

        ax.set_title(
            f'Q-Q Plot: {name}\n{qq_note}  |  skew={skew_val:.4f}  kurt={kurt_val:.4f}',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel('Theoretical Normal Quantiles', fontsize=11)
        ax.set_ylabel('Sample Quantiles', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def plot_acf_comparison(bar_returns: dict, lags: int = 25):
    """
    Plots ACF for each bar type side by side for visual comparison.
    Uses the minimum sample size across all bar types for a fair comparison.
    
    Parameters
    ----------
    bar_returns : dict
        Dictionary of bar type names and their log returns
    lags : int, optional
        Number of lags to plot, by default 25
    """
    
    # Find minimum sample size across all bar types
    min_size = min(len(returns) for returns in bar_returns.values())
    
    n_bars = len(bar_returns)
    fig, axes = plt.subplots(n_bars, 1, figsize=(12, 4 * n_bars))
    fig.suptitle(f'Autocorrelation Function (ACF) Comparison (n={min_size})', fontsize=16, y=1.02)
    
    for ax, (bar_type, returns) in zip(axes, bar_returns.items()):
        # Trim to minimum sample size
        trimmed_returns = returns.iloc[:min_size]
        
        plot_acf(
            trimmed_returns,
            lags=lags,
            ax=ax,
            zero=False,
            title=f'{bar_type} Bars',
            color='steelblue',
            vlines_kwargs={'colors': 'steelblue'}
        )
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.set_ylim(-0.10, 0.10)
        # Add significance bands label
        ax.annotate('Shaded area = 95% confidence interval', 
                   xy=(0.7, 0.85), 
                   xycoords='axes fraction',
                   fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_rolling_cv_comparison(
    bar_returns: Dict[str, pd.Series],
    rolling_window: int = 30,
    figsize: tuple = (16, 10),
    mean_cv_threshold: float = 1.0,
    vol_cv_threshold: float = 0.5
) -> None:
    """
    Plot rolling Coefficient of Variation (CV) of mean and volatility
    for multiple bar types side by side for stability comparison.

    Parameters:
    -----------
    bar_returns : Dict[str, pd.Series]
        Dictionary of {bar_name: log_returns_series}
        e.g. {"Time Bars": time_returns, "Dollar Bars": dollar_returns}
    rolling_window : int, default 30
        Window size for rolling statistics
    figsize : tuple, default (16, 10)
        Figure size (width, height)
    mean_cv_threshold : float, default 1.0
        Horizontal threshold line for mean CV stability
    vol_cv_threshold : float, default 0.5
        Horizontal threshold line for volatility CV stability
    """

    bar_names = list(bar_returns.keys())
    n_bars = len(bar_names)

    # Color palette for each bar type
    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Rolling CV Stability Comparison (window={rolling_window})",
        fontsize=15,
        fontweight="bold",
        y=1.01
    )

    # GridSpec: 2 rows (Mean CV, Vol CV), n_bars columns
    gs = gridspec.GridSpec(2, n_bars, figure=fig, hspace=0.45, wspace=0.35)

    for col_idx, (bar_name, returns) in enumerate(bar_returns.items()):
        color = colors[col_idx % len(colors)]

        # -------------------------------------------------------
        # Compute rolling mean and rolling std
        # -------------------------------------------------------
        rolling_mean = returns.rolling(window=rolling_window).mean()
        rolling_std  = returns.rolling(window=rolling_window).std()

        # -------------------------------------------------------
        # Compute rolling CV of mean:
        #   At each point t, CV_mean(t) = std(rolling_mean up to t)
        #                                 / |mean(rolling_mean up to t)|
        # We use an expanding window so the CV itself evolves over time.
        # -------------------------------------------------------
        rolling_mean_cv = (
            rolling_mean.expanding().std()
            / rolling_mean.expanding().mean().abs().replace(0, np.nan)
        )

        rolling_vol_cv = (
            rolling_std.expanding().std()
            / rolling_std.expanding().mean().replace(0, np.nan)
        )

        # -------------------------------------------------------
        # Row 0 — Rolling CV of Mean
        # -------------------------------------------------------
        ax_mean = fig.add_subplot(gs[0, col_idx])

        ax_mean.plot(
            rolling_mean_cv.index,
            rolling_mean_cv.values,
            color=color,
            linewidth=1.2,
            label="CV of Mean"
        )
        ax_mean.axhline(
            mean_cv_threshold,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Threshold ({mean_cv_threshold})"
        )

        # Shade unstable region
        ax_mean.fill_between(
            rolling_mean_cv.index,
            mean_cv_threshold,
            rolling_mean_cv.values,
            where=(rolling_mean_cv.values > mean_cv_threshold),
            alpha=0.15,
            color="red",
            label="Unstable Region"
        )

        ax_mean.set_title(f"{bar_name}", fontsize=11, fontweight="bold", color=color)
        ax_mean.set_ylabel("CV of Rolling Mean" if col_idx == 0 else "")
        ax_mean.set_xlabel("")
        ax_mean.tick_params(axis="x", rotation=30, labelsize=7)
        ax_mean.tick_params(axis="y", labelsize=8)
        ax_mean.legend(fontsize=7, loc="upper right")
        ax_mean.grid(True, alpha=0.3)

        # Annotate final CV value
        final_mean_cv = rolling_mean_cv.dropna().iloc[-1] if not rolling_mean_cv.dropna().empty else np.nan
        if not np.isnan(final_mean_cv):
            stable_label = "Stable ✓" if final_mean_cv < mean_cv_threshold else "Unstable ✗"
            ax_mean.annotate(
                f"Final: {final_mean_cv:.3f}\n{stable_label}",
                xy=(0.03, 0.93),
                xycoords="axes fraction",
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
            )

        # -------------------------------------------------------
        # Row 1 — Rolling CV of Volatility
        # -------------------------------------------------------
        ax_vol = fig.add_subplot(gs[1, col_idx])

        ax_vol.plot(
            rolling_vol_cv.index,
            rolling_vol_cv.values,
            color=color,
            linewidth=1.2,
            label="CV of Volatility"
        )
        ax_vol.axhline(
            vol_cv_threshold,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Threshold ({vol_cv_threshold})"
        )

        ax_vol.fill_between(
            rolling_vol_cv.index,
            vol_cv_threshold,
            rolling_vol_cv.values,
            where=(rolling_vol_cv.values > vol_cv_threshold),
            alpha=0.15,
            color="red",
            label="Unstable Region"
        )

        ax_vol.set_ylabel("CV of Rolling Volatility" if col_idx == 0 else "")
        ax_vol.set_xlabel("Index", fontsize=8)
        ax_vol.tick_params(axis="x", rotation=30, labelsize=7)
        ax_vol.tick_params(axis="y", labelsize=8)
        ax_vol.legend(fontsize=7, loc="upper right")
        ax_vol.grid(True, alpha=0.3)

        # Annotate final CV value
        final_vol_cv = rolling_vol_cv.dropna().iloc[-1] if not rolling_vol_cv.dropna().empty else np.nan
        if not np.isnan(final_vol_cv):
            stable_label = "Stable ✓" if final_vol_cv < vol_cv_threshold else "Unstable ✗"
            ax_vol.annotate(
                f"Final: {final_vol_cv:.3f}\n{stable_label}",
                xy=(0.03, 0.93),
                xycoords="axes fraction",
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
            )

    # -------------------------------------------------------
    # Row labels on the left side
    # -------------------------------------------------------
    fig.text(0.01, 0.72, "Mean CV", va="center", rotation="vertical",
             fontsize=11, fontweight="bold", color="dimgray")
    fig.text(0.01, 0.28, "Volatility CV", va="center", rotation="vertical",
             fontsize=11, fontweight="bold", color="dimgray")

    plt.tight_layout()
    plt.show()

