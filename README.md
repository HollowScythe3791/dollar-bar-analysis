# Financial Market Data Sampling Analysis

Replicating López de Prado's dollar bar claims with independent analysis. Statistical tests for normality, independence, and stability across four bar types using 59M BTC ticks.

## Overview

This repository accompanies the article **"Check Your Bars — Testing Normality, Independence, and Stability on Time, Tick, Volume, and Dollar Bars"**.

Many quantitative finance models assume that returns data is normally distributed, independent, and stable over time. If those assumptions are violated, models can produce incorrect signals, underestimate risk, and lead to overfit strategies. This project tests how the method used to aggregate raw tick data into bars affects these statistical properties.

We aggregate 59 million BTC-USD ticks from Kraken (Jan 2020 – Jan 2025) into four bar types and compare their log returns across three dimensions:

| Property | Tests Used |
|---|---|
| **Normality** | Shapiro-Wilk, histograms, QQ-plots |
| **Independence** | Ljung-Box, ACF plots |
| **Stability** | ADF, ARCH LM, Rolling CV, Levene |

## Key Findings

**Dollar bars consistently outperform all other bar types across every property tested.**

| Property | Winner | Highlight |
|---|---|---|
| Normality | Dollar Bars | Shapiro-Wilk of **0.992** vs 0.923 for time bars |
| Independence | Dollar Bars | Mean Ljung-Box p-value of **0.194** — the only bar type that fails to reject independence |
| Stability | Dollar Bars | Mean CV of **5.415**, 60% lower than time bars (21.305) |

These results support the claims made by Marcos López de Prado in *Advances in Financial Machine Learning*. Dollar bars produce returns that are more normal, more independent, and more stable than time, tick, or volume bars.

## Bar Types & Thresholds

Thresholds were chosen to produce a comparable number of bars across all four methods over the full five-year sample.

| Bar Type | Threshold |
|---|---|
| Time Bars | 1 hour |
| Tick Bars | 10,000 ticks |
| Volume Bars | 1,000 BTC |
| Dollar Bars | $100,000,000 |

## Data

The dataset comes from [Kraken Time and Sales data](https://support.kraken.com/hc/en-us/articles/360047124832-Downloadable-historical-OHLCVT-Open-High-Low-Close-Volume-Trades-data), which is freely available for download. We use **BTC-USD** trades from **January 1, 2020 to January 1, 2025** (~59 million ticks).

I also included a link to a google drive with the exact dataset used in this experiment. A link to the drive can be found in the README.md inside data/.

Each tick contains:

| Field | Description |
|---|---|
| `time` | Unix timestamp of the trade |
| `price` | Execution price |
| `volume` | Quantity of BTC traded |

> **Note:** The raw tick data is not included in this repository due to its size. Please download it directly from Kraken using the link above.

## Project Structure

'''
dollar-bar-analysis/
├── README.md
├── requirements.txt
├── LICENSE (MIT)
│
├── notebooks/
│   └── check_your_bars.ipynb    # The article itself
│
├── src/
│   ├── __init__.py
│   ├── bars.py                  # Bar construction functions
│   ├── statistics.py            # Test wrappers (Shapiro, Ljung-Box, ADF, etc.)
│   └── visualization.py         # Plotting functions
│
├── data/
│   └── README.md                # Instructions for downloading Kraken data
│
└── results/
    └── figures/                 # Exported plots
'''

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

'''bash
git clone https://github.com/YOUR_USERNAME/dollar-bar-analysis.git
cd dollar-bar-analysis
pip install -r requirements.txt
'''

### Running the Analysis

1. Download the BTC-USD tick data from [Kraken](https://support.kraken.com/hc/en-us/articles/360047124832-Downloadable-historical-OHLCVT-Open-High-Low-Close-Volume-Trades-data) and place it in the project directory.
2. Open the notebook:

'''bash
jupyter notebook notebook.ipynb
'''

3. Run all cells to reproduce the full analysis.

## Results Summary

### Normality (Shapiro-Wilk)

| Bar Type | W Statistic | Verdict |
|---|---|---|
| Time | 0.9235 | ❌ Reject |
| Tick | 0.9771 | ❌ Reject |
| Volume | 0.9821 | ❌ Reject |
| **Dollar** | **0.9918** | ❌ Reject (closest to normal) |

### Independence (Ljung-Box)

| Bar Type | Mean p-value | Median p-value | Dependent Lags | Rank |
|---|---|---|---|---|
| **Dollar** | **0.194** | **0.176** | 6 | 1 |
| Volume | 0.060 | 0.046 | 14 | 2 |
| Tick | 0.019 | 0.003 | 21 | 3 |
| Time | 0.000 | 0.000 | 25 | 4 |

### Stability

| Bar Type | Stationary (ADF) | ARCH LM | Mean CV | Vol CV | Levene | Score |
|---|---|---|---|---|---|---|
| Time | ✓ | ✗ | 21.305 | 0.668 | ✗ | 1/5 |
| Tick | ✓ | ✗ | 8.508 | 0.405 | ✗ | 2/5 |
| Volume | ✓ | ✗ | 9.028 | 0.384 | ✗ | 2/5 |
| **Dollar** | **✓** | ✗ | **5.415** | **0.344** | ✗ | **2/5** |

## What's Next

The instability that remains across all bar types — volatility clustering and structural breaks — cannot be solved by bar sampling alone. The next article will explore:

- **Dynamic thresholds** for dollar bar construction instead of a static $100M
- The **symmetric CUSUM filter** introduced by López de Prado
- Comparing sampling methods *within* the dollar bar framework to further minimize instability

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Martinez, G. — [Tick Bars](https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba) | [Volume & Dollar Bars](https://towardsdatascience.com/financial-machine-learning-part-1-labels-7eeed050f32e)
- [Kraken Historical Data](https://support.kraken.com/hc/en-us/articles/360047124832-Downloadable-historical-OHLCVT-Open-High-Low-Close-Volume-Trades-data)

## License

This project is licensed under the [MIT License](LICENSE).

