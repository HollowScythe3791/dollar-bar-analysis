"""
Bar construction functions for time, tick, volume, and dollar bars.
"""

import pandas as pd
import numpy as np


# ===========================================================
# BAR CREATION FUNCTIONS
# ===========================================================

def create_time_bars(df: pd.DataFrame, interval: int, fill_gaps: bool = True) -> pd.DataFrame:
    """
    Convert raw tick data into time bars with optional gap filling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw tick data with columns: 'time', 'price', 'volume'
        where 'time' is unix timestamp in seconds
    interval : int
        Time interval in seconds for each bar (e.g., 60 for 1-minute bars)
    fill_gaps : bool, default True
        If True, fills gaps where no trading occurred using forward-fill
        
    Returns:
    --------
    pd.DataFrame
        Time bars with columns: 'time', 'open', 'high', 'low', 'close', 'volume'
    """
    df = df.copy()
    
    # Calculate the bar index based on time intervals
    start_time = (df['time'].iloc[0] // interval) * interval  # Align to interval boundary
    df['bar_idx'] = (df['time'] - start_time) // interval
    
    # Aggregate ticks into time bars
    time_bars = df.groupby('bar_idx').agg(
        time=('time', 'last'),       # Time of last tick (close time)
        open=('price', 'first'),     # First price in the interval
        high=('price', 'max'),       # Maximum price
        low=('price', 'min'),        # Minimum price
        close=('price', 'last'),     # Last price in the interval
        volume=('volume', 'sum')     # Total volume
    )
    
    if fill_gaps:
        # Create complete index from first to last bar
        full_index = pd.RangeIndex(
            start=int(time_bars.index.min()),
            stop=int(time_bars.index.max()) + 1,
            step=1
        )
        
        # Reindex to include all intervals
        time_bars = time_bars.reindex(full_index)
        
        # Calculate proper timestamps for missing bars
        # Each bar's close time is at the end of its interval
        time_bars['time'] = time_bars.index * interval + start_time + interval - 1
        
        # Forward fill the close price first (needed for OHLC of gap bars)
        time_bars['close'] = time_bars['close'].ffill()
        
        # For gap bars: open, high, low should all equal the forward-filled close
        # Identify gap bars (where open is NaN)
        gap_mask = time_bars['open'].isna()
        
        # Fill gap bars with previous close
        time_bars.loc[gap_mask, 'open'] = time_bars.loc[gap_mask, 'close']
        time_bars.loc[gap_mask, 'high'] = time_bars.loc[gap_mask, 'close']
        time_bars.loc[gap_mask, 'low'] = time_bars.loc[gap_mask, 'close']
        
        # Fill volume with 0 for gap bars
        time_bars['volume'] = time_bars['volume'].fillna(0)
    
    # Reset index and clean up
    time_bars = time_bars.reset_index(drop=True)
    
    return time_bars

def create_tick_bars(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Convert raw tick data into tick bars.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw tick data with columns: 'timestamp', 'price', 'volume'
    threshold : int
        Number of ticks to group into each bar
        
    Returns:
    --------
    pd.DataFrame
        Tick bars with columns: 'time', 'open', 'high', 'low', 'close', 'volume'
    """
    # Calculate the number of complete bars we can form
    n_bars = len(df) // threshold
    
    # Trim data to only include complete bars
    trimmed_length = n_bars * threshold
    df_trimmed = df.iloc[:trimmed_length].copy()
    
    # Create a bar index for grouping
    df_trimmed['bar_idx'] = np.arange(len(df_trimmed)) // threshold
    
    # Aggregate ticks into bars
    tick_bars = df_trimmed.groupby('bar_idx').agg(
        time=('time', 'last'),       # Time of last tick (close time)
        open=('price', 'first'),     # First price in the bar
        high=('price', 'max'),       # Maximum price
        low=('price', 'min'),        # Minimum price
        close=('price', 'last'),     # Last price in the bar
        volume=('volume', 'sum')     # Total volume
    ).reset_index(drop=True)
    
    return tick_bars

def create_volume_bars(pd: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Volume bars using pandas groupby - no copies, reasonable speed.
    """
    # Compute bar IDs directly on the dataframe (no copy)
    cum_vol = pd['volume'].cumsum()
    bar_ids = (cum_vol // threshold).astype(np.int32)
    
    # Single groupby aggregation
    return pd.groupby(bar_ids, sort=False).agg(
        time=('time', 'last'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('volume', 'sum')
    ).reset_index(drop=True)

def create_dollar_bars(pd: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Convert tick data into dollar bars.
    
    Parameters:
    -----------
    pd : pd.DataFrame
        DataFrame with columns ['time', 'price', 'volume']
    threshold : int
        Dollar volume threshold for each bar (price * volume)
    
    Returns:
    --------
    pd.DataFrame
        Dollar bars with columns: time, open, high, low, close, volume
    """
    # Calculate dollar volume for each tick
    pd = pd.copy()
    pd['dollar_volume'] = pd['price'] * pd['volume']
    
    # Calculate cumulative dollar volume
    pd['cumulative_dollar_volume'] = pd['dollar_volume'].cumsum()
    
    # Assign bar IDs based on threshold crossings
    # Each bar contains ticks until cumulative dollar volume crosses the next threshold
    pd['bar_id'] = (pd['cumulative_dollar_volume'] // threshold).astype(int)
    
    # Group by bar_id and aggregate
    dollar_bars = pd.groupby('bar_id').agg(
        time=('time', 'last'),      # Close time (last tick)
        open=('price', 'first'),     # First price in bar
        high=('price', 'max'),       # Highest price in bar
        low=('price', 'min'),        # Lowest price in bar
        close=('price', 'last'),     # Last price in bar
        volume=('volume', 'sum')     # Total volume in bar
    ).reset_index(drop=True)
    
    return dollar_bars


