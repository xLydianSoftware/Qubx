"""
Orderbook visualization utilities for Qubx
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, Union
from qubx.core.series import OrderBook


def plot_orderbook(orderbook: OrderBook, cumulative: bool = False, max_levels: Optional[int] = None, 
                   figsize: tuple = (12, 8), title: Optional[str] = None, show_spread: bool = True,
                   ax=None) -> Tuple[Figure, Axes]:
    """
    Plot orderbook with bids and asks visualization.
    
    Parameters:
    -----------
    orderbook : OrderBook
        The orderbook object to plot
    cumulative : bool, default False
        Whether to show cumulative volumes or individual level volumes
    max_levels : int, optional
        Maximum number of levels to display (None = show all available levels)
    figsize : tuple, default (12, 8)
        Figure size in inches (width, height)
    title : str, optional
        Custom title for the plot
    show_spread : bool, default True
        Whether to show the bid-ask spread information
    ax : matplotlib axes, optional
        Existing axes to plot on
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract data from orderbook
    bids = orderbook.bids.copy()
    asks = orderbook.asks.copy()
    top_bid = orderbook.top_bid
    top_ask = orderbook.top_ask
    tick_size = orderbook.tick_size
    
    # Filter out zero volumes for cleaner display
    non_zero_bids = bids[bids > 0]
    non_zero_asks = asks[asks > 0]
    
    # Limit levels if specified
    if max_levels:
        max_levels = min(max_levels, len(non_zero_bids), len(non_zero_asks))
        non_zero_bids = non_zero_bids[:max_levels]
        non_zero_asks = non_zero_asks[:max_levels]
    
    n_bid_levels = len(non_zero_bids)
    n_ask_levels = len(non_zero_asks)
    
    # Calculate price levels
    bid_prices = np.array([top_bid - i * tick_size for i in range(n_bid_levels)])
    ask_prices = np.array([top_ask + i * tick_size for i in range(n_ask_levels)])
    
    # Calculate cumulative volumes if requested
    if cumulative:
        bids_display = np.cumsum(non_zero_bids)
        asks_display = np.cumsum(non_zero_asks)
        ylabel = "Cumulative Volume"
    else:
        bids_display = non_zero_bids
        asks_display = non_zero_asks
        ylabel = "Volume"
    
    # Set colors - improved palette
    bid_color = '#2E8B57'    # Sea green for bids
    ask_color = '#DC143C'    # Crimson for asks  
    spread_color = '#FFD700'  # Gold for spread
    bid_edge = '#1F5F3F'     # Darker green edge
    ask_edge = '#8B0000'     # Darker red edge
    
    # Plot bid side (left side) with improved styling
    if len(bid_prices) > 0:
        ax.barh(bid_prices, -bids_display, height=tick_size * 0.85, 
                color=bid_color, alpha=0.8, label='Bids',
                edgecolor=bid_edge, linewidth=0.5)
    
    # Plot ask side (right side) with improved styling
    if len(ask_prices) > 0:
        ax.barh(ask_prices, asks_display, height=tick_size * 0.85, 
                color=ask_color, alpha=0.8, label='Asks',
                edgecolor=ask_edge, linewidth=0.5)
    
    # Add spread visualization
    if show_spread:
        mid_price = (top_bid + top_ask) / 2
        
        # Add horizontal line at mid price with improved styling
        ax.axhline(y=mid_price, color=spread_color, linestyle='--', alpha=0.9, 
                  linewidth=2.5, label=f'Mid: ${mid_price:.2f}')
        
        # Add subtle best bid and ask lines
        ax.axhline(y=top_bid, color=bid_color, linestyle=':', alpha=0.6, 
                  linewidth=1.2)
        ax.axhline(y=top_ask, color=ask_color, linestyle=':', alpha=0.6, 
                  linewidth=1.2)
    
    # Customize the plot with improved styling
    ax.set_xlabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title(title or f'Order Book - ${top_bid:.2f} | ${top_ask:.2f}' + 
                 (f' (Spread: ${top_ask - top_bid:.2f})' if show_spread else ''),
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis to be symmetric around 0
    max_vol = max(max(bids_display) if len(bids_display) > 0 else 0, 
                  max(asks_display) if len(asks_display) > 0 else 0)
    ax.set_xlim(-max_vol * 1.15, max_vol * 1.15)
    
    # Add vertical line at x=0 with better styling
    ax.axvline(x=0, color='#333333', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Add grid with improved styling
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='#666666')
    ax.set_axisbelow(True)
    
    # Add legend with better positioning
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    # Format price axis for better readability
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.0f'))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # Skip volume labels to avoid cluttering the plot
    # if bid_bars is not None and len(bid_prices) > 0:
    #     _add_volume_labels(ax, bid_bars, bid_prices, -bids_display, 'left')
    # if ask_bars is not None and len(ask_prices) > 0:
    #     _add_volume_labels(ax, ask_bars, ask_prices, asks_display, 'right')
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax


def plot_orderbook_depth(orderbook: OrderBook, max_levels: Optional[int] = None, 
                        figsize: tuple = (14, 6), title: Optional[str] = None) -> Tuple[Figure, Axes]:
    """
    Plot orderbook depth chart showing cumulative volumes at different price levels.
    
    Parameters:
    -----------
    orderbook : OrderBook
        The orderbook object to plot
    max_levels : int, optional
        Maximum number of levels to display
    figsize : tuple, default (14, 6)
        Figure size in inches
    title : str, optional
        Custom title for the plot
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Extract data
    bids = orderbook.bids.copy()
    asks = orderbook.asks.copy()
    top_bid = orderbook.top_bid
    top_ask = orderbook.top_ask
    tick_size = orderbook.tick_size
    
    # Limit levels if specified
    if max_levels:
        max_levels = min(max_levels, len(bids), len(asks))
        bids = bids[:max_levels]
        asks = asks[:max_levels]
    
    n_levels = len(bids)
    
    # Calculate price levels
    bid_prices = np.array([top_bid - i * tick_size for i in range(n_levels)])
    ask_prices = np.array([top_ask + i * tick_size for i in range(n_levels)])
    
    # Calculate cumulative volumes
    cum_bids = np.cumsum(bids)
    cum_asks = np.cumsum(asks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors - matching the main orderbook plot
    bid_color = '#2E8B57'    # Sea green for bids
    ask_color = '#DC143C'    # Crimson for asks
    mid_color = '#FFD700'    # Gold for mid price
    
    # Plot depth curves with improved styling
    ax.fill_between(bid_prices, cum_bids, step='post', alpha=0.7, 
                   color=bid_color, label='Bid Depth', 
                   edgecolor='#1F5F3F', linewidth=1)
    ax.fill_between(ask_prices, cum_asks, step='pre', alpha=0.7, 
                   color=ask_color, label='Ask Depth',
                   edgecolor='#8B0000', linewidth=1)
    
    # Add mid price line with improved styling
    mid_price = (top_bid + top_ask) / 2
    ax.axvline(x=mid_price, color=mid_color, linestyle='--', 
              linewidth=2.5, alpha=0.9, label=f'Mid: ${mid_price:.2f}')
    
    # Customize plot with improved styling
    ax.set_xlabel('Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Volume', fontsize=12, fontweight='bold')
    ax.set_title(title or f'Order Book Depth - ${top_bid:.2f} | ${top_ask:.2f}',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='#666666')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Format price axis for better readability
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%.0f'))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # Set price range
    min_price = min(bid_prices[-1], ask_prices[0]) * 0.999
    max_price = max(bid_prices[0], ask_prices[-1]) * 1.001
    ax.set_xlim(min_price, max_price)
    
    plt.tight_layout()
    
    return fig, ax


def _add_volume_labels(ax, bars, prices, volumes, side):
    """
    Add volume labels to orderbook bars.
    
    Parameters:
    -----------
    ax : matplotlib axes
        The axes to add labels to
    bars : matplotlib bar container
        The bar objects
    prices : array
        Price levels
    volumes : array
        Volume values
    side : str
        'left' for bids, 'right' for asks
    """
    for bar, price, volume in zip(bars, prices, volumes):
        if abs(volume) > 0:  # Only label non-zero volumes
            # Position label
            x_pos = volume * 0.5 if side == 'right' else volume * 0.5
            
            # Format volume (show fewer decimals for large numbers)
            vol_abs = abs(volume)
            if vol_abs >= 1000:
                vol_text = f"{vol_abs/1000:.1f}K"
            elif vol_abs >= 1:
                vol_text = f"{vol_abs:.1f}"
            else:
                vol_text = f"{vol_abs:.3f}"
            
            # Add label
            ax.text(x_pos, price, vol_text, ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')


def plot_orderbook_comparison(orderbooks: list, labels: Optional[list] = None, 
                             cumulative: bool = False, figsize: tuple = (15, 10)) -> Tuple[Figure, list]:
    """
    Plot multiple orderbooks for comparison.
    
    Parameters:
    -----------
    orderbooks : list of OrderBook objects
        List of orderbooks to compare
    labels : list of str, optional
        Labels for each orderbook
    cumulative : bool, default False
        Whether to show cumulative volumes
    figsize : tuple, default (15, 10)
        Figure size
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    n_books = len(orderbooks)
    
    if labels is None:
        labels = [f"Book {i+1}" for i in range(n_books)]
    
    # Create subplots
    fig, axes = plt.subplots(1, n_books, figsize=figsize, sharey=True)
    
    if n_books == 1:
        axes = [axes]
    
    for i, (orderbook, label) in enumerate(zip(orderbooks, labels)):
        plot_orderbook(orderbook, cumulative=cumulative, 
                      title=label, ax=axes[i], show_spread=True)
    
    plt.tight_layout()
    return fig, axes