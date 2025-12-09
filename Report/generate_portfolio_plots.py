"""
Generate portfolio composition plots for the LaTeX report
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('figures', exist_ok=True)

# Define colors
COLORS_OILGAS = plt.cm.Blues(np.linspace(0.3, 0.9, 25))
COLORS_FOODBEV = plt.cm.Greens(np.linspace(0.3, 0.9, 30))


def build_global_isic_colors(prefixes=('OilGas', 'FoodBeverage'), years=('2015','2019','2024')):
    """Scan portfolio_exports files to build a consistent ISIC list and color map."""
    all_isic = set()
    for prefix in prefixes:
        for year in years:
            path = f'../portfolio_exports/{prefix}_{year}_with_ISIC.csv'
            try:
                df = pd.read_csv(path)
            except FileNotFoundError:
                continue
            if 'ISIC_Section' in df.columns:
                all_isic.update(df['ISIC_Section'].dropna().unique())

    # Sort for deterministic ordering
    all_isic = sorted([s for s in all_isic if pd.notna(s)])

    # Create a color palette large enough for all ISIC items
    n = max(1, len(all_isic))
    palette = plt.cm.tab20(np.linspace(0, 1, n))
    color_map = {isic: palette[i] for i, isic in enumerate(all_isic)}

    return all_isic, color_map


# Build global ISIC list and color mapping once so both portfolios use identical colors
GLOBAL_ISIC_LIST, GLOBAL_ISIC_COLORS = build_global_isic_colors()

def load_portfolio(filepath):
    """Load portfolio CSV and extract name and weight"""
    df = pd.read_csv(filepath)
    # Use Weight (%) column if available, otherwise Weight
    if 'Weight (%)' in df.columns:
        df['weight_pct'] = df['Weight (%)']
    else:
        df['weight_pct'] = df['Weight'] * 100
    return df[['Name', 'weight_pct']].sort_values('weight_pct', ascending=True)

def plot_portfolio_comparison(portfolios_dict, title, filename, colors):
    """Create horizontal bar chart comparing portfolio weights across years"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=False)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    years = ['2015', '2019', '2024']
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        df = portfolios_dict[year]
        
        # Take top 15 holdings for readability
        df_top = df.tail(15)
        
        bars = ax.barh(range(len(df_top)), df_top['weight_pct'], 
                       color=colors[idx * 5 + 5], edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                           for name in df_top['Name']], fontsize=9)
        ax.set_xlabel('Weight (%)', fontsize=11)
        ax.set_title(f'{year}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(df_top['weight_pct']) * 1.15)
        
        # Add value labels
        for bar, val in zip(bars, df_top['weight_pct']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=8)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: figures/{filename}")

def plot_isic_composition(portfolios_dict, title, filename, colormap):
    """Create stacked bar chart showing ISIC sector composition over time"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    years = ['2015', '2019', '2024']
    prefix = filename.split('_')[0]

    # Use a global ISIC->color mapping so colors are consistent across portfolios
    # Gather all ISIC sections across both portfolios' files (we expect this mapping
    # to be created once at program start and stored in GLOBAL_ISIC_LIST and
    # GLOBAL_ISIC_COLORS below).
    all_isic = GLOBAL_ISIC_LIST
    colors_map = GLOBAL_ISIC_COLORS

    # Calculate weights by ISIC section for each year
    data = {isic: [] for isic in all_isic}
    data['Unmapped/Missing'] = []
    totals = []
    
    for year in years:
        df = pd.read_csv(f'../portfolio_exports/{prefix}_{year}_with_ISIC.csv')
        if 'Weight (%)' in df.columns:
            df['weight_pct'] = df['Weight (%)']
        else:
            df['weight_pct'] = df['Weight'] * 100
        
        total_weight = df['weight_pct'].sum()
        totals.append(total_weight)
        isic_weights = df.groupby('ISIC_Section')['weight_pct'].sum()
        
        for isic in all_isic:
            data[isic].append(isic_weights.get(isic, 0))
        
        # Add unmapped portion to reach 100%
        data['Unmapped/Missing'].append(100 - total_weight)
    
    # Create stacked bar chart
    x = np.arange(len(years))
    bottom = np.zeros(len(years))
    
    for i, isic in enumerate(all_isic):
        color = colors_map.get(isic, 'lightgray')
        ax.bar(x, data[isic], bottom=bottom, label=isic[:40], color=color, width=0.6)
        bottom += np.array(data[isic])
    
    # Add unmapped portion in gray
    if any(v > 0.5 for v in data['Unmapped/Missing']):
        ax.bar(x, data['Unmapped/Missing'], bottom=bottom, label='Data not available', 
               color='lightgray', width=0.6, hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{year}\n({totals[i]:.1f}% coverage)' for i, year in enumerate(years)], fontsize=11)
    ax.set_ylabel('Portfolio Weight (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: figures/{filename}")

# Load Oil & Gas portfolios
print("Loading Oil & Gas portfolios...")
oilgas = {
    '2015': load_portfolio('../portfolio_exports/OilGas_2015_with_ISIC.csv'),
    '2019': load_portfolio('../portfolio_exports/OilGas_2019_with_ISIC.csv'),
    '2024': load_portfolio('../portfolio_exports/OilGas_2024_with_ISIC.csv')
}

# Load Food & Beverage portfolios
print("Loading Food & Beverage portfolios...")
foodbev = {
    '2015': load_portfolio('../portfolio_exports/FoodBeverage_2015_with_ISIC.csv'),
    '2019': load_portfolio('../portfolio_exports/FoodBeverage_2019_with_ISIC.csv'),
    '2024': load_portfolio('../portfolio_exports/FoodBeverage_2024_with_ISIC.csv')
}

# Generate plots
print("\nGenerating portfolio composition plots...")
plot_portfolio_comparison(oilgas, 'Oil & Gas Portfolio Composition (Top 15 Holdings)', 
                         'oilgas_composition.png', COLORS_OILGAS)
plot_portfolio_comparison(foodbev, 'Food & Beverage Portfolio Composition (Top 15 Holdings)', 
                         'foodbev_composition.png', COLORS_FOODBEV)

# Generate ISIC sector composition plots
print("\nGenerating ISIC sector composition plots...")
plot_isic_composition(oilgas, 'Oil & Gas Portfolio - ISIC Sector Composition', 
                     'OilGas_isic_composition.png', 'Blues')
plot_isic_composition(foodbev, 'Food & Beverage Portfolio - ISIC Sector Composition', 
                     'FoodBeverage_isic_composition.png', 'Greens')

print("\n✅ All plots generated successfully!")
