# Cross-Sector Impact Engine - Visualization Reference

A comprehensive guide to all visualization methods in `cross_sector_impact_engine.py`.

---

## Table of Contents
1. [Network/Flow Visualizations](#1-networkflow-visualizations)
2. [Activity-Level Analysis Charts](#2-activity-level-analysis-charts)
3. [Category-Based Analysis Charts](#3-category-based-analysis-charts)
4. [Portfolio Analysis Charts](#4-portfolio-analysis-charts)
5. [Presets and Filters Reference](#5-presets-and-filters-reference)

---

## 1. Network/Flow Visualizations

### `create_flow_visualization()`
**Purpose:** Creates a directed network graph showing how environmental pressures cascade through ecosystems to affect other industries.

**Visual Output:** 5-layer directed graph (NetworkX)
- Layer 1: **Source Activity** (red) - the starting industry
- Layer 2: **Pressures** (orange) - environmental pressures it creates
- Layer 3: **Ecosystem Components** (green) - natural systems affected
- Layer 4: **Ecosystem Services** (blue) - services those components provide
- Layer 5: **Affected Activities** (pink) - other industries that depend on those services

**Signature:**
```python
engine.create_flow_visualization(
    target_activity,           # str: ISIC Section name (e.g., "Mining and quarrying")
    min_intensity='M',         # str: VL, L, M, H, VH - minimum for Activity→Pressure
    timescale='All',           # str: All, Short term, Mid term, Long term
    direct_indirect='All',     # str: All, Direct, Indirect, Both
    spatial='All',             # str: All, Local, Regional, Global
    comp_svc_rating='All',     # str: All, R (High), A (Medium), G (Low)
    ecosystem_type='All',      # str: Filter by ecosystem type
    svc_act_intensity='M',     # str: VL, L, M, H, VH - minimum for Service→Activity
    highlight_rank=1           # int: 1=most affected, 2=2nd most, 0=no highlight
)
```

**Example:**
```python
# Show how Mining affects other industries, highlight most affected
engine.create_flow_visualization(
    "Mining and quarrying",
    min_intensity='M',
    svc_act_intensity='M',
    highlight_rank=1
)
```

---

### `visualize_activity_impact()`
**Purpose:** Wrapper that creates flow visualization + prints detailed summary statistics.

**Signature:**
```python
flow_data = engine.visualize_activity_impact(
    activity_name,        # str: ISIC Section name
    min_intensity='M'     # str: VL, L, M, H, VH
)
```

**Returns:** Dictionary with:
- `pressures`: List of environmental pressures
- `components`: Affected ecosystem components
- `services`: Affected ecosystem services
- `affected_activities`: Other industries affected
- `activity_scores`: Weighted scores per affected activity

---

## 2. Activity-Level Analysis Charts

### `plot_affected_activities_by_preset()`
**Purpose:** Multi-subplot horizontal bar chart comparing affected activities rankings across different presets.

**Visual Output:** Grid of subplots, one per preset, showing top N affected activities.

**Signature:**
```python
fig = engine.plot_affected_activities_by_preset(
    activities=None,      # List[str]: Source activities (None=all)
    presets=None,         # List[str]: Preset names (None=all)
    top_n=15,             # int: Top affected activities per subplot
    figsize=(18, 12)      # tuple: Figure size
)
```

**Example:**
```python
fig = engine.plot_affected_activities_by_preset(
    presets=['Baseline', 'Direct_Only', 'High_Sensitivity'],
    top_n=10
)
```

---

### `plot_source_activity_impacts()`
**Purpose:** Multi-subplot horizontal bar chart comparing SOURCE activity impacts (which industries CAUSE the most impact).

**Visual Output:** Grid of subplots showing which source activities have highest impact scores.

**Signature:**
```python
fig, df = engine.plot_source_activity_impacts(
    activities=None,      # List[str]: Source activities to analyze
    presets=None,         # List[str]: Preset names to compare
    top_n=15,             # int: Top source activities per subplot
    figsize=(18, 12)      # tuple: Figure size
)
```

---

### `plot_cumulative_affected_activities()`
**Purpose:** Single stacked horizontal bar chart showing cumulative impact on affected activities across all presets.

**Visual Output:** Stacked bars where each color segment = contribution from one preset.

**Signature:**
```python
fig, df = engine.plot_cumulative_affected_activities(
    activities=None,      # List[str]: Source activities
    presets=None,         # List[str]: Presets to stack
    top_n=15,             # int: Top affected activities
    figsize=(14, 10)      # tuple: Figure size
)
```

---

### `plot_cumulative_source_impacts()`
**Purpose:** Single stacked horizontal bar chart showing cumulative SOURCE activity impacts across all presets.

**Visual Output:** Stacked bars showing which source activities have highest cumulative impact.

**Signature:**
```python
fig, df = engine.plot_cumulative_source_impacts(
    activities=None,      # List[str]: Source activities
    presets=None,         # List[str]: Presets to stack
    top_n=15,             # int: Top source activities
    figsize=(14, 10)      # tuple: Figure size
)
```

---

## 3. Category-Based Analysis Charts

### `plot_affected_by_category()`
**Purpose:** Plot affected activities for a single category (Scope, Risk, Timeline, or Spatial) with subplots per preset in that category.

**Categories:**
- **Scope:** Direct_Only, Indirect_Only, Both
- **Risk:** High_Sensitivity, Medium_Sensitivity, Low_Sensitivity
- **Timeline:** Short_Term, Mid_Term, Long_Term
- **Spatial:** Local_Only, Regional, Global

**Signature:**
```python
fig, data = engine.plot_affected_by_category(
    category,             # str: 'Scope', 'Risk', 'Timeline', or 'Spatial'
    activities=None,      # List[str]: Source activities
    top_n=15,             # int: Top affected activities per preset
    figsize=None          # tuple: Auto-calculated if None
)
```

**Example:**
```python
# Compare affected activities across Risk sensitivity levels
fig, data = engine.plot_affected_by_category('Risk', top_n=10)

# Compare across Scope (Direct vs Indirect)
fig, data = engine.plot_affected_by_category('Scope', top_n=15)
```

---

### `plot_source_impacts_by_category()`
**Purpose:** Plot SOURCE activity impacts for a single category, showing which activities CAUSE the most impact.

**Signature:**
```python
fig, data = engine.plot_source_impacts_by_category(
    category,             # str: 'Scope', 'Risk', 'Timeline', or 'Spatial'
    activities=None,      # List[str]: Source activities
    top_n=15,             # int: Top source activities per preset
    figsize=None          # tuple: Auto-calculated if None
)
```

---

## 4. Portfolio Analysis Charts

### `plot_portfolio_impact_stacked()`
**Purpose:** Stacked horizontal bar chart showing which AFFECTED ACTIVITIES are impacted by a portfolio, with contributions from each holding.

**Visual Output:** 
- Y-axis: Affected activities (sorted by total impact)
- X-axis: Weighted impact score
- Bar segments: Colored by portfolio holding (with company logos)
- Right side: ISIC Section logos for affected activities

**Signature:**
```python
fig, result = engine.plot_portfolio_impact_stacked(
    csv_path,                    # str: Path to portfolio CSV
    preset=None,                 # str: Preset name (e.g., 'Baseline')
    top_n=15,                    # int: Top affected activities
    figsize=(16, 12),            # tuple: Figure size
    intensity='M',               # str: VL, L, M, H, VH
    timescale='All',             # str: Timescale filter
    direct_indirect='All',       # str: Impact type filter
    spatial='All',               # str: Spatial scale filter
    rating='All',                # str: Sensitivity rating
    svc_act_intensity='M',       # str: Min dependency intensity
    min_contribution_pct=2.0,    # float: Group holdings <X% as "Other"
    group_by='holding',          # str: 'holding' or 'isic'
    portfolio_type='oil_gas'     # str: 'oil_gas' or 'food_beverage'
)
```

**Example:**
```python
# Oil & Gas portfolio - which industries are affected
fig, result = engine.plot_portfolio_impact_stacked(
    csv_path='portfolio_exports/STOXX_Europe_600_Oil_Gas_Holdings.csv',
    preset='Baseline',
    top_n=15,
    figsize=(28, 12),
    min_contribution_pct=2.0,
    portfolio_type='oil_gas'
)

# Same but grouped by ISIC source activity instead of holding
fig, result = engine.plot_portfolio_impact_stacked(
    csv_path='portfolio_exports/STOXX_Europe_600_Oil_Gas_Holdings.csv',
    preset='Baseline',
    group_by='isic',
    portfolio_type='oil_gas'
)
```

**Returns:** Tuple of (figure, result_dict)
- `result['affected_activities']`: {activity: {holding: score}}
- `result['summary']`: Portfolio stats (total_holdings, total_weight, total_impact_score)

---

### `plot_portfolio_source_impact_stacked()`
**Purpose:** Stacked horizontal bar chart showing SOURCE ACTIVITIES (ISIC Sections) the portfolio maps to, with contributions from each holding.

**Visual Output:**
- Y-axis: Source ISIC activities (what the portfolio companies DO)
- X-axis: Impact score
- Bar segments: Colored by portfolio holding (with company logos)
- Left side: ISIC Section logos for source activities

**Signature:**
```python
fig, result = engine.plot_portfolio_source_impact_stacked(
    csv_path,                    # str: Path to portfolio CSV
    preset=None,                 # str: Preset name
    top_n=10,                    # int: Top source activities
    figsize=(16, 10),            # tuple: Figure size
    intensity='M',               # str: VL, L, M, H, VH
    timescale='All',             # str: Timescale filter
    direct_indirect='All',       # str: Impact type filter
    spatial='All',               # str: Spatial scale filter
    rating='All',                # str: Sensitivity rating
    svc_act_intensity='M',       # str: Min dependency intensity
    min_contribution_pct=2.0,    # float: Group holdings <X% as "Other"
    portfolio_type='oil_gas'     # str: 'oil_gas' or 'food_beverage'
)
```

**Example:**
```python
# Food & Beverage portfolio - what source activities it represents
fig, result = engine.plot_portfolio_source_impact_stacked(
    csv_path='portfolio_exports/STOXX-Europe-600-Food--Beverage-UCITS-ETF-DE_fund.csv',
    preset='Baseline',
    top_n=10,
    figsize=(28, 12),
    min_contribution_pct=2.0,
    portfolio_type='food_beverage'
)
```

---

## 5. Presets and Filters Reference

### Intensity Levels (ENCORE 2-6 Scale)
| Code | Description | Numeric |
|------|-------------|---------|
| VL | Very Low | 2 |
| L | Low | 3 |
| M | Medium | 4 |
| H | High | 5 |
| VH | Very High | 6 |

### Available Presets (from `presets_config.jsonc`)

**Baseline:**
- `Baseline` - Default balanced analysis

**By Scope:**
- `Direct_Only` - Only direct impacts
- `Indirect_Only` - Only indirect impacts  
- `Both` - Direct and indirect combined

**By Risk/Sensitivity:**
- `High_Sensitivity` - R rating only (high sensitivity)
- `Medium_Sensitivity` - A rating only
- `Low_Sensitivity` - G rating only

**By Timeline:**
- `Short_Term` - Short term impacts
- `Mid_Term` - Mid term impacts
- `Long_Term` - Long term impacts

**By Spatial:**
- `Local_Only` - Local scale only
- `Regional` - Regional scale
- `Global` - Global scale

### Filter Parameters
| Parameter | Options | Description |
|-----------|---------|-------------|
| `intensity` | VL, L, M, H, VH | Min intensity for Activity→Pressure |
| `svc_act_intensity` | VL, L, M, H, VH | Min intensity for Service→Activity |
| `timescale` | All, Short term, Mid term, Long term | Pressure→Component filter |
| `direct_indirect` | All, Direct, Indirect, Both | Impact type filter |
| `spatial` | All, Local, Regional, Global | Spatial scale filter |
| `rating` | All, R, A, G | Sensitivity rating (R=High, A=Medium, G=Low) |

---

## Quick Reference: Choosing the Right Visualization

| Question | Method |
|----------|--------|
| How does one industry affect others? | `create_flow_visualization()` |
| Which industries are most affected overall? | `plot_affected_activities_by_preset()` |
| Which industries cause the most impact? | `plot_source_activity_impacts()` |
| Compare direct vs indirect effects? | `plot_affected_by_category('Scope')` |
| Compare short vs long term? | `plot_affected_by_category('Timeline')` |
| What does my portfolio impact? | `plot_portfolio_impact_stacked()` |
| What sectors is my portfolio in? | `plot_portfolio_source_impact_stacked()` |

---

## Data Requirements

### Portfolio CSV Format
Required columns:
- `Name` or `Holding`: Company/holding name
- `Weight (%)` or similar: Portfolio weight percentage
- `Sector`: Industry sector for ISIC mapping

### Logo Assets
- **ISIC logos:** `assets/ISIC_logos/{letter}.png` (A-U)
- **Oil & Gas holdings:** `assets/holdings_logos_Oil_Gas/{company}.png`
- **Food & Beverage holdings:** `assets/holdings_logos_Food_Beverage/{company}.png`

---

*Last updated: December 2024*
