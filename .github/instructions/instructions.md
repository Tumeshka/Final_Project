---
applyTo: '**'
---

# Portfolio Environmental Impact Analysis System

## Project Overview
This is a portfolio environmental impact analysis system using ENCORE environmental database. The system creates network visualizations showing how investment portfolios create environmental impacts through economic activities, pressures, mechanisms, and ecosystem services.

## Architecture
- **Backend**: `portfolio_impact_engine.py` - contains all analysis logic
- **Frontend**: `Portfolio_Dashboard_Simple.ipynb` - minimal 2-cell interface for visualization
- **Data**: `ENCORE_data/` - environmental impact database with 18 datasets

## ENCORE Database Structure

### Core Activity Tables (271 economic activities each)
**ISIC Classification System**: All tables use 5-level ISIC (International Standard Industrial Classification):
- `ISIC Section` (21 major sectors): Agriculture, Mining, Manufacturing, etc.
- `ISIC Division` (88 divisions): Crop production, Coal mining, Food manufacturing, etc.  
- `ISIC Group` (238 groups): Growing of cereals, Mining of hard coal, etc.
- `ISIC Class` (419 classes): Growing of rice, Underground coal mining, etc.
- `ISIC Unique code`: Composite identifier (e.g., "A_1_14_141")

#### 1. Dependencies (03. Dependency links.csv) - Shape: (271, 32)
**Economic activities' dependence on ecosystem services**
- **Ecosystem Services (26 columns)**: Biomass provisioning, Water supply, Soil quality regulation, Climate regulation, Pollination, Flood mitigation, etc.
- **Content**: Qualitative descriptions of how each economic activity depends on ecosystem services
- **Usage**: Identifies which ecosystem services each industry relies on

#### 2. Dependency Ratings (06. Dependency mat ratings.csv) - Shape: (271, 31)  
**Materiality ratings for dependencies**
- **Same 26 ecosystem service columns** as dependency links
- **Values**: VH (Very High), H (High), M (Medium), L (Low), VL (Very Low), ND (No Dependency), N/A
- **Usage**: Quantifies intensity of dependency relationships

#### 3. Pressures (05. Pressure links.csv) - Shape: (271, 20)
**Economic activities' environmental pressures**
- **Environmental Pressures (14 columns)**:
  - `Emissions of GHG`
  - `Emissions of non-GHG air pollutants` 
  - `Emissions of toxic soil and water pollutants`
  - `Emissions of nutrient soil and water pollutants`
  - `Area of land use`
  - `Area of freshwater use`
  - `Area of seabed use`
  - `Volume of water use`
  - `Generation and release of solid waste`
  - `Other abiotic resource extraction`
  - `Other biotic resource extraction (e.g. fish, timber)`
  - `Introduction of invasive species`
  - `Disturbances (e.g noise, light)`
- **Content**: Qualitative descriptions of environmental impacts
- **Usage**: Identifies which pressures each industry creates

#### 4. Pressure Ratings (07. Pressure mat ratings.csv) - Shape: (271, 19)
**Materiality ratings for pressures**
- **Same 14 pressure columns** as pressure links
- **Values**: VH (Very High), H (High), M (Medium), L (Low), VL (Very Low), ND (No Dependency), N/A
- **Usage**: Quantifies intensity of environmental pressure relationships

### Definition Tables
#### 5. Ecosystem Services Definitions (02.) - Shape: (48, 3)
**Hierarchical ecosystem services taxonomy**
- Level 1 categories (Provisioning, Regulating, Cultural services)
- Level 2 specific services with detailed definitions
- Based on SEEA-EA (System of Environmental-Economic Accounting)

#### 6. Pressure Definitions (04.) - Shape: (15, 2)  
**Environmental pressure categories and definitions**
- Each of 14 pressure types with detailed definitions and metrics
- Examples of measurement units and impact mechanisms

#### 7. Ecosystem Components Definitions (10.) - Shape: (8, 2)
**Ecosystem component categories**
- Water, Soil, Land geomorphology, Species, Atmosphere, etc.
- Descriptions of each component type

#### 8. Mechanisms Definitions (12.) - Shape: (18, 2)
**Environmental change mechanisms**
- Disease, Drought, Flooding, Species composition changes, etc.
- Describes how pressures translate to ecosystem impacts

### Relationship Mapping Tables  
#### 9. Ecosystem Services ↔ Components (11.) - Shape: (1,395, 7)
**Links ecosystem services to ecosystem components**
- Which ecosystem components provide which services
- Ecosystem types (Forest, Grassland, Wetland, Marine, etc.)
- Rating quality and link assessments

#### 10. Pressures → Components (13.) - Shape: (88, 10)
**How environmental pressures affect ecosystem components**
- Links pressures to mechanisms to components
- Timescale (Short/Mid/Long term), spatial characteristics
- Direct vs indirect impacts with scientific references

#### 11. Upstream Links (16.) - Shape: (18,488, 8)
**Value chain upstream dependencies**
- Direct operations → Tier 1 upstream → Tier 2 upstream
- Shows supply chain environmental dependencies
- ISIC hierarchy for each tier

#### 12. Downstream Links (17.) - Shape: (15,057, 8)  
**Value chain downstream relationships**
- Direct operations → Tier 1 downstream → Tier 2 downstream
- Shows customer/distribution environmental impacts
- ISIC hierarchy for each tier

### Classification Crosswalks
#### 13. EXIOBASE-NACE-ISIC Crosswalk (14.) - Shape: (822, 11)
**Industry classification mapping**
- Maps between EXIOBASE (economic modeling), NACE (European), ISIC (international) codes
- Enables integration with economic datasets

#### 14. Impact Drivers Crosswalk - Shape: (16, 3)
**Pressure terminology evolution**  
- Maps pressure names between ENCORE versions (2018-2023 vs 2024)
- Tracks terminology updates and changes

## Data Interconnections

**Primary Key**: `ISIC Unique code` links all main tables (03, 05, 06, 07)

**Network Flow**: 
1. **Economic Activities** (ISIC codes) → 
2. **Environmental Pressures** (05, 07) → 
3. **Mechanisms** (12, 13) → 
4. **Ecosystem Components** (10, 13) → 
5. **Ecosystem Services** (02, 11) → 
6. **Dependencies** (03, 06) → 
7. **Economic Activities** (circular flow)

**Value Chains**: Activities connect via upstream (16) and downstream (17) supply chain links

**Cross-References**: Scientific literature linked throughout pressure-component relationships (13)

## Code Style Guidelines

### General Rules
- No unnecessary text output, prints, or verbose logging
- No decorative icons or emojis in code
- Keep functions focused and minimal
- Ask specific questions before major changes
- Make minimal targeted edits rather than large rewrites

### Notebook Standards
- Use 2-cell structure: setup cell + visualization cell
- Parameters at top of visualization cell for easy modification
- No progress messages or status updates
- Direct to visualization output only

### Function Design
- Single responsibility principle
- Clear parameter names and types
- Minimal error handling - fail fast
- No chatty output or confirmation messages

### Visualization Requirements
- Clustered layout with nodes grouped by type
- Portfolio node positioned on far left
- Clean node labels without redundant prefixes
- Support for customizable figure size, node size, edge width
- Dark theme by default

## Key Components
- `PortfolioImpactEngine`: Core network creation and visualization
- `PortfolioDashboard`: Portfolio management and analysis
- `create_comprehensive_impact_network()`: Main network generation function
- `visualize_portfolio_network()`: Clustered visualization with clean labels

## Modification Guidelines
- Always test changes with existing portfolios (Energy, Agriculture, Technology, Financial, Manufacturing)
- Preserve clustered layout and clean labeling
- Maintain parameter configurability in notebook interface
- Ask before changing core network structure or ENCORE data processing
- Keep notebook interface minimal and focused

## Questions to Ask Before Changes
1. Does this change affect the core ENCORE circular flow model?
2. Will this impact existing portfolio visualizations?
3. Does this add unnecessary complexity or output?
4. Is this the minimal change needed to achieve the goal?