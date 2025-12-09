# Environmental Impact Portfolio Analysis System

## Project Overview

This project develops an interactive visualization system for analyzing environmental impacts of financial portfolios using the ENCORE (Exploring Natural Capital Opportunities, Risks and Exposure) database. The system transforms complex environmental dependency and impact data into intuitive network visualizations that help financial professionals understand sustainability risks in their investment portfolios.

## What the Project Does

### 1. **Portfolio Environmental Impact Mapping**
- Maps economic activities in investment portfolios to their environmental pressures
- Visualizes dependencies on ecosystem services and natural capital
- Shows complete impact pathways from economic activities through to ecosystem degradation

### 2. **Interactive Network Visualization**
- Creates dynamic, interactive network graphs using Cytoscape.js
- Enables drag-and-drop manipulation of nodes for exploration
- Provides clustered layouts that group related environmental components
- Color-codes different types of environmental impacts for clarity

### 3. **Multi-Sector Portfolio Analysis**
- Supports analysis across 5 key economic sectors:
  - Energy (oil, gas, renewables)
  - Agriculture (crops, livestock, forestry)
  - Technology (electronics, software, telecommunications)
  - Financial Services (banking, insurance, investment)
  - Manufacturing (automotive, chemicals, textiles)

### 4. **Comprehensive Data Integration**
- Integrates 18 ENCORE datasets covering:
  - Economic activity classifications (NACE/ISIC codes)
  - Environmental pressure definitions and links
  - Ecosystem service dependencies
  - Natural capital component relationships
  - Supply chain impact pathways

## How to Use the System

### **Step 1: Setup and Installation**
1. Open the `Portfolio_Dashboard_Simple.ipynb` notebook
2. Run Cell 1 to initialize the system:
   ```python
   # This loads all ENCORE data and creates the analysis engine
   from portfolio_impact_engine import create_full_system
   engine, dashboard, analyzer, exporter = create_full_system()
   ```

### **Step 2: Configure Your Analysis**
In Cell 2, adjust the parameters for your specific needs:

```python
# SET YOUR PARAMETERS HERE:
PORTFOLIO_THEME = 'Energy'          # Choose sector to analyze
FIGURE_SIZE = (1200, 1200)          # Visualization canvas size
NODE_SIZE_MULTIPLIER = 1.3          # Scale node sizes
EDGE_WIDTH_MULTIPLIER = 1.8         # Scale edge thickness
DARK_THEME = True                   # Dark or light background
MAX_ACTIVITIES = 12                 # Network complexity control
```

### **Step 3: Generate Interactive Visualization**
Run Cell 2 to create your interactive network:
- The system automatically generates a clustered network layout
- Each node type is color-coded (gold=portfolio, green=activities, orange=pressures, etc.)
- Node labels are cleaned and shortened for readability

### **Step 4: Explore the Network**
Use the interactive features to analyze impact pathways:
- **Drag nodes** to rearrange and focus on specific relationships
- **Pan and zoom** to navigate large networks
- **Follow arrows** to trace impact flows from activities to environmental degradation
- **Identify clusters** of related environmental impacts

### **Step 5: Analyze Results**
The visualization reveals:
- Which economic activities create the most environmental pressure
- How activities are interconnected through shared ecosystem dependencies
- Critical natural capital components at risk
- Supply chain relationships and indirect impacts

## Technical Architecture

### **Core Components**
1. **Portfolio Impact Engine** (`portfolio_impact_engine.py`)
   - Data processing and network construction
   - Clustered layout algorithms for readable visualizations
   - Integration with ENCORE database schemas

2. **Interactive Dashboard** (Jupyter Notebook)
   - Two-cell interface for maximum simplicity
   - Parameter controls for customization
   - Real-time visualization generation

3. **ENCORE Database Integration**
   - 18 CSV datasets covering complete environmental impact framework
   - Crosswalk tables linking economic activities to environmental pressures
   - Comprehensive ecosystem service and natural capital mappings

### **Visualization Technology**
- **ipycytoscape**: Professional network visualization with native node interaction
- **NetworkX**: Graph construction and analysis backend
- **Jupyter Widgets**: Interactive parameter controls

## Key Features

### **1. Simplified Interface**
- Minimal 2-cell notebook structure
- Clear parameter controls
- One-click visualization generation

### **2. Professional Visualizations**
- Color-coded node types for instant recognition
- Clustered layouts that group related components
- Clean, shortened labels for readability
- Customizable styling (dark/light themes, sizing)

### **3. Interactive Exploration**
- Drag-and-drop node repositioning
- Smooth pan and zoom controls
- Real-time network manipulation
- Hover information for detailed node data

### **4. Comprehensive Analysis**
- Complete ENCORE framework implementation
- Supply chain impact tracing
- Multi-sector portfolio comparison capabilities
- Risk assessment and pathway analysis

## Use Cases

### **1. Financial Risk Assessment**
- Identify climate and environmental risks in investment portfolios
- Assess exposure to ecosystem service dependencies
- Evaluate sustainability of economic activities

### **2. Portfolio Optimization**
- Compare environmental impact profiles across sectors
- Identify lower-impact alternatives for investment decisions
- Understand interconnected risks across portfolio holdings

### **3. Regulatory Compliance**
- Support TCFD (Task Force on Climate-related Financial Disclosures) reporting
- Align with EU Taxonomy and SFDR requirements
- Demonstrate environmental risk management

### **4. Educational and Research**
- Visualize complex environmental-economic relationships
- Support sustainability finance curriculum
- Enable academic research on portfolio environmental impacts

## Future Enhancements

### **Planned Features**
- Quantitative impact scoring system
- Time-series analysis for temporal impact tracking
- Integration with real portfolio data feeds
- Automated report generation
- Machine learning-based risk prediction

### **Technical Improvements**
- Performance optimization for large portfolios
- Additional layout algorithms
- Mobile-responsive visualizations
- Advanced filtering and search capabilities

## Conclusion

This Environmental Impact Portfolio Analysis System bridges the gap between complex environmental science data and practical financial decision-making. By transforming the comprehensive ENCORE database into interactive, intuitive visualizations, it empowers financial professionals to understand and manage environmental risks in their portfolios effectively.

The system's strength lies in its combination of scientific rigor (using the authoritative ENCORE framework) with practical usability (simple interface and interactive exploration). This makes sophisticated environmental impact analysis accessible to practitioners without requiring deep technical expertise in environmental science.

---

**Project Components:**
- `Portfolio_Dashboard_Simple.ipynb` - Main interactive interface
- `portfolio_impact_engine.py` - Core analysis engine
- `ENCORE_data/` - Complete environmental impact database
- `portfolio_exports/` - Analysis results and exports

**Requirements:**
- Python 3.8+
- Jupyter Notebook
- ipycytoscape, networkx, pandas, numpy
- ENCORE database (included)