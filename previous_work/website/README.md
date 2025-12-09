# Portfolio Environmental Impact Analysis Tool

A web-based interactive visualization tool for analyzing environmental impacts of investment portfolios using the ENCORE (Exploring Natural Capital Opportunities, Risks and Exposure) framework.

## Features

- **Portfolio Analysis**: Analyze environmental impacts across 5 key sectors (Energy, Agriculture, Technology, Financial, Manufacturing)
- **Interactive Visualization**: Circular flow network visualization showing the complete ENCORE system
- **Dark Theme Interface**: Professional, clean interface optimized for data analysis
- **Responsive Design**: Works on desktop and mobile devices

## Technology

- **Frontend**: HTML5, CSS3, JavaScript ES6
- **Visualization**: Plotly.js for interactive network graphs
- **Hosting**: Netlify for fast, reliable deployment
- **Framework**: Based on ENCORE environmental impact database

## Usage

1. Select a portfolio type from the dropdown
2. Toggle label visibility as needed
3. Explore the circular flow: Portfolio → Activities → Pressures → Mechanisms → Natural Capital → Ecosystem Services
4. Use interactive features to pan, zoom, and examine impact relationships

## Circular Flow Model

The visualization implements the complete ENCORE circular system:
- **Economic Activities** → generate **Environmental Pressures**
- **Pressures** → cause **Mechanisms of Change**
- **Mechanisms** → impact **Natural Capital**
- **Natural Capital** → provides **Ecosystem Services**
- **Ecosystem Services** → support **Economic Activities** (completing the circle)

This creates a comprehensive view of how investment portfolios interact with and depend on natural systems.

## Data Source

Based on the ENCORE database developed by Global Canopy and UNEP-WCMC, providing comprehensive environmental impact and dependency data across economic sectors.

## Development

This tool is part of a larger sustainable finance analysis system that includes:
- Jupyter notebook interfaces for detailed analysis
- Python backend with NetworkX graph algorithms
- Comprehensive ENCORE database integration
- Export capabilities for research and reporting

## License

Academic project - Sustainable Finance and Networks course