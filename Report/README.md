# LaTeX Report Compilation Guide

## Quick Start

```bash
# Navigate to Report directory
cd Report

# Compile with pdflatex + biber (recommended)
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Alternative: latexmk (handles everything automatically)

```bash
latexmk -pdf main.tex
```

## Structure

```
Report/
├── main.tex          # Main document with all sections
├── references.bib    # Bibliography in BibLaTeX format
└── README.md         # This file
```

## Figure Paths

The document is configured to find figures from:
- `../Portfolio_Analysis/Cross_Sector_Impact/output_plots/`
- `../Portfolio_Analysis/OilGas/`
- `../Portfolio_Analysis/FoodBeverage/`

## Sections to Complete

1. **Abstract** - Summary of research
2. **Section 1: Introduction** - Motivation and research question
3. **Section 2: Background** - TNFD, ENCORE, nature-related risks
4. **Section 3: Methodology** - Network construction, presets, portfolio selection
5. **Section 4: Cross-Sector Impact Analysis** - Figures already included
6. **Section 5: Oil & Gas Portfolio Analysis** - Figures already included
7. **Section 6: Food & Beverage Portfolio Analysis** - Figures already included
8. **Section 7: Discussion** - Comparative analysis, implications
9. **Section 8: Conclusion** - Key findings, limitations, future work
10. **Appendix** - Data sources, full holdings lists

## Notes

- All `% TODO:` comments mark sections that need your input
- Figure placement uses `[H]` for strict positioning
- Tables use `booktabs` for professional styling
- Bibliography uses `authoryear` citation style
