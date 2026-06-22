# FoodReward - 2026

Code and data repository accompanying the manuscript "Birdsong Modification with Food Reward".  
Authors: Franziska Heubach & Lena Veit, 2026

**Note:** Figures can look aesthetically different from those in the manuscript due to changes later performed in Inkscape.

## Repository Structure
```text
data/
  ├── sequences/
      ├── Bird1/
          └── Training_bkd/
      ├── Bird2/
          └── Training_J6/
          └── ...
      ├── Bird3/
          └── ...
      ├── Bird4/
          └── ...
  ├── click_accuracy.xlsx
scripts/
  ├── Figure_1.py
  ├── Figure_2.py
  ├── Figure_3.py
  └── util/
      └── helper_fct.py
requirements.txt
README.md
```

## Requirements 

Core dependencies:  
  - Python ≥ 3.9  
  - numpy, pandas, matplotlib, scipy, openpyxl, networkx, statsmodels


For all requirements see requirements.txt or use  
```bash
pip install -r requirements.txt
```
