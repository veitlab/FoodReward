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
              └── ...batch.csv
              └── ...keep_notselect.csv
              └── ...sequence.csv
              └── ...sequence_catch.csv
      ├── Bird2/
          └── Training_J6/
          └── ...
      ├── Bird3/
          └── ...
      └── Bird4/
          └── ...
  ├── click_accuracy.xlsx
scripts/
  ├── Figure_1.py
  ├── Figure_2.py
  ├── Figure_3.py
  ├── ...
  └── util/
      └── helper_fct.py
figures/
  ├── Figure_1/
          └── Fig1_C.svg
          └── ...
  ├── ...
requirements.txt
README.md
```
Scripts should be run from the *scripts* folder.    
The folder *figures* will be created once a script was run.

## Requirements 

Core dependencies:  
  - Python ≥ 3.9  
  - numpy, pandas, matplotlib, scipy, openpyxl, networkx, statsmodels


For all requirements see requirements.txt or use  
```bash
pip install -r requirements.txt
```

## Contact  
For questions, issues or feedback regarding the code, please contact:  
Franziska Heubach: franziska.heubach@student.uni-tuebingen.de  
