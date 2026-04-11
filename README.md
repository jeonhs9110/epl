# FOBO AI — Football Outcome Betting Optimiser

## Quick Start (At Home — Skip All Training)

```bash
git clone https://github.com/jeonhs9110/epl.git
cd epl
```

**Install PyTorch:**
```bash
# GPU (NVIDIA CUDA 12.4)
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# CPU only (no GPU at home)
pip install torch
```

**Install other dependencies:**
```bash
pip install flask pandas numpy scikit-learn xgboost lightgbm joblib scipy matplotlib selenium
```

**Run:**
```bash
python run_pipeline.py
```

**Answer the prompts exactly like this to skip all training and go straight to Flask:**
```
Enable POWER MODE? → n
Scrape? → s
Upcoming fixtures? → n
Retrain? → y
TEST MODE? → n
OPTIMIZE SEQUENCE LENGTHS? → n
SKIP DL Training? → y
SKIP PPO Training? → y
SKIP XGBoost training? → y
```

Flask will start at → http://localhost:5000

---

## Leagues Covered
Premier League, Championship, La Liga, La Liga 2, Bundesliga, 2. Bundesliga,
Serie A, Serie B, Ligue 1, Ligue 2, Eredivisie, Champions League, Europa League

## Models (saved in `models/`)
| File | Description |
|------|-------------|
| `FOBO_LEAGUE_AWARE_current.pth` | Best-loss DL model (TransformerGNN) |
| `FOBO_LEAGUE_AWARE_best_acc.pth` | Best-accuracy DL model (92.3%) |
| `ppo_agent.pth` | PPO reinforcement learning agent |
| `xgb_classifier.json` | XGBoost hybrid ensemble |
| `lgbm_classifier.joblib` | LightGBM hybrid ensemble |
| `calibrator.joblib` | Platt probability calibrator |

## Full Documentation
See `FOBO_AI_Pipeline_Documentation.pdf` for architecture diagrams and design rationale.
