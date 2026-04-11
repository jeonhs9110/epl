import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib
import sys

# Optional: LightGBM (faster training, handles class imbalance natively)
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[INFO] LightGBM not installed. Using XGBoost only. (pip install lightgbm to enable)")

# Import your existing model logic
# Ensure prediction_model is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import prediction_model as pm
from prediction_model import LeagueAwareModel, get_master_data, get_dataloader, calculate_probabilities

DEVICE = pm.DEVICE
print(f"Using Device: {DEVICE}")

# CONFIG
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_DIR, 'models')
os.makedirs(_MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(_MODELS_DIR, 'FOBO_LEAGUE_AWARE_current.pth')
HYBRID_MODEL_PATH = os.path.join(_MODELS_DIR, 'xgb_classifier.json')
CALIBRATOR_PATH = os.path.join(_MODELS_DIR, 'calibrator.joblib')
LGBM_MODEL_PATH = os.path.join(_MODELS_DIR, 'lgbm_classifier.joblib')

def extract_embeddings_dataset(model, loader):
    model.eval()
    embeddings = []
    labels = []
    odds_data = []
    deltas = []
    
    print("Extracting Scaled Embeddings & Calculating Feature Deltas...")
    
    with torch.no_grad():
        for batch in loader:
            # Move to device
            b_h_seq = batch['h_seq'].to(DEVICE)
            b_a_seq = batch['a_seq'].to(DEVICE)
            b_h_id = batch['h_id'].to(DEVICE)
            b_a_id = batch['a_id'].to(DEVICE)
            b_l_id = batch['l_id'].to(DEVICE)
            b_odds = batch['odds'].to(DEVICE)
            b_h_elo = batch['h_elo'].to(DEVICE)
            b_a_elo = batch['a_elo'].to(DEVICE)
            
            # Target
            hg, ag = batch['hg'], batch['ag']
            
            # 1. Get Model Embeddings (Features)
            # [Batch, 2080] = 6*EMBED_DIM(256) + LEAGUE_EMBED_DIM(32) + 2*EMBED_DIM(256) tactical
            feats = model.extract_features(b_h_seq, b_a_seq, b_h_id, b_a_id, b_l_id, b_h_elo, b_a_elo)

            # 2. Get Model Predictions (Lambdas, Rho, xG) for Feature Delta
            # Model returns: lambdas, rho, xg_params, h_s, a_s
            lambdas, rho, _, _, _ = model(b_h_seq, b_a_seq, b_h_id, b_a_id, b_l_id, b_odds, b_h_elo, b_a_elo)
            
            # CPU conversion for loop
            lambdas_np = lambdas.cpu().numpy()
            rho_np = rho.cpu().numpy() # [B, 1] or [B]
            odds_np = b_odds.cpu().numpy()
            
            batch_deltas = []
            
            for i in range(len(lambdas_np)):
                h_lam = lambdas_np[i, 0]
                a_lam = lambdas_np[i, 1]
                
                # Check rho shape (could be [1] or scalar)
                r_val = rho_np[i] if rho_np.ndim > 0 else rho_np.item()
                if isinstance(r_val, np.ndarray): r_val = r_val.item()
                
                # Calculate True Model Probabilities
                probs = calculate_probabilities(h_lam, a_lam, r_val)
                p_home = probs['home_win']
                p_draw = probs['draw']
                p_away = probs['away_win']
                
                # Calculate Market Implied Probabilities
                o1 = odds_np[i, 0]
                ox = odds_np[i, 1]
                o2 = odds_np[i, 2]
                
                imp_home = (1.0 / o1) if o1 > 0 else 0.0
                imp_draw = (1.0 / ox) if ox > 0 else 0.0
                imp_away = (1.0 / o2) if o2 > 0 else 0.0
                
                # Feature Delta = Model - Market
                # Positive Delta = Model sees more chance than Market (Value)
                d_home = p_home - imp_home
                d_draw = p_draw - imp_draw
                d_away = p_away - imp_away
                
                # Also include raw probabilities as features?
                # The user asked for "Feature_Delta". Let's give that.
                # [DeltaH, DeltaD, DeltaA, ProbH, ProbD, ProbA]
                batch_deltas.append([d_home, d_draw, d_away, p_home, p_draw, p_away])
            
            # Collect Data
            embeddings.append(feats.cpu().numpy())
            odds_data.append(odds_np)
            deltas.append(np.array(batch_deltas))
            
            # Determine Labels (0: Home, 1: Draw, 2: Away)
            batch_labels = []
            for h, a in zip(hg, ag):
                if h > a: l = 0
                elif h == a: l = 1
                else: l = 2
                batch_labels.append(l)
            labels.append(np.array(batch_labels))
            
    # Concatenate
    X_emb = np.vstack(embeddings)
    X_odds = np.vstack(odds_data)
    X_deltas = np.vstack(deltas)
    y = np.concatenate(labels)
    
    # Combine Embeddings + Odds + Deltas for XGBoost Input
    # [N, 2080 + 3 + 6] = [N, 2089]
    X_final = np.hstack([X_emb, X_odds, X_deltas])
    
    return X_final, y

def train_hybrid():
    # 1. Get Data FIRST to build Adjacency Matrix
    print("Loading PyTorch Model...")
    _, le_team, le_league = get_master_data()
    if le_team is None: return False
    
    num_teams = len(le_team.classes_)
    num_leagues = len(le_league.classes_)

    train_loader, _ = get_dataloader(batch_size=512)
    if train_loader is None: return False
    
    # Access dataset to get Adjacency Matrix
    dataset = train_loader.dataset
    print(f"Dataset Loaded. Adj Matrix Shape: {dataset.adj.shape}")
    
    model = LeagueAwareModel(num_teams, num_leagues, dataset_adj=dataset.adj).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Loaded 'Best Loss' Model weights.")
        except RuntimeError as e:
            print(f"\n[ERROR] Model Mismatch detected: {e}")
            print(">> Triggering AUTO-RETRAINING of Deep Learning Model (train_dl.py)...")
            
            try:
                import train_dl
                success = train_dl.train_deep_model()
                if success:
                    print(">> Retraining Complete. Reloading weights...")
                    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                    print(">> Weights loaded successfully.")
                else:
                    return False
            except Exception as e2:
                print(f"Failed to auto-retrain: {e2}")
                return False
    else:
        print("Pre-trained model not found! Train PyTorch model first.")
        return False

    # 3. Extract Features
    X, y = extract_embeddings_dataset(model, train_loader)
    print(f"Data Prepared. Shape: {X.shape}")
    
    # 4. Calculate Class Weights for imbalanced data (Draw is typically underrepresented)
    from collections import Counter
    class_counts = Counter(y)
    total = len(y)
    # scale_pos_weight is for binary; for multi-class we pass sample_weight
    sample_weights = np.array([total / (3 * class_counts[yi]) for yi in y])

    # Split for validation AND calibration (always needed for calibrator)
    from sklearn.model_selection import train_test_split
    X_train_full, X_calib, y_train_full, y_calib, sw_train_full, sw_calib = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
        X_train_full, y_train_full, sw_train_full, test_size=0.1, random_state=42
    )
    X_calib_fit, X_calib_eval, y_calib_fit, y_calib_eval = train_test_split(
        X_calib, y_calib, test_size=0.15, random_state=0
    )

    # 4a. Train XGBoost (or load existing if skip requested)
    skip_xgb = os.environ.get('FOBO_SKIP_XGB_TRAIN', 'false').lower() == 'true'
    clf = xgb.XGBClassifier(objective='multi:softprob', num_class=3)

    if skip_xgb and os.path.exists(HYBRID_MODEL_PATH):
        print(">> Loading existing XGBoost model (skipping training)...")
        clf.load_model(HYBRID_MODEL_PATH)
    else:
        print("Training XGBoost Classifier (Hybrid Ensemble)...")
        tree_method = 'hist'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"XGBoost Config: Tree Method='{tree_method}', Device='{device}'")
        clf.set_params(
            n_estimators=800, learning_rate=0.04, max_depth=6,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.80,
            reg_alpha=0.1, reg_lambda=1.2, tree_method=tree_method,
            device=device, eval_metric='mlogloss', early_stopping_rounds=30
        )
        print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Calib-fit={len(X_calib_fit)}, Calib-eval={len(X_calib_eval)}")
        clf.fit(X_train, y_train, sample_weight=sw_train,
                eval_set=[(X_val, y_val)], verbose=50)
    
    # 4b. Optionally train LightGBM as second ensemble member
    clf_lgb = None
    if LGBM_AVAILABLE:
        print("\nTraining LightGBM Classifier (secondary ensemble member)...")
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=800,
            num_leaves=63,
            learning_rate=0.04,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight='balanced',
            objective='multiclass',
            num_class=3,
            device='gpu' if torch.cuda.is_available() else 'cpu',
            verbose=-1
        )
        clf_lgb.fit(X_train, y_train, sample_weight=sw_train)
        lgb_acc = accuracy_score(y_calib_eval, clf_lgb.predict(X_calib_eval))
        print(f"  LightGBM Calibration Accuracy: {lgb_acc:.4f}")
        joblib.dump(clf_lgb, LGBM_MODEL_PATH)

    # 5. Save XGB immediately — before anything else can crash
    clf.save_model(HYBRID_MODEL_PATH)
    print(f"XGBoost model saved to {HYBRID_MODEL_PATH}")

    # 6. Evaluate Uncalibrated
    y_pred = clf.predict(X_calib_eval)
    print("\n[Uncalibrated XGB] Classification Report:")
    print(classification_report(y_calib_eval, y_pred, target_names=['Home', 'Draw', 'Away']))

    # 7. Probability Calibration — manual Platt scaling via LogisticRegression on XGB's
    # predicted probabilities. This never refits XGB, avoiding the eval_set/early-stopping issue.
    print("\nTraining Probability Calibrator (Sigmoid/Platt Scaling)...")
    from sklearn.linear_model import LogisticRegression
    probs_uncal_fit = clf.predict_proba(X_calib_fit)
    platt = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    platt.fit(probs_uncal_fit, y_calib_fit)
    joblib.dump(platt, CALIBRATOR_PATH)

    # Evaluate on HELD-OUT calib-eval set
    probs_uncal = clf.predict_proba(X_calib_eval)
    probs_cal   = platt.predict_proba(probs_uncal)

    # One-hot y for brier score
    y_calib_oh = np.eye(3)[y_calib_eval]

    brier_uncal = np.mean((probs_uncal - y_calib_oh)**2)
    brier_cal   = np.mean((probs_cal   - y_calib_oh)**2)

    print(f"Brier Score (held-out, lower = better): Uncalibrated={brier_uncal:.4f}, Calibrated={brier_cal:.4f}")
    if brier_uncal > 0:
        print(f"Calibration Improvement: {((brier_uncal - brier_cal)/brier_uncal)*100:.2f}%")
    
    if os.path.exists(HYBRID_MODEL_PATH) and os.path.exists(CALIBRATOR_PATH):
        print(f"Saved models to {HYBRID_MODEL_PATH} and {CALIBRATOR_PATH}")
        return True
    else:
        print("Error: Model file check failed.")
        return False

if __name__ == "__main__":
    train_hybrid()
