"""
Offline RL for Optimal Carbon Type Selection
=============================================

This script integrates data preprocessing, augmentation (imitative data),
offline batch Q-learning (fitted Q-iteration with a target network), and
a simple policy interface to recommend the optimal carbon type that maximizes
battery capacity using an offline dataset of cycling experiments.

Assumptions & Notes
-------------------
- Reward: 'capacity_mAh_g'
- Action: 'carbon_type' (categorical)
- State features (provided by user):
    - Numeric: 'coulombic_eff_percent', 'active_material_amount_mg', 'pressure_MPa',
               'temperature_C', 'c_rate', 'state_of_helath_percent',
               'depth_of_discharge_percent', 'cycle_index'
    - Categorical: 'manufacturing_quality_note', 'anode', 'cathode', 'active_material'
- Cycle horizon H is max of 'cycle_index' in the dataset.
- Episodes are defined by grouping over all static experimental conditions including
  action 'carbon_type' and sorting by 'cycle_index'.
- Augmentation: small Gaussian noise on numeric state features and reward.

Outputs
-------
- Trains a Q-network to approximate Q(s, a) over the offline dataset.
- Provides functions:
    - recommend_carbon(state_dict) -> (best_carbon, q_values_dict)
    - verify_on_state(state_dict) -> prints ranked carbon types by Q-value.
- Optionally saves artifacts (encoders, scalers, label encoder) for reuse.

Usage
-----
$ python optimal_carbon_offline_rl.py --data /path/to/cycling_data.xlsx

If no --data is given, defaults to 'cycling_data.xlsx' in the current directory.

"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

# Scikit-learn imports (version compatibility for OneHotEncoder)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Torch (offline RL, Q-network). If not installed, exit with a helpful message.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise SystemExit(
        "This script requires PyTorch. Please install it via "
        "'pip install torch' (CPU) or the appropriate CUDA build.\n"
        f"Import error was: {e}"
    )

# --------------------------
# Configuration & Constants
# --------------------------
NUMERIC_STATE_COLS = [
    "coulombic_eff_percent",
    "active_material_amount_mg",
    "pressure_MPa",
    "temperature_C",
    "c_rate",
    "state_of_health_percent",
    "depth_of_discharge_percent",
    "cycle_index",               # include position in the trajectory
]

CATEGORICAL_STATE_COLS = [
    "manufacturing_quality_note",
    "anode",
    "cathode",
    "active_material",
]

ACTION_COL = "carbon_type"
REWARD_COL = "capacity_mAh_g"
CYCLE_COL = "cycle_index"

# Q-learning hyperparameters
GAMMA = 0.99
HIDDEN = 128
LR = 1e-3
BATCH_SIZE = 256
EPOCHS = 20
TARGET_UPDATE_EVERY = 1  # epochs


# --------------------------
# Utility Functions
# --------------------------
def safe_fill_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical values with 'unknown'."""
    for col in CATEGORICAL_STATE_COLS + [ACTION_COL]:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("unknown")
    return df


def safe_fill_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with medians or reasonable defaults.
    For SoH, if entire episode missing, derive from capacity relative to peak.
    """
    # Fill standard numeric columns with medians
    for col in NUMERIC_STATE_COLS:
        if col in df.columns:
            if df[col].isna().all():
                # fallback default
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(df[col].median())

    # Derive SoH if still missing in some rows using per-(anode,cathode,action) groups
    if "state_of_health_percent" in df.columns:
        soh_mask = df["state_of_health_percent"].isna()
        if soh_mask.any() and REWARD_COL in df.columns:
            group_cols = ["anode", "cathode", ACTION_COL]
            # Fill missing group cols to avoid grouping on NaN
            for g in group_cols:
                if g not in df.columns:
                    df[g] = "unknown"
            peak_cap_by_group = df.groupby(group_cols)[REWARD_COL].transform("max").replace(0, np.nan)
            derived_soh = (df[REWARD_COL] / peak_cap_by_group) * 100.0
            df.loc[soh_mask, "state_of_health_percent"] = derived_soh[soh_mask]
            # If still NaN, fill with overall median
            df["state_of_health_percent"] = df["state_of_health_percent"].fillna(df["state_of_health_percent"].median())

    # Ensure cycle index present
    if CYCLE_COL in df.columns:
        df[CYCLE_COL] = df[CYCLE_COL].fillna(1).astype(int).clip(lower=1)
    return df


def build_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure episodes are definable and sorted by cycle index.
    Episode grouping keys: all static conditions incl. action.
    """
    missing_cols = [c for c in (NUMERIC_STATE_COLS + CATEGORICAL_STATE_COLS + [ACTION_COL, REWARD_COL, CYCLE_COL]) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in the dataset: {missing_cols}")

    # Group keys include static conditions + action (carbon_type)
    group_keys = [
        "anode", "cathode", "active_material", "active_material_amount_mg",
        "pressure_MPa", "temperature_C", "c_rate", "manufacturing_quality_note",
        ACTION_COL
    ]
    # Ensure these columns exist
    for g in group_keys:
        if g not in df.columns:
            if g in CATEGORICAL_STATE_COLS:
                df[g] = "unknown"
            else:
                df[g] = 0.0

    # Sort within each episode by cycle index
    df_sorted = df.sort_values(group_keys + [CYCLE_COL]).reset_index(drop=True)
    return df_sorted


def fit_state_encoders(df: pd.DataFrame):
    """Fit encoders for categorical state variables and compute numeric stats."""
    # OneHotEncoder sparse_output was introduced in sklearn>=1.2; handle compat
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")

    enc.fit(df[CATEGORICAL_STATE_COLS])
    cat_feature_names = enc.get_feature_names_out(CATEGORICAL_STATE_COLS)

    # Numeric scaling stats
    numeric = df[NUMERIC_STATE_COLS].astype(float)
    means = numeric.mean()
    stds = numeric.std().replace(0, 1.0)  # avoid division by zero
    return enc, cat_feature_names, means, stds


def make_state_matrix(df: pd.DataFrame, enc, means, stds) -> np.ndarray:
    """Create full state feature matrix by concatenating standardized numeric and one-hot categorical."""
    numeric = df[NUMERIC_STATE_COLS].astype(float)
    numeric_std = (numeric - means) / stds
    cat = df[CATEGORICAL_STATE_COLS].astype(str)
    cat_oh = enc.transform(cat)
    X = np.concatenate([numeric_std.values, cat_oh], axis=1)
    return X


def build_transitions(df_sorted: pd.DataFrame, state_mat: np.ndarray, action_labels: np.ndarray):
    """Construct offline transitions (s, a, r, s_next, done) from sorted episodes."""
    transitions = []
    # Group episode with same static conditions + action
    group_keys = [
        "anode", "cathode", "active_material", "active_material_amount_mg",
        "pressure_MPa", "temperature_C", "c_rate", "manufacturing_quality_note",
        ACTION_COL
    ]
    for _, group in df_sorted.groupby(group_keys, dropna=False):
        # within each group, they are already sorted by cycle_index
        idxs = group.index.to_list()
        for i, idx in enumerate(idxs):
            s = state_mat[idx]
            a = action_labels[idx]
            r = float(group.loc[idx, REWARD_COL])
            if i < len(idxs) - 1:
                s_next = state_mat[idxs[i + 1]]
                done = False
            else:
                s_next = np.zeros_like(s)
                done = True
            transitions.append((s, a, r, s_next, done))
    return transitions


def augment_transitions(transitions, num_numeric_features: int, aug_factor: float = 1.0):
    """Create imitative augmented transitions by adding small Gaussian noise to numeric state features and reward.

    aug_factor: multiplier on dataset size (1.0 -> creates same number of augmented samples as originals)
    """
    if aug_factor <= 0:
        return []
    aug_trans = []
    n = len(transitions)
    k = int(n * aug_factor)
    rng = np.random.default_rng(42)
    for i in range(k):
        s, a, r, s_next, done = transitions[i % n]
        s_aug = s.copy()
        s_next_aug = s_next.copy()
        # add small noise to numeric slice [0:num_numeric_features)
        s_aug[:num_numeric_features] += rng.normal(0, 0.05, size=num_numeric_features)
        if not done:
            s_next_aug[:num_numeric_features] += rng.normal(0, 0.05, size=num_numeric_features)
        r_aug = float(r * (1.0 + rng.normal(0, 0.01)))
        aug_trans.append((s_aug, a, r_aug, s_next_aug, done))
    return aug_trans


# --------------------------
# Q-Network & Training
# --------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_q_network(transitions, state_dim: int, action_dim: int,
                    gamma: float = GAMMA, lr: float = LR,
                    batch_size: int = BATCH_SIZE, epochs: int = EPOCHS,
                    target_update_every: int = TARGET_UPDATE_EVERY,
                    device: str = "cpu"):
    """Train a Q-network on offline transitions using fitted Q-iteration with a target network."""
    # Prepare tensors
    states = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
    actions = torch.tensor([t[1] for t in transitions], dtype=torch.long)
    rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    next_states = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
    dones = torch.tensor([t[4] for t in transitions], dtype=torch.bool)

    states, actions, rewards, next_states, dones = [
        t.to(device) for t in (states, actions, rewards, next_states, dones)
    ]

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    N = states.size(0)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            s_b = states[idx]
            a_b = actions[idx]
            r_b = rewards[idx]
            ns_b = next_states[idx]
            d_b = dones[idx]

            q_vals = q_net(s_b)
            q_sa = q_vals.gather(1, a_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_vals = target_net(ns_b)
                max_next_q, _ = next_q_vals.max(dim=1)
                target = r_b + gamma * max_next_q * (~d_b)

            loss = criterion(q_sa, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if epoch % target_update_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        avg_loss = total_loss / max(1, num_batches)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[Epoch {epoch:02d}] avg Bellman error: {avg_loss:.6f}")

    return q_net


# --------------------------
# Policy Inference Utilities
# --------------------------
class Policy:
    """Wraps the trained Q-network and feature encoders to provide recommendations."""
    def __init__(self, q_net, enc, means, stds, label_encoder, device="cpu"):
        self.q_net = q_net
        self.enc = enc
        self.means = means
        self.stds = stds.replace(0, 1.0)
        self.le = label_encoder
        self.device = device

    def _state_to_vec(self, state_dict: dict) -> np.ndarray:
        # Build numeric vector in order
        numeric_vals = []
        for col in NUMERIC_STATE_COLS:
            val = state_dict.get(col, np.nan)
            if pd.isna(val):
                # fallback to mean if missing
                val = float(self.means.get(col, 0.0))
            # standardize
            mu = float(self.means.get(col, 0.0))
            sd = float(self.stds.get(col, 1.0)) or 1.0
            numeric_vals.append((float(val) - mu) / sd)
        # Build categorical vector (1 row)
        cat_vals = [str(state_dict.get(col, "unknown")) for col in CATEGORICAL_STATE_COLS]
        cat_oh = self.enc.transform([cat_vals])[0]
        return np.concatenate([np.array(numeric_vals, dtype=float), cat_oh], axis=0)

    def recommend(self, state_dict: dict):
        """Return (best_carbon, {carbon: q_value}) for a given state."""
        x = self._state_to_vec(state_dict)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(x_t).cpu().numpy().flatten()
        best_idx = int(np.argmax(q_vals))
        best_carbon = self.le.inverse_transform([best_idx])[0]
        return best_carbon, {c: float(q) for c, q in zip(self.le.classes_, q_vals)}

    def verify_on_state(self, state_dict: dict):
        """Prints carbon types ranked by predicted Q-value for the given state."""
        best_carbon, q_map = self.recommend(state_dict)
        print("Predicted Q-values by carbon type (descending):")
        for k, v in sorted(q_map.items(), key=lambda kv: -kv[1]):
            print(f"  {k:>12s}: {v:.4f}")
        print(f"\nRecommended optimal carbon: {best_carbon}")


# --------------------------
# Main Pipeline
# --------------------------
def object_to_array(obj_list):
    """Helper to convert a list of arrays/objects to an ndarray with dtype=object for saving."""
    return np.array([np.array(x, dtype=object) for x in obj_list], dtype=object)


def main(args):
    data_path = args.data if args.data else "cycling_data.xlsx"
    if not os.path.exists(data_path):
        raise SystemExit(f"Data file not found: {data_path}")

    # Load dataset
    df = pd.read_excel(data_path)
    print(f"Loaded dataset: {data_path} with shape {df.shape}")

    # Basic sanity checks
    required_cols = set(NUMERIC_STATE_COLS + CATEGORICAL_STATE_COLS + [ACTION_COL, REWARD_COL])
    missing = required_cols.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # Fill missing values
    df = safe_fill_categoricals(df)
    df = safe_fill_numerics(df)

    # Determine horizon H
    H = int(df[CYCLE_COL].max())
    print(f"Detected finite-horizon length H = {H} (max cycle_index)")

    # Build episode-sorted dataframe
    df_sorted = build_episodes(df)

    # Fit encoders and build state matrix
    enc, cat_feature_names, means, stds = fit_state_encoders(df_sorted)
    X = make_state_matrix(df_sorted, enc, means, stds)
    num_numeric_features = len(NUMERIC_STATE_COLS)

    # Encode action labels
    le = LabelEncoder()
    action_labels = le.fit_transform(df_sorted[ACTION_COL].astype(str))
    print(f"Discovered {len(le.classes_)} carbon types:", list(le.classes_))

    # Build offline transitions
    transitions = build_transitions(df_sorted, X, action_labels)
    print(f"Built {len(transitions)} offline transitions from {len(df_sorted)} rows.")

    # Augmentation (imitative data)
    aug = augment_transitions(transitions, num_numeric_features, aug_factor=args.aug_factor)
    offline_dataset = transitions + aug
    print(f"Total offline dataset size after augmentation: {len(offline_dataset)} "
          f"(aug_factor={args.aug_factor})")

    # Train Q-network
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Training device: {device}")
    q_net = train_q_network(
        offline_dataset,
        state_dim=X.shape[1],
        action_dim=len(le.classes_),
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        target_update_every=args.target_update,
        device=device,
    )

    # Wrap policy
    policy = Policy(q_net, enc, means, stds, le, device=device)

    # Demo recommendation using dataset medians/modes at cycle 1
    # After loading df in main():
    demo_state = {
        "cycle_index": 1,
        "coulombic_eff_percent": 98.02,
        "active_material_amount_mg": 15.723,
        "pressure_MPa": 5.0,
        "temperature_C": 25.7,
        "c_rate": 2.0,  # high discharge rate (75th percentile for CNT)
        "state_of_health_percent": 85.6,
        "depth_of_discharge_percent": 80.0,
        "manufacturing_quality_note": "standard coating; high purity",
        "anode": "Li-metal",
        "cathode": "NMC811 composite",
        "active_material": "Li2S"
    }

    print("\nThe input feature is given by:")
    print(f"\nCycle index: {demo_state['cycle_index']}")
    print(f"Coulombic efficiency: {demo_state['coulombic_eff_percent']}")
    print(f"Active material: {demo_state['active_material']}")
    print(f"The amount of active material: {demo_state['active_material_amount_mg']}")
    print(f"Pressure: {demo_state['pressure_MPa']}")
    print(f"Temperature: {demo_state['temperature_C']}")
    print(f"Discharge rate: {demo_state['c_rate']}")
    print(f"State of health: {demo_state['state_of_health_percent']}")
    print(f"Depth of discharge: {demo_state['depth_of_discharge_percent']}")
    print(f"Manufacturing quality: {demo_state['manufacturing_quality_note']}")
    print(f"Anode and cathode: {demo_state['anode']} and {demo_state['cathode']}\n")
    print("\n=== Demo Recommendation on a baseline state ===")
    best_carbon, q_map = policy.recommend(demo_state)
    print(f"Recommended carbon type: {best_carbon}")
    policy.verify_on_state(demo_state)

    # Optionally save artifacts
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(q_net.state_dict(), os.path.join(args.save_dir, "q_net.pt"))
        # Save encoders & scalers using numpy (simple npz)
        np.savez(
            os.path.join(args.save_dir, "artifacts.npz"),
            means=means.values.astype(float),
            stds=stds.values.astype(float),
            numeric_cols=np.array(NUMERIC_STATE_COLS, dtype=object),
            cat_cols=np.array(CATEGORICAL_STATE_COLS, dtype=object),
            classes=np.array(le.classes_, dtype=object),
            # Save OneHotEncoder categories_
            ohe_categories=object_to_array(enc.categories_),
        )
        print(f"Saved model and artifacts to: {args.save_dir}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline RL for optimal carbon type discovery")
    parser.add_argument("--data", type=str, default="cycling_data.xlsx",
                        help="Path to cycling_data.xlsx")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--target_update", type=int, default=TARGET_UPDATE_EVERY,
                        help="How often (epochs) to copy weights to target network")
    parser.add_argument("--aug_factor", type=float, default=1.0,
                        help="Augmented transitions as fraction of original (e.g., 1.0 doubles the data)")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save model artifacts")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)
