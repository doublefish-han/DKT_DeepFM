from pathlib import Path
import pandas as pd, numpy as np, torch

BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN = BASE_DIR / "data/processed/deepfm_input.csv"  # contains split column
HSTATE= BASE_DIR / "data/processed/dkt_hidden_states.pt"
OUT   = BASE_DIR / "data/processed/deepfm_with_mastery.csv"          # non-leaky (h_{t-1})
OUT_LEAKY = BASE_DIR / "data/processed/deepfm_with_mastery_leaky.csv" # leaky (h_t), for audit only
SPLIT_DIR = BASE_DIR / "data/processed/train_test_split"
BASE_FEATURES = ["user_id","problem_id","skill_id","weekday","hour","opportunity","duration","correct"]

def main():
    if not CLEAN.exists():
        raise FileNotFoundError(f"Clean dataset not found: {CLEAN}")
    if not HSTATE.exists():
        raise FileNotFoundError(f"Hidden state file not found: {HSTATE}")

    df = pd.read_csv(CLEAN)
    if "split" not in df.columns:
        raise KeyError("Expected 'split' column in deepfm_input.csv; rerun preprocessing.")
    
    df["idx"] = df.groupby("user_id").cumcount()

    
    H = torch.load(HSTATE)          # list of [Li, H]
    
    user_order = df.groupby("user_id", sort=False).size().index.tolist()
    assert len(user_order)==len(H), "User count does not match the length of DKT hidden states. Please check the preprocessing order."

    
    K = min(16, H[0].shape[1])
    user2H = {u: h[:, :K].numpy() for u, h in zip(user_order, H)}

    mats = []
    mats_leaky = []
    for u, g in df.groupby("user_id", sort=False):
        mat = user2H[u]
        L = len(g)
        
        if mat.shape[0] < L:
            pad = np.repeat(mat[-1][None, :], L - mat.shape[0], axis=0)
            mat = np.concatenate([mat, pad], axis=0)
        else:
            mat = mat[:L]
        
        shifted = np.vstack([np.zeros((1, mat.shape[1]), dtype=mat.dtype), mat[:-1]]) if L > 0 else mat
        
        unshifted = mat

        mats.append(
            pd.DataFrame(shifted, columns=[f"mastery_{i}" for i in range(K)]).set_index(g.index)
        )
        mats_leaky.append(
            pd.DataFrame(unshifted, columns=[f"mastery_{i}" for i in range(K)]).set_index(g.index)
        )
    mastery = pd.concat(mats).sort_index()
    mastery_leaky = pd.concat(mats_leaky).sort_index()

    out = pd.concat([df, mastery], axis=1)
    out_leaky = pd.concat([df, mastery_leaky], axis=1)

    
    mastery_cols = [c for c in mastery.columns]
    keep = BASE_FEATURES + ["split"] + mastery_cols

    fused = out[keep]
    fused.to_csv(OUT, index=False)
    print("Saved:", OUT)

    fused_leaky = out_leaky[keep]
    fused_leaky.to_csv(OUT_LEAKY, index=False)
    print("Saved (leaky audit only):", OUT_LEAKY)

    
    split_files = {
        "train": SPLIT_DIR / "train.csv",
        "val": SPLIT_DIR / "val.csv",
        "test": SPLIT_DIR / "test.csv",
    }
    if all(p.exists() for p in split_files.values()):
        for name, split_path in split_files.items():
            split_df = pd.read_csv(split_path)
            try:
                merged = split_df.merge(fused, on=BASE_FEATURES, how="left", validate="one_to_one")
                merged_leaky = split_df.merge(fused_leaky, on=BASE_FEATURES, how="left", validate="one_to_one")
            except ValueError as err:
                raise ValueError(
                    f"Failed to align {name} split with mastery features. "
                    "Please check for duplicate rows or missing key columns."
                ) from err
            missing = merged[mastery_cols].isna().any(axis=1).sum()
            if missing:
                raise ValueError(
                    f"{name} split has {missing} rows without mastery features after alignment."
                )
            out_path = SPLIT_DIR / f"{name}_with_mastery.csv"
            merged.to_csv(out_path, index=False)
            print("Saved aligned split:", out_path)

            missing_leaky = merged_leaky[mastery_cols].isna().any(axis=1).sum()
            if missing_leaky:
                raise ValueError(
                    f"{name} split has {missing_leaky} rows without mastery features after leaky alignment."
                )
            out_path_leaky = SPLIT_DIR / f"{name}_with_mastery_leaky.csv"
            merged_leaky.to_csv(out_path_leaky, index=False)
            print("Saved aligned split (leaky audit only):", out_path_leaky)
    else:
        print(f"Warning: split directory {SPLIT_DIR} missing train/val/test CSV; skipped aligned outputs.")

if __name__ == "__main__":
    main()
