import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import pickle

# ---------- helpers ----------

CANDIDATE_COLS = {
    "user_id":      ["user_id", "student", "student_id", "anon_student_id", "studentId"],
    "problem_id":   ["problem_id", "problem", "problemId", "item_id", "item"],
    "skill_id":     ["skill_id", "skill", "kc", "kc_id", "knowledge_component", "skill_name"],
    "correct":      ["correct", "is_correct", "label", "outcome", "first_attempt"],
    "timestamp":    ["timestamp", "time", "ms_first_response", "start_time", "order_id", "seq"],
    "opportunity":  ["opportunity", "opportunity_count", "opportunity_skill", "attempt_count"],
    "duration":     ["time_taken", "timeTaken", "duration", "ms_response_time"]
}

def find_col(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    # try case-insensitive
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_time(series):
    """Try best-effort timestamp parsing; fall back to numeric/ordinal."""
    # if already numeric, return as is
    if np.issubdtype(series.dtype, np.number):
        return pd.to_datetime(series, unit="ms", errors="coerce")
    # try iso/datetime
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().all():
        # cannot parse -> create synthetic order by original index
        return pd.to_datetime(np.arange(len(series)), unit="s")
    return dt

# ---------- main ----------

def main(args):
    raw_csv = os.path.join(args.data_root, "raw", "anonymized_full_release_competition_dataset.csv")
    out_dir = os.path.join(args.data_root, "processed")
    ensure_dir(out_dir)
    split_dir = os.path.join(out_dir, "train_test_split")
    ensure_dir(split_dir)

    print(f"🔹 Loading: {raw_csv}")
    df = pd.read_csv(raw_csv, low_memory=False)

    # map columns
    col_user = find_col(df, CANDIDATE_COLS["user_id"])
    col_item = find_col(df, CANDIDATE_COLS["problem_id"])
    col_skill = find_col(df, CANDIDATE_COLS["skill_id"])
    col_corr = find_col(df, CANDIDATE_COLS["correct"])
    col_time = find_col(df, CANDIDATE_COLS["timestamp"])
    col_opp  = find_col(df, CANDIDATE_COLS["opportunity"])
    col_dur  = find_col(df, CANDIDATE_COLS["duration"])

    required = [col_user, col_item, col_skill, col_corr]
    if any(c is None for c in required):
        raise ValueError(
            f"Missing required columns. Found: user={col_user}, item={col_item}, "
            f"skill={col_skill}, correct={col_corr}"
        )

    # rename to standard names
    rename_map = {}
    if col_user  != "user_id":    rename_map[col_user]  = "user_id"
    if col_item  != "problem_id": rename_map[col_item]  = "problem_id"
    if col_skill != "skill_id":   rename_map[col_skill] = "skill_id"
    if col_corr  != "correct":    rename_map[col_corr]  = "correct"
    if col_time  and col_time  != "timestamp":  rename_map[col_time]  = "timestamp"
    if col_opp   and col_opp   != "opportunity":rename_map[col_opp]   = "opportunity"
    if col_dur   and col_dur   != "duration":   rename_map[col_dur]   = "duration"

    df = df.rename(columns=rename_map)

    # keep only relevant columns
    keep_cols = ["user_id", "problem_id", "skill_id", "correct"]
    if "timestamp"  in df.columns: keep_cols.append("timestamp")
    if "opportunity" in df.columns: keep_cols.append("opportunity")
    if "duration"    in df.columns: keep_cols.append("duration")
    df = df[keep_cols]

    # drop NA and invalid
    df = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct"])
    # coerce correct to {0,1}
    df["correct"] = df["correct"].astype(float).round().clip(0,1).astype(int)

    # timestamp
    if "timestamp" in df.columns:
        dt = parse_time(df["timestamp"])
        # if parse failed entirely, create synthetic order per user:
        if dt.isna().all():
            print("⚠️  timestamp unparsable; using per-user sequence index as time.")
            df["timestamp"] = df.groupby("user_id").cumcount()
        else:
            df["timestamp"] = dt.view("int64")  # nanoseconds integer for sorting
    else:
        # create synthetic order within each user
        df["timestamp"] = df.groupby("user_id").cumcount()

    # simple duplicates removal
    df = df.drop_duplicates(subset=["user_id","timestamp","problem_id"]).reset_index(drop=True)

    # sort by user, time
    df = df.sort_values(["user_id","timestamp"]).reset_index(drop=True)

    # Label-encode ids (save encoders for reproducibility)
    encoders = {}
    for col in ["user_id","problem_id","skill_id"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, os.path.join(out_dir, "encoders.joblib"))

    # enrich simple calendar/context if timestamp was real time
    try:
        # back to datetime for features
        dt = pd.to_datetime(df["timestamp"])
        df["weekday"] = dt.dt.weekday
        df["hour"]    = dt.dt.hour
    except Exception:
        df["weekday"] = 0
        df["hour"]    = 0

    # fill optional columns
    if "opportunity" not in df.columns:
        df["opportunity"] = df.groupby(["user_id","skill_id"]).cumcount()
    if "duration" not in df.columns:
        df["duration"] = np.nan

    # save clean full
    clean_path = os.path.join(out_dir, "assist2017_full_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"✅ Saved clean full: {clean_path}  (rows={len(df):,})")

    # -------- DeepFM table --------
    deepfm_cols = ["user_id","problem_id","skill_id",
                   "weekday","hour","opportunity","duration","correct"]
    deepfm_df = df[deepfm_cols].copy()
    deepfm_path = os.path.join(out_dir, "deepfm_input.csv")
    deepfm_df.to_csv(deepfm_path, index=False)
    print(f"✅ Saved DeepFM input: {deepfm_path}")

    # split
    train_df, test_df = train_test_split(deepfm_df, test_size=args.test_size, random_state=42, stratify=deepfm_df["correct"])
    train_df.to_csv(os.path.join(split_dir,"train.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir,"test.csv"), index=False)
    print(f"✅ Saved splits: train/test = {len(train_df):,}/{len(test_df):,}")

    grouped = df.groupby("user_id", sort=False)
    dkt_data = []
    for uid, g in grouped:
        skills  = g["skill_id"].tolist()
        correct = g["correct"].tolist()
        dkt_data.append({"user_id": uid, "skills": skills, "correct": correct})
    dkt_path = os.path.join(out_dir, "dkt_sequences.pkl")
    with open(dkt_path, "wb") as f:
        pickle.dump(dkt_data, f)
    print(f"✅ Saved DKT sequences: {dkt_path}  (users={len(dkt_data):,})")

    print("🎉 Preprocessing done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(".."), help="Path to project data/ root (parent of raw/ and processed/). Use absolute if running elsewhere.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio for DeepFM.")
    args = parser.parse_args()

    # If you run from project root (DKT_DeepFM/), set data_root='data'
    # If you run from data/ folder, set data_root='.'
    main(args)
