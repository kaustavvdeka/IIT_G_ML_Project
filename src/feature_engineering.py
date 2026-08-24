"""
Feature Engineering Pipeline for PLAYHACK Sports Injury Prediction.

Transforms multi-granular wearable, training, sleep, and metadata CSVs into a single,
leakage-safe, athlete-level feature dataset anchored at obs_window_end.

Each feature extractor is strictly bounded to the 30-day observation window.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.config import ALLOWED_TARGET_COLUMNS, DEFAULT_OBS_END
from src.leakage_guard import leakage_guard, validate_date_boundary
from src.utils import find_col, LOGGER


def get_window_slice(
    df: pd.DataFrame,
    date_col: str,
    obs_end: str,
    days: int,
) -> pd.DataFrame:
    """
    Slices DataFrame strictly within [obs_end - days + 1, obs_end].
    Zero data from the future risk window can ever be included.
    """
    if df.empty or date_col not in df.columns:
        return df

    obs_end_dt = datetime.strptime(str(obs_end)[:10], "%Y-%m-%d")
    start_date_str = (obs_end_dt - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    obs_end_str = obs_end_dt.strftime("%Y-%m-%d")

    mask = (df[date_col] >= start_date_str) & (df[date_col] <= obs_end_str)
    sliced_df = df[mask].copy()

    validate_date_boundary(sliced_df, date_col, obs_end_str, table_name=f"slice_{days}d")
    return sliced_df


# ==============================================================================
# 1. ACTIVITY FEATURES
# ==============================================================================
def build_activity_features(
    activity_df: pd.DataFrame,
    athlete_ids: List[int],
    obs_end: str,
) -> pd.DataFrame:
    """
    Extracts daily activity features across 7-day, 14-day, and 30-day windows.
    Includes rolling averages, variability, extremes, and workload trend ratios.
    """
    obs_end_str = str(obs_end)[:10]
    LOGGER.info(f"Building activity features anchored at {obs_end_str}...")
    base_df = pd.DataFrame({"athlete_id": athlete_ids})

    if activity_df.empty:
        return base_df

    act = activity_df.copy()
    v_col = find_col(act, "very_active_minutes", required=False, default="VeryActiveMinutes")
    f_col = find_col(act, "fairly_active_minutes", required=False, default="FairlyActiveMinutes")
    l_col = find_col(act, "lightly_active_minutes", required=False, default="LightlyActiveMinutes")

    v_active = act[v_col] if v_col in act.columns else 0
    f_active = act[f_col] if f_col in act.columns else 0
    l_active = act[l_col] if l_col in act.columns else 0
    act["total_active_minutes"] = v_active + f_active + l_active

    steps_col = find_col(act, "steps", required=False, default="TotalSteps")
    dist_col = find_col(act, "distance", required=False, default="TotalDistance")
    cal_col = find_col(act, "calories", required=False, default="Calories")
    sed_col = find_col(act, "sedentary_minutes", required=False, default="SedentaryMinutes")

    act_30d = get_window_slice(act, "date", obs_end_str, days=30)
    act_14d = get_window_slice(act, "date", obs_end_str, days=14)
    act_7d = get_window_slice(act, "date", obs_end_str, days=7)

    # 30-day Aggregations
    f_30d = act_30d.groupby("athlete_id").agg(
        avg_steps_30d=(steps_col, "mean"),
        std_steps_30d=(steps_col, "std"),
        max_steps_30d=(steps_col, "max"),
        min_steps_30d=(steps_col, "min"),
        avg_distance_30d=(dist_col, "mean"),
        avg_calories_30d=(cal_col, "mean"),
        avg_sedentary_minutes_30d=(sed_col, "mean"),
        avg_active_minutes_30d=("total_active_minutes", "mean"),
    ).reset_index()
    f_30d["std_steps_30d"] = f_30d["std_steps_30d"].fillna(0)

    # 14-day Aggregations
    f_14d = act_14d.groupby("athlete_id").agg(
        avg_steps_14d=(steps_col, "mean"),
    ).reset_index()

    # 7-day Aggregations
    f_7d = act_7d.groupby("athlete_id").agg(
        avg_steps_7d=(steps_col, "mean"),
        std_steps_7d=(steps_col, "std"),
        avg_distance_7d=(dist_col, "mean"),
        avg_calories_7d=(cal_col, "mean"),
        avg_sedentary_minutes_7d=(sed_col, "mean"),
        avg_active_minutes_7d=("total_active_minutes", "mean"),
    ).reset_index()
    f_7d["std_steps_7d"] = f_7d["std_steps_7d"].fillna(0)

    features = base_df.merge(f_30d, on="athlete_id", how="left")
    features = features.merge(f_14d, on="athlete_id", how="left")
    features = features.merge(f_7d, on="athlete_id", how="left")

    features["steps_change_7d_vs_30d"] = features["avg_steps_7d"] / (features["avg_steps_30d"] + 1e-5)
    features["activity_change_7d_vs_30d"] = features["avg_active_minutes_7d"] / (features["avg_active_minutes_30d"] + 1e-5)
    features["calories_change_7d_vs_30d"] = features["avg_calories_7d"] / (features["avg_calories_30d"] + 1e-5)

    return features


# ==============================================================================
# 2. HEART RATE FEATURES
# ==============================================================================
def build_heart_rate_features(
    hr_df: pd.DataFrame,
    athlete_ids: List[int],
    obs_end: str,
) -> pd.DataFrame:
    """
    Extracts heart rate metrics from hourly records across 7d, 14d, and 30d windows.
    """
    obs_end_str = str(obs_end)[:10]
    LOGGER.info(f"Building heart rate features anchored at {obs_end_str}...")
    base_df = pd.DataFrame({"athlete_id": athlete_ids})

    if hr_df.empty:
        return base_df

    val_col = find_col(hr_df, "heart_rate", required=False, default="AvgHeartRate")
    min_col = find_col(hr_df, "min_heart_rate", required=False, default="MinHeartRate")
    max_col = find_col(hr_df, "max_heart_rate", required=False, default="MaxHeartRate")

    hr_30d = get_window_slice(hr_df, "date", obs_end_str, days=30)
    hr_14d = get_window_slice(hr_df, "date", obs_end_str, days=14)
    hr_7d = get_window_slice(hr_df, "date", obs_end_str, days=7)

    # 30-day HR features
    f_30d = hr_30d.groupby("athlete_id").agg(
        avg_hr_30d=(val_col, "mean"),
        std_hr_30d=(val_col, "std"),
        min_hr_30d=(min_col if min_col in hr_30d.columns else val_col, "min"),
        max_hr_30d=(max_col if max_col in hr_30d.columns else val_col, "max"),
    ).reset_index()
    f_30d["std_hr_30d"] = f_30d["std_hr_30d"].fillna(0)

    # 14-day HR features
    f_14d = hr_14d.groupby("athlete_id").agg(
        avg_hr_14d=(val_col, "mean"),
    ).reset_index()

    # 7-day HR features
    f_7d = hr_7d.groupby("athlete_id").agg(
        avg_hr_7d=(val_col, "mean"),
        std_hr_7d=(val_col, "std"),
    ).reset_index()
    f_7d["std_hr_7d"] = f_7d["std_hr_7d"].fillna(0)

    features = base_df.merge(f_30d, on="athlete_id", how="left")
    features = features.merge(f_14d, on="athlete_id", how="left")
    features = features.merge(f_7d, on="athlete_id", how="left")

    features["hr_change_7d_vs_30d"] = features["avg_hr_7d"] / (features["avg_hr_30d"] + 1e-5)
    return features


# ==============================================================================
# 3. SLEEP FEATURES
# ==============================================================================
def build_sleep_features(
    sleep_df: pd.DataFrame,
    athlete_ids: List[int],
    obs_end: str,
) -> pd.DataFrame:
    """
    Extracts sleep duration, time in bed, sleep efficiency, and sleep variability.
    """
    obs_end_str = str(obs_end)[:10]
    LOGGER.info(f"Building sleep features anchored at {obs_end_str}...")
    base_df = pd.DataFrame({"athlete_id": athlete_ids})

    if sleep_df.empty:
        return base_df

    mins_col = find_col(sleep_df, "sleep_minutes", required=False, default="TotalMinutesAsleep")
    bed_col = find_col(sleep_df, "time_in_bed", required=False, default="TotalTimeInBed")

    sleep_30d = get_window_slice(sleep_df, "date", obs_end_str, days=30)
    sleep_14d = get_window_slice(sleep_df, "date", obs_end_str, days=14)
    sleep_7d = get_window_slice(sleep_df, "date", obs_end_str, days=7)

    # 30-day Sleep features
    f_30d = sleep_30d.groupby("athlete_id").agg(
        avg_sleep_minutes_30d=(mins_col, "mean"),
        std_sleep_minutes_30d=(mins_col, "std"),
        avg_time_in_bed_30d=(bed_col, "mean"),
        minimum_sleep_30d=(mins_col, "min"),
    ).reset_index()
    f_30d["std_sleep_minutes_30d"] = f_30d["std_sleep_minutes_30d"].fillna(0)

    # 14-day Sleep features
    f_14d = sleep_14d.groupby("athlete_id").agg(
        avg_sleep_minutes_14d=(mins_col, "mean"),
    ).reset_index()

    # 7-day Sleep features
    f_7d = sleep_7d.groupby("athlete_id").agg(
        avg_sleep_minutes_7d=(mins_col, "mean"),
        avg_time_in_bed_7d=(bed_col, "mean"),
    ).reset_index()

    features = base_df.merge(f_30d, on="athlete_id", how="left")
    features = features.merge(f_14d, on="athlete_id", how="left")
    features = features.merge(f_7d, on="athlete_id", how="left")

    features["sleep_change_7d_vs_30d"] = features["avg_sleep_minutes_7d"] / (features["avg_sleep_minutes_30d"] + 1e-5)
    features["sleep_efficiency_30d"] = features["avg_sleep_minutes_30d"] / (features["avg_time_in_bed_30d"] + 1e-5)
    features["sleep_variability_30d"] = features["std_sleep_minutes_30d"] / (features["avg_sleep_minutes_30d"] + 1e-5)
    return features


# ==============================================================================
# 4. TRAINING LOAD & SESSION FEATURES
# ==============================================================================
def build_training_features(
    training_df: pd.DataFrame,
    athlete_ids: List[int],
    obs_end: str,
) -> pd.DataFrame:
    """
    Extracts training session counts, session types (practice, scrimmage, gym),
    total training minutes, and acute-to-chronic workload ratios (ACWR).
    """
    obs_end_str = str(obs_end)[:10]
    LOGGER.info(f"Building training session features anchored at {obs_end_str}...")
    base_df = pd.DataFrame({"athlete_id": athlete_ids})

    if training_df.empty:
        for col in [
            "training_sessions_7d", "training_sessions_14d", "training_sessions_30d",
            "practice_sessions_30d", "scrimmage_sessions_30d", "gym_sessions_30d",
            "training_minutes_7d", "training_minutes_14d", "training_minutes_30d",
            "training_load_7d", "training_load_14d", "training_load_30d",
            "training_load_change_7d_vs_30d", "training_load_change_7d_vs_previous_7d",
        ]:
            base_df[col] = 0.0
        return base_df

    df = training_df.copy()
    start_col = find_col(df, "start_hour", required=False, default="start_hour")
    end_col = find_col(df, "end_hour", required=False, default="end_hour")
    type_col = find_col(df, "sport_session_type", required=False, default="sport_session_type")

    if start_col in df.columns and end_col in df.columns:
        df["duration_hours"] = np.maximum(df[end_col] - df[start_col], 0.5)
    else:
        df["duration_hours"] = 1.5
    df["duration_minutes"] = df["duration_hours"] * 60.0

    type_intensity = {"practice": 1.0, "scrimmage": 1.4, "gym": 1.1, "match": 1.5}
    if type_col in df.columns:
        df["intensity_weight"] = df[type_col].astype(str).str.lower().map(lambda t: type_intensity.get(t, 1.0))
    else:
        df["intensity_weight"] = 1.0

    df["training_load"] = df["duration_hours"] * df["intensity_weight"]

    tr_30d = get_window_slice(df, "date", obs_end_str, days=30)
    tr_14d = get_window_slice(df, "date", obs_end_str, days=14)
    tr_7d = get_window_slice(df, "date", obs_end_str, days=7)

    # Previous 7 days window: [obs_end - 13, obs_end - 7]
    obs_end_dt = datetime.strptime(obs_end_str, "%Y-%m-%d")
    prev_7d_start_str = (obs_end_dt - timedelta(days=13)).strftime("%Y-%m-%d")
    prev_7d_end_str = (obs_end_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    tr_prev_7d = df[(df["date"] >= prev_7d_start_str) & (df["date"] <= prev_7d_end_str)]

    # 30-day Aggregations
    f_30d = tr_30d.groupby("athlete_id").agg(
        training_sessions_30d=("duration_minutes", "count"),
        training_minutes_30d=("duration_minutes", "sum"),
        training_load_30d=("training_load", "sum"),
    ).reset_index()

    # Session type counts over 30d
    if type_col in tr_30d.columns:
        tr_30d_types = tr_30d.assign(session_type_clean=tr_30d[type_col].astype(str).str.lower())
        practice_cnt = tr_30d_types[tr_30d_types["session_type_clean"].str.contains("practice")].groupby("athlete_id").size()
        scrimmage_cnt = tr_30d_types[tr_30d_types["session_type_clean"].str.contains("scrimmage")].groupby("athlete_id").size()
        gym_cnt = tr_30d_types[tr_30d_types["session_type_clean"].str.contains("gym")].groupby("athlete_id").size()

        f_30d["practice_sessions_30d"] = f_30d["athlete_id"].map(practice_cnt).fillna(0)
        f_30d["scrimmage_sessions_30d"] = f_30d["athlete_id"].map(scrimmage_cnt).fillna(0)
        f_30d["gym_sessions_30d"] = f_30d["athlete_id"].map(gym_cnt).fillna(0)
    else:
        f_30d["practice_sessions_30d"] = 0
        f_30d["scrimmage_sessions_30d"] = 0
        f_30d["gym_sessions_30d"] = 0

    # 14-day Aggregations
    f_14d = tr_14d.groupby("athlete_id").agg(
        training_sessions_14d=("duration_minutes", "count"),
        training_minutes_14d=("duration_minutes", "sum"),
        training_load_14d=("training_load", "sum"),
    ).reset_index()

    # 7-day Aggregations
    f_7d = tr_7d.groupby("athlete_id").agg(
        training_sessions_7d=("duration_minutes", "count"),
        training_minutes_7d=("duration_minutes", "sum"),
        training_load_7d=("training_load", "sum"),
    ).reset_index()

    # Previous 7d load for acute:chronic ratio
    f_prev_7d = tr_prev_7d.groupby("athlete_id").agg(
        training_load_prev_7d=("training_load", "sum"),
    ).reset_index()

    features = base_df.merge(f_30d, on="athlete_id", how="left").fillna(0)
    features = features.merge(f_14d, on="athlete_id", how="left").fillna(0)
    features = features.merge(f_7d, on="athlete_id", how="left").fillna(0)
    features = features.merge(f_prev_7d, on="athlete_id", how="left").fillna(0)

    chronic_weekly_avg = (features["training_load_30d"] / 30.0) * 7.0
    features["training_load_change_7d_vs_30d"] = features["training_load_7d"] / (chronic_weekly_avg + 1e-5)
    features["training_load_change_7d_vs_previous_7d"] = features["training_load_7d"] / (features["training_load_prev_7d"] + 1e-5)

    features = features.drop(columns=["training_load_prev_7d"], errors="ignore")
    return features


# ==============================================================================
# 5. WEIGHT & BODY COMPOSITION FEATURES
# ==============================================================================
def build_weight_features(
    weight_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    athlete_ids: List[int],
    obs_end: str,
) -> pd.DataFrame:
    """
    Extracts latest weight, 30-day average weight, weight change, BMI, and body fat.
    Falls back gracefully to baseline metadata weight if weight log is sparse.
    """
    obs_end_str = str(obs_end)[:10]
    LOGGER.info(f"Building weight features anchored at {obs_end_str}...")
    base_df = pd.DataFrame({"athlete_id": athlete_ids})

    base_weight_map = {}
    height_map = {}
    if not metadata_df.empty:
        w_col = find_col(metadata_df, "weight", required=False, default="weight_kg_baseline")
        h_col = find_col(metadata_df, "height", required=False, default="height_cm")
        if w_col in metadata_df.columns:
            base_weight_map = dict(zip(metadata_df["athlete_id"], metadata_df[w_col]))
        if h_col in metadata_df.columns:
            height_map = dict(zip(metadata_df["athlete_id"], metadata_df[h_col]))

    if weight_df.empty:
        base_df["latest_weight"] = base_df["athlete_id"].map(base_weight_map)
        base_df["average_weight_30d"] = base_df["latest_weight"]
        base_df["weight_change_30d"] = 0.0
        h_series = base_df["athlete_id"].map(height_map).fillna(175.0)
        base_df["latest_bmi"] = base_df["latest_weight"] / ((h_series / 100.0) ** 2)
        base_df["latest_body_fat"] = np.nan
        return base_df

    w_30d = get_window_slice(weight_df, "date", obs_end_str, days=30)
    w_col = find_col(w_30d, "weight", required=False, default="WeightKg")
    bmi_col = find_col(w_30d, "bmi", required=False, default="BMI")
    fat_col = find_col(w_30d, "body_fat", required=False, default="Fat")

    w_sorted = w_30d.sort_values(by=["athlete_id", "date"])

    agg_dict = {}
    if w_col in w_30d.columns:
        agg_dict["latest_weight"] = (w_col, "last")
        agg_dict["average_weight_30d"] = (w_col, "mean")
    if bmi_col in w_30d.columns:
        agg_dict["latest_bmi"] = (bmi_col, "last")
    if fat_col in w_30d.columns:
        agg_dict["latest_body_fat"] = (fat_col, "last")

    if agg_dict:
        f_w = w_sorted.groupby("athlete_id").agg(**agg_dict).reset_index()
    else:
        f_w = pd.DataFrame({"athlete_id": athlete_ids})

    features = base_df.merge(f_w, on="athlete_id", how="left")

    features["latest_weight"] = features["latest_weight"].fillna(features["athlete_id"].map(base_weight_map))
    features["average_weight_30d"] = features["average_weight_30d"].fillna(features["latest_weight"])

    baseline_series = features["athlete_id"].map(base_weight_map)
    features["weight_change_30d"] = (features["latest_weight"] - baseline_series).fillna(0.0)

    h_series = features["athlete_id"].map(height_map).fillna(175.0)
    features["latest_bmi"] = features["latest_bmi"].fillna(features["latest_weight"] / ((h_series / 100.0) ** 2))

    return features


# ==============================================================================
# 6. ATHLETE METADATA FEATURES
# ==============================================================================
def build_metadata_features(
    metadata_df: pd.DataFrame,
    athlete_ids: List[int],
) -> pd.DataFrame:
    """
    Extracts and standardizes demographic, position, sport, and historical injury attributes.
    """
    LOGGER.info("Building athlete metadata features...")
    base_df = pd.DataFrame({"athlete_id": athlete_ids})

    if metadata_df.empty:
        return base_df

    df = metadata_df.copy()
    keep_cols = ["athlete_id"]

    mappings = {
        "sport": "sport",
        "age": "age",
        "gender": "gender",
        "height": "height_cm",
        "weight": "weight_kg_baseline",
        "dominant_side": "dominant_side",
        "years_playing": "years_playing",
        "position": "position",
        "team_id": "team_id",
        "prior_season_injuries": "prior_season_injury_count",
    }

    rename_dict = {}
    for alias_key, standard_name in mappings.items():
        matched_col = find_col(df, alias_key, required=False)
        if matched_col and matched_col in df.columns:
            rename_dict[matched_col] = standard_name
            if standard_name not in keep_cols:
                keep_cols.append(standard_name)

    df = df.rename(columns=rename_dict)
    valid_cols = [c for c in keep_cols if c in df.columns]
    meta_subset = df[valid_cols].drop_duplicates(subset=["athlete_id"])

    features = base_df.merge(meta_subset, on="athlete_id", how="left")
    return features


# ==============================================================================
# 7. COMBINE ALL FEATURE MODULES
# ==============================================================================
def combine_features(
    raw_data: Dict[str, pd.DataFrame],
    obs_end: Optional[Union[str, pd.Timestamp]] = None,
    include_labels: bool = True,
) -> pd.DataFrame:
    """
    Orchestrates end-to-end feature extraction from all multi-granular sources.
    Performs leakage checks and outputs a unified athlete-level DataFrame.
    """
    LOGGER.info("[5/8] Building unified feature dataset...")

    labels_df = raw_data.get("train_labels", pd.DataFrame())
    meta_df = raw_data.get("athlete_metadata", pd.DataFrame())
    act_df = raw_data.get("daily_activity", pd.DataFrame())

    if not labels_df.empty and "athlete_id" in labels_df.columns:
        athlete_ids = labels_df["athlete_id"].dropna().unique().tolist()
    elif not meta_df.empty and "athlete_id" in meta_df.columns:
        athlete_ids = meta_df["athlete_id"].dropna().unique().tolist()
    elif not act_df.empty and "athlete_id" in act_df.columns:
        athlete_ids = act_df["athlete_id"].dropna().unique().tolist()
    else:
        raise ValueError("Could not determine list of athlete IDs from input data.")

    athlete_ids = sorted([int(x) for x in athlete_ids])
    LOGGER.info(f"Processing features for {len(athlete_ids)} athletes...")

    if obs_end is None:
        if not labels_df.empty and "obs_window_end" in labels_df.columns:
            obs_end_str = str(labels_df["obs_window_end"].iloc[0])[:10]
        else:
            obs_end_str = str(DEFAULT_OBS_END)[:10]
    else:
        obs_end_str = str(obs_end)[:10]

    LOGGER.info(f"Observation window anchor date: {obs_end_str}")

    # Build individual feature sets
    act_features = build_activity_features(raw_data.get("daily_activity", pd.DataFrame()), athlete_ids, obs_end_str)
    hr_features = build_heart_rate_features(raw_data.get("hourly_heartrate", pd.DataFrame()), athlete_ids, obs_end_str)
    sleep_features = build_sleep_features(raw_data.get("sleep_day", pd.DataFrame()), athlete_ids, obs_end_str)
    tr_features = build_training_features(raw_data.get("training_sessions", pd.DataFrame()), athlete_ids, obs_end_str)
    w_features = build_weight_features(raw_data.get("weight_logs", pd.DataFrame()), meta_df, athlete_ids, obs_end_str)
    meta_features = build_metadata_features(meta_df, athlete_ids)

    # Merge all feature tables
    combined = pd.DataFrame({"athlete_id": athlete_ids})
    combined["obs_window_end"] = obs_end_str

    for feat_df in [act_features, hr_features, sleep_features, tr_features, w_features, meta_features]:
        cols_to_merge = [c for c in feat_df.columns if c != "athlete_id" and c not in combined.columns]
        if cols_to_merge:
            combined = combined.merge(feat_df[["athlete_id"] + cols_to_merge], on="athlete_id", how="left")

    LOGGER.info(f"[6/8] Running leakage checks on {len(combined.columns)} extracted features...")
    leakage_guard(combined, allowed_targets=["obs_window_end"] + ALLOWED_TARGET_COLUMNS)

    if include_labels and not labels_df.empty:
        target_cols = [
            c for c in ["injured_in_risk_window", "onset_day_offset", "recovery_duration"]
            if c in labels_df.columns
        ]
        if target_cols:
            combined = combined.merge(labels_df[["athlete_id"] + target_cols], on="athlete_id", how="left")
            LOGGER.info(f"Attached target labels: {target_cols}")

    LOGGER.info(f"Final feature dataset assembled: {combined.shape[0]} rows, {combined.shape[1]} columns.")
    return combined
