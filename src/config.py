"""
Central Configuration and Schema Definitions for PLAYHACK Injury Prediction.

This module defines:
- Column aliases for robust reading across various CSV exports.
- Observation and Risk Window constants.
- Model hyperparameters, candidate models, and pipeline paths.
"""

import os
from typing import Dict, List

# ==============================================================================
# 1. DIRECTORY AND FILE PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

FEATURES_PARQUET_PATH = os.path.join(OUTPUTS_DIR, "features.parquet")
BENCHMARK_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "benchmark_results.json")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
PREDICTIONS_PATH = os.path.join(OUTPUTS_DIR, "predictions.csv")
SUBMISSION_PATH = os.path.join(OUTPUTS_DIR, "submission.csv")

# Ensure required output directories exist
for directory in [OUTPUTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)


# ==============================================================================
# 2. OBSERVATION & RISK WINDOW CONFIGURATION
# ==============================================================================
# The default observation window end date for the PLAYHACK dataset
DEFAULT_OBS_END = "2026-02-03"

# Duration of time windows in days
OBSERVATION_WINDOW_DAYS = 30
RISK_WINDOW_DAYS = 30

# Feature aggregation window sizes (in days prior to and including obs_end)
FEATURE_WINDOWS = [7, 14, 30]


# ==============================================================================
# 3. FLEXIBLE COLUMN ALIAS MAPPING
# ==============================================================================
# Allows the pipeline to ingest varying column names from different CSV sources.
COLUMN_ALIASES: Dict[str, List[str]] = {
    # Athlete identifier
    "athlete_id": [
        "athlete_id",
        "Athlete_ID",
        "Id",
        "id",
        "ID",
        "athleteId",
        "AthleteId",
    ],
    # Date and timestamp columns
    "activity_date": [
        "ActivityDate",
        "activity_date",
        "Date",
        "date",
        "SleepDay",
        "sleep_day",
        "ActivityHour",
        "activity_hour",
    ],
    # Activity metrics
    "steps": [
        "TotalSteps",
        "total_steps",
        "steps",
        "Steps",
        "StepTotal",
        "step_total",
    ],
    "calories": [
        "Calories",
        "calories",
        "total_calories",
        "TotalCalories",
    ],
    "distance": [
        "TotalDistance",
        "total_distance",
        "distance",
        "Distance",
    ],
    "very_active_minutes": [
        "VeryActiveMinutes",
        "very_active_minutes",
        "very_active_mins",
    ],
    "fairly_active_minutes": [
        "FairlyActiveMinutes",
        "fairly_active_minutes",
        "fairly_active_mins",
    ],
    "lightly_active_minutes": [
        "LightlyActiveMinutes",
        "lightly_active_minutes",
        "lightly_active_mins",
    ],
    "sedentary_minutes": [
        "SedentaryMinutes",
        "sedentary_minutes",
        "sedentary_mins",
    ],
    # Heart rate metrics
    "heart_rate": [
        "AvgHeartRate",
        "avg_heart_rate",
        "HeartRate",
        "heart_rate",
        "Value",
        "value",
    ],
    "min_heart_rate": [
        "MinHeartRate",
        "min_heart_rate",
    ],
    "max_heart_rate": [
        "MaxHeartRate",
        "max_heart_rate",
    ],
    # Sleep metrics
    "sleep_minutes": [
        "TotalMinutesAsleep",
        "total_minutes_asleep",
        "minutes_asleep",
        "sleep_minutes",
    ],
    "time_in_bed": [
        "TotalTimeInBed",
        "total_time_in_bed",
        "time_in_bed",
    ],
    "sleep_records": [
        "TotalSleepRecords",
        "total_sleep_records",
        "sleep_records",
    ],
    # Training session metrics
    "session_id": [
        "session_id",
        "SessionId",
        "sessionId",
        "id",
    ],
    "sport_session_type": [
        "sport_session_type",
        "session_type",
        "SessionType",
        "type",
        "activity_type",
    ],
    "start_hour": [
        "start_hour",
        "StartHour",
        "start_time",
        "start",
    ],
    "end_hour": [
        "end_hour",
        "EndHour",
        "end_time",
        "end",
    ],
    # Weight and body composition
    "weight": [
        "WeightKg",
        "weight_kg",
        "weight",
        "Weight",
        "weight_kg_baseline",
    ],
    "bmi": [
        "BMI",
        "bmi",
        "Bmi",
    ],
    "body_fat": [
        "Fat",
        "fat",
        "BodyFat",
        "body_fat",
    ],
    "is_manual_report": [
        "IsManualReport",
        "is_manual_report",
    ],
    # Athlete metadata
    "sport": [
        "sport",
        "Sport",
    ],
    "position": [
        "position",
        "Position",
    ],
    "gender": [
        "gender",
        "Gender",
        "sex",
        "Sex",
    ],
    "age": [
        "age",
        "Age",
    ],
    "height": [
        "height_cm",
        "HeightCm",
        "height",
        "Height",
    ],
    "dominant_side": [
        "dominant_side",
        "DominantSide",
        "side",
    ],
    "years_playing": [
        "years_playing",
        "YearsPlaying",
        "experience",
    ],
    "team_id": [
        "team_id",
        "TeamId",
        "team",
    ],
    "prior_season_injuries": [
        "prior_season_injury_count",
        "prior_season_injuries",
        "prior_injuries",
        "previous_injuries",
    ],
    # Targets and observation anchors
    "injured_in_risk_window": [
        "injured_in_risk_window",
        "injured",
        "injury",
        "target_injured",
    ],
    "onset_day_offset": [
        "onset_day_offset",
        "onset_day",
        "onset_offset",
        "onset",
    ],
    "recovery_duration": [
        "recovery_duration",
        "recovery_days",
        "duration",
        "recovery",
    ],
    "obs_window_end": [
        "obs_window_end",
        "observation_window_end",
        "obs_end",
        "window_end",
    ],
}


# ==============================================================================
# 4. EXPECTED RAW CSV FILES
# ==============================================================================
RAW_FILES = {
    "daily_activity": "dailyActivity_merged.csv",
    "hourly_steps": "hourlySteps_merged.csv",
    "hourly_calories": "hourlyCalories_merged.csv",
    "hourly_intensities": "hourlyIntensities_merged.csv",
    "hourly_heartrate": "hourlyHeartrate_merged.csv",
    "sleep_day": "sleepDay_merged.csv",
    "weight_log": "weightLogInfo_merged.csv",
    "training_sessions": "training_sessions.csv",
    "athlete_metadata": "athlete_metadata.csv",
    "train_labels": "train_labels.csv",
}


# ==============================================================================
# 5. SUSPICIOUS LEAKAGE PATTERNS
# ==============================================================================
# Column tokens that indicate data from the future/risk window
LEAKAGE_COLUMN_PATTERNS = [
    "future",
    "risk",
    "injury_future",
    "post",
    "outcome",
    "onset",
    "recovery",
    "target",
    "label",
]

# Explicit allowed target columns when working with label datasets
ALLOWED_TARGET_COLUMNS = [
    "injured_in_risk_window",
    "onset_day_offset",
    "recovery_duration",
]
