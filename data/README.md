# PLAYHACK Dataset Structure

This directory contains the multi-source wearable and session records:

1. **`dailyActivity_merged.csv`**: Daily aggregated steps, active minutes, calories, and distances.
2. **`hourlyHeartrate_merged.csv`**: Hourly heart rate metrics (average, min, max).
3. **`hourlySteps_merged.csv`**: Hourly step counts.
4. **`hourlyCalories_merged.csv`**: Hourly calorie expenditures.
5. **`hourlyIntensities_merged.csv`**: Hourly intensity tracking.
6. **`sleepDay_merged.csv`**: Daily sleep duration and time in bed.
7. **`weightLogInfo_merged.csv`**: Weight logs, BMI, and body fat measurements.
8. **`training_sessions.csv`**: Sport training sessions (practice, scrimmage, gym) with start/end hours.
9. **`athlete_metadata.csv`**: Athlete demographics, sport, position, baseline weight, and injury history.
10. **`train_labels.csv`**: Labels for the 30-day risk window:
    - `injured_in_risk_window`: Binary indicator (0/1).
    - `onset_day_offset`: Day of injury inside risk window (1-30).
    - `recovery_duration`: Duration of recovery in days.

### Observation vs Risk Windows
- **Observation Window**: Days 1–30 (`2026-01-05` to `2026-02-03`). Used strictly for feature creation.
- **Risk Window**: Days 31–60 (`2026-02-04` to `2026-03-05`). Target prediction window.
