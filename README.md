# Porto Alegre Weather Forecasting: Machine Learning for Localized Precipitation

## Description
This Python-based project processes historical daily meteorological observations (~25 years) from INMET's Jardim Botânico Automated Station (A801) in Porto Alegre, RS. By modeling multi-variable atmospheric shifts using an optimized `RandomForestClassifier`, the pipeline forecasts the specific precipitation category for the next day.

## Objective
To build an end-to-end multiclass classification pipeline capable of predicting tomorrow's weather intensity (`no_rain`, `light_rain`, `moderate_rain`, or `heavy_rain`) based on historical time-series data while rigorously avoiding temporal data leakage.

## Scientific Background
Weather dynamics in southern Brazil, particularly around the Porto Alegre metropolitan basin, are governed by subtropical weather patterns and frequent polar front transitions. Local precipitation events are preceded by distinct thermodynamic signatures:
- **Barometric Shifts:** Fluctuations in atmospheric pressure (`PRESSAO ATMOSFERICA MEDIA`) indicate incoming frontal boundaries.
- **Moisture Convergence:** Relative humidity and dew point temperature (`TEMPERATURA DO PONTO DE ORVALHO`) signal atmospheric saturation.
- **Wind Regimes:** Daily mean wind speeds and peak gusts (`VENTO, RAJADA MAXIMA`) mark air mass instability.

By engineering **3-day rolling window features** (temperature averages, wind speed averages, and accumulated precipitation) alongside seasonal indicators (`month`), the model captures short-term accumulation dynamics leading up to rain events.

## Methodology
- **Data Acquisition & Cleaning:** Historical daily data from INMET Station **A801** (2001–2026). Continuous date re-indexing (`asfreq('D')`) ensures rolling metrics remain temporally precise before filtering out unobserved dates.
- **Target Engineering:** Precipitation depth is mapped into 4 distinct categories (`no_rain`, `light_rain`, `moderate_rain`, `heavy_rain`). Target labels are shifted back by 1 day (`target_tomorrow`).
- **Pipeline Architecture:** Scikit-Learn `Pipeline` integrating `SimpleImputer` (median/mode) and `StandardScaler` / `OneHotEncoder` within a `ColumnTransformer`.
- **Validation & Tuning:** Chronological 80/20 train-test split combined with a 5-fold `TimeSeriesSplit` cross-validation. Hyperparameters were tuned via `RandomizedSearchCV` using `f1_macro` optimization and balanced class weighting (`class_weight='balanced'`).
- **Real-world Inference Routine:** A dedicated forecasting module parses a short-term observation file (`recent_data.csv`), computes rolling features, and outputs class probability distributions for tomorrow's forecast.

## Results
The tuned Random Forest model achieved an **Overall Accuracy of ~61.7%** across the 4 precipitation classes on unseen test data.

Class balancing enabled the model to maintain solid recall on critical extreme events (`heavy_rain`), providing meaningful probabilistic forecasts rather than defaulting to dominant dry-day predictions.