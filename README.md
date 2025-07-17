# Bike Count Prediction in Paris

**Felix Brochier and Gregoire Bidault - École Polytechnique Machine Learning Course Project**

## Project Description

This repository contains a comprehensive machine learning solution for predicting hourly bike counts at various counting stations throughout Paris. The project leverages historical bike count data combined with external factors such as weather conditions, French holidays, and COVID-19 restrictions to build accurate predictive models.

The solution includes multiple modeling approaches from naive baselines to advanced ensemble methods, with extensive feature engineering and hyperparameter optimization.

### Problem Statement
- **Objective**: Predict hourly bike counts at different counter locations in Paris
- **Data**: Historical bike count data from 2020-2021 with weather and contextual information
- **Target**: Logarithmic bike count (`log_bike_count`) to handle count data distribution
- **Evaluation**: Mean Squared Error (MSE) on log-transformed predictions

### Methodology
The project explores multiple machine learning approaches:
1. **Naive Baseline**: Using weekly patterns (same day/hour from previous week)
2. **Advanced Models**: LightGBM, XGBoost, Random Forest with sophisticated feature engineering
3. **Ensemble Methods**: Stacking multiple models for improved performance
4. **Hyperparameter Optimization**: Using Optuna for systematic parameter tuning

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/felixbroch/Bike_count_prediction.git
   cd Bike_count_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the baseline model**
   ```bash
   jupyter lab Notebooks/Naive_Baseline.ipynb
   ```

4. **Explore advanced models**
   ```bash
   jupyter lab Notebooks/stacking.ipynb
   ```

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Virtual environment (recommended)

### Environment Setup
```bash
# Create virtual environment
conda create -n bike-prediction python=3.10
conda activate bike-prediction

# Install dependencies
pip install -r requirements.txt
```

### Data Setup
The repository includes some data files in the `data/` folder:
- `train.parquet` - Training data with bike counts
- `final_test.parquet` - Test data for predictions
- `sample_submission.csv` - Submission format example
- `external_data.csv` - Weather and external data

Additional data sources are also available in the `external_data/` folder.

## Project Structure

```
├── data/                           # Data files
│   ├── train.parquet              # Training dataset
│   ├── final_test.parquet         # Test dataset
│   ├── external_data.csv          # Weather and external data
│   └── sample_submission.csv      # Submission format example
├── external_data/                 # External data processing
│   ├── external_data.csv          # Weather data
│   └── example_estimator.py       # Example model implementation
├── Notebooks/                     # Jupyter notebooks
│   ├── bike_counters_starting_kit.ipynb    # Initial exploration
│   ├── Naive_Baseline.ipynb                # Baseline model (Score: 0.674)
│   ├── data_visualisation_time_series.ipynb # Data visualization
│   ├── pipeline_test.ipynb                 # Pipeline testing
│   ├── stacking.ipynb                      # Ensemble methods
│   └── without_pipeline_test.ipynb         # Model comparison
├── utils.py                       # Core utility functions and pipelines
├── best_params.py                 # Optimized hyperparameters
├── requirements.txt               # Python dependencies
├── LICENSE                        # License file
└── README.md                      # This file
```

## Usage

### Running the Baseline Model
```bash
jupyter lab Notebooks/Naive_Baseline.ipynb
```
This notebook demonstrates the naive baseline approach achieving a score of 0.674.

### Training Advanced Models
```bash
# For pipeline-based models
jupyter lab Notebooks/pipeline_test.ipynb

# For ensemble methods
jupyter lab Notebooks/stacking.ipynb
```

### Using Utility Functions
```python
from utils import get_and_process_data, create_pipeline_TV
from best_params import parameters_LGBM, parameters_RandomForest

# Load and preprocess data
X, y, X_test = get_and_process_data()

# Create optimized pipeline using best parameters
pipeline = create_pipeline_TV(X)

# Fit and predict
pipeline.fit(X, y)
predictions = pipeline.predict(X_test)
```

### Using Pre-optimized Parameters
The `best_params.py` file contains hyperparameters optimized through Optuna for different models:
- `parameters_LGBM` - LightGBM parameters
- `parameters_RandomForest` - Random Forest parameters
- Additional optimized parameters for other models

## Key Features and Methods

### Feature Engineering
- **Temporal Features**: Year, month, day, weekday, hour, weekend indicator
- **Holiday Features**: French school holidays and public holidays
- **Weather Integration**: Temperature, humidity, wind, precipitation data
- **COVID-19 Impact**: Lockdown and curfew period indicators
- **Construction Work**: Specific road work affecting bike routes

### Model Approaches
1. **Naive Baseline**: Weekly pattern-based predictions
2. **Traditional ML**: Random Forest, XGBoost with manual feature engineering
3. **Modern Preprocessing**: TableVectorizer for automated feature processing
4. **Ensemble Methods**: Stacking multiple models for better performance
5. **Hyperparameter Tuning**: Optuna optimization for each model type

### Data Preprocessing Pipeline
- External weather data integration with missing value handling
- Datetime feature extraction and holiday detection
- COVID-19 restriction period encoding
- Construction work period indicators
- Automated feature selection and preprocessing

## Results and Performance

| Model | Score (MSE) | Description |
|-------|-------------|-------------|
| Naive Baseline | 0.674 | Weekly pattern-based predictions |
| LightGBM | ~0.55 | Optimized gradient boosting |
| XGBoost | ~0.56 | Extreme gradient boosting |
| Random Forest | ~0.58 | Ensemble of decision trees |
| Stacking Ensemble | ~0.53 | Combined multiple models |

*Note: Exact scores may vary due to random seeds and cross-validation splits*

## Key Insights

1. **Weekly Patterns**: Strong weekly seasonality in bike usage
2. **Weather Impact**: Temperature and precipitation significantly affect cycling
3. **Holiday Effects**: Reduced bike usage during school/public holidays  
4. **COVID-19 Impact**: Lockdowns and curfews dramatically reduced cycling
5. **Construction Work**: Road work caused localized drops in bike counts
6. **Ensemble Benefits**: Combining multiple models improved prediction accuracy

## Future Improvements

- **Additional Features**: Integration of more external data sources (events, traffic, etc.)
- **Time Series Methods**: ARIMA, LSTM, or Prophet for temporal modeling
- **Location-Specific Models**: Separate models for different counter locations
- **Real-time Predictions**: Streaming prediction system with live weather data
- **Seasonal Adjustments**: Better handling of seasonal patterns and trends

## Dependencies

Key libraries used in this project:
- **Data Processing**: pandas, numpy, scikit-learn, pyarrow
- **Machine Learning**: lightgbm, xgboost, sklearn, h2o
- **Optimization**: optuna, flaml for hyperparameter tuning
- **Feature Engineering**: skrub TableVectorizer
- **Visualization**: matplotlib, seaborn, folium
- **French Data**: jours_feries_france, vacances_scolaires_france
- **Environment**: jupyter, jupyterlab, ipython
- **Utilities**: joblib for model persistence

All dependencies are listed in `requirements.txt` and can be installed with:
```bash
pip install -r requirements.txt
```

## Authors

This project was developed by Felix Brochier and Gregoire Bidault at École Polytechnique as part of the Machine Learning 1 course curriculum.

## License

This project is licensed under the terms specified in the LICENSE file.