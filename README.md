# Global Weather Forecasting and Anomaly Detection

## Overview
This project focuses on analyzing global weather data, building forecasting models, and detecting anomalies using machine learning techniques. The dataset is sourced from Kaggle and contains historical weather data across different regions. The main goals of this project include:

- Performing data cleaning and exploratory data analysis (EDA)
- Implementing forecasting models (ARIMA, SARIMA, LSTM)
- Conducting anomaly detection using Isolation Forest, Z-Score, and Autoencoders
- Performing feature importance analysis using SHAP values
- Studying long-term climate trends and visualizing global weather patterns

## Dataset
The dataset used in this project is the **Global Weather Repository** from Kaggle. The key features include:

- **Date (`last_updated`)**: The timestamp of weather records
- **Temperature (`temperature_celsius`)**: Measured in Celsius
- **Precipitation (`precip_mm`)**: Measured in millimeters
- **Humidity (`humidity`)**: Percentage of atmospheric humidity
- **Wind Speed (`wind_kph`)**: Wind speed in kilometers per hour
- **Pressure (`pressure_mb`)**: Atmospheric pressure in millibars
- **Country (`country`)**: Geographic location of the weather observation

## Data Preprocessing
- Removed unnecessary columns (`temperature_fahrenheit`, `wind_mph`, `pressure_in`, etc.)
- Converted the `last_updated` column to datetime format
- Handled missing values by removing or imputing where necessary
- Aggregated data for time series forecasting

## Exploratory Data Analysis (EDA)
- Visualized temperature and precipitation distributions using histograms
- Analyzed correlation between different weather variables
- Created heatmaps to understand feature relationships

## Forecasting Models
### 1. ARIMA (Autoregressive Integrated Moving Average)
- Trained ARIMA(5,1,0) on historical temperature data
- Evaluated performance using MAE and RMSE
- Forecasted temperature for the next 30 days

### 2. SARIMA (Seasonal ARIMA)
- Implemented SARIMA(5,1,0)(1,1,1,12) to capture seasonal trends
- Achieved improved forecasting accuracy over standard ARIMA

### 3. LSTM (Long Short-Term Memory)
- Preprocessed data by normalizing temperature values
- Defined an LSTM model with two stacked LSTM layers and dropout regularization
- Trained on past 30 days' data to predict future temperatures
- Generated a 30-day forecast and visualized results

## Anomaly Detection
### 1. Isolation Forest
- Identified anomalies by detecting points that deviate significantly from normal data patterns
- Marked anomalies in the dataset and analyzed extreme weather events

### 2. Z-Score Method
- Used statistical thresholding (Z-score > 3) to detect outliers in weather features
- Visualized outliers using boxplots

### 3. Autoencoder-Based Anomaly Detection
- Trained an autoencoder to reconstruct input features
- Computed reconstruction errors and flagged high-error instances as anomalies

## Feature Importance Analysis (SHAP)
- Trained a Random Forest model on weather features
- Used SHAP (SHapley Additive exPlanations) to interpret the influence of each feature on temperature prediction
- Generated SHAP summary plots to visualize feature impact

## Climate Analysis
- Aggregated temperature and precipitation data by year
- Plotted long-term climate trends to study global warming patterns
- Mapped global temperature distribution using Geopandas and contextily

## Results
- ARIMA and SARIMA models successfully predicted temperature trends with reasonable accuracy
- LSTM captured complex temporal dependencies but required careful hyperparameter tuning
- Anomaly detection methods identified extreme weather events effectively
- SHAP analysis provided insights into key weather factors affecting temperature

## Dependencies
The project requires the following libraries:
```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn statsmodels shap geopandas contextily kagglehub
```

## Usage
1. Load the dataset from Kaggle
2. Run the preprocessing and EDA scripts
3. Train forecasting models (ARIMA, SARIMA, LSTM)
4. Perform anomaly detection using Isolation Forest, Z-Score, and Autoencoders
5. Analyze feature importance using SHAP
6. Study long-term climate trends

## Conclusion
This project demonstrates a comprehensive approach to weather forecasting and anomaly detection using statistical and machine learning methods. The insights derived from this analysis can help improve climate predictions and better understand global weather patterns.

## Future Work
- Improve model accuracy by tuning hyperparameters
- Integrate additional climate factors such as CO2 levels and solar radiation
- Develop an interactive dashboard for real-time weather forecasting

