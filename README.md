# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/seattle_weather_1948-2017.csv')

# Convert DATE column to datetime and sort by date
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.sort_values('DATE')

# Set DATE as index
data.set_index('DATE', inplace=True)

# Plot the Temperature Time Series
plt.plot(data.index, data['TMAX'], label='Max Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.legend()
plt.show()

# Check for stationarity using Dickey-Fuller test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['TMAX'])

# Plot ACF and PACF
plot_acf(data['TMAX'])
plt.show()
plot_pacf(data['TMAX'])
plt.show()

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['TMAX'][:train_size], data['TMAX'][train_size:]

# Fit the SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/639c765a-84a1-4ad9-b7f0-7e40b5a40034)

![image](https://github.com/user-attachments/assets/5979a4c9-f167-4d93-be19-733345d0d9d1)

![image](https://github.com/user-attachments/assets/919dc6ae-15cf-4d70-a7cf-1c632be69715)


![image](https://github.com/user-attachments/assets/b1dbdf45-e2f1-476c-a0e4-8ca676142563)



### RESULT:
Thus the program run successfully based on the SARIMA model.

