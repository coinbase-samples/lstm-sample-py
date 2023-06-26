This repository contains an example LSTM (Long Short-Term Memory) model for predicting ETH-USD prices using hourly data acquired from [data.coinbase.com](https://data.coinbase.com).

A corresponding blog post that details each individual step of this model will be made available shortly, at which time this sentence will be updated.

# Installation
Clone the repository:

```bash
git clone https://github.com/coinbase-samples/lstm-sample-py
```

Install the required libraries:

```bash
cd lstm-sample-py
pip install -r requirements.txt
```

# Data Acquisition
This model has been designed and tested on more than three years of historical hourly data for ETH-USD to predict one week of future close prices. You can acquire this dataset and others from Coinbase's [Data Marketplace](https://data.coinbase.com/categories/exchange-data/packages/ohlcv-hourly).

While OHLCV data is used, only the timestamp and close columns are needed to build this model. Please make sure to store this data in the parent repository.

# Model Configuration
Several elements require testing and potential modification in the construction of this model:

**Train-Test Split:** The current split ratio is 75:25. Depending on your data and the amount of data you have, you might want to adjust this ratio.

**Time Step:** Currently set to 168 (which equals one week of hourly data). You might need to adjust this value based on the frequency and patterns of your data.

**Neurons:** The number of neurons in the LSTM layer is currently set to 50. You should adjust this value and find the optimal number for your specific task.

**Dropout Rate:** The dropout rate is currently set to 0.2, meaning 20% of the neurons will be deactivated randomly at each training step. Adjusting this rate might also influence the model's performance.

**Epochs:** Currently set to 100. You might also want to adjust this value depending on the number of times you wish to train the model. 

**Early Stopping:** The model uses early stopping to prevent overfitting. The 'patience' parameter is set to 10, which means the model will stop training if there is no improvement in the validation loss after 10 consecutive epochs.

# Usage
After installing the requirements, making sure your data is in the correct format and adjusting this code to read your specific filename, you can simply run the script to train the model and see results. Iterations on the results of your model will require adjusting the above elements and re-running the model to observe how results change. 