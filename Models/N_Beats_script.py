# ===================================================
# ===================== IMPORTS =====================
# ===================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
import torch
import random 


torch.manual_seed(10)
random.seed(10)

from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, rmse
from darts.datasets import EnergyDataset
# matplotlib inline

def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None):
    fig,ax = plt.subplots()
    fig.canvas.draw()
    
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)

    # Plot lines
    pred_series_inv.plot(label=("Predicted"), color="teal")
    ts_transformed.univariate_component(0).plot(label="True", color="darkorange")


    plt.xlabel("Day", fontsize=14, fontweight="bold")
    plt.ylabel("Stock Price", fontsize=14, fontweight="bold")
    plt.title("Predicted Vs True Stock Prices", fontsize=14, fontweight="bold")

    ax.set_xticklabels(np.arange(0,len(pred_series),90));
    ax.grid(False)
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("1")
    plt.xticks(rotation=0)
    plt.legend(frameon=True)

# Create folder for graphs
folderName = "Graphs"
if not os.path.exists("Graphs"):
    os.makedirs(folderName)
    print("Graphs folder created!")

# ===================================================
# ==================== DATA SETUP ===================
# ===================================================

# Download csv files
train_df = pd.read_csv("../Datasets/train_data.csv")
test_df = pd.read_csv("../Datasets/test_data.csv")

train_len = len(train_df)
test_len = len(test_df)

first_train_date = train_df["Date"][0]
first_test_date = test_df["Date"][0]

# Create new df of everything with new date column
all_df = train_df.append(test_df)
date_range = pd.date_range(start=first_train_date, periods = len(all_df))
all_df["Date2"] = date_range

test_start_date = all_df.iloc[train_len]["Date2"]


filler = MissingValuesFiller()
all_series = filler.transform(
    TimeSeries.from_dataframe(all_df, "Date2", ["Open"])
).astype(np.float32)

# Set up train and test series
train_series = all_series[:train_len]

# Set up a scaler
scaler = StandardScaler()
applyScaler = Scaler(scaler)
applyScaler.fit(train_series)

scaled_train_series = applyScaler.fit_transform(train_series)

# ===================================================
# ==================== SETUP MODEL ==================
# ===================================================

# ============= Experiment 1 =============
# seq_len_list = [5,10,30,50,70] # past
# forecast_steps_list = [10,20,test_len] # future

# ============= Experiment 2 =============
seq_len_list = [20] # past
forecast_steps_list = [10,20,30,40,50,60] # future

num_epochs = 200

# Go through each sequence length
for curr_seq_len in seq_len_list:
    for curr_for_steps in forecast_steps_list:

        test_series = all_series[(train_len - curr_seq_len):(train_len + curr_for_steps)]
        scaled_test_series = applyScaler.fit_transform(test_series)

        print("= = = = = = = = = = = = = = = = = = = = = = = = =")
        print("sequence Length" + str(curr_seq_len) + " ForeCast Steps" + str(curr_for_steps))
        print("= = = = = = = = = = = = = = = = = = = = = = = = =")
        model_nbeats = NBEATSModel(
            input_chunk_length = curr_seq_len,
            output_chunk_length = curr_for_steps,
            generic_architecture=True, # Generic version of the model is used
            num_stacks=5, #10
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            n_epochs=num_epochs,
            nr_epochs_val_period=10,
            batch_size=150,
            model_name="nbeats_run",
            random_state=1
        )

        # ===================================================
        # ==================== TRAIN MODEL ==================
        # ===================================================
        # train_series = all_series[:train_len]
        # test_series = all_series[(train_len - curr_seq_len):(train_len + curr_for_steps)]

        model_nbeats.fit(scaled_train_series, 
                          val_series = scaled_test_series, 
                          verbose = True)

        # ===================================================
        # ==================== TEST MODEL ==================
        # ===================================================
        pred_series = model_nbeats.historical_forecasts(
            scaled_test_series,
            start=pd.Timestamp(test_start_date), # The first test date
            forecast_horizon=1,
            stride=1, # Do every test day
            retrain=False,
            verbose=True,
        )

        # Unscale everything
        all_series_inv = applyScaler.inverse_transform(all_series)
        test_series_inv = applyScaler.inverse_transform(scaled_test_series)
        pred_series_inv = applyScaler.inverse_transform(pred_series)

        if curr_for_steps == test_len:
            # Display a graph of predicted vs. actual
            display_forecast(pred_series_inv, test_series_inv[curr_seq_len:], "7 day")

            fileName = "Graphs/"+"seqLen_" + str(curr_seq_len) +".png"
            plt.savefig(fileName, bbox_inches="tight")

        # ===================================================
        # ===================== SAVE RMSE ===================
        # ===================================================

        # Record rmse values in csv file
        rmse_value = round(rmse(test_series_inv[curr_seq_len:], pred_series_inv),3)
        print("The RMSE value is:", rmse_value)

        resultsStr = str(curr_seq_len) + ", " + str(curr_for_steps)+ ", " + str(rmse_value) + "\n"
        print(resultsStr)

        with open("resultsExp2.csv", "a") as res_file:
            res_file.write(resultsStr)