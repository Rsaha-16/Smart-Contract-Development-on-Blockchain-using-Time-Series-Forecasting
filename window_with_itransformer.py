import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer
from neuralforecast.losses.pytorch import MAE
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl

# Custom callback to save training losses at each epoch
class SaveTrainingLossCallback(pl.Callback):
    def __init__(self, log_file='epoch_loss_log_window_itransformer_model.txt'):
        self.training_losses = []
        self.log_file = log_file
        self.window_number = 0
        with open(self.log_file, 'w') as f:
            f.write('Epoch,Train_Loss,Window\n')

    def on_train_epoch_end(self, trainer, pl_module):
        # Save the training loss at the end of each epoch
        train_loss = trainer.callback_metrics['train_loss'].item()
        self.training_losses.append(train_loss)
        print(f"Epoch {trainer.current_epoch}: Train Loss = {train_loss}")
        
        # Log the loss to the file
        with open(self.log_file, 'a') as f:
            f.write(f'{trainer.current_epoch},{train_loss},{self.window_number}\n')

    def set_window_number(self, window_number):
        self.window_number = window_number

# Initialize callbacks
save_loss_callback = SaveTrainingLossCallback()
pl_trainer_kwargs = {"callbacks": [save_loss_callback], "accelerator": "cpu", "devices": 1}

# Load and preprocess the data
csv_file_path = '/home/raj/Rajarshi/Term Project/rajarshi_code/rajarshi_code/data/SBIN.NS_day_2022.csv'
sbi_data = pd.read_csv(csv_file_path, parse_dates=['Date'])
sbi_data.dropna(inplace=True)
sbi_data.set_index('Date', inplace=True)
sbi_data = sbi_data.asfreq('B', method='pad')

# Create scalers
scaler_close = MinMaxScaler()
sbi_data['Open_Close_Diff'] = sbi_data['Open'] - sbi_data['Close']
sbi_data['Close'] = scaler_close.fit_transform(sbi_data[['Close']])

# Initialize variables
training_end_date = sbi_data.index.max() - pd.DateOffset(months=2)  # Train using last 2 months of data
final_predictions = []

# Define the window management and model training class
class ModelTrainer:
    def __init__(self, data, scaler_close, save_loss_callback, pl_trainer_kwargs):
        self.data = data
        self.scaler_close = scaler_close
        self.save_loss_callback = save_loss_callback
        self.pl_trainer_kwargs = pl_trainer_kwargs

    def train_model(self, train_data, window_number):
        # Set the window number for the callback
        self.save_loss_callback.set_window_number(window_number)

        # Prepare the training data
        Y_train_df = train_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        Y_train_df['unique_id'] = 'SBIN'

        # Initialize and train the iTransformer model
        model = iTransformer(
            h=15,  # Output horizon (prediction length)
            input_size=150,  # Input window size
            n_series=1,  # Number of time series (SBIN in this case)
            hidden_size=512,  # Adjusted for iTransformer
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=2048,
            factor=1,
            dropout=0.1,
            use_norm=True,
            loss=MAE(),
            learning_rate=0.001,
            max_steps=500,  # Adjusted as per your code
            **{'callbacks': [self.save_loss_callback]}  # Pass the callback directly here
        )

        # NeuralForecast object to handle model training
        nf = NeuralForecast(models=[model], freq='B')
        nf.fit(df=Y_train_df)

        # Generate future dataframe automatically
        futr_df = nf.make_future_dataframe()

        # Generate predictions
        forecasts = nf.predict(futr_df=futr_df)
        
        pred_values = self.scaler_close.inverse_transform(forecasts[['iTransformer']].values)
        dates = futr_df['ds']

        return dates, pred_values

# Initialize the trainer
trainer = ModelTrainer(sbi_data, scaler_close, save_loss_callback, pl_trainer_kwargs)

# Training and prediction loop
window_number = 1
while True:
    train_data = sbi_data.loc[:training_end_date]
    print(f"Training window {window_number}: from {train_data.index.min()} to {train_data.index.max()}")

    # Train the model and get predictions
    dates, pred_values = trainer.train_model(train_data, window_number)

    if len(dates) == 0:
        print("No future dates were generated. Exiting the loop.")
        break

    # Store the predictions
    predictions_df = pd.DataFrame({'Date': dates, 'Predicted Value': pred_values.flatten()})
    final_predictions.append(predictions_df)

    # Update training_end_date for the next window only if dates exist
    training_end_date = dates.iloc[-1] if len(dates) > 0 else training_end_date

    # Break if we reach the end of the data
    if training_end_date >= sbi_data.index.max():
        break

    window_number += 1

# Combine predictions and save
all_predictions_df = pd.concat(final_predictions, ignore_index=True)
output_csv_file = 'prediction_using_window_method_itransformer_model.csv'
all_predictions_df.to_csv(output_csv_file, index=False)

print(f"Predictions saved to {output_csv_file}")

# Print the logged training losses
print("Training Losses:", save_loss_callback.training_losses)
