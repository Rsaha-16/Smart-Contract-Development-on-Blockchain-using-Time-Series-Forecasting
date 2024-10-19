import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import Informer, iTransformer
from neuralforecast.losses.pytorch import MAE
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pytorch_lightning as pl

# Custom callback to save training losses
class SaveTrainingLossCallback(pl.Callback):
    def __init__(self, model_id, log_file='epoch_loss_log_ensemble_model.txt'):
        self.training_losses = []
        self.model_id = model_id  # Unique identifier for the model
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write('Epoch,Model_ID,Train_Loss\n')  # Updated header to include Model_ID

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss'].item()
        self.training_losses.append(train_loss)
        print(f"Epoch {trainer.current_epoch}: Model {self.model_id} Train Loss = {train_loss}")
        
        # Log the loss to the file with model ID
        with open(self.log_file, 'a') as f:
            f.write(f'{trainer.current_epoch},{self.model_id},{train_loss}\n')  # Log with Model_ID

# Initialize callbacks
save_loss_callback_informer = SaveTrainingLossCallback(model_id='Informer_Model')
save_loss_callback_itransformer = SaveTrainingLossCallback(model_id='iTransformer_Model')
pl_trainer_kwargs = {"accelerator": "cpu", "devices": 1}

# Load and preprocess the data
csv_file_path = '/home/raj/Rajarshi/Term Project/rajarshi_code/rajarshi_code/data/SBIN.NS_day_2022.csv'
sbi_data = pd.read_csv(csv_file_path, parse_dates=['Date'])
sbi_data.dropna(inplace=True)
sbi_data.set_index('Date', inplace=True)
sbi_data = sbi_data.asfreq('B', method='pad')

# Create derived features
sbi_data['Open_Close_Diff'] = sbi_data['Open'] - sbi_data['Close']

# Prepare data for both models
features = sbi_data[['Close', 'Open_Close_Diff', 'Volume', 'RSI']].copy()
scaler = MinMaxScaler()
features['Close'] = scaler.fit_transform(features[['Close']])
Y_train_df = features.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
Y_train_df['unique_id'] = 'SBIN'

# Define and train the Informer model
model_informer = Informer(
    h=7, input_size=60, hidden_size=128, conv_hidden_size=32,
    n_head=4, dropout=0.2, encoder_layers=2, decoder_layers=1,
    factor=3, distil=True, loss=MAE(), learning_rate=5e-4, max_steps=1000,
    **{'callbacks': [save_loss_callback_informer]}
)

# Define and train the iTransformer model
model_itransformer = iTransformer(
    h=7, input_size=60, n_series=1, hidden_size=512, n_heads=8,
    e_layers=2, d_layers=1, d_ff=2048, factor=1, dropout=0.1, use_norm=True,
    loss=MAE(), learning_rate=0.0009, max_steps=1000,
    **{'callbacks': [save_loss_callback_itransformer]}
)

# Create NeuralForecast objects and fit models
nf_informer = NeuralForecast(models=[model_informer], freq='B')
nf_itransformer = NeuralForecast(models=[model_itransformer], freq='B')

nf_informer.fit(df=Y_train_df)
nf_itransformer.fit(df=Y_train_df)

# Generate future dataframe for predictions
futr_df = nf_informer.make_future_dataframe()

# Predict using both models
forecasts_informer = nf_informer.predict(futr_df=futr_df)
forecasts_itransformer = nf_itransformer.predict(futr_df=futr_df)

# Stack the predictions from both models (no inverse scaling yet)
stacked_predictions = np.hstack((forecasts_informer[['Informer']].values, forecasts_itransformer[['iTransformer']].values))

# Prepare the target values (for MLP training)
true_values = Y_train_df['y'][-7:].values.reshape(-1, 1).flatten()

# Train an MLP to combine the predictions
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp.fit(stacked_predictions, true_values)

# Predict using MLP (still in scaled form)
final_predictions_scaled = mlp.predict(stacked_predictions)

# Rescale the final MLP predictions back to original scale using the fitted scaler
final_predictions = scaler.inverse_transform(final_predictions_scaled.reshape(-1, 1)).flatten()

# Save the final predictions with corresponding dates
predictions_df = pd.DataFrame({'Date': futr_df['ds'], 'Predicted Value': final_predictions})
output_csv_file = 'ensemble_prediction_informer_itransformer_model.csv'
predictions_df.to_csv(output_csv_file, index=False)

print(f"Ensemble Predictions saved to {output_csv_file}")

# Print the logged training losses for inspection
print("Training Losses Informer Model:", save_loss_callback_informer.training_losses)
print("Training Losses iTransformer Model:", save_loss_callback_itransformer.training_losses)
