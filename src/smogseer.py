import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from utils import plot_comparison, plot_training_history


# Load your dataset
ds = xr.open_dataset('../data/S5PL2_5D.nc')

# Stack the features into a single DataArray
features = ['SO2', 'NO2', 'CH4', 'O3', 'CO', 'HCHO']
data = xr.concat([ds[feature] for feature in features], dim='feature')
data = data.transpose('time', 'lat', 'lon', 'feature')

# Convert to NumPy arrays
X_data = data.values.astype(np.float32)

# Normalize the input data
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)

# Impute nan values with the mean of the respective feature
X_data_reshaped = X_data.reshape(-1, X_data.shape[-1])
imputer = SimpleImputer(strategy='mean')
X_data_imputed = imputer.fit_transform(X_data_reshaped)
X_data_imputed = X_data_imputed.reshape(X_data.shape)

# Add the time dimension to the input data
X_data_imputed = np.expand_dims(X_data_imputed, axis=1)

# Load your actual target data
ds_target = xr.open_dataset('../data/S5PL2_5D.nc')
target_data = ds_target['AER_AI_340_380'].values.astype(np.float32)

# Normalize target data to [0, 1]
target_scaler = MinMaxScaler()
target_data = target_scaler.fit_transform(target_data.reshape(-1, 1)).reshape(target_data.shape)

# Impute nan values in target data
target_data_reshaped = target_data.reshape(-1, target_data.shape[-1])
target_data_imputed = imputer.fit_transform(target_data_reshaped)
target_data_imputed = target_data_imputed.reshape(target_data.shape)

# Ensure the target data shape is (num_samples, num_timesteps, num_latitudes, num_longitudes, 1)
target_data_imputed = target_data_imputed.reshape((target_data.shape[0], 1, target_data.shape[1], target_data.shape[2], 1))

# Remove samples with nan values in target data
non_nan_target_indices = ~np.isnan(target_data_imputed).any(axis=(1, 2, 3, 4))
X_data_clean = X_data_imputed[non_nan_target_indices]
y_data_clean = target_data_imputed[non_nan_target_indices]

# Ensure target values are within the valid range [0, 1]
print("Target data range: ", y_data_clean.max(), y_data_clean.min())

# Split data into training and validation sets
split_ratio = 0.8
split_idx = int(split_ratio * X_data_clean.shape[0])

X_train, X_val = X_data_clean[:split_idx], X_data_clean[split_idx:]
y_train, y_val = y_data_clean[:split_idx], y_data_clean[split_idx:]

np.save("../data/X_val.npy", X_val)
np.save("../data/Y_val.npy", y_val)

# Define the model with correct input shape
inp = layers.Input(shape=(1, X_data_clean.shape[2], X_data_clean.shape[3], X_data_clean.shape[4]))

x = layers.BatchNormalization()(inp)
x = layers.ConvLSTM2D(
    filters=16,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="tanh",
    recurrent_activation="sigmoid",
    kernel_initializer="glorot_uniform"
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="tanh",
    recurrent_activation="sigmoid",
    kernel_initializer="glorot_uniform"
)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)

model = keras.models.Model(inp, x, name="smogseer-1-ConvLSTM2D")

# Use a reduced learning rate and gradient clipping
optimizer = keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
)

# Print the model summary
model.summary()

# Data Generator Class

class DataGenerator(Sequence):
    def __init__(self, X_data, y_data, batch_size):
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.indices = np.arange(X_data.shape[0])
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = self.X_data[batch_indices]
        batch_y = self.y_data[batch_indices]
        return batch_X, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

batch_size = 1
train_generator = DataGenerator(X_train, y_train, batch_size)
val_generator = DataGenerator(X_val, y_val, batch_size)

# Define callbacks for monitoring and adjusting learning rate
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-7
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, verbose=1, restore_best_weights=True
    ),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

# Train the model using data generators
history = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=callbacks)

# Save the model
model.save('../models/smogseer50.keras')

# Load the model
model = load_model('../models/smogseer50.keras')

# Run predictions on validation data
predictions = model.predict(X_val)



# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")


# Visualize a few samples
num_samples_to_plot = 5
for i in range(num_samples_to_plot):
    plot_comparison(y_val, predictions, i, f'../static/comparison_plot_{i}.png')

# Plot training history
plot_training_history(history, '../static/training_history.png')
