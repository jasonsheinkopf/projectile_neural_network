import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
from torch import nn
from tqdm import tqdm
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split

def train_neural_net(results: dict, device, layers, epochs, neurons):
  '''
  Takes dict of projectile examples, trains a neural net, and returns the model and results dataframe
  '''
  # create df from dictionaries in list and view first 5 entries
  input_df = pd.DataFrame(results)

  def normalize_df_to_tensors(df):
    '''Normalized values to range [0, 1] and returns as X, y tensors as well as dict of scaling factors.'''
    # create new column for the max value of each variable
    df['max_angle'] = df['angle'].max()
    df['max_distance'] = df['distance'].max()
    df['max_vel_start'] = df['vel_start'].max()

    # divide each column by the largest value in each column, so they ranges [0, 1] and save to new columns
    df['angle_norm'] = df['angle'] / df['max_angle']
    df['distance_norm'] = df['distance'] / df['max_distance']
    df['vel_start_norm'] = df['vel_start'] / df['max_vel_start']

    # check value range again to make sure they range [0, 1]
    print(f"Non-normalized max values: Angle: {df['angle'].max()}, Dist: {df['distance'].max()}, Vel: {df['vel_start'].max()}")
    print(f"Normalized max values: Angle: {df['angle_norm'].max()}, Dist: {df['distance_norm'].max()}, Vel: {df['vel_start_norm'].max()}")

    # extract values from normalized df
    X_values = df[['angle_norm', 'distance_norm']].values
    y_values = df['vel_start_norm'].values

    # convert to tensors
    X = torch.tensor(X_values, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32).unsqueeze(dim=1)

    return X, y, df

  # convert dataframe to tensors
  X, y, normalized_df = normalize_df_to_tensors(input_df)

  def denormalize_tensors_to_df(df, X: torch.Tensor, y: torch.Tensor, pred: torch.Tensor=None) -> pd.DataFrame:
    '''Convert tensor values back to their original scale and save to dataframe. X[angle, distance] y[velocity]'''
    # multiply by scale factor and convert to numpy arrays
    angle_np = (X[:, 0] * df.max_angle[0]).to('cpu').numpy()
    distance_np = (X[:, 1] * df.max_distance[0]).to('cpu').numpy()
    vel_start_np = (y * df.max_vel_start[0]).to('cpu').numpy().squeeze()

    # create df for denormalized values
    denorm_df = pd.DataFrame(data={'vel_start': vel_start_np, 'angle': angle_np, 'distance': distance_np})

    # if prediction is provided
    if pred is not None:
      # denormalize
      prediction_np = (pred * df.max_vel_start[0]).to('cpu').numpy().squeeze()
      # add to the df
      denorm_df['vel_preds'] = prediction_np

    return denorm_df

  # denormalize tensors
  denorm_df = denormalize_tensors_to_df(input_df, X, y)
  denorm_df.head()

  # combine X and y into a tuple so they stay grouped together
  dataset = torch.utils.data.TensorDataset(X, y)

  # set size of train and test stes
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size

  # create train and test datasets
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

  # try a simple model
  class RegressionModelV0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
      # initialize parent class nn.Module
      super().__init__()
      # create linear layer to feed the two X values into (angle, distance)
      self.layer1 = nn.Linear(input_size, hidden_size)
      # create nonlinear activation function
      self.relu1 = nn.ReLU()
      # create second linear layer to feed the hidden layers into the output
      self.layer2 = nn.Linear(hidden_size, hidden_size)
      # create nonlinear activation function
      self.relu2 = nn.ReLU()
      # create second linear layer to feed the hidden layers into the output
      self.layer3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
      x = self.layer1(x)
      x = self.relu1(x)
      for i in range(layers - 2):
        x = self.layer2(x)
        x = self.relu2(x)
      x = self.layer3(x)
      return x

  # set manual seed for reproducibility
  torch.manual_seed(57)

  # create instance of model 0
  model_0 = RegressionModelV0(input_size=2, hidden_size=neurons, output_size=1).to(device)

  # define a loss function (mean squared error)
  loss_fn = nn.MSELoss()

  # choose optimiser
  optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)


  def predict_all_test(test_dataset, model, input_df, loss_fn):
      # create list to store results dfs
      results_df_list = []
      # create variable to track loss
      test_loss = 0
      # iterate over all samples in test dataset
      for sample in tqdm(test_dataset):
        # extract X and y
        X_sample, y_sample = sample
        # put sample on target device
        X_sample, y_sample = X_sample.to(device), y_sample.to(device)
        # set model to eval mode
        model.eval()
        # turn off gradient tracking
        with torch.no_grad():
          # predict
          predicted_y = model(X_sample)
          # calculate loss and add to running todal
          test_loss += loss_fn(predicted_y, y_sample)
          # add dimension to X
          X_sample = X_sample.unsqueeze(dim=0)
          # denormalize result and save to df
          denorm_sample_df = denormalize_tensors_to_df(input_df, X_sample, y_sample, predicted_y)
          # append individual sample df to list
          results_df_list.append(denorm_sample_df)
      # concatenate list of results dataframes
      test_pred_df = pd.concat(results_df_list, axis=0, ignore_index=True)
      # divide accumulated loss by num of items
      test_loss /= len(test_dataset)

      return test_pred_df, test_loss.item()

  # predict entire test data with current model
  test_preds_df, test_loss = predict_all_test(test_dataset, model_0, input_df, loss_fn)
  test_preds_df.head()

  def plot_results(input_df, train_dataset, test_preds_df):
      # Denormalize datasets to DataFrame
      train_df = denormalize_tensors_to_df(input_df, train_dataset[:][0], train_dataset[:][1])

      marker_size = 3

      # Create scatter plot for train data in blue
      fig = px.scatter_3d(train_df, x='distance', y='angle', z='vel_start',
                                labels={'distance': 'Distance', 'angle': 'Angle', 'vel_start': 'Velocity'},
                                title='Train Data (Blue) Model Predictions (Red)', size_max=marker_size)

      # Update marker size for test data
      fig.update_traces(marker=dict(size=marker_size))

      # Add test data to the scatter plot in red
      fig.add_trace(px.scatter_3d(test_preds_df, x='distance', y='angle', z='vel_preds',
                              color_discrete_sequence=['red']).data[0].update(marker_size=2))

      # Update marker size for predictions
      fig.update_traces(marker=dict(size=marker_size))

      # Show the interactive plot
      fig.show()

  # count epochs across multiple runs of training cell
  total_training_epochs = 0

  torch.manual_seed(57)

  epochs = epochs

  epoch_count = []
  loss_values = []
  test_loss_values = []

  # extract train and test sets
  X_train, y_train = train_dataset[:]
  X_test, y_test = test_dataset[:]

  # put train and test sets on device
  X_train, y_train = X_train.to(device), y_train.to(device)
  X_test, y_test = X_test.to(device), y_test.to(device)

  # iterate over range of epochs
  for epoch in range(epochs):
    total_training_epochs += 1
    ### Train
    model_0.train()
    # forward pass
    y_pred = model_0(X_train)
    # calculate loss
    loss = loss_fn(y_pred, y_train)
    # optimizer zero grade
    optimizer.zero_grad()
    # backpropagation
    loss.backward()
    # step
    optimizer.step()

    ### Test
    model_0.eval()
    # turn off gradient tracking
    with torch.inference_mode():
      # forward pass
      test_pred = model_0(X_test)
      # calculate loss
      test_loss = loss_fn(test_pred, y_test)

    if epoch % (epochs / 10) == 0:
      epoch_count.append(epoch)
      loss_values.append(loss)
      test_loss_values.append(test_loss)
      print(f'Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}')

  print(f'{total_training_epochs=}')

  # predict entire test data with current model
  test_preds_df, test_loss = predict_all_test(test_dataset, model_0, input_df, loss_fn)

  # plot test data
  plot_results(input_df, train_dataset, test_preds_df)

  return model_0, input_df

def nn_predict (model_0, angle, distance, input_df, device):
  # normalize features
  normalized_angle = angle / input_df['max_angle'][0]
  normalized_distance = distance / input_df['max_distance'][0]

  # create feature tensor
  X_trial = torch.tensor([[normalized_angle, normalized_distance]], dtype=torch.float32).to(device)

  model_0.eval()
  # turn off gradient tracking
  with torch.inference_mode():
    # forward pass
    y_trial_normalized = model_0(X_trial).item()

  y_trial = y_trial_normalized * input_df['max_vel_start'][0]

  return y_trial