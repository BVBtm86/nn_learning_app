import os
import tensorflow as tf
import random
from keras import models, layers, regularizers
from keras.utils.np_utils import to_categorical
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import pandas as pd

# available_datasets = ["MNIST", "Imdb", "Reuters", "Boston"]
available_datasets = ["MNIST", "Reuters", "Boston"]
data_description = [
    "dataset of 60K handwritten images has served as the basis for benchmarking classification algorithms.",
    # "dataset of 50K movie reviews for natural language processing or Text analytics.",
    "dataset of 11k newswires from Reuters useed for multiclass classification.",
    "dataset contains information collected by the U.S Census Service concerning housing in the area of "
    "Boston Mass used for regression tasks."]
final_layer_units = [10, 1, 46, 1]
final_layer_activation = ["softmax", "sigmoid", "softmax", "linear"]
model_loss = ["sparse_categorical_crossentropy", "binary_crossentropy", "categorical_crossentropy", "mse"]
model_metric = ["accuracy", "accuracy", "accuracy", "mae"]


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1909)
    tf.random.set_seed(1909)
    np.random.seed(1909)
    random.seed(1909)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, j in enumerate(sequences):
        results[i, j] = 1.
    return results


def nn_modelling(data_name, train_data, train_labels, train_size, no_epochs, no_batches,
                 unit_layers, activation_layers, reg_layers, drop_layers):
    reset_random_seeds()
    # ##### Data Processing
    if data_name == 'MNIST':
        x_train = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
        x_train = x_train.astype('float32') / 255
        y_train = np.asarray(train_labels)
    elif data_name == 'Imdb':
        x_train = vectorize_sequences(train_data)
        y_train = np.asarray(train_labels).astype('float32')
    elif data_name == 'Reuters':
        x_train = vectorize_sequences(train_data)
        y_train = to_categorical(np.asarray(train_labels).astype('float32'))
    else:
        x_train = train_data.copy()
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean) / std
        y_train = np.asarray(train_labels).astype('float32')

    # ##### Create Training and Validation Dataset
    x_model, x_valid, y_model, y_valid = train_test_split(x_train, y_train,
                                                          test_size=np.round((100 - train_size) / train_size, 2),
                                                          random_state=1909)

    # ##### Create Model Architecture
    if len(x_model.shape) > 2:
        model_features = x_model.shape[1] * x_model.shape[2]
    else:
        model_features = x_model.shape[1]
    model_nn = models.Sequential()

    for i in range(len(unit_layers)):
        if i == 0:
            if reg_layers[i] != "L1":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i],
                                          kernel_regularizer=regularizers.L1(0.001),
                                          input_shape=(model_features,)))
            elif reg_layers[i] != "L2":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i],
                                          kernel_regularizer=regularizers.L2(0.001),
                                          input_shape=(model_features,)))
            elif reg_layers[i] != "L1+L2":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i],
                                          kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                                          input_shape=(model_features,)))
            else:
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i],
                                          input_shape=(model_features,)))
        else:
            if reg_layers[i] != "L1":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          kernel_regularizer=regularizers.L1(0.001),
                                          activation=activation_layers[i]))
            if reg_layers[i] != "L2":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          kernel_regularizer=regularizers.L2(0.001),
                                          activation=activation_layers[i]))
            if reg_layers[i] != "L1+L2":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                                          activation=activation_layers[i]))
            else:
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i]))

        # Add Dropout if selected
        if drop_layers[i] != None:
            model_nn.add(layers.Dropout(int(drop_layers[i].replace("%", "")) / 100))

    # Output layer
    model_nn.add(layers.Dense(units=final_layer_units[available_datasets.index(data_name)],
                              activation=final_layer_activation[available_datasets.index(data_name)]))

    # ##### Compile and fit Model
    model_nn.compile(optimizer='rmsprop',
                     loss=model_loss[available_datasets.index(data_name)],
                     metrics=model_metric[available_datasets.index(data_name)])

    model_history = model_nn.fit(x_model,
                                 y_model,
                                 batch_size=no_batches,
                                 epochs=no_epochs,
                                 validation_data=(x_valid, y_valid),
                                 use_multiprocessing=False,
                                 verbose=2)

    final_results = model_history.history

    # ##### Loss and Metric
    loss_results_train = final_results['loss']
    loss_results_validation = final_results['val_loss']
    if data_name != "Boston":
        metric_train = final_results['accuracy']
        metric_validation = final_results['val_accuracy']
        metric_name = 'Accuracy'
    else:
        metric_train = final_results['mae']
        metric_validation = final_results['val_mae']
        metric_name = 'MAE'

    # ##### Plot Results
    # ### Loss Plot
    loss_trace_1 = go.Scatter(x=list(range(1, no_epochs + 1)),
                              y=loss_results_train,
                              mode='lines+markers',
                              name='Train',
                              line_color='#6600cc',
                              hovertemplate="Epoch %{x} Loss: %{y:.4f}<extra></extra>")

    loss_trace_2 = go.Scatter(x=list(range(1, no_epochs + 1)),
                              y=loss_results_validation,
                              mode='lines+markers',
                              name='Validation',
                              line_color='#1e1e1e',
                              hovertemplate="Epoch %{x} Loss: %{y:.4f}<extra></extra>")

    loss_layout = go.Layout(title=f"<b>Loss</b> Results",
                            yaxis=dict(tickformat='.2f',
                                       hoverformat=".2f",
                                       title='Loss'),
                            xaxis=dict(title='Epochs'),
                            title_font=dict(size=20),
                            plot_bgcolor='#e7e7e7',
                            autosize=False,
                            width=600,
                            height=500)

    loss_data_plot = [loss_trace_1, loss_trace_2]
    loss_fig = go.Figure(data=loss_data_plot, layout=loss_layout)
    loss_fig.update_xaxes(showgrid=False, zeroline=False)
    loss_fig.update_layout(showlegend=False)

    # ### Metric PLot
    if data_name != "Boston":
        metric_trace_1 = go.Scatter(x=list(range(1, no_epochs + 1)),
                                    y=metric_train,
                                    mode='lines+markers',
                                    name='Train',
                                    line_color='#6600cc',
                                    hovertemplate="Epoch %{x} Accuracy: %{y:.2%}<extra></extra>")
        metric_trace_2 = go.Scatter(x=list(range(1, no_epochs + 1)),
                                    y=metric_validation,
                                    mode='lines+markers',
                                    name='Validation',
                                    line_color='#1e1e1e',
                                    hovertemplate="Epoch %{x} Accuracy: %{y:.2%}<extra></extra>")

    else:
        metric_trace_1 = go.Scatter(x=list(range(1, no_epochs + 1)),
                                    y=metric_train,
                                    mode='lines+markers',
                                    name='Train',
                                    line_color='#6600cc',
                                    hovertemplate="Epoch %{x} MAE: %{y:.4f}<extra></extra>")
        metric_trace_2 = go.Scatter(x=list(range(1, no_epochs + 1)),
                                    y=metric_validation,
                                    mode='lines+markers',
                                    name='Validation',
                                    line_color='#1e1e1e',
                                    hovertemplate="Epoch %{x} MAE}: %{y:.4f}<extra></extra>")

    metric_layout = go.Layout(title=f"<b>{metric_name}</b> Results",
                              yaxis=dict(tickformat='.2f',
                                         hoverformat=".2f",
                                         title=f'{metric_name}'),
                              xaxis=dict(title='Epochs'),
                              title_font=dict(size=20),
                              plot_bgcolor='#e7e7e7',
                              autosize=False,
                              width=600,
                              height=500)

    metric_data_plot = [metric_trace_1, metric_trace_2]
    metric_fig = go.Figure(data=metric_data_plot, layout=metric_layout)
    metric_fig.update_xaxes(showgrid=False, zeroline=False)

    # ##### Create Table Results
    train_results_df = pd.DataFrame([[int(i) for i in range(1, no_epochs + 1)],
                                     loss_results_train,
                                     loss_results_validation,
                                     metric_train,
                                     metric_validation]).T
    train_results_df.columns = ['No of Epochs', 'Train Loss', 'Validation Loss',
                                f'Train {metric_name}', f'Validation {metric_name}']
    train_results_df['No of Epochs'] = train_results_df['No of Epochs'].astype(int)

    return final_results, loss_fig, metric_fig, train_results_df


def nn_evaluate(data_name, train_data, train_labels, test_data, test_labels, no_epochs, no_batches,
                unit_layers, activation_layers, reg_layers, drop_layers):
    reset_random_seeds()
    # ##### Data Processing
    if data_name == 'MNIST':
        x_train = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
        x_train = x_train.astype('float32') / 255
        y_train = np.asarray(train_labels)
        x_test = test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
        x_test = x_test.astype('float32') / 255
        y_test = np.asarray(test_labels)
    elif data_name == 'Imdb':
        x_train = vectorize_sequences(train_data)
        y_train = np.asarray(train_labels).astype('float32')
        x_test = vectorize_sequences(test_data)
        y_test = np.asarray(test_labels).astype('float32')
    elif data_name == 'Reuters':
        x_train = vectorize_sequences(train_data)
        y_train = to_categorical(np.asarray(train_labels).astype('float32'))
        x_test = vectorize_sequences(test_data)
        y_test = to_categorical(np.asarray(test_labels).astype('float32'))
    else:
        x_train = train_data.copy()
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean) / std
        y_train = np.asarray(train_labels).astype('float32')
        x_test = test_data.copy()
        x_test = (x_test - mean) / std
        y_test = np.asarray(test_labels).astype('float32')

    # ##### Create Model Architecture
    if len(x_train.shape) > 2:
        model_features = x_train.shape[1] * x_train.shape[2]
    else:
        model_features = x_train.shape[1]
    model_nn = models.Sequential()

    for i in range(len(unit_layers)):
        if i == 0:
            if reg_layers[i] != "L1":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i],
                                          kernel_regularizer=regularizers.L1(0.001),
                                          input_shape=(model_features,)))
            elif reg_layers[i] != "L2":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i],
                                          kernel_regularizer=regularizers.L2(0.001),
                                          input_shape=(model_features,)))
            elif reg_layers[i] != "L1+L2":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i],
                                          kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                                          input_shape=(model_features,)))
            else:
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i],
                                          input_shape=(model_features,)))
        else:
            if reg_layers[i] != "L1":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          kernel_regularizer=regularizers.L1(0.001),
                                          activation=activation_layers[i]))
            if reg_layers[i] != "L2":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          kernel_regularizer=regularizers.L2(0.001),
                                          activation=activation_layers[i]))
            if reg_layers[i] != "L1+L2":
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                                          activation=activation_layers[i]))
            else:
                model_nn.add(layers.Dense(units=unit_layers[i],
                                          activation=activation_layers[i]))

        # Add Dropout if selected
        if drop_layers[i] != None:
            model_nn.add(layers.Dropout(int(drop_layers[i].replace("%", "")) / 100))

    # Output layer
    model_nn.add(layers.Dense(units=final_layer_units[available_datasets.index(data_name)],
                              activation=final_layer_activation[available_datasets.index(data_name)]))

    # ##### Compile and fit Model
    model_nn.compile(optimizer='rmsprop',
                     loss=model_loss[available_datasets.index(data_name)],
                     metrics=model_metric[available_datasets.index(data_name)])

    model_nn.fit(x_train,
                 y_train,
                 batch_size=no_batches,
                 epochs=no_epochs,
                 use_multiprocessing=False,
                 verbose=2)

    final_train_results = model_nn.evaluate(x_train, y_train)
    final_test_results = model_nn.evaluate(x_test, y_test)

    return final_train_results, final_test_results
