import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import math
import pandas as pd
from python_scripts.arhitecture_script import NeuralNetworkModel

# available_datasets = ["MNIST", "Imdb", "Reuters", "Boston"]
available_datasets = ["MNIST"]
data_description = [
    "dataset of 60K handwritten images has served as the basis for benchmarking classification algorithms."]
    # "dataset of 50K movie reviews for natural language processing or Text analytics.",
    # "dataset of 11k newswires from Reuters useed for multiclass classification.",
    # "dataset contains information collected by the U.S Census Service concerning housing in the area of "
    # "Boston Mass used for regression tasks."


def nn_modelling(data_name, train_data, test_data, no_epochs, batch_size, no_layers, unit_layers,
                 activation_layers, drop_layers, regularization, regularization_lambda, progress):

    # ##### Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ##### Initiate Model
    input_size = train_data.data.shape[1] * train_data.data.shape[2]
    num_classes = max(train_data.targets).item() + 1
    learning_rate = 0.001
    model = NeuralNetworkModel(unit_layers, num_classes, no_layers, activation_layers, drop_layers)

    # ##### loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    # ##### Create Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    # ##### Training loop
    widget = st.empty()
    loss_values = []
    current_run = 0
    no_of_runs = no_epochs * math.ceil(len(train_data) / batch_size)
    for epoch in range(no_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            # ##### forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # ##### Regularization
            if regularization != "No":
                if regularization == "L1":
                    lambda_param = regularization_lambda
                    l_norm = sum(param.abs().sum() for param in model.parameters())
                    loss = loss + (lambda_param * l_norm)
                elif regularization == "L2":
                    lambda_param = regularization_lambda
                    l_norm = sum(param.square().sum() for param in model.parameters())
                    loss = loss + (lambda_param * l_norm)
                else:
                    lambda_1 = regularization_lambda
                    lambda_2 = 1 - lambda_1
                    l1_norm = sum(param.abs().sum() for param in model.parameters())
                    l2_norm = sum(param.square().sum() for param in model.parameters())
                    loss = loss + (lambda_1 * l1_norm) + (lambda_2 * l2_norm)

            # ##### backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_run += 1
            progress.progress((current_run / no_of_runs))

        loss_values.append(loss.item())
        widget.info(f"Epoch {epoch + 1} Completed, loss = {loss.item():.4f}")

    # ##### Plot Results
    loss_trace = go.Scatter(x=list(range(1, no_epochs + 1)),
                            y=loss_values,
                            mode='lines+markers',
                            name='Train',
                            line_color='#6600cc',
                            hovertemplate="Epoch %{x} Loss: %{y:.4f}<extra></extra>")
    loss_layout = go.Layout(title=f"<b>Loss</b> Results",
                            yaxis=dict(tickformat='.2f',
                                       hoverformat=".2f",
                                       title='Loss'),
                            xaxis=dict(title='Epochs'),
                            title_font=dict(size=20),
                            plot_bgcolor='#e7e7e7',
                            autosize=False,
                            height=600)

    loss_data_plot = [loss_trace]
    loss_fig = go.Figure(data=loss_data_plot, layout=loss_layout)
    loss_fig.update_xaxes(showgrid=False, zeroline=False)
    loss_fig.update_layout(showlegend=False)

    # ##### Testing
    with torch.no_grad():
        # ##### Train
        train_n_correct = 0
        train_n_samples = 0
        for train_images, train_labels in train_loader:
            train_images = train_images.view(train_images.size(0), -1).to(device)
            train_labels = train_labels.to(device)
            train_outputs = model(train_images)

            _, train_predictions = torch.max(train_outputs, axis=1)
            train_n_samples += train_labels.shape[0]
            train_n_correct += (train_predictions == train_labels).sum().item()

        train_acc = train_n_correct / train_n_samples

        # ##### Test
        test_n_correct = 0
        test_n_samples = 0
        for test_images, test_labels in test_loader:
            test_images = test_images.view(test_images.size(0), -1).to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images)

            _, test_predictions = torch.max(test_outputs, axis=1)
            test_n_samples += test_labels.shape[0]
            test_n_correct += (test_predictions == test_labels).sum().item()

        test_acc = test_n_correct / test_n_samples

    return loss_fig, model, train_acc, test_acc
