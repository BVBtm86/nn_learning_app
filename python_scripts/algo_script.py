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


def nn_modelling(data_name, train_data, no_epochs, batch_size, no_layers,
                 unit_layers, activation_layers, progress):

    # reg_layers, drop_layers
    # ##### Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ##### Initiate Model
    input_size = train_data.data.shape[1] * train_data.data.shape[2]
    num_classes = max(train_data.targets).item() + 1
    learning_rate = 0.001
    model = NeuralNetworkModel(input_size, unit_layers, num_classes, no_layers, activation_layers)

    # ##### loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    # ##### Train Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

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
            loss = criterion(outputs, labels)

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

    return loss_fig, model
