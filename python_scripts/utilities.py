import torchvision
import torchvision.transforms as transforms
import streamlit as st


@st.cache(allow_output_mutation=True)
def nn_data(data_name):
    if data_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data',
                                                  train=False,
                                                  transform=transforms.ToTensor())

    return train_dataset, train_dataset
