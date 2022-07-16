from keras.datasets import mnist, imdb, reuters, boston_housing
import streamlit as st


@st.cache
def nn_data(data_name):
    if data_name == 'MNIST':
        train, test = mnist.load_data()
    elif data_name == 'Imdb':
        train, test = imdb.load_data(num_words=10000)
    elif data_name == 'Reuters':
        train, test = reuters.load_data(num_words=10000)
    else:
        train, test = boston_housing.load_data()

    return train, test
