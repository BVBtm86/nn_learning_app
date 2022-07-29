from PIL import Image
from python_scripts.utilities import *
from python_scripts.algo_script import *
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide",
                   page_title="Deep Learning App",
                   page_icon="📶")

logo = Image.open('images/logo.png')

# ##### Hide Streamlit info
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ##### Progress Bar Color
st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #d5e3d6 , #0db518);
        }
    </style>""",
    unsafe_allow_html=True,
)

# ##### Button Color
button_color = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ffffff;
    color:#7575a3;
    width: 100%;
    border-color: #ffffff;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #0db518;
    color:#ffffff;
    border-color: #ffffff;
    font-weight: bold;
    width: 100%;
    }
</style>""", unsafe_allow_html=True)


# ##### Main Application
def main():
    # Option Menu Bar
    with st.sidebar:
        st.subheader("NN Task")
        nn_task = option_menu(menu_title=None,
                              options=["Home", "NN Learning"],
                              icons=["house-fill", "wrench"])
    # ##### App Description
    image_col, title_col = st.columns([1, 11])
    with title_col:
        st.title("Deep Learning")
    with image_col:
        st.markdown("")
        st.image(logo, use_column_width=True)

    if nn_task == 'Home':
        st.markdown(
            '<b><font color=#6600cc>Deep Learning</font></b> is a specific subfield of machine learning: a new take on '
            'learning representations from data that puts an emphasis on learning successive layers of increasingly '
            'meaningful representations. It represents a multistage way to learn data representations. The app will '
            'allow the user to <b><font color=#6600cc>train</font></b> and <b><font color=#6600cc>evaluate</font></b> '
            'different types of Neural Network based on some of the most famous dataset used in Deep Learning.',
            unsafe_allow_html=True)

    else:
        # ##### Dataset selection
        df_name = st.sidebar.selectbox(label="Dataset", options=available_datasets)
        st.markdown(f"<b>Description</b>: <b><font color=#6600cc>{df_name}</font></b> "
                    f"{data_description[available_datasets.index(df_name)]}", unsafe_allow_html=True)
        train_data, test_data = nn_data(data_name=df_name)

        # ##### Parameter selections
        st.sidebar.subheader("Parameters")
        no_layers = st.sidebar.slider(label="No of Layers", min_value=1, max_value=10, value=1)
        no_epochs = st.sidebar.slider(label="No of Epochs", min_value=1, max_value=100, value=20)
        batch_size = st.sidebar.slider(label="Batch Size", min_value=1, max_value=512, value=16)
        include_regularization = st.sidebar.selectbox("Include Regularization", ["No", "L1", "L2", "L1+L2"])
        if include_regularization != "No":
            regularization_parameter = st.sidebar.slider(label=f" {include_regularization} Regularization",
                                                         min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        else:
            regularization_parameter = None

        # ##### Layers Definition
        st.markdown(f"<h3><b><font color=#6600cc>Network</font></b> Hyperparameters</h3>", unsafe_allow_html=True)
        col_layers = st.columns(no_layers)

        # No of Layer Units
        no_layer_units = []
        for i, units in enumerate(col_layers):
            no_layer_units.append(units.selectbox(f"# Units for Layer {i + 1}", [4, 8, 16, 32, 64, 128], key=i))

        # Activation Layer Function
        layer_activation = []
        for i, activation in enumerate(col_layers):
            layer_activation.append(activation.selectbox(f"Activation for Layer {i + 1}",
                                                         ["relu", "tanh", "sigmoid", "softmax"], key=i))

        pytorch_activation = []
        for activation in layer_activation:
            if activation == 'relu':
                pytorch_activation.append(nn.ReLU())
            elif activation == 'tanh':
                pytorch_activation.append(nn.Tanh())
            elif activation == 'sigmoid':
                pytorch_activation.append(nn.Sigmoid())
            else:
                pytorch_activation.append(nn.Softmax())

        # DropOut
        layer_dropout = []
        for i, dropout in enumerate(col_layers):
            layer_dropout.append(dropout.selectbox(f"DropOut for Layer {i + 1}",
                                                   [None, "10%", "20%", "30%", "40%", "50%"], key=i))

        pytorch_dropout = []
        for dropout in layer_dropout:
            if dropout == '10%':
                pytorch_dropout.append(nn.Dropout(p=0.1))
            elif dropout == '20%':
                pytorch_dropout.append(nn.Dropout(p=0.2))
            elif dropout == '30%':
                pytorch_dropout.append(nn.Dropout(p=0.3))
            elif dropout == '40%':
                pytorch_dropout.append(nn.Dropout(p=0.4))
            elif dropout == '50%':
                pytorch_dropout.append(nn.Dropout(p=0.5))
            else:
                pytorch_dropout.append(nn.Dropout(p=0))

        # ##### Deep Learning Model:
        layer_input_size = [train_data.data.shape[1] * train_data.data.shape[2]]
        layer_input_size.extend(no_layer_units)
        button_col, output_col = st.columns([2, 14])
        run_dl = st.sidebar.button("Train Model")

        if run_dl:
            progress_bar = st.progress(0)
            fig_loss, model_architecture, train_accuracy, test_accuracy = \
                nn_modelling(data_name=df_name,
                             train_data=train_data,
                             test_data=test_data,
                             no_epochs=no_epochs,
                             batch_size=batch_size,
                             no_layers=no_layers,
                             unit_layers=layer_input_size,
                             activation_layers=pytorch_activation,
                             drop_layers=pytorch_dropout,
                             regularization=include_regularization,
                             regularization_lambda=regularization_parameter,
                             progress=progress_bar)

            progress_bar.empty()
            # # ##### Training Results
            model_col, chart_col = st.columns([4, 7])
            with chart_col:
                st.markdown(f"Training <b><font color=#6600cc>Loss</font></b>", unsafe_allow_html=True)
                config = {'displayModeBar': False}
                st.plotly_chart(fig_loss, config=config, use_container_width=True)

            with model_col:
                st.markdown(f"Model <b><font color=#6600cc>Architecture</font></b>", unsafe_allow_html=True)
                st.text(model_architecture)
                st.success(f"The model completed after {no_epochs} Epochs with a Train Accuracy of {train_accuracy:.2%}"
                           f" and a Test Accuracy of {test_accuracy:.2%}")


if __name__ == '__main__':
    torch.manual_seed(1909)
    main()
