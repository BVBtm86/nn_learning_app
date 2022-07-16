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


# ##### Main Application
def main():
    # Option Menu Bar
    with st.sidebar:
        st.subheader("NN Task")
        nn_task = option_menu(menu_title=None,
                              options=["Home", "Training", "Evaluate"],
                              icons=["house-fill", "wrench", "reception-4"])
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
        train_df, test_df = nn_data(data_name=df_name)

        # ##### Parameter selections
        st.sidebar.subheader("Parameters")
        layer_size = [""]
        layer_size.extend([i for i in range(1, 11)])
        no_layers = st.sidebar.selectbox(label="No of Layers", options=layer_size, index=0)
        if no_layers != "":
            if nn_task == "Training":
                training_size = st.sidebar.slider(label="Training Size %", min_value=50, max_value=99, value=80)
            no_epochs = st.sidebar.slider(label="No of Epochs", min_value=1, max_value=100, value=20)
            batch_size = st.sidebar.slider(label="Batch Size", min_value=1, max_value=512, value=16)

            # ##### Layers Definition
            st.markdown(f"<h3><b><font color=#6600cc>Network</font></b> Hyperparameters</h3>", unsafe_allow_html=True)
            col_layers = st.columns(no_layers)

            # No of Units
            layer_units = []
            for i, units in enumerate(col_layers):
                layer_units.append(units.selectbox(f"# Units for Layer {i + 1}", [4, 8, 16, 32, 64, 128], key=i))

            # No of Units
            layer_activation = []
            for i, activation in enumerate(col_layers):
                layer_activation.append(activation.selectbox(f"Activation for Layer {i + 1}",
                                                             ["relu", "tanh", "sigmoid", "linear", "softmax"], key=i))
            # Regularization
            layer_regularization = []
            for i, regularization in enumerate(col_layers):
                layer_regularization.append(regularization.selectbox(f"Regularization for Layer {i + 1}",
                                                                     [None, "L1", "L2", "L1+L2"], key=i))

            # DropOut
            layer_dropout = []
            for i, dropout in enumerate(col_layers):
                layer_dropout.append(dropout.selectbox(f"DropOut for Layer {i + 1}",
                                                       [None, "10%", "20%", "30%", "40%", "50%"], key=i))

            # ##### Deep Learning Model:
            train_data, train_labels = train_df[0], train_df[1]
            test_data, test_labels = test_df[0], test_df[1]

            button_col, output_col = st.columns([2, 14])
            with button_col:
                if nn_task == "Training":
                    run_dl = st.button("Train Model")
                else:
                    run_dl = st.button("Evaluate Model")

            if run_dl:
                if nn_task == "Training":
                    with output_col:
                        with st.spinner("Training the model..."):
                            model_history, fig_loss, \
                            fig_metric, model_df = nn_modelling(data_name=df_name,
                                                                train_data=train_data,
                                                                train_labels=train_labels,
                                                                train_size=training_size,
                                                                no_epochs=no_epochs,
                                                                no_batches=batch_size,
                                                                unit_layers=layer_units,
                                                                activation_layers=layer_activation,
                                                                reg_layers=layer_regularization,
                                                                drop_layers=layer_dropout)

                    if df_name != "Boston":
                        st.success(
                            f"Model Finished after {no_epochs} {'Epochs' if no_epochs > 1 else 'Epoch'} with a "
                            f"Train Accuracy of {model_history['accuracy'][-1]:.2%} and a Validation Accuracy of "
                            f"{model_history['val_accuracy'][-1]:.2%}")
                    else:
                        st.success(
                            f"Model Finished after {no_epochs} {'Epochs' if no_epochs > 1 else 'Epoch'} with a "
                            f"Train MAE of {model_history['mae'][-1]:.2f} and a Validation MAE of "
                            f"{model_history['val_mae'][-1]:.2f}")

                    # ##### Training Results
                    st.markdown(f"<h3><b><font color=#6600cc>Training</font></b> Results</h3>", unsafe_allow_html=True)
                    loss_col, metric_col, _ = st.columns([5, 5, 0.5])
                    config = {'displayModeBar': False}
                    with loss_col:
                        st.plotly_chart(fig_loss, config=config)
                    with metric_col:
                        st.plotly_chart(fig_metric, config=config)

                    if df_name != 'Boston':
                        st.table(
                            data=model_df.set_index('No of Epochs').style.format(subset=["Train Loss",
                                                                                         "Validation Loss"],
                                                                                 formatter="{:.3f}").
                                format(subset=["Train Accuracy", "Validation Accuracy"], formatter="{:.2%}").
                                apply(lambda x: ['background: #7575a3' if i % 2 == 1 else 'background: #ffffff'
                                                 for i in range(len(x))], axis=0).apply(
                                lambda x:
                                ['color: #000000' if i % 2 == 0 else 'color: #ffffff'
                                 for i in range(len(x))],
                                axis=0).set_table_styles(
                                [{'selector': 'th',
                                  'props': [('background-color', '#7575a3'), ('color', '#ffffff')]}]))

                    else:
                        st.table(
                            data=model_df.set_index('No of Epochs').style.format(formatter="{:.3f}").
                                apply(lambda x: ['background: #7575a3' if i % 2 == 1 else 'background: #ffffff'
                                                 for i in range(len(x))], axis=0).
                                apply(lambda x: ['color: #000000' if i % 2 == 0 else 'color: #ffffff'
                                                 for i in range(len(x))], axis=0).set_table_styles(
                                [{'selector': 'th',
                                  'props': [('background-color', '#7575a3'), ('color', '#ffffff')]}]))

                else:
                    with output_col:
                        with st.spinner("Evaluate model..."):
                            evaluate_train, evaluate_test = nn_evaluate(data_name=df_name,
                                                                        train_data=train_data,
                                                                        train_labels=train_labels,
                                                                        test_data=test_data,
                                                                        test_labels=test_labels,
                                                                        no_epochs=no_epochs,
                                                                        no_batches=batch_size,
                                                                        unit_layers=layer_units,
                                                                        activation_layers=layer_activation,
                                                                        reg_layers=layer_regularization,
                                                                        drop_layers=layer_dropout)

                    if df_name != "Boston":
                        st.success(
                            f"Final Model after {no_epochs} {'Epochs' if no_epochs > 1 else 'Epoch'} has a "
                            f"Train Accuracy of {evaluate_train[-1]:.2%} and a Test Accuracy of {evaluate_test[-1]:.2%}")
                    else:
                        st.success(
                            f"Final Model after {no_epochs} {'Epochs' if no_epochs > 1 else 'Epoch'} has a "
                            f"Train MAE of {evaluate_train[-1]:.2f} and a Test MAE of {evaluate_test[-1]:.2f}")


if __name__ == '__main__':
    main()
