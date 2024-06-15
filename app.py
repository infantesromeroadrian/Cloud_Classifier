import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.cloud_classifier import CloudClassifier
from src.cloud_predictor import CloudPredictor

# Cargar el modelo entrenado
def load_model(model, file_path):
    try:
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        return model
    except RuntimeError as e:
        st.error(f"Error loading the model: {e}")
        return None

# Configuración de la aplicación
st.set_page_config(
    page_title="Cloud Classification App",
    page_icon="☁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Cloud Classification App ☁️")
st.write("Esta aplicación clasifica imágenes de nubes en diferentes tipos.")

# Cargar el modelo y las clases
model_path = 'cloud_classifier.pth'
classes_path = 'classes.pth'
class_names = torch.load(classes_path)
num_classes = len(class_names)

trained_model = CloudClassifier(num_classes=num_classes)
trained_model = load_model(trained_model, model_path)

if trained_model is None:
    st.stop()

predictor = CloudPredictor(model=trained_model, class_names=class_names)

# Sidebar para subir la imagen
st.sidebar.header("Sube tu imagen aquí:")
uploaded_file = st.sidebar.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

# Botón para realizar la predicción
if st.sidebar.button("Clasificar Imagen") and uploaded_file is not None:
    # Abrir la imagen
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    st.write("")

    # Realizar la predicción
    with st.spinner('Clasificando...'):
        predicted_class = predictor.predict(uploaded_file)
    st.success(f'Predicted class: {predicted_class}')

    # Mostrar la predicción en un cuadro de información
    st.info(f'La imagen subida ha sido clasificada como: **{predicted_class}**')

# Información adicional en la barra lateral
st.sidebar.write("Desarrollado por [Adrian Infantes](https://www.linkedin.com/in/adrianinfantes/)")
st.sidebar.write("GitHub: [Tu Repositorio](https://github.com/tu-repo)")
st.sidebar.write("¡Gracias por usar esta aplicación! 🌤️")
