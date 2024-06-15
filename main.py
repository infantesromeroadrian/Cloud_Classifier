import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import torchvision

from src.cloud_dataset import CloudDataset
from src.cloud_classifier import CloudClassifier
from src.cloud_evaluator import CloudEvaluator
from src.cloud_predictor import CloudPredictor

def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # Desnormalizar
    img = np.clip(img, 0, 1)  # Asegurarse de que los valores estén en el rango [0, 1]
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()

def main():
    # Configuraciones y transformaciones
    data_dir = '/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/DevelopLLMs/data/raw_data/clouds_data/clouds_train'
    batch_size = 32
    img_size = (150, 150)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Creación del dataset y dataloader de entrenamiento
    dataset = CloudDataset(data_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Verificación del dataset
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    print(f"Tamaño del dataset: {len(dataset)}")
    print(f"Tamaño de un batch: {images.size()}")
    print(f"Etiquetas de un batch: {labels}")

    # Mostrar algunas imágenes del batch con sus etiquetas
    imshow(torchvision.utils.make_grid(images), title=[dataset.idx_to_class[label.item()] for label in labels])
    print([dataset.idx_to_class[label.item()] for label in labels])

    # Guardar el dataset procesado
    CloudDataset.save_processed_dataset(dataset, 'clouds_train.pth')

    # Configuraciones y entrenamiento
    num_classes = len(dataset.classes)
    model = CloudClassifier(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 12

    trained_model = CloudClassifier.train_model(model, dataloader, criterion, optimizer, num_epochs)

    # Evaluación del modelo
    data_dir_test = '/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/DevelopLLMs/data/raw_data/clouds_data/test_data/clouds_test'

    # Crear el dataset y dataloader de prueba
    test_dataset = CloudDataset(data_dir=data_dir_test, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Verificar el número de clases y los nombres de las clases
    print(f"Clases en entrenamiento: {dataset.classes}")
    print(f"Clases en prueba: {test_dataset.classes}")

    # Instanciar el evaluador
    evaluator = CloudEvaluator(model=trained_model, dataloader=test_dataloader, class_names=test_dataset.classes)

    # Realizar la evaluación
    labels, preds = evaluator.evaluate()

    # Mostrar la matriz de confusión
    evaluator.plot_confusion_matrix(labels, preds)

    # Imprimir el informe de clasificación
    evaluator.print_classification_report(labels, preds)

    # Guardar el modelo entrenado
    CloudClassifier.save_model(trained_model, 'cloud_classifier.pth')

    # Guardar el diccionario de clases
    torch.save(dataset.classes, 'classes.pth')

    # Cargar el modelo entrenado
    trained_model = CloudClassifier(num_classes=len(dataset.classes))
    trained_model = CloudClassifier.load_model(trained_model, 'cloud_classifier.pth')

    # Instanciar el predictor
    predictor = CloudPredictor(model=trained_model, class_names=torch.load('classes.pth'))

    # Hacer una predicción
    image_path = '/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/DevelopLLMs/data/raw_data/clouds_data/test_data/clouds_test/cirriform clouds/0208a6357f23cf53980f72dea42ade63.jpg'
    predicted_class = predictor.predict(image_path)
    print(f'Predicted class: {predicted_class}')

if __name__ == "__main__":
    main()
