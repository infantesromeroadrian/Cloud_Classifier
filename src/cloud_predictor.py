import torch
from torchvision import transforms
from PIL import Image


class CloudPredictor:
    def __init__(self, model, class_names, img_size=(150, 150)):
        self.model = model
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
        predicted_class = self.class_names[preds.item()]
        return predicted_class
