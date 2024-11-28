import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FoodClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.base_model = models.resnet50(pretrained=False)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Define your selected classes (must match training)
selected_classes = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
    "beignets", "bibimbap", "caesar_salad", "caprese_salad",
    "carrot_cake", "ceviche", "cheesecake", "chicken_curry",
    "chicken_quesadilla", "churros", "clam_chowder", "club_sandwich",
    "crab_cakes", "donuts", "eggs_benedict", "falafel"
]

# Load the model
def load_model(model_path):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model instance
    model = FoodClassifier(num_classes=len(selected_classes))
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# Prediction function
def predict_image(model, image_path):
    # Define transforms (must match training transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Prepare image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return selected_classes[predicted.item()]

# Example usage
def main():
    # Path to your downloaded model
    model_path = './food_classifier_model.pth'
    
    # Load model
    model = load_model(model_path)
    
    # Predict on an image
    image_path = './images/burger.jpg'
    prediction = predict_image(model, image_path)
    print(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()