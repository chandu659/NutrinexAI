import torchvision
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont

def detect_ingredients_faster_rcnn(image_path):
    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # Load image
    image = Image.open(image_path)
    
    # Convert image to tensor and add batch dimension
    input_image = F.to_tensor(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        prediction = model(input_image)
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    
    # COCO dataset class names (pre-defined in torchvision)
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Filter ingredients
    food_ingredients = [
        'apple', 'banana', 'orange', 'carrot', 'broccoli', 
        'pizza', 'sandwich', 'cake', 'donut', 'hot dog',
        'bowl', 'bottle', 'cup'
    ]
    
    # Process predictions
    for i, box in enumerate(prediction[0]['boxes']):
        score = prediction[0]['scores'][i]
        label = prediction[0]['labels'][i]
        
        # Get class name using the COCO category names
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
    
        # Draw if confidence is high and it's a food ingredient
        if score > 0.8 and class_name.lower() in food_ingredients:
            # Convert to coordinates
            xmin, ymin, xmax, ymax = box.numpy()
            
            # Draw rectangle
            draw.rectangle([xmin, ymin, xmax, ymax], outline='green', width=10)
            
            # Add label
            font = ImageFont.truetype('arial.ttf', size=30)
            left, top, right, bottom = font.getbbox(f'{class_name}: {score:.2f}')
            text_width, text_height = right - left, bottom - top
            draw.rectangle([xmin, ymin-text_height-4, xmin+text_width, ymin], fill='green')
            draw.text((xmin, ymin-20), f'{class_name}: {score:.2f}',font =font, fill='white')
    
    # Save output
    output_path = 'ingredient_detection.jpg'
    image.save(output_path)
    return output_path