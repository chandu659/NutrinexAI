## Food calorie calculator
from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image, ImageDraw
import re
import cv2
import numpy as np
import torch
import sys
sys.path.append("Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the depth model
@st.cache_resource
def load_depth_model():
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def get_depth_map(image):
    # Convert PIL Image to cv2 format
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Get depth map using the model
    model = load_depth_model()
    depth = model.infer_image(cv2_image)
    
    # Normalize depth map for visualization
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))

def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Function to process the uploaded image and create image parts
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type, 
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to parse bounding box response from Gemini
def parse_bounding_box(response):
    bounding_boxes = re.findall(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\w\s]+)\]', response)
    parsed_boxes = []
    for box in bounding_boxes:
        parts = box.split(',')
        numbers = list(map(int, parts[:-1]))
        label = parts[-1].strip()
        parsed_boxes.append((numbers, label))
    return parsed_boxes

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, bounding_boxes_with_labels):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = np.array(image)
    label_colors = {}

    for bounding_box, label in bounding_boxes_with_labels:
        width, height = image.shape[1], image.shape[0]
        ymin, xmin, ymax, xmax = bounding_box
        x1 = int(xmin / 1000 * width)
        y1 = int(ymin / 1000 * height)
        x2 = int(xmax / 1000 * width)
        y2 = int(ymax / 1000 * height)

        if label not in label_colors:
            color = np.random.randint(0, 256, (3,)).tolist()
            label_colors[label] = color
        else:
            color = label_colors[label]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        box_thickness = 3
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        text_bg_x1 = x1
        text_bg_y1 = y1 - text_size[1] - 5
        text_bg_x2 = x1 + text_size[0] + 8
        text_bg_y2 = y1

        cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

    return Image.fromarray(image)

# Streamlit app setup
st.set_page_config(page_title="NutrinexAI - Calorie and Ingredient Detector")

st.header("NutrinexAI - Calorie and Ingredient Detector")
input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=None)

submit = st.button("Analyze Image")

depthEstimate = st.button("Estimate Depth")


# Prompt for calories and nutrient detection
input_prompt = """
You are an expert in nutrition where you need to see the food items from the image
and calculate the total calories accurately. Provide calorie intake, Do not make up any information
in the following format along with the dish name and nutrient values and do not give note or any additional information except what asked in propmt:
if food image is not provided, analyze the image but do not tell the calories.

Dish Name - Name of the Dish
1. Item 1 - no of calories
2. Item 2 - no of calories
3. Nutritional Facts
    1.Total Fat : fat in grams (int)
    2.Total Carbohydrates : carbs in grams (int)
    3.Total Protein : protein in grams (int)
    4.Total Sugars : sugars in grams (int)

"""

# Prompt for ingredient detection
ingredient_prompt = """
Return bounding boxes for all objects in the image in the format
[ymin, xmin, ymax, xmax, object_name]. For ingredients, provide separate lists.
"""

def calculate_volume_from_depth(depth_map, bounding_boxes):
    """
    Calculate approximate volume of ingredients based on depth map and bounding boxes.
    """
    volumes = []
    for bounding_box in bounding_boxes:
        ymin, xmin, ymax, xmax = bounding_box[0]
        cropped_depth = depth_map[ymin:ymax, xmin:xmax]
        avg_depth = np.mean(cropped_depth)
        pixel_area = (xmax - xmin) * (ymax - ymin)
        # Estimate volume as pixel area * average depth
        volume = pixel_area * avg_depth
        volumes.append((bounding_box[1], volume))  # (ingredient label, volume)
    return volumes

def generate_calorie_prompt_with_depth(image_data, volumes):
    """
    Generate a detailed prompt including portion sizes estimated from depth.
    """
    volume_details = "\n".join([f"{label}: Estimated volume {volume:.2f} cubic units" for label, volume in volumes])
    prompt = f"""
    You are an expert in nutrition. Using the image and the provided ingredient volumes, calculate the total calories accurately.
    Ingredient volumes:
    {volume_details}
    
    Provide the response in this format:
    Dish Name - Name of the Dish
    1. Item 1 - no of calories
    2. Item 2 - no of calories
    3. Nutritional Facts
        1.Total Fat : fat in grams (int)
        2.Total Carbohydrates : carbs in grams (int)
        3.Total Protein : protein in grams (int)
        4.Total Sugars : sugars in grams (int)
    """
    return prompt

# If submit button is clicked
# If submit button is clicked
if submit:
    # Prepare the image for API calls
    image_data = input_image_setup(uploaded_file)

    # Get bounding boxes for ingredients
    ingredient_response = get_gemini_response(ingredient_prompt, image_data, input)
    bounding_boxes_with_labels = parse_bounding_box(ingredient_response)

    # Draw bounding boxes on the image
    labeled_image = draw_bounding_boxes(image, bounding_boxes_with_labels)
    st.subheader("Ingredient Detection")
    st.image(labeled_image, caption="Detected Ingredients", use_container_width=True)

    # Generate depth map
    with st.spinner("Generating depth map..."):
        depth_map = np.array(get_depth_map(image))
        depth_map_resized = cv2.resize(depth_map, (image.width, image.height))  # Match bounding box dimensions
        st.subheader("Depth Map")
        st.image(depth_map, caption="Generated Depth Map", use_container_width=True)

    # Calculate volumes from depth map
    volumes = calculate_volume_from_depth(depth_map_resized, bounding_boxes_with_labels)

    # Generate detailed prompt including depth-based volume
    refined_prompt = generate_calorie_prompt_with_depth(image_data, volumes)

    # Get calorie and nutrient information with refined prompt
    refined_response = get_gemini_response(refined_prompt, image_data, input)
    st.subheader("Refined Nutritional Information")
    st.write(refined_response)


if depthEstimate:
    st.subheader("Depth Analysis")
    with st.spinner("Generating depth map..."):
        depth_map = get_depth_map(image)
        st.image(depth_map, caption="Depth Map", use_container_width=True)

