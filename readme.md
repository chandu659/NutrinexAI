# Food Nutrition & Calorie Tracker

## Overview

This project provides an intelligent system for calculating your nutritional intake and calorie consumption. By analyzing photos of your food, it estimates portion sizes and identifies nutritional values, helping you maintain a balanced and personalized diet.

## Features

- **Instant Nutritional Analysis**: Upload a picture of your food, and the system will analyze it to provide information on calories, fat, sugars, proteins, and other nutritional properties.
- **Portion Estimation**: Using a simple 2D photo of your food, the algorithm estimates the portion size, enhancing accuracy in nutritional breakdown.
- **Food Identification**: Automatically detects the type of food to provide tailored nutritional data, even in mixed or varied dishes.

## Getting Started

1. **Upload an Image**: Take a picture of your food included in the frame.
2. **Receive Instant Feedback**: The system will process the image, identify the food, estimate the portion size, and calculate the nutritional values.

## Example Usage


1. Take a clear photo of your food.
2. Upload the photo through streamlit application.
4. Instantly receive data on:
   - **Calories**
   - **Macronutrients** (Carbohydrates, Fats, Proteins)
   - **Sugars**
   - **Other Nutrients** as applicable to the food type

## Applications

- **Weight Management**: Understand your caloric intake to manage weight.
- **Nutritional Awareness**: Gain insights into the nutritional content of different foods.


## Future Enhancements

- **Automated Meal Suggestions**: Based on daily intake, the app will suggest balanced meals.
- **Enhanced Food Database**: Broader recognition of regional and specialized foods and improve accuracy in finding operation size and nutrient values.

## Contributing

This project welcomes contributions! Please feel free to fork the repository and submit pull requests.

 ## Project Setup
 Utilize virtual environment for the consistent package version

 1. Activate the virtual environment 
 2. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```
 4. Setup environment with GEMINI_API_KEY
 
 This project consists of two model
 1. **Food Model (`food-model.py`)**: This code loads Faster R-CNN model to predict the dish name and draw bounding boxes for each ingredient of the image.
```bash
python food-model.py
```

 2. **Food_calorie_calculator.py** : This code uses gemini-1.5-pro model to detect the dish name, count calories, display nutrient values of the dish
   and which also detects ingredients and draws bounding boxes on the image.
```bash
  py -m streamlit run gemini-1.5pro.py
```

