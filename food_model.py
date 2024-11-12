import pandas as pd
import os
from PIL import Image

# Set the path to your CSV file and image folder
csv_file = 'C:/Users/meena/dl_project/NutrinexAI/dataset/nutrient/data.csv'
image_folder = 'C:/Users/meena/dl_project/NutrinexAI/dataset/nutrient/images'

# Load the CSV file into a DataFrame
data = pd.read_csv(csv_file)

# Assuming the image filenames are in a column called 'index'
dish_names = {
    "1.jpg": "blue cheese and walnut scones",
    "2.jpg": "mini focaccia bread",
    "3.jpg": "stuffed bread rolls",
}

# Loop through each row in the DataFrame
for index, row in data.iterrows():
    # Convert 'index' column to a string to create the filename
    image_filename = str(row['index']) + '.jpg'  # Assuming the 'index' column matches your image filenames
    image_path = os.path.join(image_folder, image_filename)
    
    # Retrieve the dish name from the dictionary or use a default value if not found
    dish_name = dish_names.get(image_filename, "Unknown Dish")
    
    # Display the nutritional information
    print(f"Image: {image_filename} - Dish: {dish_name}")
    print("Carbs: 30g, Fat: 11g, Fibre: 2g, Kcal: 275, Protein: 17g")
    print("Salt: 1.99g, Saturates: 6g, Sugars: 4g")
    print("------------------------------")
    
    # Optional: Display the image if it exists
    if os.path.exists(image_path):
        image = Image.open(image_path)
        image.show()  
    else:
        print("Image not found:", image_path)
