#-------------------------------------------------------------------------------
#--------------------------------IMPORTS----------------------------------------
#-------------------------------------------------------------------------------
#--------------------------Third Party Libraries--------------------------------
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
import cv2
#--------------------------Python Standard Libraries----------------------------
import concurrent.futures
from collections import Counter
import json

#-------------------------------------------------------------------------------
#-------------------------PATHS AND GLOBAL VARIABLES----------------------------
#-------------------------------------------------------------------------------

# Load the trained model
model = load_model('model.h5')

# Update this with the path to your new image

#-------------------------------------------------------------------------------

# Image Dimensions
im_dim = 299

# Map the numerical class indices to the actual class names
class_mapping = {
    0: 'gravel',
    1: 'sand',
    2: 'soil'
}
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#--------------------------FUNCTION DEFINATIONS---------------------------------
#-------------------------------------------------------------------------------

# Function to preprocess the input image for inference
# def preprocess_image(image_path, target_size=(im_dim, im_dim)):
#     img = image.load_img(image_path, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     processed_img = img_array / 255.0  # Normalize pixel values
#     return processed_img

def preprocess_image(image_path, target_size=(im_dim, im_dim)):
    # Read the image directly using TensorFlow
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)

    # Resize the image
    img = tf.image.resize(img, target_size)

    # Convert the image to float32 and normalize pixel values
    processed_img = tf.image.convert_image_dtype(img, tf.float32)
    processed_img /= 255.0

    # Expand the dimensions to create a batch of one image
    processed_img = tf.expand_dims(processed_img, axis=0)

    return processed_img

# Function to predict the category of the image
def predict_category(model, image_path, class_mapping):
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_mapping[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence

def classify_images(image_fp, model):
    upscale_size_pred = 1024
    classes = ['Gravel', 'Sand', 'Silt']
    gravel_count = 0
    sand_count = 0
    silt_count = 0
    img = cv2.imread(image_fp)
    img = cv2.resize(img, (upscale_size_pred, upscale_size_pred))

    def process_cropped_image(r, c):
        cropped_img = img[r:r + im_dim, c:c + im_dim, :]
        h, w, c = cropped_img.shape
        if h == im_dim and w == im_dim:
            classification = model_classify(cropped_img, model)
            if classification == classes[0]:
                return 'gravel'
            elif classification == classes[1]:
                return 'sand'
            elif classification == classes[2]:
                return 'silt'
        else:
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_cropped_image, r, c)
                   for r in range(0, img.shape[0], im_dim)
                   for c in range(0, img.shape[1], im_dim)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                if result == 'gravel':
                    gravel_count += 1
                elif result == 'sand':
                    sand_count += 1
                elif result == 'silt':
                    silt_count += 1

    total_count = gravel_count + sand_count + silt_count
    proportion_array = [gravel_count / total_count, sand_count / total_count, silt_count / total_count]
    return proportion_array

def model_classify(cropped_img, model):
    classes = ['Gravel', 'Sand', 'Silt']
    image_array = cropped_img / 255.
    img_batch = np.expand_dims(image_array, axis=0)
    prediction_array = model.predict(img_batch)[0]
    first_idx = np.argmax(prediction_array)
    first_class = classes[first_idx]
    return first_class

def classify_percentage(image_fp):
    proportion_array = classify_images(image_fp=image_fp, model=model)
    percent_gravel = round(proportion_array[0] * 100, 2)
    percent_sand = round(proportion_array[1] * 100, 2)
    percent_soil = round(proportion_array[2] * 100, 2)

    return percent_gravel, percent_sand, percent_soil

#-------------------------------------------------------------------------------

def find_soil_type(gravel_percent, sand_percent, silt_percent):
    # Define the soil types with their associated percentage ranges
    soil_types = {
        'Clay': (0, 12, 0, 20, 50, 100),
        'Sandy Clay': (0, 15, 70, 90, 0, 15),
        'Silty Clay': (0, 12, 0, 50, 50, 100),
        'Sandy Loam': (10, 40, 50, 70, 0, 30),
        'Silt Loam': (0, 10, 30, 50, 20, 60),
        'Loam': (0, 15, 25, 50, 25, 50),
        'Sandy Clay Loam': (0, 25, 40, 70, 0, 30),
        'Silky Clay Loam': (0, 20, 20, 50, 30, 60),
        'Sandy Silt Loam': (5, 30, 30, 60, 15, 40),
        'Loamy Sand': (25, 70, 25, 60, 0, 30),
        'Sand': (0, 15, 85, 100, 0, 10),
        'Sandy Silt': (0, 10, 50, 85, 15, 50),
        'Silt': (0, 5, 0, 45, 50, 100),
        'Gravelly Silt': (15, 80, 0, 10, 10, 85),
        'Gravelly Sand': (60, 100, 20, 70, 0, 20)
    }

    # Loop through each soil type and check if the input percentages match
    for soil_type, (min_gravel, max_gravel, min_sand, max_sand, min_silt, max_silt) in soil_types.items():
        if min_gravel <= gravel_percent <= max_gravel and min_sand <= sand_percent <= max_sand and min_silt <= silt_percent <= max_silt:
            return soil_type

    # If no match found, return None
    return None
#-------------------------------------------------------------------------------

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def color_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

def classify_color(hex_color):
  # Base color classes and their corresponding hex values
    base_colors = {
                    "Black": "#000000",
                    "Dark Brown": "#654321",
                    "Brown": "#A52A2A",
                    "Light brown": "#CD853F",
                    "Red": "#FF0000",
                    "Yellow": "#FFFF00",
                    "Orange": "#FFA500",
                    "Grey": "#808080",
                    "White": "#FFFFFF",
                    "Blue": "#0000FF",
                    "Green": "#008000"
                    }
    rgb_color = hex_to_rgb(hex_color)
    closest_color = min(base_colors.items(), key=lambda x: color_distance(rgb_color, hex_to_rgb(x[1])))
    return closest_color[0]

def preprocess(raw):
    upscale_size_colr = 100
    image = cv2.resize(raw, (upscale_size_colr, upscale_size_colr), interpolation=cv2.INTER_AREA)
    image = image.reshape(-1, 3)
    return image

def analyze(img):
    counts = Counter(map(tuple, img))
    dominant_color = max(counts, key=counts.get)
    hex_color = "#{:02x}{:02x}{:02x}".format(*dominant_color)
    return hex_color

#-------------------------------------------------------------------------------

def get_soil_info(soil_type, soil_color):
    soil_info = {
        'Clay': {
            'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to meet the nutritional needs of plants. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
             'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
        },
#-------------------------------------------------------------------------------
        'Sandy Clay': {
            'Dark Brown': {
                'pHRange': '5.5 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots, Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Balanced NPK fertilizer to meet plant needs.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer with additional magnesium and sulfur.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Phosphorus-based fertilizer to meet plant phosphorus needs.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture.'
            },
             'Red': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Phosphorus-based fertilizer to meet plant phosphorus needs.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
        },
#-------------------------------------------------------------------------------
         'Silty Clay': {
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to meet the nutritional needs of plants. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums.	',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
            'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to meet the nutritional needs of plants. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
        },
#-------------------------------------------------------------------------------
         'Sandy Loam': {
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce, Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth'
            },
             'Yellow': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Carrots, Lettuce, Spinach, Flowers like Sunflowers and Zinnias, Shrubs and small trees',
                'GeneralRecommendation': 'Implement cover cropping to enrich soil nutrients. Use organic mulch to conserve soil moisture.'
            },
            'Dark Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Broccoli, Cauliflower, Cabbage, Perennial flowers like Roses and Lilies, Fruit trees and berry bushes',
                'GeneralRecommendation': 'Practice proper irrigation to prevent water stress. Add organic compost for improved soil fertility and structure.'
            },
             'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to meet the nutritional needs of plants. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
             'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
        },
#-------------------------------------------------------------------------------
         'Silt Loam': {
            'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
             'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Gray': {
                'pHRange': '6.0 - 7.5	',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
             'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
        },
#-------------------------------------------------------------------------------
         'Loam': {
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.Most crops and garden plants',
                'CropPlantRecommendation': 'A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels.Practice crop rotation to prevent disease buildup.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce, Trees like Apple, Peach, and Pear, Ornamental plants and flowers	',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Dark Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to meet the nutritional needs of plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots, Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to provide essential nutrients. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
             'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
        },
#-------------------------------------------------------------------------------
         'Sandy Clay Loam': {
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
            'Dark Brown': {
                'pHRange': '6.0 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Iron',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Corn, Beans, Peas, Cucumbers, Squash, Pumpkins',
                'GeneralRecommendation': 'Maintain soil moisture through regular watering. Mulch the soil to conserve moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.2',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Iron',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Potatoes, Carrots',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention.'
            },
            'Light Brown': {
                'pHRange': '6.2 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Iron',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Lettuce, Spinach, Broccoli, Cauliflower, Beets',
                'GeneralRecommendation': 'Maintain soil pH to optimize nutrient availability.'
            },
            'Yellow': {
                'pHRange': '6.5 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Iron',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Peaches, Pears, Nectarines, Apricots, Plums',
                'GeneralRecommendation': 'Monitor soil pH regularly and adjust as needed.'
            },
            'Red': {
                'pHRange': '6.0 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Iron',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Beans, Peas, Tomatoes, Peppers',
                'GeneralRecommendation': 'Control erosion through proper soil management practices.'
            },
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Iron',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Grasses, Legumes, Cereals',
                'GeneralRecommendation': 'Implement crop rotation to improve soil health and fertility.'
            },
            'Blue': {
                'pHRange': '6.5 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Nitrogen-rich fertilizers or organic matter',
                'CropPlantRecommendation': 'Various vegetables and fruits',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to enrich soil nutrients.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
        },
#-------------------------------------------------------------------------------
         'Silty Clay Loam': {
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer to provide essential nutrients.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
              'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer to provide essential nutrients.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
              'Dark Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer to provide essential nutrients.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
        },
#-------------------------------------------------------------------------------
         'Sandy Silt Loam': {
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Use nitrogen-rich fertilizers to support plant growth.',
                'CropPlantRecommendation': 'Various vegetables and fruits',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'Red': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
        },
#-------------------------------------------------------------------------------
         'Loamy Sand': {
            'Green': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Apply compost or organic matter to improve soil fertility and structure. Use micronutrient-rich fertilizers for iron and other trace elements.',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach, Drought-tolerant plants like Lavender and Rosemary, Succulents and Cacti',
                'GeneralRecommendation': 'Implement cover cropping to enrich soil nutrients. Mulch the soil to conserve moisture.'
            },
            'White':{
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Calcium, Phosphorus, Potassium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers with additional phosphorus and potassium. Consider applying calcium carbonate to raise soil pH if needed.',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents, Native plants adapted to sandy conditions',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching. Supplement with micronutrients if needed.'
            },
                        'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to meet the nutritional needs of plants. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
             'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
        },
#-------------------------------------------------------------------------------
         'Sand': {
            'White': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Calcium, Phosphorus, Potassium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers with additional phosphorus and potassium. Consider applying calcium carbonate to raise soil pH if needed',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents, Native plants adapted to sandy conditions',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching. Supplement with micronutrients if needed.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce Trees like Apple, Peach, and Pear Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes - Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
            'Green': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Apply compost or organic matter to improve soil fertility and structure. Use micronutrient-rich fertilizers for iron and other trace elements.',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach, Drought-tolerant plants like Lavender and Rosemary, Succulents and Cacti',
                'GeneralRecommendation': 'Implement cover cropping to enrich soil nutrients. Mulch the soil to conserve moisture.'
            },
              'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to meet the nutritional needs of plants. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
             'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
             'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes, Groundcovers like Clover and Creeping Thyme',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
        },
#-------------------------------------------------------------------------------
         'Sandy Silt': {
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials.',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Low in Organic Matter, Low in Major to Moderate Nutrients',
                'FertilizerRecommendation': 'Light Grey colour - Use balanced NPK fertilizers to provide essential nutrients. Dark Gray colour - Apply compost or organic matter to improve soil fertility and structure.',
                'CropPlantRecommendation': 'Light Grey colour - Drought-tolerant plants like Succulents and Cacti. Dark Gray colour - Native plants adapted to sandy silt conditions.',
                'GeneralRecommendation': 'Light Grey colour - Improve water retention with organic matter and mulching. Dark Gray colour - Implement cover cropping to enrich soil nutrients.'
            },
             'Red': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Low in Organic Matter, Moderate in Major Nutrients',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs.',
                'CropPlantRecommendation': 'Native shrubs and herbaceous perennials',
                'GeneralRecommendation': 'Implement proper irrigation practices to avoid water stress.'
            },
             'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
              'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
        },
#-------------------------------------------------------------------------------
         'Silt': {
            'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers for crops requiring phosphorus. Address iron chlorosis with iron chelates. Test soil for manganese levels and supplement if deficient.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Address iron chlorosis with iron chelates or foliar sprays. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Most crops and garden plants',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Implement crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use phosphorus-rich fertilizers for crops requiring phosphorus. Consider adding sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients. Supplement with manganese sulfate if manganese deficiency occurs.',
                'CropPlantRecommendation': 'Grasses, Cereals, Legumes',
                'GeneralRecommendation': 'Practice crop rotation to manage soil health. Implement soil testing to monitor nutrient levels.'
            },
             'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to meet the nutritional needs of plants. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
            'Green': {
                'pHRange': '6.2 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Radishes, Spinach',
                'GeneralRecommendation': 'Mulch the soil to conserve moisture.'
            },
        },
#-------------------------------------------------------------------------------
         'Gravelly Silt': {
             'Brown': {
                'pHRange': '5.5 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Grapes, Berries, Melons, Lettuce, Spinach',
                'GeneralRecommendation': 'Improve drainage to prevent waterlogging. Add organic matter for nutrient retention.'
            },
             'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Grains like Wheat and Barley',
                'GeneralRecommendation': 'Ensure proper soil aeration for root development. Monitor nutrient levels regularly.'
            },
             'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Phosphorus-based fertilizer',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Beans, Squash',
                'GeneralRecommendation': 'Address drainage issues to prevent waterlogged conditions. Mulch to retain soil moisture.'
            },
             'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Corn, Sunflowers, Legumes',
                'GeneralRecommendation': 'Apply lime if soil pH is too acidic. Implement crop rotation for soil health.'
            },
             'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Nitrogen-based fertilizer',
                'CropPlantRecommendation': 'Leafy greens, Berries, Beans',
                'GeneralRecommendation': 'Avoid excessive irrigation to prevent leaching of nutrients. Use cover crops for soil enrichment.'
            },
             'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Phosphorus-based fertilizer',
                'CropPlantRecommendation': 'Squash, Pumpkins, Cucumbers, Carrots',
                'GeneralRecommendation': 'Improve soil structure with organic matter. Mulch to conserve soil moisture.'
            },
             'Green': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Iron, Sulfur',
                'FertilizerRecommendation': 'Organic matter, Compost, Micronutrient-rich fertilizers',
                'CropPlantRecommendation': 'Carrots, Beets, Radishes, Spinach',
                'GeneralRecommendation': 'Apply organic matter for nutrient supply and water retention. Monitor soil pH regularly.'
            },
             'White': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Calcium, Phosphorus, Potassium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer with extra phosphorus',
                'CropPlantRecommendation': 'Drought-resistant plants, Cacti',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching. Supplement with micronutrients if needed.'
            },
             'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
        },
#-------------------------------------------------------------------------------
         'Gravelly Sand': {
            'Gray': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Low in nutrients; deficient in organic matter',
                'FertilizerRecommendation': 'Light Gray - Apply organic matter, such as compost or well-rotted manure, to improve soil fertility and structure. Use balanced NPK fertilizers with micronutrients.'
                                             '\t\t\t Dark Gray	- Use slow-release fertilizers to reduce nutrient leaching. Supplement with micronutrients like iron and zinc.\n'
                                             '\t\t\t Reddish-Gray - Apply balanced NPK fertilizers with micronutrients.\n'
                                             '\t\t\t Brownish-Gray - Use slow-release fertilizers to minimize nutrient leaching. Supplement with micronutrients.\n'
                                             '\t\t\t Yellowish-Gray - Apply balanced NPK fertilizers with additional phosphorus and potassium. Supplement with micronutrients.\n'
                                             '\t\t\t Greenish-Gray - Use balanced NPK fertilizers with micronutrients.\n'
                                             '\t\t\t Bluish-Gray - Apply slow-release fertilizers to reduce nutrient leaching. Supplement with micronutrients.\n',

                'CropPlantRecommendation': 'Light Gray - Native plants and wildflowers\n'
                                             '\t\t\t Dark Gray - Native plants and wildflowers\n'
                                             '\t\t\t Reddish-Gray - Native grasses and sedges\n'
                                             '\t\t\t Brownish-Gray - Native grasses and sedges\n'
                                             '\t\t\t Yellowish-Gray - Drought-resistant flowers and herbs\n'
                                             '\t\t\t Greenish-Gray - Drought-resistant vegetables and fruits\n'
                                             '\t\t\t Bluish-Gray - Native shrubs and trees\n',

                'GeneralRecommendation': 'Light Gray - Enhance soil fertility with regular application of organic matter. Implement mulching to conserve moisture.\n'
                                          '\t\t\t Dark Gray - Add organic matter to improve soil fertility and water retention. Avoid excessive water application.\n'
                                          '\t\t\t Reddish-Gray - Use drought-resistant plants suitable for gravelly sand.Implement drip irrigation to conserve water.\n'
                                          '\t\t\t Brownish-Gray - Regularly monitor soil pH and nutrient levels. Practice erosion control to prevent soil loss.\n'
                                          '\t\t\t Yellowish-Gray - Improve soil fertility with regular applications of organic matter. Implement mulching to conserve moisture.\n'
                                          '\t\t\t Greenish-Gray - Enhance soil structure with organic matter. Avoid overwatering to prevent waterlogged conditions.\n'
                                          '\t\t\t Bluish-Gray - Implement proper irrigation practices to avoid water stress. Use drought-tolerant plants.\n'
            },
            'Green': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Low in nutrients; deficient in organic matter',
                'FertilizerRecommendation': 'Green - Use balanced NPK fertilizers with micronutrients.\n'
                                             '\t\t\t Brownish-Green - Use balanced NPK fertilizers with micronutrients.\n'
                                             '\t\t\t Yellowish-Green - Apply balanced NPK fertilizers with additional phosphorus and potassium. Supplement with micronutrients.\n',

                'CropPlantRecommendation': 'Green - Drought-resistant ornamental plants\n'
                                             '\t\t\t Brownish-Green - Drought-resistant herbs and succulents\n'
                                             '\t\t\t Yellowish-Green - Drought-resistant ornamental plants\n',

                'GeneralRecommendation': 'Green - Enhance soil structure with organic matter. Avoid over-fertilization to prevent nutrient runoff.\n'
                                          '\t\t\t Brownish-Green - Apply organic matter to improve soil fertility and water retention. Practice proper weed control.\n'
                                          '\t\t\t Yellowish-Green - Implement drip irrigation to minimize water usage. Monitor soil moisture levels regularly.\n'
            },
             'Black': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply compost or well-rotted manure to improve soil structure and fertility. Use nitrogen-rich organic fertilizers like blood meal or fish emulsion to enhance plant growth. Consider adding green manures like clover to fix nitrogen.',
                'CropPlantRecommendation': 'Rice, Wheat, Corn, Sugar beets, Vegetables like Cabbage, Lettuce, and Spinach, Legumes like Peas and Beans, Nutrient-hungry plants that benefit from the water retention of clay soils.',
                'GeneralRecommendation': 'Ensure proper drainage to prevent waterlogging. Add organic matter regularly to improve soil structure and nutrient content.'
            },
            'Dark Brown': {
                'pHRange': '5.5 - 7.0',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to provide essential nutrients to plants. Consider using slow-release fertilizers to reduce nutrient leaching in sandy soils.',
                'CropPlantRecommendation': 'Beans, Peas, Sweet Potatoes, Carrots,Drought-resistant plants like Succulents and Cacti, Deep-rooted plants that can access water deeper in the soil.',
                'GeneralRecommendation': 'Improve soil structure with organic matter to enhance water retention. Mulch the soil to reduce evaporation and maintain moisture.'
            },
            'Brown': {
                'pHRange': '6.0 - 7.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron',
                'FertilizerRecommendation': 'Apply balanced NPK fertilizers to meet the nutritional needs of plants. Supplement with iron chelates or foliar sprays if iron deficiency is observed.',
                'CropPlantRecommendation': 'Soybeans, Alfalfa, Cabbage, Spinach, Vegetables like Lettuce, Broccoli, and Cauliflower, Fruits like Apples and Pears, Flowers like Roses and Chrysanthemums',
                'GeneralRecommendation': 'Maintain soil pH to enhance nutrient availability. Avoid overwatering to prevent waterlogged conditions. Implement crop rotation to manage disease and pest issues.'
            },
            'Light Brown': {
                'pHRange': '6.0 - 7.8',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to ensure plants receive all essential nutrients. Supplement with magnesium sulfate (Epsom salt) for magnesium deficiency. Consider adding gypsum for calcium needs.',
                'CropPlantRecommendation': 'Tomatoes, Peppers, Melons, Lettuce,Trees like Apple, Peach, and Pear, Ornamental plants and flowers',
                'GeneralRecommendation': 'Add organic matter to increase nutrient retention and improve soil structure. Maintain adequate soil moisture for optimal plant growth.'
            },
            'Red': {
                'pHRange': '5.0 - 7.0',
                'AssociatedNutrients': 'Iron, Aluminum, Manganese, Phosphorus',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to address phosphorus deficiencies. Supplement with iron chelates if iron chlorosis is observed.',
                'CropPlantRecommendation': 'Corn, Beans, Peanuts, Apples, Perennial flowers like Daylilies and Hostas, Shade-tolerant plants that benefit from nutrient-rich soils.',
                'GeneralRecommendation': 'Manage erosion to prevent soil loss. Implement proper irrigation practices to avoid water stress.'
            },
            'Yellow': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Sulfur',
                'FertilizerRecommendation': 'Use balanced NPK fertilizers to maintain soil fertility. Consider applying elemental sulfur to lower soil pH if needed.',
                'CropPlantRecommendation': 'Most crops and garden plants, A wide range of vegetables, fruits, and flowers',
                'GeneralRecommendation': 'Regularly monitor soil pH and nutrient levels. Practice crop rotation to prevent disease buildup.'
            },
            'Orange': {
                'pHRange': '5.0 - 7.5',
                'AssociatedNutrients': 'Iron, Aluminum, Phosphorus, Sulfur',
                'FertilizerRecommendation': 'Apply phosphorus-rich fertilizers to meet plant phosphorus needs. Consider adding sulfur to lower soil pH if necessary.',
                'CropPlantRecommendation': 'Squash, Cucumbers, Tomatoes, Flowering plants like Marigold and Zinnia, Drought-resistant plants that can thrive in sandy soils.',
                'GeneralRecommendation': 'Apply mulch to conserve soil moisture and suppress weeds. Avoid over-fertilization to prevent nutrient runoff.'
            },
            'Blue': {
                'pHRange': '7.0 - 8.5',
                'AssociatedNutrients': 'Organic matter, Carbon, Nitrogen',
                'FertilizerRecommendation': 'Apply nitrogen-rich fertilizers to support plant growth. Use organic matter to enhance soil structure and water retention.',
                'CropPlantRecommendation': 'Various vegetables and fruits, Trees and shrubs, Flowering plants and perennials',
                'GeneralRecommendation': 'Avoid excessive tillage to minimize soil erosion. Use cover cropping to improve soil fertility.'
            },
            'White': {
                'pHRange': '7.0 - 8.0',
                'AssociatedNutrients': 'Phosphorus, Potassium, Calcium, Sulfur',
                'FertilizerRecommendation': 'Balanced NPK fertilizer',
                'CropPlantRecommendation': 'Drought-resistant plants like Cacti and Succulents',
                'GeneralRecommendation': 'Improve water retention with organic matter and mulching.'
            },
        },
    }

    if soil_type in soil_info and soil_color in soil_info[soil_type]:
        return soil_info[soil_type][soil_color]
    else:
        return None
#-------------------------------------------------------------------------------
def save_to_json(file_name, data):
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

#-------------------------------------------------------------------------------
#-----------------------------FUNCTION CALLS------------------------------------
#-------------------------------------------------------------------------------

def classify_image(new_image_path):
    
    image_colr = cv2.imread(new_image_path)
    image_colr = cv2.cvtColor(image_colr, cv2.COLOR_BGR2RGB)

    predicted_class, confidence = predict_category(model, new_image_path, class_mapping)
    #-------------------------------------------------------------------------------
    # Composition
    gravel_, sand_, silt_ = classify_percentage(image_fp=new_image_path)
    #-------------------------------------------------------------------------------
    soil_type = find_soil_type(gravel_, sand_, silt_)
    #-------------------------------------------------------------------------------
    modified_image = preprocess(image_colr)
    hex_input = analyze(modified_image)
    soil_color = classify_color(hex_input)
    result = get_soil_info(soil_type, soil_color)
#-------------------------------------------------------------------------------
#----------------------------------JSON FILE------------------------------------
#-------------------------------------------------------------------------------
    if result:
        output_data = {
            "Category": predicted_class,
            "Type": soil_type,
            "Composition": {
                "Gravel": gravel_,
                "Sand": sand_,
                "Silt": silt_
            },
            "Color": soil_color,
        }
        for key, value in result.items():
            output_data[key] = value
        return output_data

        # save_to_json("soil_info.json", output_data)
    else:
        output_data = {
            "Category": "! Image Data Not Available in Database",
            "Type": "! Image Data Not Available in Database",
            "Composition": {
                "Gravel": 0.0,
                "Sand": 0.0,
                "Silt": 0.0
            },
            "Color": "! Image Data Not Available in Database",
            'pHRange': "! Image Data Not Available in Database",
            'AssociatedNutrients': "! Image Data Not Available in Database",
            'FertilizerRecommendation': "! Image Data Not Available in Database",
            'CropPlantRecommendation': "! Image Data Not Available in Database",
            'GeneralRecommendation': "! Image Data Not Available in Database"
        }
        return output_data
    
        # save_to_json("soil_info.json", output_data)

