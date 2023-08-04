# Horizon-A
Horizon-A revolutionizes soil analysis and personalized crop recommendations using machine learning and image processing. It automates soil analysis, delivering valuable insights to farmers and agronomists.

Horizon-A - Intelligent Soil Analysis and Recommendation System

Description:
Horizon-A is an innovative project aimed at revolutionizing soil analysis and providing personalized recommendations for optimal crop cultivation. Leveraging the power of machine learning and image processing techniques, Horizon-A automates the process of soil analysis and delivers valuable insights to farmers, agronomists, and agricultural enthusiasts.

Process Flow:

Image Upload and Preprocessing: Users can upload soil images through the Horizon-A Android app. The images are stored in an Amazon S3 bucket. On API request, the image is fetched and preprocessed using OpenCV and NumPy to ensure compatibility with the TensorFlow ML model.

ML Model Inference: The preprocessed image is passed to a TensorFlow-powered deep learning model, based on the Xception architecture. This sophisticated model, with over 100 layers and 22 million parameters, accurately classifies the soil into three main categories: Gravel, Sand, and Silt.

Subcategorization and Segmentation: Upon classification, the image is upscaled to 1024x1024 and divided into segments. These segments are further analyzed to determine the percentage composition of gravel, sand, and silt in the soil. The combined results yield a total of 15 subcategories of soil, providing detailed insights into its composition.

Color Analysis: Employing K-means clustering, Horizon-A extracts the dominating color from the soil image. This analysis contributes to a more comprehensive understanding of the soil's properties and characteristics.

Soil Characterization and Recommendations: The combination of category, color, and subtype information is then used to identify the soil's pH range, nutrition values, and specific fertilizer recommendations. Additionally, based on the soil analysis, Horizon-A generates tailored crop and plantation recommendations. Finally, the platform provides general guidelines to support users in optimizing their agricultural practices.

Tools and Technologies:

TensorFlow and scikit-learn: Utilized for developing the machine learning model and enabling efficient soil image analysis.
Convolutional Neural Network (Xception Architecture): Empowered the image classifier to achieve accurate soil categorization.
Custom Dataset: A curated dataset of over 1000 soil images was created, covering three main categories and 15 subcategories to train the ML model effectively.
Data Augmentation and Feature Engineering: Techniques applied to enhance the model's robustness and performance.
Image Segmentation: Utilized for detailed soil analysis to determine the percentage composition of gravel, sand, and silt.
K-means Clustering: Employed for color extraction to gain deeper insights into the soil's properties.
AWS Services: Leveraged Amazon S3 for image uploads and EC2 for model deployment, ensuring seamless integration into the Android app.
Flask API: Developed to integrate the ML model into the Android app and facilitate user-friendly interactions.
Recommendation Dataset: A comprehensive dataset of over 200 unique combinations of fertilizers, soil types, plantation options, and general recommendations.
Horizon-A represents a significant step forward in modern agriculture, making soil analysis and personalized recommendations accessible to all. Whether you are a farmer seeking to maximize crop yield or an agricultural researcher analyzing soil health, Horizon-A is your go-to solution for intelligent soil analysis and precise recommendations.
