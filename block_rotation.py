import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
def load_dataset_from_csv(csv_file, image_folder):
    df = pd.read_csv(csv_file)

    image_paths = []
    labels = []
        
    for index, row in df.iterrows():
        i, j, label = row['i'], row['j'], row['label']
        # Construct the image file name based on i and j values
        image_name = f'cropped_image_{i}_{j}_{label}.jpg'  # or .jpg depending on your image format
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(label)
        else:
            print(f"Image {image_name} not found.")
    return image_paths, labels

# Step 2: Preprocess Images
def preprocess_image(image_paths, labels, image_size=(128, 128)):
    images = []
    for image_path in image_paths:
        image = load_img(image_path, target_size=image_size)
        image_array = img_to_array(image) / 255.0  # Normalize the image
        images.append(image_array)
        
    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def predict_rotation(image_path, model):
    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Create batch dimension   
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
        
    return predicted_class
image_size = (128, 128)
try:
    model = tf.keras.models.load_model('rotation_model_from_csv.h5')
except:
    # Step 1: Load CSV file and prepare dataset
    # Load dataset from CSV
    csv_file = 'block_rotation.csv'  # Replace with actual CSV file path
    image_folder = 'able_image/able_image'  # Replace with actual image folder path
    image_paths, labels = load_dataset_from_csv(csv_file, image_folder)

    # Preprocess images and split the dataset
    images, labels = preprocess_image(image_paths, labels, image_size=image_size)

    # Split data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Step 3: Define CNN Model
    def create_cnn_model():
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(3, activation='softmax')  # Assuming 3lasses
        ])
        
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model

    # Step 4: Initialize model and train
    model = create_cnn_model()

    # Fit the model on the training data
    history = model.fit(
        train_images, train_labels,
        epochs=10,
        batch_size=32,
        validation_data=(val_images, val_labels)
    )

    # Step 5: Save the trained model
    model.save('rotation_model_from_csv.h5')

    # Step 6: Prediction on a new image
    # Example usage
    model = tf.keras.models.load_model('rotation_model_from_csv.h5')
image_path = 'able_image/able_image/cropped_image_2_5.jpg'  # Replace with actual image path
predicted_label = predict_rotation(image_path, model)
print(f'Predicted label: {predicted_label}')
