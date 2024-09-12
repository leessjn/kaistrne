import os
import pandas as pd
from PIL import Image

# Step 1: Load the CSV file
def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Step 2: Rotate and Save the Images, Update the CSV
def rotate_and_update_csv(df, image_folder, output_folder, csv_file):
    new_rows = []  # To store new rows for updated CSV
    
    for index, row in df.iterrows():
        i, j, label = row['i'], row['j'], row['label']
        if (int(j)>=10):
            continue
        if (int(i)<1000):
            image_name = f'cropped_image_{i}_{j}.jpg'  # Adjust format if necessary
        else:
            image_name = f'cropped_image_{i}_{j}.png'  # Adjust format if necessary
        image_path = os.path.join(image_folder, image_name)
        
        if os.path.exists(image_path):
            img = Image.open(image_path)
            
            # Rotate by 30 degrees
            rotated_30 = img.rotate(30, expand=True)
            new_image_name_30 = f'cropped_image_{i}_{j}_{(label+1)%3}.jpg'
            rotated_30.save(os.path.join(output_folder, new_image_name_30))
            # new_rows.append({'i': i, 'j': f'{j}0', 'label': label + 1})
            
            # Rotate by 60 degrees
            rotated_60 = img.rotate(60, expand=True)
            new_image_name_60 = f'cropped_image_{i}_{j}_{(label+2)%3}.jpg'
            rotated_60.save(os.path.join(output_folder, new_image_name_60))
            # new_rows.append({'i': i, 'j': f'{j}1', 'label': label + 2})
            
            print(f"Processed and saved {new_image_name_30} and {new_image_name_60}")
        else:
            print(f"Image {image_name} not found. Skipping.")
    
    # Step 3: Append new rows to the existing CSV and save
    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(csv_file, index=False)
    print(f"Updated CSV saved to {csv_file}")

# Step 4: Set paths and execute
csv_file = 'block_rotation.csv'  # Path to the original CSV file
image_folder = 'able_image/able_image'  # Folder with original images
output_folder = 'able_image/able_image'  # Folder to save rotated images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the original CSV file
df = load_csv(csv_file)

# Rotate images, save them, and update CSV
rotate_and_update_csv(df, image_folder, output_folder, csv_file)
