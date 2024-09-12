import os
import pandas as pd
from PIL import Image
import numpy as np

# Step 1: Load the CSV file
def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Step 2: Tone up and Save the Images, Update the CSV
def toneup_and_update_csv(df, image_folder, output_folder, csv_file):
    new_rows = []  # To store new rows for updated CSV
    
    for index, row in df.iterrows():
        i, j, label = row['i'], row['j'], row['label']
        i=int(i)
        j=int(j)
        if pd.isnull(label) or label == "":
            continue
        label=int(label)
        if (int(j) >= 10):
            continue
        if (int(i)<1000):
            image_name = f'cropped_image_{i}_{j}_{label}.jpg'  # Adjust format if necessary
        else:
            image_name = f'cropped_image_{i}_{j}_{label}.png'  # Adjust format if necessary
        image_path = os.path.join(image_folder, image_name)
        
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Subtract 100 from each pixel value and clip to keep values in valid range [0, 255]
            toned_up_array = np.where(img_array > 20, np.clip(img_array + 50, 0, 255), 0).astype(np.uint8)
            toned_up_img = Image.fromarray(toned_up_array)
            
            # Save the toned-up image
            new_image_name = f'cropped_image_toned_up_{i}_{j}_{label}.jpg'
            toned_up_img.save(os.path.join(output_folder, new_image_name))
            # new_rows.append({'i': i, 'j': f'{j}2', 'label': label})
            
            print(f"Processed and saved {new_image_name}")
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
output_folder = 'toned_up'  # Folder to save toned-up images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the original CSV file
df = load_csv(csv_file)

# Tone up images, save them, and update CSV
toneup_and_update_csv(df, image_folder, output_folder, csv_file)
