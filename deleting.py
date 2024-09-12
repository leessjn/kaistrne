import os
import pandas as pd

def clean_and_rename_images(csv_file, image_folder):
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Create a list to store rows to delete
    rows_to_delete = []
    
    for index, row in df.iterrows():
        i, j, label = row['i'], row['j'], row['label']
        i=int(i)
        j=int(j)
        # Construct the old image file name (without label)
        if (int(i)<1000):
            old_image_name = f'cropped_image_{int(i)}_{int(j)}.jpg'  # Adjust format if necessary
        else:
            old_image_name = f'cropped_image_{int(i)}_{int(j)}.png'  # Adjust format if necessary
        old_image_path = os.path.join(image_folder, old_image_name)
        print(old_image_path)
        # Check if the label is NaN or empty, and if the file exists
        if pd.isnull(label) or label == "":
            if os.path.exists(old_image_path):
                os.remove(old_image_path)  # Delete the image file
                print(f"Deleted image: {old_image_name}")
            
            rows_to_delete.append(index)  # Mark the row for deletion
        else:
            label=int(label)
            # If the image exists and has a valid label, rename it to include the label
            new_image_name = f'cropped_image_{int(i)}_{int(j)}_{int(label)}.jpg'
            new_image_path = os.path.join(image_folder, new_image_name)
            
            if os.path.exists(old_image_path):
                os.rename(old_image_path, new_image_path)  # Rename the file
                print(f"Renamed {old_image_name} to {new_image_name}")
            
            # Update the CSV to reflect the new image name (if needed)
            # df.at[index, 'image_name'] = new_image_name  # Optional: if you want to keep track of new image names

    # Drop the rows with no label from the dataframe
    df_cleaned = df.drop(rows_to_delete)
    
    # Save the cleaned CSV
    df_cleaned.to_csv(csv_file, index=False)
    print(f"Updated CSV saved: {csv_file}")

# Example usage
csv_file = 'block_rotation.csv'  # Replace with actual CSV file path
image_folder = 'able_image/able_image'  # Replace with actual image folder path

clean_and_rename_images(csv_file, image_folder)
