import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")


def plot_seg_data(dataset_dir, split, num_images):
    im_split_dir = os.path.join(dataset_dir, split)
    mask_split_dir = os.path.join(dataset_dir, f"{split}_masks")
    
    if not os.path.exists(im_split_dir):
        return f"Invalid split path. Format the path as /dataset_dir/{split}"
    if not os.path.exists(mask_split_dir):
        return f"Invalid split mask path. Format the masks path as /dataset_dir/{split}_masks"
    
    image_names = sorted(os.listdir(im_split_dir))
    
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5*num_images))
    fig.suptitle(f"Images and Masks from {split} set", fontsize=16)
    
    for idx, image_name in enumerate(image_names[:num_images]):
        im_path = os.path.join(im_split_dir, image_name)
        mask_name = os.path.splitext(image_name)[0] + '_mask' + '.gif'
        mask_path = os.path.join(mask_split_dir, mask_name)
        
        try:
            im = Image.open(im_path)
            im = np.array(im)
        except Exception as e:
            print(f"Error loading image {im_path}: {e}")
            continue
        
        try:
            mask = Image.open(mask_path)
            mask = np.array(mask)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            continue
        
        axes[idx, 0].imshow(im)
        axes[idx, 0].set_title(f"Image {idx+1}")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title(f"Mask {idx+1}")
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def calc_dataset_statistics(dataset_dir):
    # Dictionary to store the number of images per split
    split_stats = {}
    
    # Traverse the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            if '_masks' not in dir_name:
                split_folder = os.path.join(root, dir_name)
                
                # Count the number of image files in the current split folder
                num_images = len([f for f in os.listdir(split_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                # Store the count in the dictionary with the split name
                split_stats[dir_name] = num_images

    # Convert the stats into a pandas DataFrame
    df = pd.DataFrame(list(split_stats.items()), columns=['Split', 'Number of Images'])
    
    # Plotting a pie chart
    plt.figure(figsize=(3, 3))
    plt.pie(df['Number of Images'], labels=df['Split'], autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title('Dataset Distribution by Split')
    plt.show()
    
    # Return the DataFrame containing the statistics
    return df