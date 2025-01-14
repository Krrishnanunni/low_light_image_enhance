

import cv2
import numpy as np
import os

def enhance_low_light_image(input_path, output_path, gamma=1.2, clip_limit=1.5, tile_grid_size=(8, 8)):
    """
    Enhances low-light images in the input folder using CLAHE and gamma correction with balanced parameters.
    
    Parameters:
        input_path (str): Path to the input folder containing low-light images.
        output_path (str): Path to the output folder to save enhanced images.
        gamma (float): Gamma correction factor (default=1.2 for mild brightening).
        clip_limit (float): Threshold for contrast limiting in CLAHE (default=1.5).
        tile_grid_size (tuple): Grid size for CLAHE (default=(8, 8)).
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        input_file = os.path.join(input_path, filename)
        
        if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
            print(f"Skipping non-image file: {filename}")
            continue

        image = cv2.imread(input_file)
        if image is None:
            print(f"Could not read image: {filename}")
            continue

        # Step 1: Convert to LAB color space for CLAHE
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        
        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l_channel = clahe.apply(l_channel)

        # Merge the CLAHE-enhanced L-channel back
        lab_enhanced = cv2.merge((enhanced_l_channel, a_channel, b_channel))
        image_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Step 2: Apply Gamma Correction
        normalized_image = image_clahe / 255.0
        gamma_corrected = np.power(normalized_image, 1 / gamma)
        enhanced_image = np.uint8(gamma_corrected * 255)

        # Save the enhanced image
        output_file = os.path.join(output_path, filename)
        cv2.imwrite(output_file, enhanced_image)
        print(f"Enhanced image saved: {output_file}")

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"

    print(f"Enhancing images from folder: {input_folder}")
    enhance_low_light_image(input_folder, output_folder, gamma=1.2, clip_limit=1.5, tile_grid_size=(8, 8))
    print("Processing complete!")
