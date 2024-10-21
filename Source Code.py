import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to detect tables using rectangles in an image
# Tables with borders are rectangles
def detect_tables_figure(img):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Perform Canny edge detection to detect the boundaries of objects
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3) 
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []

    # Iterate through detected contours
    for contour in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

        # If the contour has four corners, it might be a table
        if len(approx) == 4:
            # Calculate the area of the contour
            area = cv2.contourArea(approx)

            # If the area is above 5000 pixels, consider it as a potential table
            # Prevents the contour from considering text as table 
            if area > 5000:  
                tables.append(approx)

    # After detecting the tables, removing the tables by making the region with the tables white 
    # Goes undetected in the paragraphs extraction function    

    # Create a blank image of the same size as the grayscale image
    mask = np.zeros_like(img)

    # Filled the rectangles (potential tables) in white on the mask
    for table in tables:
        cv2.fillPoly(mask, [table], (255, 255, 255))

    # Create a copy of the grayscale image
    result_img = img.copy()

    # Draw rectangles (with white borders) on the result image
    for table in tables:
        cv2.drawContours(result_img, [table], 0, (255, 255, 255), 5)

    # Combine the result image and the mask to fill the rectangles with white color
    result_img = cv2.bitwise_or(result_img, mask)

    return result_img

# Detects if there is an figure/image based on average pixel value
def brightness_check(image):
    avg_brightness = np.mean(image)
    return avg_brightness < 200  

# Extract paragraphs from an image
def extract_paragraphs(image):
    # Calculate column-wise projection to detect spaces between text columns
    col_projection = np.sum(image == 0, axis=0)
    columns = []
    start_col = None
    consecutive_blank_cols = 0
    
    # Extract columns by detecting consecutive blank columns
    for i, col_sum in enumerate(col_projection):
        if np.sum(col_sum) > 0 and start_col is None:
            # Offset to ensure full text coverage in columns
            start_col = i - 40  
            consecutive_blank_cols = 0
        elif np.sum(col_sum) == 0 and start_col is not None:
            consecutive_blank_cols += 1
            # Loop end when the column detect 50 empty columns consecutively
            if consecutive_blank_cols >= 50:  
                end_col = i 
                column = image[:, start_col:end_col]
                columns.append(column)
                start_col = None
                consecutive_blank_cols = 0
    
    paragraphs = []
    
    # Extract paragraphs within columns
    for column in columns:
        row_projection = np.sum(column == 0, axis=1)
        start_row = None 
        # Threshold to distinguish paragraphs from non-textual elements based on the density of black pixels in a column
        density_threshold = 0.01 
        
        for i, row_sum in enumerate(row_projection):
            if np.sum(row_sum) > 0 and start_row is None:
                start_row = i - 40  # Offset for full paragraph coverage
            elif np.sum(row_sum) == 0 and start_row is not None:
                avg_density = np.mean(row_projection[i:i+50])  # Average density of text pixels in the next 50 rows
                if avg_density < density_threshold: 
                    end_row = i + 40  # Offset for full paragraph coverage
                    paragraph = column[start_row:end_row, :]
                    
                    # Filter the extracted paragraphs so figures/images are not extracted as well 
                    if not brightness_check(paragraph):
                        paragraphs.append(paragraph)
                    
                    start_row = None
    
    return paragraphs

def program(img_file_path):
    # Read the image
    img = cv2.imread(img_file_path)

    # Detect tables or figure in the image
    detected_tables_img = detect_tables_figure(img)

    # Extract paragraphs from the processed image
    paragraphs = extract_paragraphs(detected_tables_img)

    # Display the detected paragraphs in a subplot grid
    num_paragraphs = len(paragraphs)
    rows = (num_paragraphs // 3) + 1 if num_paragraphs % 3 != 0 else num_paragraphs // 3
    cols = min(num_paragraphs, 3)

    plt.figure(figsize=(12, 8))
    
    for idx, paragraph_img in enumerate(paragraphs, start=1):
        # Generate file name by including image file path and paragraph index
        # Split the img_file_path to remove the file extension and then append the paragraph index
        base_name = img_file_path.split('.')[0]
        file_name = f'{base_name}_extracted_paragraph_{idx}.png'
            
        # Save extracted paragraphs as separate images
        cv2.imwrite(file_name, paragraph_img) 

        plt.subplot(rows, cols, idx)
        plt.imshow(paragraph_img, cmap='gray')
        plt.title(f'Paragraph {idx}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Enter the image file path
image_file_path = "006.png"
program(image_file_path)