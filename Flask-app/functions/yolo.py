import torch
import numpy as np
from PIL import ImageDraw, ImageFont, ImageEnhance

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolo.pt', trust_repo=True)

def yolo_image(img):
    # Define class names and  colors
    class_names = {
        0: "Capot",
        1: "Carrosserie",
        2: "Porte",
        3: "Aile",
        4: "Phare",
        5: "Vitre",
        6: "Deformation",
        7: "Rayure"
        }

    class_colors = {
        0: (0, 0, 255),     
        1: (0, 255, 0),     
        2: (255, 255, 0),   
        3: (255, 0, 255),   
        4: (0, 255, 255),   
        5: (128, 128, 128), 
        6: (255, 165, 0),   
        7: (128, 0, 128)    
        }

    def predict_and_draw(image, confidence_threshold=0.6, text_size=15, box_width=3):
        # Model prediction on image
        result = model(np.array(image))
        df = result.pandas().xyxy[0]

        # Filter by confidence threshold
        filtered_df = df[df['confidence'] > confidence_threshold]

        # Initialize drawing
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("models/Roboto.ttf", text_size)
        legend_y = 10

        for _, row in filtered_df.iterrows():
            # Get bounding info
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            confidence, class_id = row['confidence'], row['class']
            class_name = class_names.get(int(class_id), "Unknown")  # Get class name
            color = class_colors.get(int(class_id), (255, 255, 255))  # Get color

            # Draw bounding box on image
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=box_width)
            label = f"{class_name}: {confidence:.2f}"

            # Calculate text size for label
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

            # Define background and text positioning
            background_x = 10
            background_y = legend_y
            background_width = text_width + 20
            background_height = text_height + 10

            text_x = background_x + (background_width - text_width) // 2
            text_y = background_y + (background_height - text_height) // 2

            # Draw label background
            draw.rectangle([background_x, background_y, background_x + background_width, background_y + background_height], fill=color)

            # Draw label text
            draw.text((text_x, text_y), label, fill="white", font=font)

            legend_y += background_height + 10

        return image

    # Annotate the image with box and labels
    annotated_image = predict_and_draw(img)

    # If no box, enhance image contrast and try again
    if annotated_image is None:
        enhancer = ImageEnhance.Brightness(img)
        image_contrasted = ImageEnhance.Contrast(enhancer.enhance(1.2)).enhance(1.2)
        annotated_image = predict_and_draw(image_contrasted)

    return annotated_image