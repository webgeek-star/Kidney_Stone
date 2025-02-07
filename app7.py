import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load trained YOLOv5 model
MODEL_PATH = "C:/Users/SRISHA/yolov5/runs/train/exp4/weights/best.pt"  # Update this path
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)

# Custom styles for better text visibility
def set_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.verywellhealth.com/thmb/ICEwe5sN45w2qtaln11ZM2B7yBw=/3600x2400/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1132248020-a3646ab44b7e42d9b6a267edfbead4b1.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            filter: brightness(70%);  /* Reduce background brightness */
        }
        .stTitle {
            font-size: 2.5em;  /* Larger font size */
            font-weight: bold;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7);  /* Add shadow for better visibility */
            background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent black background */
            padding: 10px;
            border-radius: 10px;
            color: white;  /* Bright text color */
        }
        .stMarkdown, .stHeader {
            background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent black background */
            padding: 10px;
            border-radius: 10px;
            color: white;  /* Bright text color */
        }
        h1, h2, h3, h4, h5, h6 {
            font-weight: bold;
            color: black;  /* Ensure headers are bright */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call custom styles
set_custom_styles()

# Conversion factor: pixels to millimeters
PIXEL_TO_MM_CONVERSION = 0.264583  # Example conversion factor (adjust based on image resolution)

# Streamlit App UI
st.title("🩺 Real-time Kidney Tumor Segmentation using YOLOv5")
st.write("Upload a CT/MRI image to detect and classify kidney tumors.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to a PIL Image
    image = Image.open(uploaded_file)

    # Ensure image is RGB (some images might be grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to YOLOv5 input size (640x640)
    image = image.resize((640, 640))

    # Convert image to OpenCV format (BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform inference
    results = model(image)

    # Draw bounding boxes
    results.render()
    processed_image = Image.fromarray(results.ims[0])

    # Display the processed image with bounding boxes
    st.image(processed_image, caption="Detection Results", use_container_width=True)

    # Extract and filter results (apply confidence filtering)
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] > 0.2]  # Filter out low-confidence detections

    # Get image center to determine left/right kidney
    image_center_x = image.shape[1] // 2  # Half of image width

    # Count the number of stones before looping
    stone_count = (detections["name"] == "stone").sum()
    stone_printed = False  # Flag to ensure we print stone count only once

    # Display results in a structured format
    st.subheader("🔍 Detection Summary")

    if len(detections) == 0:
        st.markdown("**No objects detected in the image.**")
    else:
        for _, row in detections.iterrows():
            class_name = row["name"]
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            bbox_center_x = (x1 + x2) // 2  # Get bounding box center
            bbox_width_px = x2 - x1
            bbox_height_px = y2 - y1

            # Convert tumor size to millimeters
            bbox_width_mm = bbox_width_px * PIXEL_TO_MM_CONVERSION
            bbox_height_mm = bbox_height_px * PIXEL_TO_MM_CONVERSION

            if class_name.lower() == "tumor":
                kidney_side = "Left Kidney" if bbox_center_x < image_center_x else "Right Kidney"
                st.markdown(f"**⚠️ Tumor detected in {kidney_side}!**")
                st.markdown(f"   - **Tumor Size:** {bbox_width_mm:.2f}mm × {bbox_height_mm:.2f}mm")

            elif class_name.lower() == "stone" and not stone_printed:
                st.markdown("**🪨 Stone detected!**")
                st.markdown(f"   - **Number of stones detected:** {stone_count}")
                stone_printed = True  # Prevent duplicate printing

            #else:
              #  st.markdown(f"✅ **{class_name.capitalize()} detected.**")
