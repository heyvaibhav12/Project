import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Function to calculate intensity values
def calculate_intensity(image, x, y, w, h):
    if len(image.shape) == 3:  # Convert to grayscale if it's a color image
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image  # Already grayscale

    # Extract the selected region
    region = grayscale[y:y+h, x:x+w]

    # Calculate intensity values
    max_intensity = np.max(region)
    min_intensity = np.min(region)
    avg_intensity = np.mean(region)
    return max_intensity, min_intensity, avg_intensity

# Streamlit App
st.title("Region-Based Image Intensity Analyzer")

# Upload Images
uploaded_files = st.file_uploader(
    "Upload one or more images (JPG/PNG)", type=["jpg", "png"], accept_multiple_files=True
)

if uploaded_files:
    all_data = []  # To store data for CSV export
    for uploaded_file in uploaded_files:
        # Load image using Pillow
        image = Image.open(uploaded_file)
        image_np = np.array(image)  # Convert to NumPy array for OpenCV processing

        st.header(f"Processing: {uploaded_file.name}")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("**Draw a rectangle on the image to select a region**:")

        # Create a canvas for region selection
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
            stroke_width=3,
            stroke_color="#FFFF",
            background_image=image,
            update_streamlit=True,
            height=image_np.shape[0],
            width=image_np.shape[1],
            drawing_mode="rect",
            key=f"canvas_{uploaded_file.name}",  # Unique key for each canvas
        )

        # Process the region if a rectangle is drawn
        if canvas_result and canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                shape = objects[-1]  # Only process the last rectangle drawn
                if shape["type"] == "rect":  # If the object is a rectangle
                    # Get rectangle coordinates
                    x = int(shape["left"])
                    y = int(shape["top"])
                    w = int(shape["width"])
                    h = int(shape["height"])

                    # Calculate intensity values
                    max_intensity, min_intensity, avg_intensity = calculate_intensity(
                        image_np, x, y, w, h
                    )

                    # Store data
                    region_data = {
                        "Image Name": uploaded_file.name,
                        "x": x, "y": y, "Width": w, "Height": h,
                        "Max Intensity": max_intensity,
                        "Min Intensity": min_intensity,
                        "Avg Intensity": avg_intensity
                    }
                    all_data.append(region_data)

                    # Display results
                    st.write(f"**Selected Region Coordinates:** (x={x}, y={y}, w={w}, h={h})")
                    st.write(f"**Maximum Intensity:** {max_intensity:.2f}")
                    st.write(f"**Minimum Intensity:** {min_intensity:.2f}")
                    st.write(f"**Average Intensity:** {avg_intensity:.2f}")

    # Display all collected data as a table
    if all_data:
        df = pd.DataFrame(all_data)
        st.write("### Collected Region Data")
        st.dataframe(df)

        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="region_intensity_data.csv",
            mime="text/csv"
        )
