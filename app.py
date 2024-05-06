import streamlit as st
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Set page title and favicon
st.set_page_config(page_title="Burned Areas Detection", page_icon="🔥")

# Define CSS for better styling
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #ff5733;
        padding-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        padding-bottom: 30px;
    }
    .upload {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border: 2px dashed #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .upload p {
        font-size: 18px;
        color: #777;
    }
    .result {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border: 2px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .result img {
        border-radius: 10px;
    }
    .spinner-text {
        text-align: center;
        margin-top: 20px;
        font-size: 18px;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page title and instructions
st.title("Burned Areas Detection")
st.markdown("Detect burned areas in images.")

# Load a pretrained YOLOv8n model
model = YOLO('C:/Users/ASUS/Desktop/p2m/deployment/best.pt')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display feedback while processing
    with st.spinner('Detecting burned areas...'):
        # Read the uploaded image
        image = Image.open(uploaded_file)

        # Run inference on the uploaded image
        results = model([image])  # results list

    # Visualize the results
    st.subheader("Detection Result")
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Display the annotated image
        st.image(im_rgb, caption='Result', use_column_width=True)

# Add a link to the GitHub repository
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <a href="https://github.com/yourusername/your-repo" target="_blank">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
