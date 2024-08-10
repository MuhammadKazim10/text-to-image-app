import streamlit as st
from PIL import Image
import os
from gradio_client import Client

# Initialize the Gradio client
client = Client("mukaist/DALLE-4K",api_name="/run")

def generate_image(prompt):
    result = client.predict(
        prompt=prompt,
        negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
        use_negative_prompt=True,
        style="3840 x 2160",
        seed=0,
        width=1024,
        height=1024,
        guidance_scale=6,
        randomize_seed=True,
        api_name="/run"
    )
    
    if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], list):
        image_data = result[0]
        if len(image_data) > 0 and 'image' in image_data[0]:
            image_path = image_data[0]['image']
            if os.path.exists(image_path):
                return Image.open(image_path)
    
    raise ValueError(f"Unable to process the API result: {result}")

st.title("Text-to-Image Generation with DALLE-4K")

prompt = st.text_input("Enter a description of the image you want to generate:")

if st.button("Generate"):
    with st.spinner("Generating image..."):
        try:
            image = generate_image(prompt)
            st.image(image, use_column_width=True)
        except Exception as e:
            st.error(f"Error generating image: {e}")