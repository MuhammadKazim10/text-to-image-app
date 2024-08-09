import streamlit as st
from PIL import Image
import os
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

def generate_image(prompt):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512)["images"][0]
    return image

st.title("Text-to-Image Generation")

prompt = st.text_input("Enter a description of the image you want to generate:")

if st.button("Generate"):
    with st.spinner("Generating image..."):
        try:
            image = generate_image(prompt)
            st.image(image, use_column_width=True)
        except Exception as e:
            st.error(f"Error generating image: {e}")