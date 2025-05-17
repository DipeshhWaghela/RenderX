import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
import os


import torch
print(torch.cuda.is_available())  # Should be True for GPU solution
print(torch.version.cuda)  # Shows your CUDA version

# Set up the app
st.title("Product Image Transformer")
st.write("Upload a image")

# User inputs
uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png", "jpeg"])
prompt = st.text_input("Prompt for transformation")
aspect_ratio = st.selectbox("Aspect ratio", ["1:1", "4:5", "16:9", "9:16"])

if uploaded_file and prompt:
    # Load the image
    init_image = Image.open(uploaded_file).convert("RGB")
    
    # Resize based on aspect ratio
    width, height = init_image.size
    ratio_map = {
        "1:1": (512, 512),
        "4:5": (512, 640),
        "16:9": (1024, 576),
        "9:16": (576, 1024)
    }
    new_size = ratio_map[aspect_ratio]
    init_image = init_image.resize(new_size)
    
    st.image(init_image, caption="Original Image", use_column_width=True)
    
    if st.button("Transform Image"):
        with st.spinner("Generating your image..."):
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                #use_auth_token=st.secrets["HF_TOKEN"]  
                use_auth_token = os.environ.get("HF_TOKEN")
            ).to("cuda")
            
            # Generate image
            images = pipe(
                prompt=prompt,
                image=init_image,
                strength=0.75,  # How much to transform the original
                guidance_scale=7.5
            ).images
            
            # Display and make downloadable
            st.image(images[0], caption="Transformed Image", use_column_width=True)
            
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="transformed_image.png",
                mime="image/png"
            )