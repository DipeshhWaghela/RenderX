import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io

@st.cache_resource
def load_pipeline():
    return StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        use_auth_token=st.secrets["HF_TOKEN"]
    )

# App UI
st.title("Product Image Transformer")
uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png", "jpeg"])
prompt = st.text_input("Prompt for transformation")

if uploaded_file and prompt:
    init_image = Image.open(uploaded_file).convert("RGB")
    st.image(init_image, caption="Original Image")
    
    if st.button("Transform Image"):
        with st.spinner("Generating..."):
            pipe = load_pipeline()
            image = pipe(
                prompt=prompt,
                image=init_image,
                strength=0.75
            ).images[0]
            
            st.image(image, caption="Transformed Image")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "result.png")