import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
import os
from huggingface_hub import login

# Configuration for Render's CPU environment
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")  # Set in Render dashboard
login(token=os.environ["HF_TOKEN"])

@st.cache_resource
def load_pipeline():
    # Use smaller model variant for CPU
    return StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        use_auth_token=os.environ["HF_TOKEN"],
        low_cpu_mem_usage=True
    )

# App UI
st.title("🖼️ Lite Product Transformer")
st.warning("Note: Running on CPU - Generation will be slow (~5-10 mins)")

uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png", "jpeg"])
prompt = st.text_input("Transformation prompt")

if uploaded_file and prompt:
    # Reduce image size to save memory
    init_image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    st.image(init_image, caption="Original Image (resized)")
    
    if st.button("Transform Image"):
        with st.spinner("Generating (this may take 5-10 minutes)..."):
            try:
                pipe = load_pipeline()
                
                # CPU-optimized generation
                image = pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=0.7,  # Lower strength for faster generation
                    num_inference_steps=15,  # Reduced steps
                    guidance_scale=7.0
                ).images[0]
                
                st.image(image, caption="Transformed Image")
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button("Download", buf.getvalue(), "transformed.png")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("This app requires more memory than Render's free tier provides. Consider upgrading or using smaller images.")