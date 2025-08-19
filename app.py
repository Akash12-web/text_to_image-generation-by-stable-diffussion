import os
import uuid
from flask import Flask, render_template, request
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

app = Flask(__name__)

# Ensure "static" is a clean folder
if os.path.exists("static"):
    if os.path.isdir("static"):
        # Remove old files
        for f in os.listdir("static"):
            try:
                os.remove(os.path.join("static", f))
            except Exception:
                pass
    else:
        raise RuntimeError("'static' exists but is not a folder. Please delete/rename it.")
else:
    os.makedirs("static")

# Load Stable Diffusion
model_id = "stabilityai/stable-diffusion-2-1"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        try:
            # Generate image
            image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            filename = f"static/{uuid.uuid4().hex}.png"
            image.save(filename)
            return render_template("index.html", image_path=filename, prompt=prompt)
        except Exception as e:
            return f"<h3>Error:</h3><pre>{str(e)}</pre>"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
