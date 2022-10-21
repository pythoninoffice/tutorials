from diffusers import OnnxStableDiffusionPipeline
#from PIL import Image

#baseImage = Image.open(r"in.jpg").convert("RGB") # opens an image directly from the script's location and converts to RGB color profile
#baseImage = baseImage.resize((768,512))

prompt = "A fantasy landscape, trending on artstation"
denoiseStrength = 0.8 # a float number from 0 to 1 - decreasing this number will increase result similarity with baseImage
steps = 25
scale = 7.5

pipe = OnnxStableDiffusionPipeline.from_pretrained(r"E:\PythonInOffice\amd_img2img_demo\onnx",
                                                   provider="DmlExecutionProvider")
image = pipe(prompt, strength=denoiseStrength, num_inference_steps=steps, guidance_scale=scale).images[0]
image.save("t2i.png")
