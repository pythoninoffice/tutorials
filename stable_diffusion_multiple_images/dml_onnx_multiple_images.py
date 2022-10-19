import inspect
import warnings
from typing import List, Optional, Union

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
# from diffusers import StableDiffusionSafetyChecker
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
import xlwings as xw
import random




class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format(format)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)
        
        onnx = False
        if "execution_provider" in kwargs:
            onnx = True
            self.scheduler = self.scheduler.set_format("np")
            ep = kwargs.pop("execution_provider")
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.enable_mem_pattern=False
            unet_sess = ort.InferenceSession("onnx/unet.onnx", so, providers=[ep])
            post_quant_conv_sess = ort.InferenceSession("onnx/post_quant_conv.onnx", so, providers=[ep])
            decoder_sess = ort.InferenceSession("onnx/decoder.onnx", so, providers=[ep])
            encoder_sess = ort.InferenceSession("onnx/encoder.onnx", so, providers=[ep])

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=False,
            return_tensors="pt",
        )
        
        if onnx: text_embeddings = encoder_sess.run(None, {"text_input": text_input.input_ids.numpy()})[0]
        else: text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, truncation=False, return_tensors="pt"
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if onnx: 
                uncond_embeddings = encoder_sess.run(None, {"text_input": uncond_input.input_ids.numpy()})[0]
                text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
            else: 
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, generator=generator, device=self.device)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)
        if onnx: latents = latents.numpy() # use pytorch rand to get consistent results

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            if onnx and do_classifier_free_guidance: latent_model_input = np.concatenate([latents] * 2)
            else: latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                if onnx: latent_model_input = latent_model_input.astype('float32')

            # predict the noise residual
            if onnx:
                inp = {"latent_model_input": latent_model_input, 
                       "t": np.array([t], dtype=np.int64), 
                       "encoder_hidden_states": text_embeddings}
                noise_pred = unet_sess.run(None, inp)[0]
            else:
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            # perform guidance
            if do_classifier_free_guidance:
                if onnx: noise_pred_uncond, noise_pred_text = np.array_split(noise_pred, 2)
                else: noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        if onnx:
            latents = post_quant_conv_sess.run(None, {"latents": latents.astype("float32")})[0]
            image = decoder_sess.run(None, {"latents": latents})[0]
            image = np.clip((image / 2 + 0.5), 0, 1)
            image = np.transpose(image, (0, 2, 3, 1))
        else:
            image = self.vae.decode(latents)
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        # image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image,}# "nsfw_content_detected": has_nsfw_concept}


if __name__ == '__main__':
    #img_url = r'C:\Users\jay\Desktop\PythonInOffice\stable_diffusion_amd\diffusers-dml\examples\inference\test2.jpg'
    #url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    #response = requests.get(url)
    #init_image = Image.open(BytesIO(response.content)).convert("RGB")
    #init_image = init_image.resize((768, 512))

    # init_image = Image.open(img_url)
    # init_image = init_image.resize((512,512))
    
    
    df_prompts = pd.read_csv(r'C:\Users\jay\Desktop\stable_diffusion\stable-diffusion-main\scripts\prompts.csv')
    img_files = r'C:\Users\jay\Desktop\stable_diffusion\stable-diffusion-main\outputs\txt2img-samples\multiple'
    prompts = df_prompts['prompts']
    num =3
    wb = xw.Book()
    ws = wb.sheets[0]
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=lms, use_auth_token=True)
    
    ws.range(1,1).value = 'Prompts'
    ws.range(f"2:{len(prompts)+1}").row_height = 130
    ws.range(f'B:{letters[num]}').column_width = 30


    r = 2
    for p in prompts:
        c = 2
        for s in range(1,num+1):
            seed = random.randint(0,4294967295)
            torch.manual_seed(seed)
            #seed = 100
            image = pipe(prompt, height=512, width=512, num_inference_steps=30, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider")["sample"][0]
            image.save(f"{r}_{c}.png") 
            ws.range(r,1).value = p
            ws.range(r,c).value = seed
            ws.pictures.add(rf'{img_files}\{r}_{c}.png', left = ws.range(r,c).left, top = ws.range(r,c).top, scale = 0.3)
            c += 1
        r += 1


    wb.save(r'C:\Users\jay\Desktop\stable_diffusion\stable-diffusion-main\scripts\prompts_grid_amd.xlsx')

    
    #torch.manual_seed(42)

    # prompt = ["""background dark, block houses, eastern Europe, city highly detailed oil painting,
    # unreal 5 render, rhads, bruce pennington, studio ghibli, tim hildebrandt, digital art,
    # octane render, beautiful composition, trending on artstation, award-winning photograph, masterpiece"""]
    

    
    # image2 = pipe(prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider")["sample"][1]
    # image.save("Dml_02.png")
    # Works on AMD Windows platform
    # Image width and height is set to 512x512
    # If you need images of other sizes (size must be divisible by 8), make sure to save the model with that size in save_onnx.py
    # For example, if you need height=512 and width=768, change create_onnx.py with height=512 and width=768 and run the prompt below with height=512 and width=768
    # Default values are height=512, width=512, num_inference_steps=50, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider"

    # prompt = "a photo of an astronaut riding a horse on mars"
    # image = pipe(prompt, height=512, width=768, num_inference_steps=50, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider")["sample"][0]
    # image.save("Dml_1.png")
  