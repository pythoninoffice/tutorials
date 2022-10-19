import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.half()
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    has_nsfw_concept = [i if i == False else not i for i in has_nsfw_concept]
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

class SD:
    def __init__(self):
        self.config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
        self.model = load_model_from_config(self.config , r"C:\Users\jay\Desktop\stable-diffusion-webui\models\Stable-diffusion\novelai-model.ckpt")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.sampler = PLMSSampler(self.model) ## assume --plms by default
        self.n_samples = 1
        self.output_path = '' ## TODO
        self.precision = 'autocast'
        self.precision_scope = autocast if self.precision=="autocast" else nullcontext
        self.start_code = None
        # if self.fixed_code:
        #     self.start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        self.W = 512
        self.H = 512    
        self.ddim_steps = 50
        self.scale = 7.5
        self.ddim_eta = 0
        self.n_iter = 1
        self.C = 4
        self.f = 8
        self.n_samples = 1
        self.batch_size = self.n_samples
        self.outpath = "outputs/txt2img-samples"
        self.sample_path = os.path.join(self.outpath, "multiple")
        os.makedirs(self.sample_path, exist_ok=True)
        self.base_count = len(os.listdir(self.sample_path))



    def makeimg(self, prompt, filename, seed=None):
        if seed != None:
            seed_everything(seed)
        self.data = [prompt]  #one prompt at a time
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    #all_samples = list()
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(self.data, desc="data"):
                            uc = None
                            if self.scale != 1.0:
                                uc = self.model.get_learned_conditioning(self.batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [self.C, self.H // self.f, self.W // self.f]
                            samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=self.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=self.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=self.ddim_eta,
                                                            x_T=self.start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            #x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            x_checked_image = x_samples_ddim
                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                #img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(self.sample_path, f"{filename}.png"))
                                self.base_count += 1

                            # if not opt.skip_grid:
                            #     all_samples.append(x_checked_image_torch)

                    # if not opt.skip_grid:
                    #     # additionally, save as grid
                    #     grid = torch.stack(all_samples, 0)
                    #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    #     grid = make_grid(grid, nrow=n_rows)

                    #     # to image
                    #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    #     img = Image.fromarray(grid.astype(np.uint8))
                    #     img = put_watermark(img, wm_encoder)
                    #     img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    #     grid_count += 1

                    toc = time.time()              