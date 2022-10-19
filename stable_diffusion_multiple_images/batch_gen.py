import pandas as pd
import xlwings as xw
from scripts.txt2img_batch import SD
import random

df_prompts = pd.read_csv(r'C:\Users\jay\Desktop\stable_diffusion\stable-diffusion-main\scripts\prompts.csv')
img_files = r'C:\Users\jay\Desktop\stable_diffusion\stable-diffusion-main\outputs\txt2img-samples\multiple'

prompts = df_prompts['prompts']
num =10
sd_nvidia = SD()
wb = xw.Book()
ws = wb.sheets[0]
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

ws.range(1,1).value = 'Prompts'
ws.range(f"2:{len(prompts)+1}").row_height = 130
ws.range(f'B:{letters[num]}').column_width = 30

r = 2
for p in prompts:
    c = 2
    for s in range(1,num+1):
        seed = random.randint(0,4294967295)
        #seed = 100
        sd_nvidia.makeimg(p, f'{r}_{c}',seed = seed)
        ws.range(r,1).value = p
        ws.range(r,c).value = seed
        ws.pictures.add(rf'{img_files}\{r}_{c}.png', left = ws.range(r,c).left, top = ws.range(r,c).top, scale = 0.3)
        c += 1
    r += 1


wb.save(r'C:\Users\jay\Desktop\stable_diffusion\stable-diffusion-main\scripts\prompts_grid.xlsx')