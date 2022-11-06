import discord
import os, sys
from discord.ext import commands
import asyncio
from discord.ext import tasks
import pathlib
import hashlib
import random
from scripts.txt2img_bot import SDBot

TOKEN = 'your_token_here'

intents = discord.Intents.default()
intents.message_content = True
#client = discord.Client(intents=intents)

client = commands.Bot(command_prefix='.', intents=intents)#, owner_id = ) # this is optional
queues = []
blocking = False
sd_bot = SDBot()
loop = None

@client.event
async def on_ready():
    print('bot ready')




def sd_gen(ctx, queues):
    global blocking

    print(queues)
    if len(queues) > 0:
        blocking = True
        prompt = queues.pop(0)
        mention = list(prompt.keys())[0]
        prompt = list(prompt.values())[0] #convert from dictoinary to string
        filename = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:20]
        if 'seed' in prompt.lower():
            try:
                seed = int(prompt.split('seed')[1].split('=')[1].strip())
            except:
                seed = random.randint(0,4294967295)
            prompt = prompt.split('seed')[0]
        else:
            seed = random.randint(0,4294967295)
            
        sd_bot.makeimg(prompt, filename, seed)
        save_path = r'C:\img'
        channel = client.get_channel(1038609371367747624) #garden_1 chanel
        with open(rf'{save_path}\{filename}.png', 'rb') as f:
            pic = discord.File(f)
            asyncio.run_coroutine_threadsafe(channel.send(f'{mention} "{prompt}", seed= {seed}', file=pic), loop)        
        sd_gen(ctx, queues)
    else:
        blocking = False        
        
def que(ctx, prompt):
    user_id = ctx.message.author.mention
    queues.append({user_id:prompt})
    print(f'{prompt} added to queue')



def check_num_in_que(ctx):
    user = ctx.message.author.mention
    user_list_in_que = [list(i.keys())[0] for i in queues]
    return user_list_in_que.count(user)



@client.command()
async def makeimg(ctx, prompt):
    num = check_num_in_que(ctx)
    if num >=10:
        await ctx.send(f'{ctx.message.author.mention} you have 10 items in queue, please allow your requests to finish before adding more to the queue.')
    else:
    
        global loop
        loop = asyncio.get_running_loop()
        print(loop)      
        que(ctx, prompt)
        await ctx.send(f'{prompt} added to queue')

        if blocking:
            print('this is blocking')
            await ctx.send("currently generating image, please wait")
        else:
            await asyncio.gather(asyncio.to_thread(sd_gen,ctx,queues))
    


@client.command()
async def status(ctx):

    total_num_queued_jobs = len(queues)
    que_user_ids = [list(a.keys())[0] for a in queues]
    if ctx.message.author.mention in que_user_ids:
        user_position = que_user_ids.index(ctx.message.author.mention)
        msg = f"{ctx.message.author.mention} Your job is currently {user_position}/{total_num_queued_jobs} in queue. Estimated time until image is ready: {user_position * 40/60 + 0.5} minutes."
    else:
        msg = f"{ctx.message.author.mention}you don't have a job queued."

    await ctx.send(msg)
              


@client.command()
async def showque(ctx):
    await ctx.send(queues)
    print(queues)

@client.command()
async def chanellstats(ctx):
    print(ctx.channel.id)
    await ctx.send(ctx.channel.id)

client.run(TOKEN)
