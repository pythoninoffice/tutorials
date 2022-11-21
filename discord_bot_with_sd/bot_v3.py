import discord
import os, sys
from discord.ext import commands
from discord import app_commands
import asyncio
from discord.ext import tasks
import pathlib
import hashlib
import random
from webui import initialize
import modules.shared as shared
from modules import txt2img


TOKEN = 'discord_token'

intents = discord.Intents.default()
intents.message_content = True
#client = discord.Client(intents=intents)

client = commands.Bot(command_prefix='.', intents=intents)
queues = []
blocking = False
initialize()
loop = None

@client.event
async def on_ready():
    print('bot ready')
    await client.tree.sync()




def sd_gen(ctx, queues):
    global blocking

    print(queues)
    if len(queues) > 0:
        blocking = True
        user_input = queues.pop(0)
        mention = list(user_input.keys())[0]
        prompt = list(user_input.values())[0]['prompt'] #convert from dictoinary to string
        negative_prompt = list(user_input.values())[0]['negative_prompt']
        seed = list(user_input.values())[0]['seed']
        filename = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:20]
        
        if seed == -1:
            seed = random.randint(0,4294967295)


        img = txt2img.txt2img(f"mdjrny-v4 style, {prompt}",
                negative_prompt, # negative prompt
                '',
                '',
                50, #steps
                0,
                True,
                False,
                1, #'next'
                1,
                7.5,
                seed, #seed
                0,
                0,
                0,
                0,
                False,
                512,
                512,
                False,
                0.7,
                0,
                0,
                0, False, False, False, '', 1, '', 0, '', True, False, False
                )[0][0]

        img.save(rf'C:\img\{filename}.png')
        save_path = r'C:\img'
        channel = client.get_channel(1038609371367747624) #test channel
        with open(rf'{save_path}\{filename}.png', 'rb') as f:
            pic = discord.File(f)
            msg_to_user = f"""{mention} prompt="{prompt}",
                                        negative prompt="{negative_prompt}",
                                        seed= {seed}"""
            asyncio.run_coroutine_threadsafe(channel.send(msg_to_user, file=pic), loop)        
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



@client.hybrid_command()
async def makeimg(ctx, prompt, negative_prompt = '', seed=-1):
    user_input = {'prompt':prompt,
                  'negative_prompt': negative_prompt,
                  'seed': seed}

    num = check_num_in_que(ctx)
    if num >=10:
        await ctx.send(f'{ctx.message.author.mention} you have 10 items in queue, please allow your requests to finish before adding more to the queue.')
    else:
    
        
        global loop
        loop = asyncio.get_running_loop()
        print(loop)      
        que(ctx, user_input)
        #await ctx.send(f'{user_input} added to queue')
        reaction_list = [':thumbsup:',':laughing:', ':wink:', ':heart:', ':pray:', ':eggplant::sweat_drops:', ':100:', ':sloth:', ':snake:', ':underage:']
        reaction_choice = reaction_list[random.randrange(10)]
        await ctx.send(f'{reaction_choice} {ctx.message.author.mention}')
        if blocking:
            print('this is blocking')
            #await ctx.send("currently generating image, please wait")
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

@client.hybrid_command()
async def chanellstats(ctx):
    print(ctx.channel.id)
    await ctx.send(ctx.channel.id)

client.run(TOKEN)
