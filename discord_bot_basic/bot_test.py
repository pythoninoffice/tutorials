import discord
from discord.ext import commands

TOKEN = 'GET_THIS_FROM https://discord.com/developers/applications'
intents = discord.Intents.default()
intents.message_content = True


client = commands.Bot(command_prefix='.', intents=intents)




@client.event
async def on_ready():
    print('bot ready')
    
@client.event
async def on_message(message):
    #channel = client.get_channel(1030698526075785298)
    #await channel.send('testing')
    print(message.author, message.content, message.channel.id)
#     pass


@client.command()
async def hello(ctx):
    channel = client.get_channel(1033759192596619386)
    await channel.send(f'hello there {ctx.author.mention}')
    
    
client.run(TOKEN)