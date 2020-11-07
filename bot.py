import asyncio
import discord
from evaluate import generate
import json
from concurrent.futures import ThreadPoolExecutor

client = discord.AutoShardedClient()
config_general=json.loads(open("config.json","r").read())["discord"]

@client.event
async def on_ready():
    print('Logged in as '+client.user.name+' (ID:'+str(client.user.id)+') | Connected to '+str(len(client.guilds))+' servers')
    print('--------')
    print("Discord.py verison: " + discord.__version__)
    print('--------')
    print(str(len(client.shards))+" shard(s)")
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="my stupid self ✨✨"))
    
@client.event
async def on_message(message):
    loop = asyncio.get_event_loop()
    if message.author.bot == False and message.guild != None:
        if message.content.lower().startswith(config_general["prefix"]):
            if message.channel.id == 771859532136710154:
                await message.channel.trigger_typing()
                out = await loop.run_in_executor(ThreadPoolExecutor(), generate, message.content[len(config_general["prefix"]):])
                await message.channel.send(out["output"][:-len(" |endofgeneration|")].replace(" |br| ", "\n").replace(" |aigenerationstart| ", "\n"))
                await message.channel.send(f'```{out["input_encoded"]}\n{out["output_encoded"]}\nDone in {out["time"]} seconds```')

client.run(config_general["token"])