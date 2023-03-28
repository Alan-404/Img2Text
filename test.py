#%%
from preprocessing.image import ImageProcessor
# %%
image_processor = ImageProcessor()
# %%
import io
# %%
infors = io.open("D:\Datasets\img2text/captions.txt").read().strip().split("\n")
# %%
infors = infors[1:]
# %%
infors
# %%
links = []
texts = []
# %%
for item in infors:
    info = item.split(",")
    links.append(info[0])
    texts.append(info[1])
# %%
links
# %%
for i in range(len(links)):
    links[i] = f"D:\Datasets\img2text\Images\{links[i]}"
# %%
num_data = 5000
# %%
links = links[:num_data+1]
texts = texts[:num_data+1]
# %%

# %%
images = image_processor.process(links)
# %%
images
# %%
type(images)
# %%
from preprocessing.text import TextProcessor

# %%
text_procesor = TextProcessor(tokenizer_path='./tokenizer/tokenizer.pkl')
# %%
texts = text_procesor.process(texts, max_len=41, start_token=True, end_token=True)
# %%
texts
# %%
import pickle
# %%
with open('./clean/text.pkl', 'wb') as file:
    pickle.dump(texts, file, protocol=pickle.HIGHEST_PROTOCOL)
# %%
with open("./clean/image.pkl", 'wb') as file:
    pickle.dump(images, file, protocol=pickle.HIGHEST_PROTOCOL)
# %%
text_procesor.tokenizer.token_index
# %%
from torchvision import models
# %%
resnet = models.resnet50(pretrained=True)
# %%
modules = list(resnet.children())[:-2]
# %%
import torch
# %%
pretrained = torch.nn.Sequential(*modules)
# %%
a = torch.rand((32, 3, 256, 256))
# %%
b = pretrained(a)
# %%
b.size()
# %%

# %%
