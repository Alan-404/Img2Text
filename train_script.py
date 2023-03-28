#%%
import pickle
import numpy as np
from model.transform_capper import TransformerImage
# %%
with open("./clean/text.pkl", 'rb') as file:
    texts = pickle.load(file)
# %%
with open("./clean/image.pkl", 'rb') as file:
    images = pickle.load(file)
# %%
from preprocessing.text import TextProcessor
# %%
text_processor = TextProcessor("./tokenizer/tokenizer.pkl")
# %%
text_processor.loadd_tokenizer("./tokenizer/tokenizer.pkl")
# %%
text_processor.tokenizer.num_tokens
# %%
model = TransformerImage(
    token_size=text_processor.tokenizer.num_tokens+1
)
#%%
import torch
# %%
images = torch.tensor(images, dtype=torch.float32)
texts = torch.tensor(texts)
# %%
images = images.permute((0, 3, 1, 2))
# %%
model.fit(images, texts, epochs=10, batch_size=16, mini_batch=32)
# %%
images.size()
# %%
