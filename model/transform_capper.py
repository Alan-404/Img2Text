import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model.components.encoder import Encoder
from model.components.decoder import Decoder

from typing import Callable

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class TransformerImageModel(nn.Module):
    def __init__(self, 
                 token_size: int, 
                 n: int, d_model: int, 
                 heads: int, 
                 d_ff: int, 
                 dropout_rate: float, 
                 eps: float, 
                 activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(token_size=token_size, n=n, d_model=d_model, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.to(device)
    def forward(self, image: Tensor, text: Tensor):
        encoder_output = self.encoder(image)
        decoder_output = self.decoder(text, encoder_output)

        return decoder_output
    
class TransformerImage:
    def __init__(self, 
                 token_size: int,
                 n: int = 6, 
                 d_model: int = 512, 
                 heads: int = 8, 
                 d_ff: int = 2048, 
                 dropout_rate: float = 0.1, 
                 eps: float = 0.1, 
                 activation: Callable[[Tensor], Tensor] = F.relu) -> None:
        self.model = TransformerImageModel(
            token_size=token_size,
            n=n,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            eps=eps,
            activation=activation
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters())

        self.loss = 0.0

    def build_dataset(self, inputs: Tensor, labels: Tensor, batch_size: int):
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size ,shuffle=True)

        return dataloader
    
    def loss_function(self, outputs: Tensor, labels: Tensor) -> Tensor:
        batch_size = labels.size(0)

        loss = 0.0
        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], labels[batch])
        loss = loss / batch_size

        return loss

    def train_step(self, images: Tensor, texts: Tensor, labels: Tensor):
        self.optimizer.zero_grad()

        outputs = self.model(images, texts)

        loss = self.loss_function(outputs, labels)
        loss.backward()

        self.optimizer.step()

        self.loss += loss.item()

    def fit(self, images: Tensor, texts: Tensor, epochs: int = 1, batch_size: int = 1, mini_batch: int = 1):
        dataset = self.build_dataset(images, texts, batch_size=batch_size)

        total = len(dataset)
        delta = total - (total//mini_batch)*mini_batch

        for epoch in range(epochs):
            for index, data in enumerate(dataset, 0):
                encoder_inputs = data[0].to(device)
                decoder_inputs = data[1][:, :-1].to(device)
                labels = data[1][:, 1:].to(device)

                print(encoder_inputs.size())
                print(decoder_inputs.size())
                print(labels.size())

                self.train_step(encoder_inputs, decoder_inputs, labels)
                if index%mini_batch == mini_batch - 1:
                    print(f"Epoch: {epoch} Batch: {index+1} Loss: {(self.loss/mini_batch):.4f}")
                    self.loss = 0.0
                elif index == total-1:
                    print(f"Epoch: {epoch} Batch: {index+1} Loss: {(self.loss/delta):.4f}")
                    self.loss = 0.0

        


