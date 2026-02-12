# models.py

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

from transformer import PositionalEncoding


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)
    
    
class NeuralLanguageModel(LanguageModel):
    def __init__(self, transformer, vocab_index, num_positions):
        """
        :param transformer: trained Transformer network
        :param vocab_index: an Indexer of the character vocabulary (27 characters)
        :param num_positions: max sequence length that will be fed to the model
        """
        self.transformer = transformer
        self.vocab_index = vocab_index
        self.num_positions = num_positions

    def get_next_char_log_probs(self, context):
        """
        :param context: input string providing previous characters
        :return: log probability distribution for the next character
        """
        self.transformer.eval()
        
        # Filters context to the num_positions - 1 most recent characters
        context = context[-(self.num_positions - 1):]
        input = " " + context
        
        # Convert characters to indices
        input_tensor = torch.LongTensor([self.vocab_index.index_of(c) for c in input])
        
        # Run model
        log_probabilities = self.transformer.forward(input_tensor)
        next_char_log_probs = log_probabilities[-1]
            
        return next_char_log_probs.detach().cpu().numpy().astype(np.float64)  # add float type to avoid error


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, nhead, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model
        :param d_model: dimension of the inputs and outputs of the layer
        :param nhead: number of heads in the mulitiheadattention models
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        
        # Map character index to d_model dimensional vector
        # [num_positions] -> [num_positions, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Add positional information to embeddings
        self.positional_encoding = PositionalEncoding(d_model, num_positions=num_positions, batched=False)
        
        # Encoder transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear projection of encoder to vocabulary size
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Convert logits to log probabilities
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: softmax log probabilities
        """
        # Map character indices to vectors then add positional encodings
        input_vecs = self.positional_encoding(self.embedding(indices))
        input_vecs = input_vecs.unsqueeze(1)  # add batch dimension (no batching, so just 1)
        
        # Create mask so letters cannot attend to future letters
        mask = nn.Transformer.generate_square_subsequent_mask(indices.shape[0]).to(input_vecs.device)
        
        attention = self.transformer(input_vecs, mask=mask)
        attention = attention.squeeze(1)  # remove batch dimension
        
        log_probs = self.log_softmax(self.output_projection(attention))
        
        return log_probs
    
    
def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # Model parameters
    vocab_size = 27
    num_positions = 20
    d_model = 64
    nhead = 4
    num_layers = 1
    
    # Hyperparameters
    learning_rate = 0.0001
    num_epochs = 15

    model = Transformer(vocab_size, num_positions, d_model, nhead, num_layers)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        # Randomize order each epoch
        random.seed(epoch)
        ex_indexes = list(range(0, len(train_text) - num_positions, num_positions))
        random.shuffle(ex_indexes)
        
        total_loss = 0.0
        loss_fcn = nn.NLLLoss()
        
        # Training loop
        for ex_index in ex_indexes:
            context = train_text[ex_index : ex_index + num_positions]  # need to predict
            input = " " + context[:-1]
            
            input_tensor = torch.LongTensor([vocab_index.index_of(c) for c in input])
            context_tensor = torch.LongTensor([vocab_index.index_of(c) for c in context])  # correct next char
            
            log_probabilities = model.forward(input_tensor)
            loss = loss_fcn(log_probabilities, context_tensor)  # compute loss using context
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Print loss
        print(f"Epoch {epoch}: total loss = {total_loss:.2f}")
            
    model.eval()
    return NeuralLanguageModel(model, vocab_index, num_positions)
