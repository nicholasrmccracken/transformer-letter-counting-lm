# transformer.py

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        
        # Map character index to d_model dimensional vector
        # [num_positions] -> [num_positions, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Add positional information to embeddings
        self.positional_encoding = PositionalEncoding(d_model, num_positions=num_positions, batched=False)
        
        # Controls which transformer layer is used (attention or relative position)
        self.transformer_layers = nn.ModuleList([
            TransformerLayerAttention(d_model, d_internal) for _ in range(num_layers)
        ])
        
        # Map each position vector to output class
        self.linear = nn.Linear(d_model, num_classes)
        
        # Convert logits to log probabilities
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Map character indices to vectors then add positional encodings
        input_vecs = self.positional_encoding(self.embedding(indices))
        
        # Transformer layers
        attention_maps = []
        for transformer_layer in self.transformer_layers:
            input_vecs, attention_weights = transformer_layer(input_vecs)
            attention_maps.append(attention_weights)
        
        log_probs = self.log_softmax(self.linear(input_vecs))
        
        return log_probs, attention_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayerAttention(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()

        self.d_model = d_model
        self.d_internal = d_internal
        
        # Linear projections to compute Q, K, V
        self.query_projection = nn.Linear(d_model, d_internal)
        self.key_projection = nn.Linear(d_model, d_internal)
        self.value_projection = nn.Linear(d_model, d_internal)
        
        # Softmax over self attention matrix
        self.softmax = nn.Softmax(dim=1)
        
        # Linear projection of attention output back to d_model
        # Necessary for residual layers
        self.output_projection = nn.Linear(d_internal, d_model)
        
        # Feedforward layer
        self.linear1 = nn.Linear(d_model, d_internal)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_internal, d_model)

    def forward(self, input_vecs):
        """
        :param input_vecs: A sequence of input vectors
        :return: A tuple of the updated sequence representations and the attention map matrix
        """
        # Self attention
        Q = self.query_projection(input_vecs)
        K = self.key_projection(input_vecs)
        V = self.value_projection(input_vecs)
        
        attention_scores = torch.matmul(Q, K.transpose(0, 1))              # Q*K^T
        attention_scores = attention_scores / np.sqrt(self.d_internal)     # Q*K^T / sqrt(d)
        attention_weights = self.softmax(attention_scores)                 # softmax(Q*K^T / sqrt(d))
        attention_weighted_sum = torch.matmul(attention_weights, V)        # softmax(Q*K^T / sqrt(d)) * V
        attention_output = self.output_projection(attention_weighted_sum)
        
        # Residual connection
        residual_output = input_vecs + attention_output
        
        # Feedforward (linear layer -> nonlinearity -> linear layer)
        ff_output = self.linear2(self.activation(self.linear1(residual_output)))

        # Final residual connection
        output_vecs = residual_output + ff_output
        
        return output_vecs, attention_weights
    
    
class TransformerLayerRelativePosition(nn.Module):
    def __init__(self, d_model, d_internal, num_positions=20):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        :param num_positions: max sequence length that will be fed to the model; should be 20
        """
        super().__init__()

        self.d_model = d_model
        self.d_internal = d_internal
        self.num_positions = num_positions
        
        # Linear projections to compute Q, K, V
        self.query_projection = nn.Linear(d_model, d_internal)
        self.key_projection = nn.Linear(d_model, d_internal)
        self.value_projection = nn.Linear(d_model, d_internal)
        
        # Relative offsets which range from -(num_positions - 1) to (num_positions - 1)
        self.relative_key = nn.Embedding(2 * num_positions - 1, d_internal)
        
        # Softmax over self attention matrix
        self.softmax = nn.Softmax(dim=1)
        
        # Linear projection of attention output back to d_model
        # Necessary for residual layers
        self.output_projection = nn.Linear(d_internal, d_model)

        # Feedforward layer
        self.linear1 = nn.Linear(d_model, d_internal)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_internal, d_model)

    def forward(self, input_vecs):
        """
        :param input_vecs: A sequence of input vectors
        :return: A tuple of the updated sequence representations and the attention map matrix
        """
        # Self attention
        Q = self.query_projection(input_vecs)
        K = self.key_projection(input_vecs)
        V = self.value_projection(input_vecs)
                
        # Relative position
        position_indices = torch.arange(input_vecs.shape[0], device=input_vecs.device)
        
        # Relative offset where [i, j] = i - j
        relative_offsets = position_indices[:, None] - position_indices[None, :]
        
        # Clamp relative offsets to the length of the sequence
        max_offset = self.num_positions - 1
        relative_offsets = relative_offsets.clamp(-max_offset, max_offset)
        
        # Shift range of relative offsets to be all positive
        # [-max_offset, max_offset] -> [0, 2 * max_offset]
        relative_offsets = relative_offsets + max_offset
        
        # Find learned vector for each relative offset (i - j) to get the embedding vector
        relative_key_vectors = self.relative_key(relative_offsets)
        
        # Use query to convert relative position vector to scalar score
        # relative_scores[i, j] = Q[i] * relative_key_vectors[i, j]; not just a normal dot product, use einsum
        relative_scores = torch.einsum("id,ijd->ij", Q, relative_key_vectors)
        
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) + relative_scores  # Q*K^T
        attention_scores = attention_scores / np.sqrt(self.d_internal)           # Q*K^T / sqrt(d)
        attention_weights = self.softmax(attention_scores)                       # softmax(Q*K^T / sqrt(d))
        attention_weighted_sum = torch.matmul(attention_weights, V)              # softmax(Q*K^T / sqrt(d)) * V
        attention_output = self.output_projection(attention_weighted_sum)
        
        # Residual connection
        residual_output = input_vecs + attention_output
        
        # Feedforward (linear layer -> nonlinearity -> linear layer)
        ff_output = self.linear2(self.activation(self.linear1(residual_output)))

        # Final residual connection
        output_vecs = residual_output + ff_output
        
        return output_vecs, attention_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    """
    :param args: command-line args, passed through here for your convenience
    :param train: list of LetterCountingExample objects (training data)
    :param dev: list of LetterCountingExample objects (development data)
    :return: trained Transformer model
    """
    # Model parameters
    vocab_size = 27
    num_positions = 20
    d_model = 64
    d_internal = 32
    num_classes = 3
    num_layers = 1
    
    # Hyperparameters
    learning_rate = 0.0001
    num_epochs = 15

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        # Randomize order each epoch
        random.seed(epoch)
        ex_indexes = list(range(len(train)))
        random.shuffle(ex_indexes)
        
        total_loss = 0.0
        loss_fcn = nn.NLLLoss()
        
        # Training loop
        for ex_index in ex_indexes:
            ex = train[ex_index]
            
            log_probs, _ = model.forward(ex.input_tensor)  # do not need attention maps here
            loss = loss_fcn(log_probs, ex.output_tensor)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print loss and dev accuracy based on small sample
        print(f"\nEpoch {epoch}: total loss = {total_loss:.2f}")
        decode(model, dev[:100], do_print=False)
            
    model.eval()
    
    # Exploration: Generate attention charts  
    decode(model, dev[:num_layers], do_plot_attn=True)
        
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False, do_attention_normalization_test=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
        do_attention_normalization_test = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        if do_attention_normalization_test:
            normalizes = attention_normalization_test(attn_maps)
            print("%s normalization test on attention maps" % ("Passed" if normalizes else "Failed"))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


def attention_normalization_test(attn_maps):
    """
    Tests that the attention maps sum to one over rows
    :param attn_maps: the list of attention maps
    :return:
    """
    for attn_map in attn_maps:
        total_prob_over_rows = torch.sum(attn_map, dim=1)
        if torch.any(total_prob_over_rows < 0.99).item() or torch.any(total_prob_over_rows > 1.01).item():
            print("Failed normalization test: probabilities not sum to 1.0 over rows")
            print("Total probability over rows:", total_prob_over_rows)
            return False
    return True
