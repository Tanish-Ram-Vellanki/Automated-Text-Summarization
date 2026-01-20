# Automated-Text-Summarization
Automated text summarization using a scratch custom Transformer encoder–decoder architecture with self-attention and cross-attention, trained two separate models on dialogue and CNN news datasets seperately.
Dialog Summarization Dataset Link : https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum
CNN news Dataset Link : https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

Project Overview
The goal of this project is to build an automated text summarizer that processes input texts and generates corresponding summaries in human speech voice.

The summarizer is based on a Transformer architecture consisting of custom multi-layered encoders and decoders. These components:

Work on the output of the preceding layers.
Use self-attention and cross-attention mechanisms to process the data.
Workflow Summary
Dataset Loading and Preprocessing:

Input Format: A .csv file containing columns id, dialogues, summaries, and topics.
Columns id and topics were excluded during preprocessing.
Text length analysis:
97% of dialogue texts have a length ≤ 1584.(for dialogue summarization dataset in Small.ipynb)
97% of summary texts have a length ≤ 283. (for dialogue summarization dataset in Small.ipynb)
97% of article texts have a length ≤ 8560.(for CNN news summarization dataset in codebig.ipynb)
97% of summary texts have a length ≤ 558.(for CNN news summarization dataset in codebig.ipynb)
Only texts meeting these length criteria were processed further.
A new dataset was created with dialogue texts and their corresponding summaries.
Batch Loading:

Data was loaded into batches of size 2 using DataLoader.
Tokenization:

A T5Tokenizer was used to tokenize texts:
Adds start and end tokens.
Pads the sequences to maintain consistent lengths.
Maximum token lengths:
Dialogue: 450 tokens. (for the dataset of dialogue summarization in Small.ipynb).
Summary: 450 tokens. (for the dataset of dialogue summarization in Small.ipynb).
Article: 1800 tokens. (for the dataset of CNN news summarization in codebig.ipynb).
Summary: 1800 tokens. (for the dataset of CNN news summarization in codebig.ipynb).
Padding and Masks:

Padding ensures constant input sizes for the encoder and decoder.
Padding masks:
Encoder Mask: Ignores padding tokens during attention computation.
Decoder Masks:
Suppresses padded tokens.
Masks future tokens to prevent the model from seeing ahead.
Tensor Creation:

Input (input_ids) and output (output_ids) tensors were created for tokenized texts.
All batches were concatenated into large tensors:
tensor_tokenized_inputs.
tensor_tokenized_outputs. These outputs contain the tokens tensors for each dialogue or artile (tensor_tokenized_inputs) and for each summary (tensor_tokenized_outputs).
Transformer Architecture
Encoder
Multi-Head Attention:

Calculates self-attention the fourmula is given in the code in the class of Multiheadattention, Maskedattention,Multiheadcrossattention.
Q, K, and V matrices are derived from input embeddings.
Outputs from all heads are concatenated and passed through a linear layer.
Dropout and Residual Connections:

Drops some neurons (probability = 0.1) for the generalization of the model.
Residual connections add the original input to the output of the attention layer to address vanishing gradients.
Layer Normalization:

Stabilizes training by normalizing inputs to subsequent layers.
Feedforward Layers:

Sequence of linear transformations, ReLU activations, and dropout.
Residual connections and normalization are applied again.
Layer Stacking:

Multiple encoder layers are stacked.
The output of the last encoder layer is sent to the decoder.
Decoder
Masked Self-Attention:

Prevents the decoder from "seeing" future tokens by masking.
Ensures the model predicts tokens sequentially.
Cross-Attention:

Combines the decoder’s queries with the encoder’s outputs to refine predictions.
Layer Configuration:

Similar to the encoder for the dropout and residual connections,layer normalization,feedfoward layers and layer stacking.
Embedding and Positional Encoding
Texts are converted into word embeddings (dense vector representations).
Positional encodings (using sine and cosine functions) are added to embeddings to retain word order.
Training
Loss Function:

Cross-entropy loss is used to compute the difference between predicted and actual summaries.
Backpropagation:

Loss gradients are propagated back to update model parameters,positional encoding layers, embedding_matrix_inputs and embedding_matrix_outputs using adam optimizer with a leanring rate of 0.001.
Batch Processing:

Training occurs in batches of size 1, with parameters updated after each batch.
Model Results
Encoder-Decoder Synchronization:
Multi-layered architecture ensures efficient processing of long texts and accurate summary generation.
Loss Convergence:
The loss reduced significantly during training, demonstrating successful learning.
Final Loss for model trained with smaller dataset(Dialog summarization) = 0.0031
Final loss for model trained with Larger dataset(CNN news summarization) = 0.0029
Future Enhancements
Real-time text-to-speech integration.
Support for multiple languages.
Fine-tuning to improve summary coherence and fluency.
This project showcases the power of Transformers in natural language understanding and generation, providing an accessible solution for summarizing speech-based texts.
