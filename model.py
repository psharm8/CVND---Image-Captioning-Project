import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        device = features.device
        # clean hidden state
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

        # print("captions shape", captions.shape)
        # print("features shape", features.shape)

        # captions shape is [batch_size, caption_len + 2]
        #  +2 is for <start> and <end>
        # embedding will generate [batch_size, caption_len + 2, embed_size]
        # But later we need to concat features of shape [batch_size, 1, embed_size]
        # Which produces output having one extra value in the seconds dimension
        # So I drop one word from the captiion, the <end> word
        # which makes it match the expected output shape.
        # I do not know the implications of removing the end word from the caption.
        # Generates [batch_size, caption_len + 1, embed_size]; removed <end> word.
        captionEmbedding = self.embedding(captions[:,:-1])
        # print("embedding shape", captionEmbedding.shape)
        
        # feature size is [batch_size, embed_size]
        # add 2nd dimension -> [batch_size, 1, embed_size]
        features = features.unsqueeze(1)

        # concat features with word embedding
        # shape becomes [batch_size, caption_len + 2 + 1, embed_size]
        inputs = torch.cat((features, captionEmbedding), dim=1)
        # print("inputs shape", inputs.shape)

        # feed the lstm
        outputs, hidden = self.lstm(inputs, hidden)
        # print("output shape", outputs.shape)
        
        # linear mapping from hidden size to vocab size
        outputs = self.fc(outputs)
        # print("linear output shape", outputs.shape)

        return outputs


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted = []
        
        for _ in range(max_len) :
            # Get the output state from LSTM
            output, states = self.lstm(inputs, states)
            # Map it to vocab space
            output = self.fc(output)
            # Get the idx with max probability
            prob, idx = output.max(dim=2)
            # append to predicted word list
            predicted.append(idx.item())
            # generate embedding to be fed back to RNN for next word prediction
            inputs = self.embedding(idx)

        return predicted

