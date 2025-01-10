from torch import nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes = 0):
        super(LSTMAutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.encoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, batch_first=True)
        self.reconstruction_linear = nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)
        #self.classification_linear = nn.Linear(in_features=hidden_size, out_features=n_classes, bias=False)

    def forward(self, x):
        _, (context, _) = self.encoder(x)
        context = context.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        decoder_output, _ = self.decoder(context)

        reconstruction = self.reconstruction_linear(decoder_output)
        class_output = self.classification_linear(decoder_output) if self.n_classes > 0 else None

        return reconstruction, class_output
