from torch import nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(LSTMAutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)

    def forward(self, x):
        _, (context, _) = self.encoder(x)
        context = context.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        decoder_output, _ = self.decoder(context)
        output = self.linear(decoder_output)

        return output
