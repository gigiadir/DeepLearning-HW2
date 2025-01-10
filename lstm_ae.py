from torch import nn

class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_AE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers=1, batch_first=True, proj_size=1)

    def forward(self, x):
        _, (context, _) = self.encoder(x)
        context = context.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(context)

        return output
