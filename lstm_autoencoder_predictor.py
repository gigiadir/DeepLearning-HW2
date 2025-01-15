from torch import nn

class LSTMAutoencoderPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMAutoencoderPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.reconstruction_linear = nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)

        self.predictor_linear = nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)


    def forward(self, x, y = None):
        _, (context, _) = self.encoder(x)

        context_repeated = context.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        decoder_output, _ = self.decoder(context_repeated)
        reconstruction = self.reconstruction_linear(decoder_output)

        if y is not None:
            _, (prediction_context, _) = self.encoder(y)
            prediction_context_repeated = context.repeat(x.size(1), 1, 1).permute(1, 0, 2)
            prediction_output, _ = self.decoder(prediction_context_repeated)
            prediction = self.predictor_linear(prediction_output)

            return reconstruction, prediction

        prediction = self.predictor_linear(decoder_output)
        prediction = prediction[:, -1, :].unsqueeze(1)
        return reconstruction, prediction