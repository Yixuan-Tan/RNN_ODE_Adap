import torch
import torch.nn as nn

class RNNODE(nn.Module):
    def __init__(self, input_dim=1, n_latent=128, n_hidden=128):
        super(RNNODE, self).__init__()
        self.fc1 = nn.Linear(input_dim + n_latent, n_hidden)
        self.tanh = nn.Tanh()

    def forward(self, t, h, x):
        out = self.fc1(torch.cat((x, h), dim=1))
        out = self.tanh(out)
        return out

    def initHidden(self):
        return


class OutputNN(nn.Module):
    def __init__(self, input_dim=2, n_latent=128):
        super(OutputNN, self).__init__()
        self.fc = nn.Linear(n_latent, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.n_latent = n_latent

    def forward(self, h):
        out = self.fc(h)
        return out


class RNN(nn.Module):
    def __init__(self, input_dim=2, n_latent=128):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=n_latent, batch_first=True)
        self.linear = nn.Linear(n_latent, input_dim)

    def forward(self, x, h_0):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        h_0 = h_0.reshape(1, h_0.shape[0], h_0.shape[1])

        _, final_hidden_state = self.rnn(x, h_0)
        output = self.linear(final_hidden_state)

        return output.reshape(output.shape[1], output.shape[2]), final_hidden_state.reshape(h_0.shape[1], h_0.shape[2])


class LSTM(nn.Module):
    def __init__(self, input_dim=2, n_latent=128):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=n_latent, batch_first=True)
        self.linear = nn.Linear(n_latent, input_dim)

    def forward(self, x, h_0, c_0):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        #         print(x.shape)
        h_0 = h_0.reshape(1, h_0.shape[0], h_0.shape[1])
        c_0 = c_0.reshape(1, c_0.shape[0], c_0.shape[1])

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        output = self.linear(output)

        #         print(output.shape)
        output = output.reshape(output.shape[0], output.shape[2])
        final_hidden_state = final_hidden_state.reshape(h_0.shape[1], h_0.shape[2])
        final_cell_state = final_cell_state.reshape(c_0.shape[1], c_0.shape[2])

        return output, final_hidden_state, final_cell_state

