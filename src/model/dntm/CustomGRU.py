import torch


class CustomGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, device=None):
        super().__init__()
        # input-hidden parameters
        self.W_ir = torch.nn.Parameter(torch.zeros((hidden_size, input_size)))
        self.W_iz = torch.nn.Parameter(torch.zeros((hidden_size, input_size)))
        self.W_in = torch.nn.Parameter(torch.zeros((hidden_size, input_size)))
        self.b_ir = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_iz = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_in = torch.nn.Parameter(torch.zeros((hidden_size, 1)))

        # hidden-hidden parameters
        self.W_hr = torch.nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.W_hz = torch.nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.W_hn = torch.nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.b_hr = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_hz = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_hn = torch.nn.Parameter(torch.zeros((hidden_size, 1)))

        # memory-hidden parameters
        self.W_mr = torch.nn.Parameter(torch.zeros((hidden_size, memory_size)))
        self.W_mz = torch.nn.Parameter(torch.zeros((hidden_size, memory_size)))
        self.W_mn = torch.nn.Parameter(torch.zeros((hidden_size, memory_size)))
        self.b_mr = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_mz = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_mn = torch.nn.Parameter(torch.zeros((hidden_size, 1)))

    def forward(self, input, hidden, memory_reading):
        sigm = torch.nn.Sigmoid()
        tanh = torch.nn.Tanh()

        r = sigm(self.W_ir @ input + self.b_ir +
                 self.W_hr @ hidden + self.b_hr +
                 self.W_mr @ memory_reading + self.b_mr)
        z = sigm(self.W_iz @ input + self.b_iz +
                 self.W_hz @ hidden + self.b_hz +
                 self.W_mz @ memory_reading + self.b_mz)
        n = tanh(self.W_in @ input + self.b_in +
                 self.W_mn @ memory_reading + self.b_mn
                 + r * (self.W_hn @ hidden + self.b_hn))
        return (1 - z) * n + z * hidden

