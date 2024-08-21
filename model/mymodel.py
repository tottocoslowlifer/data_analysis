from torch import nn


# モデルの定義
class NeuralNetwork(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        first_dim, second_dim, slope
    ):
        super().__init__()
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, first_dim),
            nn.LeakyReLU(slope),
            nn.Linear(first_dim, second_dim),
            nn.LeakyReLU(slope),
            nn.Linear(second_dim, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        outputs = self.linear_relu_stack(x)
        return outputs
