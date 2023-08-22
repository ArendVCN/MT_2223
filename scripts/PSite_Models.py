from torch import nn

'''
Several CNN prediction model architectures have been constructed for testing, more may be added or revisited
Each model has at least the following commonalities: convolutional layers, non-linear activation functions (ReLU),
a dropout value to reduce over-fitting, MaxPooling that halves the field size and finally a fully connected layer
that consists of an initial Flatten and subsequent Linear layers (also with dropout and ReLU).
'''


def adjust_input_size(length, rep, k, pad, pool: bool = False):
    """
    Adjusts the input number for the fully connected layer based on the hyperparameters of the Convolutional blocks
    """
    # Repeat as many times as required, usually this mean the number of convolutional blocks
    for _ in range(rep + 1):
        # Depending on the kernel size and padding, the receptive field length will be constricted
        length = (length - (k - 1) + pad*2)

        # If a MaxPooling occurred, the receptive field length is halved
        if pool:
            length //= 2

    return length


class PSitePredictV1(nn.Module):
    """
    Version 1 consists of 2 ConvBLocks each containing 2 convolutional layers, 2 ReLU functions, a dropout layer and a
    max pooling layer. The fully connected layer is flattened and contains 2 linearization layers, a dropout layer
    and a ReLU function.
    Default dropout values are applied and the number of hidden units is fixed for all layers.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.Dropout1d(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.Dropout1d(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7,
                      out_features=hidden_units),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.convblock2(self.convblock1(x)))


class PSitePredictV2(nn.Module):
    """
    Similar to PSitePredictV1, but with only 1 Conv1d layer in the Convolutional block. An obligatory first ConvBlock is
    normalized with LayerNorm. Upgraded to take specific values for kernel_size, padding, dropout probability
    and the number of ConvBlock repeats (with repeats=0, only the initial ConvBlock is used)
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, field_length: int,
                 kernel: int = 3, pad_idx: int = 0, dropout: float = 0.5, repeats: int = 1) -> None:
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=kernel,
                      stride=1,
                      padding=pad_idx
                      ),
            nn.Dropout1d(dropout),
            nn.LayerNorm(adjust_input_size(field_length, 0, kernel, pad_idx)),
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=kernel,
                      stride=1,
                      padding=pad_idx
                      ),
            nn.Dropout1d(dropout),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.repeats = repeats

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*adjust_input_size(field_length, repeats, kernel, pad_idx, pool=True),
                      out_features=hidden_units),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pool(self.convblock1(x))
        for repeat in range(self.repeats):
            x = self.pool(self.convblock2(x))
        x = self.classifier(x)
        return x


class PSitePredictV3(nn.Module):
    """
    Same as PSitePredict Version 2, but with a fixed number of 2 ConvBlocks and no normalization.
    In the second ConvBlock, the number of hidden units has doubled, while in the fully connected layer,
    it has been quintupled.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, field_length: int,
                 kernel: int = 3, pad_idx: int = 0, dropout: float = 0.5) -> None:
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=kernel,
                      stride=1,
                      padding=pad_idx
                      ),
            nn.Dropout1d(dropout),
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units*2,
                      kernel_size=kernel,
                      stride=1,
                      padding=pad_idx
                      ),
            nn.Dropout1d(dropout),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*2*adjust_input_size(field_length, 1, kernel, pad_idx, pool=True),
                      out_features=hidden_units*5),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units*5,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.pool(self.convblock1(x))
        x = self.pool(self.convblock2(x))
        x = self.classifier(x)
        return x


class PSitePredictV4(nn.Module):
    """
    Same as PsitePredict Version 3, but the number of hidden units in the fully connected layer is sextupled.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, field_length: int,
                 kernel: int = 3, pad_idx: int = 0, dropout: float = 0.5) -> None:
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=kernel,
                      stride=1,
                      padding=pad_idx
                      ),
            nn.Dropout1d(dropout),
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units*2,
                      kernel_size=kernel,
                      stride=1,
                      padding=pad_idx
                      ),
            nn.Dropout1d(dropout),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*2*adjust_input_size(field_length, 1, kernel, pad_idx, pool=True),
                      out_features=hidden_units*6),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units*6,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.pool(self.convblock1(x))
        x = self.pool(self.convblock2(x))
        x = self.classifier(x)
        return x
