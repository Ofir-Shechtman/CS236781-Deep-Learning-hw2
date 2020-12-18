import torch
import torch.nn as nn
import itertools as it

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: list,
            pool_every: int,
            hidden_dims: list,
            conv_params: dict = dict(kernel_size=3, padding=1),
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = dict(kernel_size=2, stride=2, padding=1)
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        # print(f'feature_extractor in_size: {self.in_size}')
        # print(self.in_size)
        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions.
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        channels = [in_channels] + self.channels
        for i in range(len(self.channels)):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], **self.conv_params))  # , (in_h, in_w)
            layers.append(ACTIVATIONS.get(self.activation_type)(**self.activation_params))
            if (i + 1) % self.pool_every == 0:
                layers.append(POOLINGS.get(self.pooling_type)(**self.pooling_params))
        # print(f'len(self.channels): {len(self.channels)}, self.pool_every: {self.pool_every}')
        # if len(self.channels) % self.pool_every == 0:
        #    layers.append(ACTIVATIONS.get(self.activation_type)(**self.activation_params))
        seq = nn.Sequential(*layers)
        # print(f'feature_extractor: {seq}')
        return seq

    def _make_classifier(self):
        layers = []
        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        '''layer = self.feature_extractor[-1]
        print(self.pooling_params)
        kernel_size=self.pooling_params.get('kernel_size', 2)
        polling = len(self.channels) // self.pool_every
        factor = kernel_size ** polling
        print(factor)
        in_channels, in_h, in_w, = tuple(self.in_size)
        print(in_h, in_w)
        features_number = self.channels[-1] * (in_h // factor) * (in_w // factor)
        print(features_number)
        print((1, *self.in_size))'''
        # print(f'calling feature_extractor with zero vector of {self.in_size}, {torch.zeros((1, *self.in_size))}')
        features_number = self.feature_extractor(torch.zeros((1, *self.in_size)))
        size_in = features_number.reshape(features_number.shape[0], -1).shape[1]
        # print(f'features_number: {features_number.shape}, size_in: {size_in}')
        layers.append(nn.Linear(size_in, self.hidden_dims[0]))
        layers.append(ACTIVATIONS.get(self.activation_type)(**self.activation_params))
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(ACTIVATIONS.get(self.activation_type)(**self.activation_params))
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # print(f'x: {x.shape}')
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # print(self.feature_extractor(x).shape)
        x_shape = self.feature_extractor(x).reshape(x.shape[0], -1)
        # print(x_shape.shape)
        out = self.classifier(x_shape)
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
            self,
            in_channels: int,
            channels: list,
            kernel_sizes: list,
            batchnorm=False,
            dropout=0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        main_layers = []
        full_channels = [in_channels] + channels
        for in_channel, out_channel, kernel_size in zip(full_channels, channels[:-1], kernel_sizes):
            main_layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, bias=True, padding=(kernel_size - 1) // 2))
            main_layers.append(nn.Dropout2d(p=dropout))
            if batchnorm:
                main_layers.append(nn.BatchNorm2d(out_channel))
            main_layers.append(ACTIVATIONS.get(activation_type)(**activation_params))
        main_layers.append(nn.Conv2d(full_channels[-2], full_channels[-1], kernel_size=kernel_sizes[-1], bias=True,
                                     padding=(kernel_sizes[-1] - 1) // 2))
        self.main_path = nn.Sequential(*main_layers)

        shortcut_layers = []
        # features_number = self.main_path(torch.zeros((6, 3, 3, 3)))
        # out_shape = features_number.reshape(features_number.shape[0], -1).shape[1]
        if in_channels != channels[-1]:
            shortcut_layers.append(nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False))
        self.shortcut_path = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        channels = [in_channels] + self.channels
        jumps = list(range(0, len(channels) - 1, self.pool_every))
        kernel_size = self.conv_params.get('kernel_size')
        for i in jumps:
            layers.append(ResidualBlock(channels[i], channels[i + 1:i + self.pool_every + 1],
                                        kernel_sizes=[kernel_size] * len(channels[i + 1:i + self.pool_every + 1]),
                                        batchnorm=self.batchnorm, dropout=self.dropout,
                                        activation_type=self.activation_type, activation_params=self.activation_params))
            if i != jumps[-1]:
                layers.append(POOLINGS.get(self.pooling_type)(**self.pooling_params))
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every, hidden_dims)

        # TODO: Change whatever you want about the ConvClassifier to try to
        #  improve it's results on CIFAR-10.
        #  For example, add batchnorm, dropout, skip connections, change conv
        #  filter sizes etc.
        # ====== YOUR CODE: ======

    def _make_feature_extractor(self):
        resnet = ResNetClassifier(self.in_size, self.out_classes, self.channels, self.pool_every, self.hidden_dims,
                                  activation_type='lrelu',
                                  activation_params=dict(negative_slope=0.01),
                                  conv_params=dict(kernel_size=5, padding=1),
                                  pooling_type='avg',
                                  batchnorm=True, dropout=0.4,
                                  )
        return resnet._make_feature_extractor()

    def _make_classifier(self):
        layers = []

        features_number = self.feature_extractor(torch.zeros((1, *self.in_size)))
        size_in = features_number.reshape(features_number.shape[0], -1).shape[1]
        # print(f'features_number: {features_number.shape}, size_in: {size_in}')
        layers.append(nn.Linear(size_in, self.hidden_dims[0]))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout2d(p=0.3))
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        seq = nn.Sequential(*layers)
        return seq
    # ========================
