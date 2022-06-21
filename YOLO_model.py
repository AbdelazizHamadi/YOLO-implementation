import torch
import torch.nn as nn

architecture_config = [

    # tuple (conv layer) : (kernel size, num_filters, stride, padding)
    (7, 64, 2, 3),
    # M : maxpool layer (2x2 - stride 2) as YOLO paper
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",

    # list : [tuple, tuple, num_repeat]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):

    # **kwargs to pass other arguments after like kernel_size, stride...
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, data):

        return self.leakyrelu(self.batchnorm(self.conv(data)))


class YOLOv1(nn.Module):

    def __init__(self, architecture, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, data):

        data = self.darknet(data)
        return self.fcs(torch.flatten(data, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:

            if type(x) == tuple:

                kernel_size, out_channels, stride, padding = x  # get Conv_details

                layers += [
                    CNNBlock(in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
                ]

                in_channels = out_channels

            elif type(x) == str:
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=2)
                ]

            elif type(x) == list:

                conv1 = x[0]  # tuple
                kernel_size_1, out_channels_1, stride_1, padding_1 = conv1  # get Conv_details

                conv2 = x[1]  # tuple
                kernel_size_2, out_channels_2, stride_2, padding_2 = conv2

                num_repeat = x[2]  # Conv repeats (integer)

                for _ in range(num_repeat):
                    layers += [
                        CNNBlock(in_channels=in_channels,
                                 out_channels=out_channels_1,
                                 kernel_size=kernel_size_1,
                                 stride=stride_1,
                                 padding=padding_1)
                    ]
                    # re-initialize input channels
                    in_channels = out_channels_1

                    layers += [
                        # in_channels of this layer is the output
                        CNNBlock(in_channels=in_channels,
                                 out_channels=out_channels_2,
                                 kernel_size=kernel_size_2,
                                 stride=stride_2,
                                 padding=padding_2)
                    ]

                    # re-initialize input channels for the next iteration
                    in_channels = out_channels_2

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):

        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 539),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(539, S * S * (C + B * 5))  # (S, S, 30) where C + B*5 = 30
        )


def test(S=7, B=2, C=1):
    model = YOLOv1(architecture=architecture_config, in_channels=3, split_size = S, num_boxes=B, num_classes = C)
    x = torch.randn((2, 3, 448, 448))

    return model(x)
    #print(model(x).reshape(30, 7, 7))
