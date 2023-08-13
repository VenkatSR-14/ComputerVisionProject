import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# In the architecture there are 2 3x3 convolutions
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # BatchNormalization will be done
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        # We want to do model.eval and all those processes which we cannot
        # do in a regular list, so we use a module-list.
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        # According to the architecture, the input channels are mapped to
        # 64, 128, 256, 512. That is why we are providing those features for ups and downs.
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of the UNET
        # Here we do transpose convolutions which are faster and cheaper

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)  # 512 to 1024
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # *
    # One problem we can think of is, when the input is not exactly divisible by 2,
    # The up layer will have a number exactly divisible by 2
    # Why? The max pool layer will round of the number of features to an even number
    # Now when adding the skip connection, there will be a problem since the dimensions won't match
    # Always choose an input perfectly divisible by 16 for this purpose.
    # We can make our implementation general in that way
    # *#

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):  # There is a double conv then up step so the 2
            # In these series of steps, you do the up sampling,
            # Then you concatenate the skip connection.
            # This is shown in the diagram
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]  # The idx step is 2 so you divide by 2
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])  # This is to handle the uneven image sizes.
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert (x.shape == preds.shape)


if __name__ == "__main__":
    test()
