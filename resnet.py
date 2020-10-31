class ResidualBlock(nn.Module):
    def __init__(self, input_channel, middle_channel, output_channel, s=1):
        super(ResidualBlock, self).__init__()
        self.stride = s

        self.skip_connection = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, stride=s, bias=False),
            nn.BatchNorm2d(output_channel)
        )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=middle_channel, kernel_size=1, stride=s, bias=False),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channel, out_channels=middle_channel, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channel, out_channels=output_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )

    def forward(self, x):
        if self.stride != 1:
            skip = self.skip_connection(x)
        else:
            skip = x

        conv = self.block(x)

        x = skip + conv
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 7)
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=1)

        self.resBlock1 = ResidualBlock(64, 64, 256, 2)
        self.resBlock2 = ResidualBlock(256, 64, 256, 1)

        self.resBlock3 = ResidualBlock(256, 128, 512, 2)
        self.resBlock4 = ResidualBlock(512, 128, 512, 1)

        self.resBlock5 = ResidualBlock(512, 256, 1024, 2)
        self.resBlock6 = ResidualBlock(1024, 256, 1024, 1)

        self.resBlock7 = ResidualBlock(1024, 512, 2048, 2)
        self.resBlock8 = ResidualBlock(2048, 512, 2048, 1)

        self.avg_pool = nn.AvgPool2d(3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.classifier = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock2(x)

        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = self.resBlock4(x)
        x = self.resBlock4(x)

        x = self.resBlock5(x)
        x = self.resBlock6(x)
        x = self.resBlock6(x)
        x = self.resBlock6(x)
        x = self.resBlock6(x)
        x = self.resBlock6(x)

        x = self.resBlock7(x)
        features = self.resBlock8(x)

        x = self.avg_pool(features)
        
        x = x.view(features.size(0), -1)
        
        fc = self.fc1(x)
        fc = self.fc2(fc)
        out = self.fc3(fc)
        out = self.classifier(out)

        return out, features
