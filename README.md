# TinyImageNet-Pytorch-ResNet

TinyImageNet Dataset을 이용하여 pytorch로 ResNet을 만들고 학습합니다.

## TinyImageNet

TinyImageNet은 기존의 224x224 이미지가 총 1300개씩 1000개의 클래스가 있는 ImageNet이 너무 용량이 크고 학습하기 어려워서 나온 작은 사이즈의 Dataset입니다.

TinyImageNet은 64x64 이미지가 총 1000개씩 100개의 클래스가 있습니다. ImageNet보다 적은 용량입니다. (ImageNet은 train set만 130GB이고 TinyImageNet은 총 합쳐도 400MB정도입니다.)

TinyImageNet은 https://tiny-imagenet.herokuapp.com/ 여기서 받을 수 있지만, train 데이터와 validation 데이터가 0~100 클래스 폴더로 나뉘어 있는 https://www.kaggle.com/c/thu-deep-learning/data 여기의 데이터를 사용하였습니다.

## ResNet

ResNet은 2015년 ILSVRC에서 우승을 차지한 CNN모델입니다. ResNet 이전의 모델인 VGGNet은 모델의 깊이 즉, Layer의 수를 늘리려고 노력하였습니다. 

하지만 ResNet 개발팀은 Layer의 수가 일정 이상 넘어가면 오히려 비효율적이게 되는 것을 알게 되었고 이를 보완하기 위해 나온 것이 ResNet입니다.

ResNet은 Skip Connection 혹은 Shortcut 이라는 새로운 길을 만들어냅니다. 

![img1](https://github.com/kjo26619/TinyImageNet-Pytorch-Resnet/blob/main/Image/skip%20connection.png)

이 Skip Connection에서 ResNet의 이름인 Residual Network의 뜻을 가지게 되었는데, Residual(잔차)를 최소화한다는 의미입니다.

기존의 Neural Network는 입력 x가 들어오면 출력 y가 나오는 함수 H(x)를 찾아내는 것이 목표였습니다. 

하지만, ResNet에서는 기존의 Convolution Layer를 지나는 F(x)와 Skip Connection을 지나온 x를 더함으로써 H(x) = F(x) + x를 결과로 만들어냅니다.

이를 학습의 단계로 가게 된다면, 네트워크는 F(x) = H(x) - x를 0으로 만들게끔 학습이 진행됩니다.

기존의 Neural Network는 H(x) = x 가 나오게끔 학습을 합니다. 이 원리에서 ResNet에서는 F(x)에 x가 더해지므로 학습은 H(x)=x를 만들기 위해 F(x)를 0으로 만들게 학습하는 것입니다.

이것이 어떠한 효과를 발휘하냐면, Gradient Vanishing 현상을 어느정도 해결합니다.

즉, Layer의 수가 일정 이상 넘어가면 Training이 안되는 이유가 Gradient Vanishing 문제라는 것입니다.

Neural Network는 학습의 과정에서 미분을 이용하여 최적의 매개변수를 찾아나가는데, 이 것이 일정 Layer 수를 넘어가면 0으로 수렴하게되는 문제가 발생하는 것입니다.

ResNet에서는 Skip Connection으로 더했을 시 F(x) + x가 되는 것이고 이를 미분하면 F'(x) + 1이 되므로, Gradient가 0으로 수렴하는 일이 사라지게 되는 것 입니다.

개발팀은 다양한 ResNet을 제시하였는데 구조는 다음과 같습니다.

![img2](https://github.com/kjo26619/TinyImageNet-Pytorch-Resnet/blob/main/Image/resnet_arch.png)

이 중에서 ResNet50 이상부터는 1x1 Conv, 3x3 Conv와 1x1Conv를 사용하는 Residual Block을 사용하는데 이러한 구조를 Bottleneck 구조라고 합니다.

## Bottleneck Architecture

![img3](https://github.com/kjo26619/TinyImageNet-Pytorch-Resnet/blob/main/Image/bottleneck.png)

Bottleneck 구조는 ResNet에서 Residual Block으로 늘어난 Layer의 수 만큼 연산 시간을 많이 차지하는 문제를 해결하기 위해 적용하였습니다.

1x1 Convolution Layer를 사용하면 Dimension을 줄일 수 있기 때문입니다.

Dimension을 줄인 뒤 3x3 Convolution Layer를 작업함으로써 더 낮은 연산 시간을 갖게 되는 것입니다. 그 이후 1x1 Convolution Layer로 다시 Dimension을 확장시켜 줍니다.

