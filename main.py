import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc

torch.cuda.device(0)

def data_load():
    train_set = torchvision.datasets.ImageFolder(
        root='./TinyImageNet/train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
    )

    test_set = torchvision.datasets.ImageFolder(
        root='./TinyImageNet/val',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=50, num_workers=8)

    test_loader = DataLoader(test_set, shuffle=True, batch_size=50)

    return train_loader, test_loader

def new_plot(title, xlabel, ylabel, data_1, data_2, data_1_label, data_2_label):
    plt.figure(figsize=(10,5))
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data_1, label=data_1_label)
    plt.plot(data_2, label=data_2_label)
    plt.legend()
    
    plt.show()



def main():
    gc.collect()
    torch.cuda.empty_cache()
    train_loader, test_loader = data_load()
    epochs = 20
    PATH = './TinyImageNet/result2.pt'
    load_model = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(torch.cuda.get_device_name(device))
    
    
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    model = ResNet()
    try:
        if load_model:
            model.load_state_dict(torch.load(PATH))
            model = model.to(device)
        else:
            model = model.to(device)

            criterion = torch.nn.CrossEntropyLoss().cuda()
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)

            for epoch in range(epochs):
                running_loss = 0.0
                temp_loss = 0.0
                train_accuracy = 0.0
                temp_accuracy  = 0.0
                total_train = 0
                correct_train = 0
                count = 0
                for i, img in enumerate(train_loader, 0):
                    inputs, labels = img
                    inputs, labels = inputs.to(device), labels.to(device)

                    y_pred, feature = model(inputs)

                    loss = criterion(y_pred, labels)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    _, predicted = torch.max(y_pred, 1)
                    total_train += labels.size(0)
                    correct_train += predicted.eq(labels.data).sum().item()
                    #print(labels.size(0))
                    #print(predicted.eq(labels.data).sum().item())
                    temp_loss += loss.item()
                    temp_accuracy += (correct_train / total_train) * 100
                    running_loss += loss.item()
                    train_accuracy += (correct_train / total_train) * 100
                    count+=1
                    if i % 50 == 49:
                        print('[%d, %5d] loss: %.3f  accuracy : %.3f' %
                              (epoch + 1, i + 1, temp_loss / 50, temp_accuracy / 50))
                        temp_loss = 0.0
                        temp_accuracy = 0.0

                train_accuracy_list.append(train_accuracy / count)
                train_loss_list.append(running_loss / count)

                print("validation ===============================")
                correct_val = 0.0
                total_val = 0.0
                val_loss = 0.0
                count = 0
                accuracy_sum = 0
                with torch.no_grad():
                    for test_data in test_loader:
                        img, labels = test_data
                        img, labels = img.to(device), labels.to(device)

                        out, _ = model(img)
                        _, predicted = torch.max(out, 1)
                        total_val += labels.size(0)
                        correct_val += predicted.eq(labels.data).sum().item()

                        accuracy_sum+=(correct_val / total_val) * 100

                        loss = criterion(out, labels)
                        val_loss += loss.item()

                        count+=1

                val_loss /= count

                print('Validation Accuracy: %.3f'%(accuracy_sum / count), "Validation Loss: %.3f"%(val_loss))

                val_accuracy_list.append(accuracy_sum / count)
                val_loss_list.append(val_loss)

                print("==========================================")

        correct_test = 0.0
        total_test = 0.0
        test_loss = 0.0
        count = 0
        accuracy_sum = 0
        with torch.no_grad():
            for test_data in test_loader:
                img, labels = test_data
                img, labels = img.to(device), labels.to(device)

                out, _ = model(img)
                _, predicted = torch.max(out, 1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels.data).sum().item()

                accuracy_sum+=(correct_test / total_test) * 100

                loss = criterion(out, labels)
                test_loss += loss.item()

                count+=1

        test_loss /= count

        print('Test Accuracy: %.3f'%(accuracy_sum / count), "Test Loss: %.3f"%(test_loss))

        new_plot('Train Loss & Validation Loss', 'epochs', 'Traing loss', train_loss_list, val_loss_list, 'train', 'validation')
        new_plot('Train Accuracy & Validation Accuracy', 'epochs', 'Accuracy', train_accuracy_list, val_accuracy_list, 'train', 'validation')
    except Exception as e:
        del model
        del train_loader
        del test_loader
        del criterion
        del opt
        del inputs
        gc.collect()

        torch.cuda.empty_cache()
        print(e)
        


if __name__ == '__main__':
    main()
