import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 定义模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    # 定义超参数
    input_size = 42
    hidden_size = 64
    output_size = 6
    learning_rate = 0.001
    num_epochs = 200
    batch_size = 32

    # 加载数据
    train_data_dir = './pose_recog_trian_data'
    train_data = []
    train_labels = []
    for filename in os.listdir(train_data_dir):
        if filename.endswith('.npy'):
            data = np.load(os.path.join(train_data_dir, filename))
            data[1::2] -= data[1]
            data[0::2] -= data[0]
            train_data.append(data)
            label = int(filename.split('_')[1])
            one_hot_label = np.zeros(output_size)
            one_hot_label[label] = 1
            train_labels.append(one_hot_label)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_data = torch.randn(1000, input_size)
    train_labels = torch.randint(0, output_size, (1000,))
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # 推理过程
    test_data_dir = './pose_recog_test_data'
    test_data = []
    test_labels = []
    for filename in os.listdir(test_data_dir):
        if filename.endswith('.npy'):
            data = np.load(os.path.join(test_data_dir, filename))
            data[1::2] -= data[1]
            data[0::2] -= data[0]
            test_data.append(data)
            test_labels.append(int(filename.split('_')[1]))
    test_data = torch.tensor(test_data, dtype=torch.float32)
    outputs = model(test_data)
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted labels:', predicted)
    print('Truth labels:', test_labels)

    torch.save(model.state_dict(), 'model_params.pth')