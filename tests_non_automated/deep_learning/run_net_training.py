import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tests_non_automated.deep_learning.feed_forward import FeedForward
from neat.dataset.regression_example import RegressionExample1Dataset
from tests.config_files.config_files import create_configuration
import matplotlib.pyplot as plt

config = create_configuration(filename='/siso.json')

lr = 0.01
weight_decay = 0.0005
n_epochs = 2000

dataset = RegressionExample1Dataset()
batch_size = 50000
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

network = FeedForward(n_input=config.n_input, n_output=config.n_output,
                      n_neurons_per_layer=10,
                      n_hidden_layers=1)

criterion = nn.MSELoss()

optimizer = Adam(network.parameters(), lr=lr, weight_decay=weight_decay)


def train():
    network.train()
    loss_epoch = 0
    for x_batch, y_batch in data_loader:
        # x_batch = x_batch.reshape((-1, config.n_input))
        x_batch = x_batch.float()
        y_batch = y_batch.float()

        # forward pass
        output = network(x_batch)
        loss = criterion(output, y_batch)
        loss_epoch += loss.data.item()

        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

    return loss_epoch


def predict(x_pred):
    network.eval()

    x_batch = x_pred.reshape((-1, config.n_input))
    x_batch = x_batch.float()

    # forward pass
    with torch.no_grad():
        y_pred = network(x_batch)
    return y_pred


# train
for epoch in range(n_epochs):
    loss_epoch = train()
    if epoch % 10 == 0:
        print(f'Epoch = {epoch}. MSE: {loss_epoch}')

# save weights
filename = 'network.pt'
torch.save(network.state_dict(), f'./models/{filename}')

# predict
x_test = torch.Tensor(dataset.x)
y_true = dataset.y
y_pred = predict(x_test).numpy()

plt.figure(figsize=(20, 20))
plt.plot(x_test.numpy(), y_true, 'r*')
plt.plot(x_test.numpy(), y_pred, 'b*')
plt.show()
