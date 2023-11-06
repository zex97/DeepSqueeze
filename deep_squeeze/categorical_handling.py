import numpy as np
import torch
import torch.nn as nn
import logging
from tqdm import tqdm


class CategoryClassifier(nn.Module):
    def __init__(self, code_size, cardinality):
        super().__init__()
        hidden_layer_width = cardinality * 2
        self.activation = nn.ReLU()

        self.input_hidden = nn.Linear(in_features=int(code_size), out_features=hidden_layer_width)
        self.middle1 = nn.Linear(in_features=hidden_layer_width, out_features=hidden_layer_width)
        self.hidden_output = nn.Linear(in_features=hidden_layer_width, out_features=cardinality)

    def forward(self, x):
        x = self.activation(self.input_hidden(x))
        x = self.activation(self.middle1(x))
        x = self.hidden_output(x)
        return x


def train_categorical(model, device, codes, labels, epochs=30, batch_size=32, lr=1e-4):
    train_data = []
    for i in range(len(codes)):
        train_data.append([codes[i], labels[i]])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch_ind, batch in enumerate(tqdm(train_loader, disable=True)):

            codes_batch = batch[0].to(device)
            labels_batch = batch[1].to(device)
            y_pred = model(codes_batch)
            loss = criterion(y_pred, labels_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss

        logging.info(f"Epoch: {epoch + 1} / {epochs} | "
                      f"{type(criterion).__name__}: {float(epoch_loss / len(train_loader)):.3f}\n")

    final_loss = epoch_loss / len(train_loader)
    logging.debug(f"Categorical training finished. Final loss: {float(final_loss):.3f}")

    return model


def materialize_c_failures(model, codes, column, device):
    codes = torch.from_numpy(codes).float().to(device)
    y_column = torch.argmax(model(codes), dim=1)
    res = np.array(torch.FloatTensor(column) - y_column)
    logging.info(f"Categorical materialization result: {np.unique(res, return_counts=True)}")
    return res

