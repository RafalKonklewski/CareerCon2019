import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from time import gmtime, strftime
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


#####################################################################
le = LabelEncoder()

train_x = pd.read_csv("data/X_train.csv")
train_y = pd.read_csv("data/y_train.csv")
train_y = train_y.set_index('series_id').drop(['group_id'], axis=1)
train_x = train_x.set_index('row_id').drop(['series_id', 'measurement_number'], axis=1)

# train_x = train_x.head(2560*1000)
# train_y = train_y.head(20*1000)
####Encode labels
train_y = le.fit_transform(train_y).reshape(-1,1)
train_y = pd.DataFrame({'y':train_y[:,0]})

#####Standardize training dataset
for col in train_x.columns:
    if 'orient' in col:
        scaler = StandardScaler()
        train_x[[col]] = scaler.fit_transform(train_x[[col]])

##### TESTING ON SAMPLE
# train_x = train_x.head(2560*100)
# train_y = train_y.head(20*100)

##### SETTING PARAMETERS
seq_len = 128
batch_size = 64
h1 = 512

##### PREPARING TORCH DATASETS AND DATALOADERS
train_x_temp = np.array(train_x.values).reshape(-1, seq_len, 10)
train_y_temp = np.array(train_y.values).reshape(-1)
X_train, X_valid, y_train, y_valid = train_test_split(train_x_temp, train_y_temp, test_size=0.3, random_state=666)

y_true = y_train
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_valid_tensor = torch.FloatTensor(X_valid)
y_valid_tensor = torch.FloatTensor(y_valid)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
##### SHUFFLE TRAINING DATA TO AVOID OVERFITTING TO DATA ORDER
train_dataloader = DataLoader(train_dataset, num_workers = 8, shuffle = True, batch_size = batch_size)
valid_dataloader = DataLoader(valid_dataset, num_workers = 8, shuffle = False, batch_size = batch_size)


######### MODEL DEFINITION
class LSTM(nn.Module):

    def __init__(self, seq_len, hidden_dim, batch_size, output_dim=1,
                    num_features=10, dropout = 0.1, num_layers=1, label_size = 9):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.label_size = label_size
        self.output_dim = output_dim
        self.num_features = num_features
        # DataLoader batch is of shape [batch_size, sequence_length, num_features]
        # LSTM LAYER
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers,
                            bidirectional=False, dropout = dropout, batch_first=True)

        # Linear layers and activation functions
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.linear2 = nn.Linear(seq_len, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p = dropout)

    def init_hidden(self):
        hid = (torch.randn(self.num_layers, self.hidden_dim),
                torch.randn(self.num_layers, self.hidden_dim))
        return hid

    def forward(self, input):
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        lstm_out, self.hidden = self.lstm(input.view(self.seq_len,
                                                  -1, self.num_features))
        # print("size after lstm" + str(lstm_out.size()))
        y_pred = self.linear(lstm_out)
        y_pred = self.relu(y_pred)
        # print("size after 1 linear" + str(y_pred.size()))
        y_pred = self.linear2(y_pred.view(-1, self.label_size, self.seq_len))
        # print("size after 2 linear" + str(y_pred.size()))

        return y_pred.view(-1, self.label_size)

model = LSTM(seq_len=seq_len, hidden_dim=h1, batch_size = batch_size,
             output_dim=9, dropout = 0.25, num_layers=3)

#model.load_state_dict(torch.load('/models/lstm_model.txt'))
model.to('cuda')

print("LSTM model details", model)

#######################TRAINING PARAMETERS
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.1, patience=20, verbose=True)
#####################
# Train model
#####################
num_epochs = 200
#hist = np.zeros(num_epochs)
hist = np.zeros((2, num_epochs))

predictions = np.array([])


model.train()
for t in tqdm(range(num_epochs)):
    for step, batch in enumerate(train_dataloader):
        optimiser.zero_grad()

        model.init_hidden()

        x = batch[0].cuda()
        y = batch[1].cuda()

        # Forward pass
        y_pred = model(x)
        # print("1" + str(y.size()))
        # print("2" + str(y_pred.size()))
        loss = loss_fn(y_pred, y.long())
        if t % 10 == 0 or t == num_epochs-1:
            predictions = np.append(predictions, y_pred.detach()
                                    .cpu().numpy().reshape(-1), axis = 0)
        hist[0][t] = loss.item()

         # Backward pass
        loss.backward()

         # Update parameters
        optimiser.step()
    # Calculate validation cross entropy and accuracy
    val_loss_array = []
    val_acc = []
    for batch_i, valid_batch in enumerate(valid_dataloader):
        data = valid_batch[0].cuda()
        target = valid_batch[1].cuda()
        output = model(data)
        # print("3" + str(output.size()))
        # print("4" + str(target.size()))
        # print(valid_batch.size())


        val_loss = loss_fn(output, target.long())
        val_loss_array.append(val_loss.item())
        a = target.data.cpu().numpy()
        b = output.detach().cpu().numpy().argmax(1)
        val_acc.append(accuracy_score(a, b))

    hist[1][t] = np.mean(val_acc)

    if t % 10 == 0 or t == num_epochs - 1:
        reshaped = predictions.reshape(-1, 9)
        labels_predicted = np.argmax(reshaped, axis=1).reshape(-1, 1).reshape(-1)
        acc = accuracy_score(y_true = y_true, y_pred=labels_predicted)
        #hist[1][t] = acc
        tqdm.write("Epoch: {}, Train Cross Entropy: {}, Train Accuracy: {}, Valid Cross Entropy: {}, Valid Accuracy: {}"
                   .format(t, loss.item(), acc, np.mean(val_loss_array), np.mean(val_acc)))
        # print("pred: " + str(y_pred[0]) + "real: " + str(y[0]))
        predictions = np.array([])
    if t % 50 == 0:
        current_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        path = '/home/rafal/CareerCon2019/models/lstm_model{}.txt'.format(current_time)
        path_histogram = '/home/rafal/CareerCon2019/histograms/histogram{}.txt'.format(current_time)
        torch.save(model.state_dict(), path)
        np.savetxt(path_histogram, hist)
    # scheduler.step(np.mean(val_loss_array))




################## SUBMISSION
test_x = pd.read_csv("data/X_test.csv")
test_x = test_x.set_index('row_id').drop(['series_id', 'measurement_number'], axis=1)

for col in train_x.columns:
    if 'orient' in col:
        scaler = StandardScaler()
        test_x[[col]] = scaler.fit_transform(test_x[[col]])

test_x_tensor = torch.FloatTensor(test_x.values).reshape(-1, seq_len, 10)

dataset_test = TensorDataset(test_x_tensor)
dataloader_test = DataLoader(dataset_test, batch_size = batch_size,
                        num_workers = 6, shuffle = False)

preds_tensor = torch.tensor([])
model.eval()
with torch.no_grad():
    for batch in dataloader_test:
        test_batch = batch[0].cuda()
        predict = model(test_batch)
        preds_tensor = torch.cat((preds_tensor, predict.cpu()))

predict = preds_tensor
preds = le.inverse_transform(np.argmax(predict.cpu().numpy(), axis=1).reshape(-1,1))

preds_df = pd.DataFrame({"surface": preds})
preds_df.index.name = 'series_id'

preds_df.to_csv('/home/rafal/CareerCon2019/submission.csv')



################## PLOT LOSS HISTOGRAM
plt.title("Validation accuracy")
plt.plot(hist[1])
plt.show()
