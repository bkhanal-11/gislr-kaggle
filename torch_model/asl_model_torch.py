import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import onnx_tf
import onnx

from feature_preprocessing import FeaturePreprocess

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset from ASL Pretrained
features_path = "/kaggle/input/asl-pretrained/processed_features.npy"
labels_path = "/kaggle/input/asl-pretrained/processed_labels.npy"

features = torch.from_numpy(np.load(features_path))
labels = torch.from_numpy(np.load(labels_path))

MAX_DATA_LEN = 60000

# For testing due to memory overflow
data = features[:MAX_DATA_LEN]
labels = labels[:MAX_DATA_LEN]

# Split the data into training and validation sets
SEED = 11
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=SEED)

del data
del labels

# Create PyTorch data loaders
train_data = TensorDataset(X_train.float(), y_train.float())
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)

val_data = TensorDataset(X_val.float(), y_val.float())
val_dataloader = DataLoader(val_data, batch_size=128, shuffle=False, drop_last=True)

del X_train, X_val, y_train, y_val

# Model

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(DenseBlock, self).__init__()
        self.dense = nn.Linear(in_channels, out_channels)
        self.layernorm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.dense(x)
        x = self.layernorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1216, hidden_size=250, num_layers=5, num_classes=250, dropout_rate=0.1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.avgpool2d = nn.AvgPool2d((4,1))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.avgpool2d(x)
    
        x = x.view(x.size(0), x.size(1), -1)
        
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class SignClassifier(nn.Module):
    def __init__(self, encoder_units=[3, 128, 64], dropout_rate=0.1):
        super(SignClassifier, self).__init__()
        self.dense = nn.ModuleList([])    
        for i in range(len(encoder_units[:-1])):
            layer = DenseBlock(in_channels = encoder_units[i], out_channels=encoder_units[i + 1], dropout_rate=dropout_rate) 
            self.dense.append(layer)
        self.classifier = LSTMClassifier()
    
    def forward(self, x):
        for layer in self.dense:
            x = layer(x)
        
        output = self.classifier(x)
        return output

# Training Model
model = SignClassifier().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCH = 100

# Train the model
for epoch in range(EPOCH):
    model.train()
    training_loss = 0.0
    train_correct_predictions = 0
    train_total_predictions = 0
    
    for feature, label in train_dataloader:
        feature, label = feature.to(device), label.to(device)
        
        optimizer.zero_grad()
        outputs = model(feature)
        
        num_classes = 250
        input_rounded = torch.round(label).long().to(device)
        identity_matrix = torch.eye(num_classes).to(device)
        one_hot = identity_matrix[input_rounded]

        loss = criterion(outputs, one_hot.to(device))
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        # Calculate the number of correct predictions
        predictions = torch.argmax(outputs, dim=1)
        train_correct_predictions += (predictions == input_rounded).sum().item()
        train_total_predictions += input_rounded.size(0)
    
    avg_training_loss = training_loss / len(train_dataloader)
    training_accuracy = train_correct_predictions / train_total_predictions

    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
        
    with torch.no_grad():
        for feature, label in val_dataloader:
            feature, label = feature.to(device), label.to(device)
            
            outputs = model(feature)
            
            num_classes = 250
            input_rounded = torch.round(label).long().to(device)
            identity_matrix = torch.eye(num_classes).to(device)
            one_hot = identity_matrix[input_rounded]
            
            loss = criterion(outputs, one_hot.to(device))
            val_running_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            val_correct_predictions += (predictions == input_rounded).sum().item()
            val_total_predictions += input_rounded.size(0)
        
        avg_val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = val_correct_predictions / val_total_predictions
        
    print(f"Epoch {epoch + 1}: Training loss: {avg_training_loss:.3f} | Training accuracy: {training_accuracy:.3f} | Validation loss: {avg_val_loss:.3f} | Validation accuracy: {val_accuracy:.3f}")

# Save Model
torch.save(model.state_dict(), '/kaggle/working/model_weights.pth')

# Model loading for TFLite conversion
eval_model = SignClassifier().to(device)
eval_model.load_state_dict(torch.load('/kaggle/working/model_weights.pth'))

# Pytorch to ONNX
feature_converter = FeaturePreprocess()
feature_converter.eval()
preprocess_sample = torch.rand((37, 543, 3)).to(device) 
onnx_preprocess_path = 'preprocess.onnx'
torch.onnx.export(feature_converter,
                  preprocess_sample,
                  onnx_preprocess_path,
                  opset_version=12,
                  input_names = ['inputs'],
                  output_names = ['outputs'],
                  dynamic_axes={'inputs': {0: 'frames'}})

eval_model.eval()
model_sample = torch.rand((1, 37, 77, 3)).to(device)
onnx_model_path = 'model.onnx'
torch.onnx.export(eval_model,
                  model_sample,
                  onnx_model_path,
                  opset_version=12,
                  input_names = ['inputs'],
                  output_names = ['outputs'],
                  dynamic_axes={'inputs': {0: 'batch_size'}})

onnx_preprocess = onnx.load(onnx_preprocess_path)
onnx.checker.check_model(onnx_preprocess)
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

# ONNX to TFLite
tf_preprocess_path = 'tf_preprocess'
tf_preprocess = onnx_tf.backend.prepare(onnx_preprocess)
tf_preprocess.export_graph(tf_preprocess_path)

tf_model_path = 'tf_model'
tf_model = onnx_tf.backend.prepare(onnx_model)
tf_model.export_graph(tf_model_path)

class InferenceModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = tf.saved_model.load(tf_preprocess_path)
        self.model = tf.saved_model.load(tf_model_path)
        self.preprocess.trainable = False
        self.model.trainable = False
    
    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')
    ])
    def call(self, x):
        outputs = {}
        preprocessed = self.preprocess(**{'inputs':x})['outputs']
        pred = self.model(**{'inputs':tf.expand_dims(preprocessed, 0)})['outputs'][0,:]
        #pred = tf.nn.softmax(pred)
        return {
            'outputs': pred
        }

tf_inference = InferenceModel()
tf_inference_path = 'tf_inference'
tf.saved_model.save(tf_inference, tf_inference_path, signatures={'serving_default': tf_inference.call})

model_converter = tf.lite.TFLiteConverter.from_saved_model(tf_inference_path) # path to the SavedModel directory
tflite_model = model_converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)