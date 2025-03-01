import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.transforms import ToTensor

# Constants
NUM_SAMPLES = 1000
SHAPES = ['square', 'triangle']
IMAGE_SIZE = (256, 256)

# Function to generate images with different shapes
def generate_shape_image(shape, size=IMAGE_SIZE):
    image = np.zeros(size + (3,), dtype=np.uint8)
    color = tuple(np.random.randint(0, 256, 3).tolist())

    if shape == 'circle':
        center = tuple(np.random.randint(0, size[0], 2).tolist())
        radius = np.random.randint(10, 100)
        cv2.circle(image, center, radius, color, thickness=-1)
    elif shape == 'square':
        top_left = np.random.randint(0, size[0] - 100, 2)
        width = height = np.random.randint(20, 100)
        bottom_right = top_left + np.array([width, height])
        cv2.rectangle(image, tuple(top_left), tuple(bottom_right), color, thickness=-1)
    elif shape == 'triangle':
        points = np.random.randint(0, size[0], (3, 2))
        cv2.fillPoly(image, [points], color)

    return image

# Function to create dataset
# Function to create dataset
def create_dataset(num_samples, shapes, generate_shape_image=generate_shape_image):
    dataset = []
    labels = []
    for _ in range(num_samples):
        shape = np.random.choice(shapes)
        image = generate_shape_image(shape)
        dataset.append(image)
        labels.append(shapes.index(shape))

    transform = transforms.Compose([ToTensor()])
    dataset = torch.stack([transform(image) for image in dataset])
    labels = torch.tensor(labels)

    return dataset, labels

# Function to split dataset into train and validation
def split_dataset(dataset, labels, train_ratio=0.8):
    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(dataset, labels), [train_size, val_size]
    )
    return train_dataset, val_dataset

# Function to create data loaders
def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Function to predict shape
def predict_shape(model, image, shapes):
    transform = transforms.Compose([ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    predicted_class_idx = torch.argmax(output, dim=1).item()
    predicted_shape = shapes[predicted_class_idx]

    return predicted_shape

class SimpleShapeClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(64 * 64 * 64, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 64 * 64)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Create dataset
dataset, labels = create_dataset(NUM_SAMPLES, SHAPES)

# Split dataset into train and validation
train_dataset, val_dataset = split_dataset(dataset, labels)

# Create data loaders
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

# Define model
model = SimpleShapeClassifier(num_classes=len(SHAPES))

# Define trainer
trainer = pl.Trainer(max_epochs=10, default_root_dir='./logs')

# Train model
trainer.fit(model, train_loader, val_loader)

# Test model
test_dataset = torch.utils.data.TensorDataset(dataset, labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
trainer.test(model=model, dataloaders=test_loader)

# Predict shape for new unseen image
print("Generating new image [unseen shape = circle]")
new_image_unseen = generate_shape_image('circle')
predicted_shape = predict_shape(model, new_image_unseen, SHAPES)
print(f"Predicted shape: {predicted_shape}")

# Predict shape for new seen image
print("Generating new image [seen shape = triangle]")
new_image_seen = generate_shape_image('triangle')
predicted_shape = predict_shape(model, new_image_seen, SHAPES)
print(f"Predicted shape: {predicted_shape}")

# New shapes for transfer learning
NEW_SHAPES = ['star', 'pentagon']

# Function to generate images with new shapes
def generate_new_shape_image(shape, size=IMAGE_SIZE):
    image = np.zeros(size + (3,), dtype=np.uint8)
    color = tuple(np.random.randint(0, 256, 3).tolist())

    if shape == 'star':
        # Generate a star
        pts = np.array([[10,5], [20,30], [70,20], [50,50], [40,70], [10,50], [20,20]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(image,[pts],True,color)
    elif shape == 'pentagon':
        # Generate a pentagon
        pts = np.array([[50,50], [150,50], [200,150], [150,200], [50,200]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(image,[pts],True,color)

    return image

# Create new dataset with new shapes
new_dataset, new_labels = create_dataset(NUM_SAMPLES, NEW_SHAPES, generate_shape_image=generate_new_shape_image)

# Split new dataset into train and validation
new_train_dataset, new_val_dataset = split_dataset(new_dataset, new_labels)

# Create new data loaders
new_train_loader, new_val_loader = create_data_loaders(new_train_dataset, new_val_dataset)

class TransferShapeClassifier(SimpleShapeClassifier):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__(num_classes, learning_rate)
        # Freeze the convolutional layers
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        # Replace the last fully connected layer to match the number of classes in the new dataset
        self.fc2 = torch.nn.Linear(128, num_classes)

# Define new model
new_model = TransferShapeClassifier(num_classes=len(NEW_SHAPES))

# Define new trainer
new_trainer = pl.Trainer(max_epochs=10, default_root_dir='./logs')

# Train new model
new_trainer.fit(new_model, new_train_loader, new_val_loader)

# Test new model
new_test_dataset = torch.utils.data.TensorDataset(new_dataset, new_labels)
new_test_loader = DataLoader(new_test_dataset, batch_size=32, shuffle=False)
new_trainer.test(model=new_model, dataloaders=new_test_loader)

# Predict shape for new unseen image
print("Generating new image [unseen shape = hexagon]")
new_image_unseen = generate_new_shape_image('hexagon')
predicted_shape = predict_shape(new_model, new_image_unseen, NEW_SHAPES)
print(f"Predicted shape: {predicted_shape}")

# Predict shape for new seen image
print("Generating new image [seen shape = star]")
new_image_seen = generate_new_shape_image('star')
predicted_shape = predict_shape(new_model, new_image_seen, NEW_SHAPES)
print(f"Predicted shape: {predicted_shape}")
print("Generating new image [seen shape = pentagon]")
new_image_seen = generate_new_shape_image('pentagon')
predicted_shape = predict_shape(new_model, new_image_seen, NEW_SHAPES)
print(f"Predicted shape: {predicted_shape}")
