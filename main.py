import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import math
import numpy as np


class CyclicLR(Callback):
    def __init__(self, base_lr=1e-3, max_lr=6e-3, step_size=2000., mode='triangular', gamma=0.99):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.gamma = gamma  # Decay rate for max_lr
        self.cycle_count = 0  # Keep track of the number of cycles
        self.history = {}

    def clr(self):
        cycle = math.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        elif self.mode == 'triangular_exponential_decay':
            # Apply exponential decay to max_lr
            decayed_max_lr = self.max_lr * (self.gamma ** self.cycle_count)
            lr = self.base_lr + (decayed_max_lr - self.base_lr) * max(0, (1 - x))
            if self.clr_iterations % (2 * self.step_size) == 0:
                self.cycle_count += 1  # Increment cycle count at the end of each cycle
        return lr

    def on_train_begin(self, logs=None):
        logs = logs or {}
        if self.clr_iterations == 0:
            self.clr_iterations = 0.
        else:
            self.clr_iterations = self.trn_iterations

        self.trn_iterations = 0.
        self.model.optimizer.lr = self.base_lr

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        lr = self.clr()
        self.model.optimizer.lr = lr
        self.history.setdefault('lr', []).append(lr)
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data by scaling the images to the range of [0, 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Convert labels to one-hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the neural network architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Input layer that flattens the 28x28 images
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons
    Dense(64, activation='relu'),   # Second hidden layer with 64 neurons
    Dense(10, activation='softmax')  # Output layer with 10 neurons for 10 classes
])

# Define the custom callback for changing optimizers
class KnowledgeTranmisionCallback(Callback):
    def __init__(self, number_of_phases, mode = 'optimizer'):
        super(KnowledgeTranmisionCallback, self).__init__()
        self.optimizers = [
            # tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9),
            tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001)
        ]
        self.mode = mode
        self.number_of_phases = number_of_phases
        self.epochs_per_phase = None

    def on_train_begin(self, logs=None):
        # Calculate epochs per phase
        self.epochs_per_phase = self.params['epochs'] // self.number_of_phases

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epochs_per_phase == 0 and (epoch + 1) != self.params['epochs']:
            if self.mode == 'optimizer':
                # Select a random optimizer
                old_lr = self.model.optimizer.lr
                new_optimizer = np.random.choice(self.optimizers)
                self.model.optimizer = new_optimizer
                self.model.optimizer.lr = old_lr
                print(f"\nTransitioning to new optimizer: {new_optimizer.get_config()['name']} at epoch {epoch + 1}")
            elif self.mode == 'layer':
                return

# Initialize the callback
knowledge_callback = KnowledgeTranmisionCallback(number_of_phases=2, mode='optimizer')

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# Train the model
# # Initialize the cyclic learning rate callback
clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=200., mode='triangular_exponential_decay', gamma=0.94)
# callbacks=[clr, change_optimizer_callback]
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, callbacks=[clr, knowledge_callback])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)

(test_loss, test_acc)

plt.figure(figsize=(10, 6))
plt.plot(clr.history['lr'])
plt.xlabel('Batch number')
plt.ylabel('Learning rate')
plt.title('Learning Rate Schedule')
plt.show(block=True)
