import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define a basic residual block
def residual_block(x, filters, kernel_size=3, stride=1):
    identity = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, identity])
    x = ReLU()(x)
    return x

# Build the ResNet model
def build_resnet(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    for _ in range(3):
        x = residual_block(x, filters=64)
        
    x = residual_block(x, filters=128, stride=2)
    for _ in range(3):
        x = residual_block(x, filters=128)
        
    x = residual_block(x, filters=256, stride=2)
    for _ in range(5):
        x = residual_block(x, filters=256)
        
    x = residual_block(x, filters=512, stride=2)
    for _ in range(2):
        x = residual_block(x, filters=512)
        
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the ResNet model
model = build_resnet(input_shape=(32, 32, 3), num_classes=10)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
