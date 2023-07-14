from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import pandas as pd
# impoting l2 regularizer
from tensorflow.keras.regularizers import l2
import os

# define input shape and batch size
input_shape = (625, 625, 3)
batch_size = 32
model_type = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNetV3Large', 'NASNetLarge', 'NASNetMobile', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'VGG16', 'VGG19', 'Xception']
ep = 50
learning_rate = 0.00000001

# paths
train_dir = "datasets/images"
# test_dir = "datasets/images/test" # didn't work with ImageDataGeneartor.flow_from_dataframe
csv_dir = "datasets/labels/labels_train.csv"
label_names_dir = "datasets/labels/categories.csv"

# read csv data for loading image label information
df = pd.read_csv(csv_dir)
df_labels = pd.read_csv(label_names_dir)
print(df_labels.head())
label_names = list(df_labels["Labels"])
x_col = df.columns[0]
y_cols = list(df.columns[1:len(label_names)+1])

# load input images and split into training, test and validation
datagen = ImageDataGenerator(rescale=1./255,validation_split=.25)

train_generator = datagen.flow_from_dataframe(
    df,
    directory=train_dir,
    x_col=x_col,
    y_col=y_cols,
    subset="training",
    target_size=input_shape[0:2],
    color_mode="rgb",
    class_mode="raw", # for multilabel output
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
    validate_filenames=False
)

test_generator = datagen.flow_from_dataframe(
    df,
    directory=train_dir,
    x_col=x_col,
    y_col=y_cols,
    subset="validation",
    target_size=input_shape[0:2],
    color_mode="rgb",
    class_mode="raw",
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
    validate_filenames=False
)

# build model
n_outputs = len(label_names)
for i in model_type:
    # Creating the model
    if model_type == 'DenseNet121':
        model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(625,625,3)
        )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    if model_type == 'DenseNet121':
        model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(625,625,3)
        )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'DenseNet169':
        model = DenseNet169(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'DenseNet201':
        model = DenseNet201(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])


    elif model_type == 'InceptionResNetV2':
        model = InceptionResNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'InceptionV3':
        model = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'MobileNetV3Large':
        model = MobileNetV3Large(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'NASNetLarge':
        model = NASNetLarge(
                weights='imagenet',
                include_top=False,
                input_shape=(331,331,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])        

    elif model_type == 'NASNetMobile':
        model = NASNetMobile(
                include_top=False,
                input_shape=(224,224,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'VGG19':
        model = VGG19(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'Xception':
        model = Xception(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])
        
    elif model_type == 'ResNet50':
        model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'ResNet101':
        model = ResNet101(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'ResNet152':
        model = ResNet152(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'ResNet50V2':
        model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'ResNet101V2':
        model = ResNet101V2(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'ResNet152V2':
        model = ResNet152V2(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    elif model_type == 'InceptionV3':
        model = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])


    elif model_type == 'VGG16':
        model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.Flatten()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    else:
        model_type == 'EfficientNetB0'
        model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(625,625,3)
                )
        for layer in model.layers:
                layer.trainable = True
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(128, activation = 'relu',kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(n_outputs, activation = 'sigmoid',kernel_regularizer=l2(0.01))(x)
        model = Model(model.input, x)
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])


    # define training and validation steps
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = test_generator.samples // test_generator.batch_size

    # train model
    hist = model.fit(train_generator,epochs=ep, steps_per_epoch=steps_per_epoch,validation_data=test_generator,validation_steps=validation_steps).history
    if not os.path.exists(f"models/{i}"):
        os.makedirs(f"models/{i}")
    print(hist)
    # saving history to csv
    hist_df = pd.DataFrame(hist)
    hist_csv_file = f"models/{i}/{i}_history.csv"
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    model.summary()
    # save model
    model.save(f"models/{i}/{i}.h5")
