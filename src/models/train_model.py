import tensorflow.keras as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_svm(
    data,
    test_size: float = 0.33,
    probability: bool = False,
    random_state=42
):
    X = list(map(lambda x: x[1], data))
    y = list(map(lambda x: x[2], data))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    clf = make_pipeline(
        StandardScaler(),
        SVC(gamma='auto', probability=probability)
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, classification_report(y_test, y_pred)


def train_resnet(
    data,
    batch_size: int = 100,
    epochs: int = 10,
    lr: float = 0.01,
    test_size: float = 0.1,
    save_model_path: str = None,
    pretrained_model: K.Model = None,
    random_state: int = 42
):

    X = np.array(list(map(lambda x: x[1], data)))
    y = np.array(list(map(lambda x: x[2], data)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    if pretrained_model:
        model = pretrained_model
    else:
        input_t = K.Input(shape=(X[0].shape))
        res_model = K.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_t
        )

        for layer in res_model.layers:
            layer.trainable = False

        model = K.models.Sequential()
        model.add(res_model)
        model.add(K.layers.Flatten())
        # model.add(K.layers.Dense(10, activation='relu'))
        # model.add(K.layers.Dense(5, activation='relu'))
        model.add(K.layers.Dense(2, activation='softmax'))

    model.compile(
        optimizer=K.optimizers.Adam(lr),
        loss=K.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test)
    )

    if save_model_path:
        model.save_weights(save_model_path)

    return model, history
