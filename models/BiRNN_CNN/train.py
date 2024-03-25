from models.BiRNN_CNN import preprocessing
from models.BiRNN_CNN import architecture
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_model(train_data_path, validation_data_path, emeddings_path):
    lr = 0.0105
    epochs = 10
    batch_size = 64
    droprate = 0.68

    train_data_list, val_data_list, labels_train, labels_val = preprocessing.get_dataset(train_data_path,
                                                                                        validation_data_path,
                                                                                        emeddings_path)
    word2idx,\
        case2idx,\
        char2idx,\
        label2idx,\
        word_embeddings,\
        case_embeddings = preprocessing.get_dicts_and_embeddings(train_data_path,
                                                                validation_data_path,
                                                                emeddings_path)

    model = architecture.build_model(word2idx=word2idx,
                                    case2idx=case2idx,
                                    char2idx=char2idx,
                                    label2idx=label2idx,
                                    word_embeddings=word_embeddings,
                                    case_embeddings=case_embeddings,
                                    droprate=droprate)

    opt = keras.optimizers.Adam(learning_rate=lr)
    loss = keras.losses.SparseCategoricalCrossentropy()

    model.compile(loss=loss, optimizer=opt)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, monitor="val_loss")]

    model.summary()

    history = model.fit(x=train_data_list,
                        y=labels_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_data_list, labels_val),
                        callbacks=callbacks)

    return model.predict(val_data_list)
