import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, Input
from keras.layers import Dropout, MaxPooling1D, Conv1D, Flatten
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing


def load_info(csv):
    """
    Load information about dataset (csv file)
    :param csv: csv file to load with dataset information
    :return: pandas.dataframe with images names and categories
    """
    df = pd.read_csv(csv)
    print('Tobacco has {} documents'.format(df.shape[0]))

    return df


def plot_hist(df):
    """
    Plot histogram of document labels
    :param df: dataframe with document's names and categories
    :return: None
    """
    plt.title('Histogram')
    sns.countplot(data=df, y='label')
    plt.savefig('hist')
    plt.show()


def show_examples(n, df):
    """
    Show n examples (image and corresponding text)
    :param n: number of examples to show
    :param df: dataframe with dataset information
    :return: None
    """
    # Number of examples to show
    n_examples = n

    # choose randomly n documents to display
    ind = np.random.choice(df.shape[0], n_examples)

    for i, j in enumerate(ind):
        im = plt.imread('Tabacco3482/' + str(df.img_path[j]))
        plt.figure(i, figsize=(15, 10))
        plt.title(df.label[j] + str(i))
        plt.axis('off')
        plt.imshow(im, cmap='gray')
        plt.show()
        txt_path = 'data/Tabacco3482/' + '.'.join(df.img_path[j].split('.')[:-1]) + '.txt'
        with open(txt_path, 'r') as f:
            lines = [line for line in f]
        print('Text of figure {}:'.format(df.label[j] + str(i)))
        for l in lines:
            print(l.rstrip())


def load_data(df, texts):
    """
    Load relevant data (text) to our problem
    :param df: dataframe with dataset information
    :param texts: dir where texts are stored
    :return: X: list of images text obtain previously by OCR
    """
    X = []
    for j in range(df.shape[0]):
        txt_path = texts + '.'.join(df.img_path[j].split('.')[:-1]) + '.txt'

        with open(txt_path, 'r') as f:
            lines = [line for line in f]
        s = [l.rstrip() for l in lines if l.rstrip() != '']
        r = ''
        for l in s:
            r += l
        X.append(r)

    return X


def split_data(x, df):
    """
    Split data in train (60%), dev (20%), test(20%)
    :param x: text list
    :param df: dataframe with labels (target)
    :return: xtrain, ytrain, xdev, ydev, xtest, ytest
    """
    xtrain, xother, ytrain, yother = train_test_split(x, df.iloc[:].label, test_size=0.4)

    xdev, xtest, ydev, ytest = train_test_split(xother, yother, test_size=0.5)
    print('Train : {}, Dev : {}, Test : {}'.format(len(xtrain), len(xdev), len(xtest)))

    return xtrain, ytrain, xdev, ydev, xtest, ytest


def get_train_test(train_raw_text, test_raw_text):
    """

    :param train_raw_text: data to train (list of documents' texts)
    :param test_raw_text: data to test
    :return: train and test sets padded
    """
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)

    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), \
           sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH)


def get_model(units, dout, emb, mtl, n_out):
    """
    Build Model
    :param units: cells number of conv1D
    :param dout: rate for dropout
    :param emb: output size of embedding layer
    :param mlt: input size
    :param n_out: number of classes to dense layer
    :return: model
    """
    inp = Input(shape=(mtl,))
    model = Embedding(mtl, emb)(inp)
    model = Dropout(dout)(model)
    model = Conv1D(filters=units, kernel_size=emb, padding='same', activation='relu')(model)
    model = MaxPooling1D(pool_size=2)(model)
    model = Flatten()(model)
    model = Dense(n_out, activation="softmax")(model)
    model = Model(inputs=inp, outputs=model)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train_fit_predict(model, x_train, x_test, y, callbacks):
    """
    Fit and predict
    :param model: model previously built
    :param x_train: data to train the model
    :param x_test: data to predict with the trained model
    :param y: target of train set (labels)
    :param callbacks: object to checkpoint
    :return: y_prediction of x_test
    """
    model.fit(x_train, y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1,
              validation_split=VALIDATION_SPLIT,
              callbacks=callbacks)

    return model.predict(x_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training / testing Model")
    parser.add_argument("-csv", type=str, help="csv file with name images and categories", required=True)
    parser.add_argument("-dir_text", type=str, help="texts directory", required=True)
    pargs = parser.parse_args()

    # Load dataset information
    df = load_info(pargs.csv)

    # Plot histogram (documents per categories)
    # plot_hist(df)

    # Show some examples
    show_examples(1, df)

    # Load texts from documents (read by OCR)
    X = load_data(df, pargs.dir_text)

    # split dataset
    xtrain, ytrain, xdev, ydev, xtest, ytest = split_data(X, df)

    # Model parameters
    MAX_FEATURES = 5000
    MAX_TEXT_LENGTH = 2048
    EMBED_SIZE = 200
    BATCH_SIZE = 16
    EPOCHS = 20
    VALIDATION_SPLIT = 0.1

    # Get the list of different classes and total classes
    CLASSES_LIST = np.unique(ytrain)
    n_out = len(CLASSES_LIST)
    print('Number of classes : {}'.format(CLASSES_LIST))

    # Encode labels
    le = preprocessing.LabelEncoder()
    le.fit(CLASSES_LIST)
    y_train = le.transform(ytrain)
    y_test = le.transform(ydev)
    y_final = le.transform(ytest)
    train_y_cat = np_utils.to_categorical(y_train, n_out)

    # get the textual data in the correct format for NNE
    x_vec_train, x_vec_test = get_train_test(xtrain, xdev)
    _, x_vec_final = get_train_test(xtrain, xtest)
    print('Samples to train : {}, Samples to test in dev : {}'.format(len(x_vec_train), len(x_vec_test)))

    # define the NN topology (units of conv1D : 32, dropout = 0.4)
    model = get_model(32, 0.4, EMBED_SIZE, MAX_TEXT_LENGTH, n_out)

    # Save weights of each epoch
    checkpoint_period = 1
    checkpoint_path = './weights/'
    callbacks = [ModelCheckpoint(checkpoint_path + "weights.{epoch:02d}.hdf5", period=checkpoint_period)]

    # Train and predict
    y_predicted = train_fit_predict(model, x_vec_train, x_vec_test, train_y_cat, callbacks).argmax(1)
    # print("Test Accuracy (dev):", accuracy_score(y_test, y_predicted))

    acc = []
    for i in range(1, EPOCHS + 1):

        # print('Epoch : {}'.format(i))
        if i < 10:
            model.load_weights('./weights/weights.0' + str(i) + '.hdf5')
        else:
            model.load_weights('weights/weights.' + str(i) + '.hdf5')

        y_predicted = model.predict(x_vec_test).argmax(1)

        # print("Test Accuracy:", accuracy_score(y_test, y_predicted))
        acc.append(accuracy_score(y_test, y_predicted))

    # Pick the best epoch (to have the best generalization)
    best_one = np.asarray(acc).argmax() + 1
    print('Best epoch : {}'.format(best_one))

    if best_one < 10:
        model.load_weights('./weights/weights.0' + str(best_one) + '.hdf5')
    else:
        model.load_weights('weights/weights.' + str(best_one) + '.hdf5')

    print('Test Accuracy (dev) : {}'.format(acc[best_one - 1]))
    y_pred = model.predict(x_vec_final).argmax(1)
    print('Confusion matrix on test :')
    print(confusion_matrix(y_final, y_pred))
    print("Test Accuracy (test):", accuracy_score(y_final, y_pred))
