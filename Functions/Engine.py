import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def train_model(model, train_data, validation_data, test_data, callback, path_model_destination,epochs, name, test):
    history = model.fit(train_data,
                        epochs=epochs,
                        steps_per_epoch=len(train_data),
                        validation_data=validation_data,
                        # Go through less of the validation data so epochs are
                        # faster (we want faster experiments!)
                        validation_steps=int(len(validation_data)),
                        callbacks=[callback],
                        verbose=1,
                        )
    return model, history
    
def predict_and_extract(model, test_data):
    """Predict classes and extract true and predicted labels"""
    y_test = []    
    y_pred = []
    cont = 0
    for test_image, test_label in test_data:
        for image in test_image:            
            y_pred.append(np.argmax((model.predict(np.expand_dims(image, axis=0)))))
            y_test.append(np.argmax(test_label[cont,:]))
            cont+=1
        cont = 0
    return y_test, y_pred

def calculate_and_print_metrics(y_test, y_pred):
    """Calculate and print classification metrics"""
    print(
        "Precision: {}%".format(
            100 *
            metrics.precision_score(
                y_test,
                y_pred,
                average="weighted")))
    
    print(
        "Recall: {}%".format(
            100 *
            metrics.recall_score(
                y_test,
                y_pred,
                average="weighted")))
    
    print(
        "f1_score: {}%".format(
            100 *
            metrics.f1_score(
                y_test,
                y_pred,
                average="weighted")))
            
    print("Error: {}%".format(metrics.mean_absolute_error(y_test, y_pred)))
    
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print('\\Report\n')
    print(report)
    print(matrix)

def results(model, test_data):
    y_test, y_pred = predict_and_extract(model, test_data)
    calculate_and_print_metrics(y_test, y_pred)
    
def plot_loss_curves(history, name, test, path_model_destination):
    """
    Plots loss and accuracy curves from the Keras model history
    Args:
        history: Keras model history object
        name: Name of the model
    Returns: 
        None: Does not return anything, saves figures to file
    Processing Logic:
        - Extracts loss and accuracy values from history
        - Plots loss vs epochs and saves figure
        - Plots accuracy vs epochs and saves figure
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(path_model_destination+ "/" + name + "/" + test +
                "Loss_" +
                str(name) + ".png")

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(path_model_destination+ "/" + name + "/" + test +
                "ACC_" +
                str(name) + ".png")