import Dataset_setup as ds
import Engine 
import Moldel_builder as mb
import Utils
import Quantization
import tensorflow as tf

#variables
classes = 4
epochs = 1
unfreeze_layers = -20
img_height = 224
img_width = 224
# Define some parameters for the loader:
batch_size = 32
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

# Models to be trained
models_name = ["Efficient", "Mobilenet", "Resnet", "VGG", "Inception"]

#datapaths and folders
path_data_source = "your_path"
test_path = "your_path"
train_path = "your_path"
model_path = "your_path"
result_path = "your_path" 
#########################
ds.versions
ds.check_data(path_data_source)
labels, tam_labels = ds.get_labels(path_data_source)
ds.view_n_images(train_path,labels,3)
train_data, validation_data, test_data = ds.split_tratin_test_set(path_data_source,
                                                                  batch_size,
                                                                  img_height,
                                                                  img_width)

######## Create folders ###########
for name in models_name:
    Utils.make_folder(name, "Models_Folder_path")
    Utils.make_folder(name, "Results_Folder_path")
############## TEST 1 #############
Efficient_model = mb.build_Efficient_model(classes, False )
Mobilenet_model = mb.build_Mobilenet_model(classes, False )
Resnet_model = mb.build_Resnet_model(classes, False )
VGG_model = mb.build_VGG16_model (classes, False )
Inception_model = mb.build_Inception_model (classes, False )
models = [Efficient_model, Mobilenet_model, Resnet_model, VGG_model, Inception_model, Inception_model]
for i, model in enumerate(models):
    model_path_out = model_path + "/" + str(models_name [i])
    model_out, history = Engine.train_model(model, 
                                            train_data, 
                                            validation_data, 
                                            test_data, 
                                            callback,
                                            model_path_out,
                                            epochs, 
                                            str(models_name [i]),
                                            "test1")
    y_test, y_pred = Engine.redict_and_extract(model_out, test_data)
    Engine.calculate_and_print_metrics(y_test,y_pred)
    Engine.results(model_out,test_data)
    Engine.plot_loss_curves(history, str(models_name [i]), "test1", model_path_out)
    model.save(model_path_out + "test1" + ".h5")
    Quantization.quantized(model, "test1", str(models_name [i]), model_path_out, train_data)
    

############## TEST 2 #############
Efficient_model = mb.build_Efficient_model(classes, True )
Mobilenet_model = mb.build_Mobilenet_model(classes, True )
Resnet_model = mb.build_Resnet_model(classes, True )
VGG_model = mb.build_VGG16_model (classes, True )
Inception_model = mb.build_Inception_model (classes, True )
models = [Efficient_model, Mobilenet_model, Resnet_model, VGG_model, Inception_model, Inception_model]

for i, model in enumerate(models):
    model_path_out = model_path + "/" + str(models_name [i])
    model_out, history = Engine.train_model(model, 
                                            train_data, 
                                            validation_data, 
                                            test_data, 
                                            callback,
                                            model_path_out,
                                            epochs, 
                                            str(models_name [i]),
                                            "test2")
    y_test, y_pred = Engine.redict_and_extract(model_out, test_data)
    Engine.calculate_and_print_metrics(y_test,y_pred)
    Engine.results(model_out,test_data)
    Engine.plot_loss_curves(history, str(models_name [i]), "test2", model_path_out)
    model.save(model_path_out + "test2" + ".h5")
    Quantization.quantized(model, "test2", str(models_name [i]), model_path_out, train_data)


############## TEST 3 #############
Efficient_model = mb.unfreeze_model(mb.build_Efficient_model(classes, True ), 20)
Mobilenet_model = mb.unfreeze_model(mb.build_Mobilenet_model(classes, True ), 20)
Resnet_model = mb.unfreeze_model(mb.build_Resnet_model(classes, True ), 20)
VGG_model = mb.unfreeze_model(mb.build_VGG16_model (classes, True ), 20)
Inception_model = mb.unfreeze_model(mb.build_Inception_model (classes, True ), 20)
models = [Efficient_model, Mobilenet_model, Resnet_model, VGG_model, Inception_model, Inception_model]

for i, model in enumerate(models):
    model_path_out = model_path + "/" + str(models_name [i])
    model_out, history = Engine.train_model(model, 
                                            train_data, 
                                            validation_data, 
                                            test_data, 
                                            callback,
                                            model_path_out,
                                            epochs, 
                                            str(models_name [i]),
                                            "test3")
    y_test, y_pred = Engine.redict_and_extract(model_out, test_data)
    Engine.calculate_and_print_metrics(y_test,y_pred)
    Engine.results(model_out,test_data)
    Engine.plot_loss_curves(history, str(models_name [i]), "test3", model_path_out)
    model.save(model_path_out + "test3" + ".h5")
    Quantization.quantized(model, "test3", str(models_name [i]), model_path_out, train_data)