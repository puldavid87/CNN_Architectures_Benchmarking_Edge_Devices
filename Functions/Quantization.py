import tensorflow as tf
import os
import numpy as np
import pathlib
import pandas as pd


#Define some parameters for the loader:
batch_size = 32
img_height = 224
img_width = 224

def representative_data_gen(train_images):
  for input_value,_ in train_images.take(20):
    input_value=np.expand_dims(input_value[0], axis=0).astype(np.float32)
    yield [input_value]

def quantized(model,mod_name, model_name, path_destination, train_images):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path(path_destination)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    direc=str(model_name + "_" + mod_name + "_" + ".tflite")
    tflite_model_file = tflite_models_dir/direc
    tflite_model_file.write_bytes(tflite_model)

    ###Post-training float16 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    direc=str(model_name + "_" + mod_name + "_" +  "_quant_f16.tflite")
    tflite_fp16_model = converter.convert()
    tflite_model_fp16_file = tflite_models_dir/direc
    tflite_model_fp16_file.write_bytes(tflite_fp16_model)

    #Post-training dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    direc=str(model_name + "_" + mod_name + "_" + "_quant.tflite")
    tflite_model_quant_file = tflite_models_dir/direc
    tflite_model_quant_file.write_bytes(tflite_quant_model)

    #Post-training integer quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    converter.representative_dataset = representative_data_gen(train_images)
    tflite_model_quant_float = converter.convert()
    direc=str(model_name + "_" + mod_name + "_"  + "_quant_float.tflite")
    tflite_model_quant_float_file = tflite_models_dir/direc
    tflite_model_quant_float_file.write_bytes(tflite_model_quant_float)

    #Convert using integer-only quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    direc=str(model_name + "_" + mod_name + "_" + "_quant_int.tflite")
    tflite_model_quant_int = converter.convert()
    tflite_model_quant_int_file = tflite_models_dir/direc
    tflite_model_quant_int_file.write_bytes(tflite_model_quant_int)

    #Load the model into the interpreters

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()

    interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
    interpreter_fp16.allocate_tensors()

    interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter_quant.allocate_tensors()

    interpreter_quant_float = tf.lite.Interpreter(model_path=str(tflite_model_quant_float_file))
    interpreter_quant_float.allocate_tensors()

    interpreter_quant_int = tf.lite.Interpreter(model_path=str(tflite_model_quant_int_file))
    interpreter_quant_int.allocate_tensors()

    models=[interpreter, interpreter_fp16, interpreter_quant,interpreter_quant_float,interpreter_quant_int]
    tam = [tflite_model_file, tflite_model_fp16_file, tflite_model_quant_file,tflite_model_quant_float_file,tflite_model_quant_int_file]
    x = 0
    for test_model in models:
        score = evaluate_model(test_model)
        print(score)
        print("######################################")
        size_tf = os.path.getsize(tam[x])
        print(size_tf)
        x += 1
 
# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter, test_images):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    cont=0
    var = 0
    accurate_count = 0
    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    input_details = interpreter.get_input_details()[0]
    for test_image,test_label in test_images:
        for image in test_image:
            test_pred = np.argmax(test_label[var,:])
            var += 1
            if input_details['dtype'] == np.uint8:
                test_image=np.expand_dims(image, axis=0).astype(np.uint8)
            else:
                test_image=np.expand_dims(image, axis=0).astype(np.float32)

            interpreter.set_tensor(input_index, test_image)
              # Check if the input type is quantized, then rescale input data to uint8
              # Run inference.
            interpreter.invoke()
        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)
          # Compare prediction results with ground truth labels to calculate accuracy.
            cont += 1
            if digit == test_pred:
                accurate_count += 1
        var = 0
    accuracy = accurate_count * 1.0 / cont
    return accuracy

