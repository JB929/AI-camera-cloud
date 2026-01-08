import tensorflow as tf
import numpy as np

MODEL_PATH = "src/core/movenet_thunder.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def movenet_detect(frame):
    # --- PREPROCESS INPUT based on what model requires ---
    img = frame[:, :, ::-1]  # BGR → RGB
    img = tf.image.resize_with_pad(img, 256, 256)

    # Check model input type
    input_type = input_details[0]['dtype']

    if input_type == np.uint8:
        # Model wants uint8 → scale to [0,255]
        img = tf.cast(img, tf.uint8)
    else:
        # Float model → normalize to [0,1]
        img = tf.cast(img, tf.float32)
        img = img / 255.0

    img = tf.expand_dims(img, axis=0)

    # Set tensor
    interpreter.set_tensor(input_details[0]['index'], img.numpy())

    # Inference
    interpreter.invoke()

    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return keypoints

