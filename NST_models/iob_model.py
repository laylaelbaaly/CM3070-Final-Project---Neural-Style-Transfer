# NST_models/iob_model.py

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

# -----------------------------
# Globals (as in the notebook)
# -----------------------------
img_height = 512   

# -----------------------------
# Utilities (from the notebook)
# -----------------------------
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

def deprocess_image(x):
    x = x.reshape((img_height, img_width, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR -> RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return Image.fromarray(x)

def gram_matrix(x):
    # x expected [1, H, W, C]
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    input_shape = tf.shape(x)
    H = tf.cast(input_shape[1], tf.float32)
    W = tf.cast(input_shape[2], tf.float32)
    C = tf.cast(input_shape[3], tf.float32)
    norm = H * W * C
    return tf.squeeze(result / norm, axis=0)

def total_variation_loss(x):
    a = tf.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = tf.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

def plot_losses():
    if not (content_losses and style_losses):
        return
    plt.figure(figsize=(10, 6))
    plt.plot(content_losses, label='Content Loss')
    plt.plot(style_losses, label='Style Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Style Transfer Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# Evaluator (for L-BFGS)
# -----------------------------
class Evaluator:
    def __init__(self, loss_fn, shape):
        self.loss_fn = loss_fn
        self.shape = shape
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        loss, grads = self.loss_fn(x.reshape(self.shape))
        self.loss_value = loss
        self.grad_values = grads.flatten().astype('float64')
        return self.loss_value

    def grads(self, x):
        return self.grad_values


def run_style_transfer(content_path, style_path, output_path,
                       content_weight, style_weight, tv_weight,
                       optimizer_type, init_type,
                       iterations=300, learning_rate=10.0):
    
    start_time = time.time()

    # derive img_width from incoming content
    global img_width, content_losses, style_losses
    width, height = load_img(content_path).size
    img_width = int(width * img_height / height)

    content_losses, style_losses = [], []

    # preprocess inputs
    content_image = preprocess_image(content_path, (img_height, img_width))
    style_image   = preprocess_image(style_path,   (img_height, img_width))

    # init combination image
    if init_type == "content":
        combo_img = tf.Variable(content_image, dtype=tf.float32)
    else:
        noise_img = tf.random.uniform(content_image.shape, 0, 255)
        noise_img = vgg19.preprocess_input(noise_img)
        combo_img = tf.Variable(noise_img, dtype=tf.float32)

    # VGG19 features
    vgg = vgg19.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False
    content_layer = 'block5_conv2'
    style_layers  = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    outputs = [vgg.get_layer(content_layer).output] + [vgg.get_layer(n).output for n in style_layers]
    feature_extractor = Model(inputs=vgg.input, outputs=outputs)

    content_features = feature_extractor(content_image)[0]
    style_features   = feature_extractor(style_image)[1:]

    def loss_and_grads(x_tensor):
        with tf.GradientTape() as tape:
            outs  = feature_extractor(x_tensor)
            c_out = outs[0]
            s_outs= outs[1:]

            c_loss = content_weight * tf.reduce_mean(tf.square(c_out - content_features))
            s_loss = 0.0
            for sF, sO in zip(style_features, s_outs):
                s_loss += tf.reduce_mean(tf.square(gram_matrix(sF) - gram_matrix(sO)))
            s_loss *= style_weight / len(style_layers)
            tv_l   = tv_weight * total_variation_loss(x_tensor)
            total  = c_loss + s_loss + tv_l

        grads = tape.gradient(total, x_tensor)
        return total, grads, c_loss, s_loss

    if optimizer_type == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        for i in range(iterations):
            loss, grads, c_l, s_l = loss_and_grads(combo_img)
            optimizer.apply_gradients([(grads, combo_img)])
            content_losses.append(float(c_l.numpy()))
            style_losses.append(float(s_l.numpy()))
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{iterations}] Total:{loss.numpy():,.2f} | Content:{c_l.numpy():,.2f} | Style:{s_l.numpy():,.2f}")
    else:
        # L-BFGS
        lbfgs_step = [0]
        def wrapped_loss(x):
            x_tensor = tf.convert_to_tensor(x.reshape((1, img_height, img_width, 3)), dtype=tf.float32)
            x_var = tf.Variable(x_tensor)
            loss, grads, c_l, s_l = loss_and_grads(x_var)
            content_losses.append(float(c_l.numpy()))
            style_losses.append(float(s_l.numpy()))
            lbfgs_step[0] += 1
            if lbfgs_step[0] % 10 == 0:
                print(f"LBFGS Step [{lbfgs_step[0]}] Total:{loss.numpy():,.2f} | Content:{c_l.numpy():,.2f} | Style:{s_l.numpy():,.2f}")
            return loss.numpy().astype("float64"), grads.numpy().flatten().astype("float64")

        evaluator = Evaluator(wrapped_loss, (1, img_height, img_width, 3))
        x_opt, _, _ = fmin_l_bfgs_b(evaluator.loss,
                                    combo_img.numpy().flatten(),
                                    fprime=evaluator.grads,
                                    maxiter=iterations)
        combo_img = tf.convert_to_tensor(x_opt.reshape((1, img_height, img_width, 3)))

    total_time = time.time() - start_time
    final_img = deprocess_image(combo_img.numpy())
    final_img.save(output_path)
    print(f"Runtime: {total_time:.2f} sec | Saved: {output_path}")

    plot_losses()
    return content_losses[-1], style_losses[-1], total_time
