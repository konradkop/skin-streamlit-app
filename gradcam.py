import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def make_gradcam_heatmap_tf2(img_array, model, last_conv_layer_name, class_index=None):
    img_array = np.asarray(img_array, dtype="float32")
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, 0)
    img_tensor = tf.convert_to_tensor(img_array)

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        preds_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
        if class_index is None:
            class_index = int(tf.argmax(preds_tensor[0]))
        class_score = preds_tensor[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    if grads is None:
        raise RuntimeError("Gradients are None.")

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy()), class_index

    heatmap = heatmap / max_val
    H, W = img_array.shape[1], img_array.shape[2]
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (H, W))
    return tf.squeeze(heatmap).numpy(), class_index

def overlay_heatmap_on_pil(orig_pil, heatmap, alpha=0.35):
    import matplotlib.cm as cm
    base = np.asarray(orig_pil.convert("RGB"), dtype="float32") / 255.0
    H0, W0, _ = base.shape

    heatmap_tf = tf.convert_to_tensor(heatmap)
    if heatmap_tf.ndim == 2:
        heatmap_tf = heatmap_tf[tf.newaxis, ..., tf.newaxis]
    heatmap_resized = tf.image.resize(heatmap_tf, (H0, W0))[0, :, :, 0].numpy()

    if heatmap_resized.max() > 0:
        heatmap_resized /= heatmap_resized.max()

    color_hm = cm.get_cmap("jet")(heatmap_resized)[..., :3]
    overlay = alpha * color_hm + (1 - alpha) * base
    return (overlay * 255).clip(0, 255).astype("uint8")
