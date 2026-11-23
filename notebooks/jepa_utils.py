# jepa_utils.py — all the machinery; main.py tells the story.
#
# Contents:
#   patchify              — (B, img, img, C) -> (B, n_patch, patch_dim)
#   mlp                   — small MLP builder (activation: GELU)
#   make_encoder          — shared encoder factory (used inside JEPA)
#   extract_features      — ANY encoder + images -> one emb-dim vector per image
#   class JEPA            — encoders, predictor, EMA, anti-collapse losses
#   print_class_balance   — checklist item: is the dataset balanced?
#   label_indices         — shared labeled subsample (keeps both arms fair)
#   probe_accuracy        — linear probe on frozen features
#   dense_baseline        — plain Dense network on raw pixels (the no-JEPA arm)
#
# All models end in raw logits (no softmax); losses use from_logits=True.

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ------------------------- small helpers -------------------------

def patchify(images, grid, patch, channels):
    """(B, img, img, C) -> (B, grid*grid, patch*patch*C)."""
    b = tf.shape(images)[0]
    x = tf.reshape(images, (b, grid, patch, grid, patch, channels))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (b, grid * grid, patch * patch * channels))


def mlp(inp_dim, hidden, out, name):
    inp = keras.Input((None, inp_dim))
    h = keras.layers.Dense(hidden, activation="gelu")(inp)
    return keras.Model(inp, keras.layers.Dense(out)(h), name=name)


def make_encoder(patch_dim, hidden=256, emb=128, name="encoder"):
    """A fresh encoder; used inside JEPA (kept here so both encoder
    creations share one factory). The no-JEPA arm does NOT use it."""
    return mlp(patch_dim, hidden, emb, name)


def extract_features(encoder, images, grid, patch, channels, step=1024):
    """One emb-dim vector per image (mean-pooled patch embeddings).
    Works with ANY encoder — JEPA's or otherwise. This is the
    img*img*channels -> emb squeeze, and it lives OUTSIDE the JEPA class."""
    out = []
    for i in range(0, len(images), step):
        p = patchify(tf.constant(images[i:i + step]), grid, patch, channels)
        out.append(tf.reduce_mean(encoder(p), axis=1).numpy())
    return np.concatenate(out)


# ------------------------- the JEPA class -------------------------

class JEPA:
    """Everything that makes JEPA 'JEPA' lives in this class.

      1. LATENT PREDICTION  — a predictor maps context-patch embeddings to
         predicted embeddings of the held-out patches (never pixels).
      2. EMA TARGET ENCODER — targets come from a slow-moving copy of the
         context encoder, wrapped in stop_gradient.
      3. ANTI-COLLAPSE      — variance/covariance penalties keep all `emb`
         dimensions alive. The 3072 -> emb squeeze is a byproduct of this,
         not the objective.
    """

    def __init__(self, img=32, channels=3, patch=4, emb=128, hidden=256,
                 ctx_frac=0.4, ema_decay=0.99, lr=1e-3, weight_decay=0.01):
        self.img, self.channels, self.patch = img, channels, patch
        self.grid = img // patch
        self.n_patch = self.grid * self.grid
        self.patch_dim = patch * patch * channels
        self.emb, self.hidden = emb, hidden
        self.ema_decay = ema_decay
        self.n_ctx = int(self.n_patch * ctx_frac)

        # context encoder (trained) and its EMA copy (never trained directly)
        self.context_encoder = make_encoder(self.patch_dim, hidden, emb,
                                            "context_encoder")
        self.target_encoder = keras.models.clone_model(self.context_encoder)
        self.target_encoder.set_weights(self.context_encoder.get_weights())
        self.target_encoder.trainable = False

        # positional embeddings let the predictor know WHERE a target sits
        self.pos_emb = tf.Variable(tf.random.normal((self.n_patch, emb)) * 0.02,
                                   name="pos_emb")

        # predictor: pooled context summary + target position -> predicted latent
        p_in = keras.Input((None, 2 * emb,))
        p = keras.layers.Dense(hidden, activation="gelu")(p_in)
        self.predictor = keras.Model(p_in, keras.layers.Dense(emb)(p), name="predictor")

        self.opt = keras.optimizers.AdamW(lr, weight_decay=weight_decay)  # Adam on TF < 2.11

    @tf.function
    def train_step(self, batch):                     # batch: (B, img, img, C)
        patches = patchify(batch, self.grid, self.patch, self.channels)
        idx = tf.random.shuffle(tf.range(self.n_patch))
        ctx_idx, tgt_idx = idx[:self.n_ctx], idx[self.n_ctx:]

        with tf.GradientTape() as tape:
            h_all = self.context_encoder(patches)    # (B, n_patch, emb)
            ctx_summary = tf.reduce_mean(tf.gather(h_all, ctx_idx, axis=1), axis=1)

            B = tf.shape(patches)[0]
            n_tgt = tf.size(tgt_idx)
            pos_tgt = tf.gather(self.pos_emb, tgt_idx)
            x_in = tf.concat((tf.tile(ctx_summary[:, None, :], (1, n_tgt, 1)),
                              tf.tile(pos_tgt[None], (B, 1, 1))), -1)
            pred = self.predictor(x_in)              # main point 1: predicted latents

            t_all = self.target_encoder(patches)
            t_tgt = tf.gather(t_all, tgt_idx, axis=1)
            t_tgt = tf.stop_gradient(t_tgt)   

            mse = tf.reduce_mean((pred - t_tgt) ** 2)

            flat = tf.reshape(pred, (-1, self.emb))  # main point 3: anti-collapse
            var_loss = tf.reduce_mean(tf.nn.relu(1.0 - tf.math.reduce_variance(flat, axis=0)))
            fc = flat - tf.reduce_mean(flat, axis=0)
            cov = tf.matmul(fc, fc, transpose_a=True) / tf.cast(tf.shape(flat)[0], tf.float32)
            cov_loss = (tf.reduce_sum(cov ** 2)
                        - tf.reduce_sum(tf.linalg.diag_part(cov) ** 2)) / tf.cast(self.emb, tf.float32)
            loss = mse + var_loss + 5.0 * cov_loss

        vars_ = (self.context_encoder.trainable_variables
                 + self.predictor.trainable_variables + [self.pos_emb])
        self.opt.apply_gradients(zip(tape.gradient(loss, vars_), vars_))

        for w_c, w_t in zip(self.context_encoder.weights, self.target_encoder.weights):
            w_t.assign(self.ema_decay * w_t + (1.0 - self.ema_decay) * w_c)
        return loss

    def fit(self, images, epochs=5, batch=256, seed=0):
        """Self-supervised pretraining — uses NO labels."""
        ds = tf.data.Dataset.from_tensor_slices(images).shuffle(10000, seed=seed).batch(batch)
        for epoch in range(epochs):
            losses = [float(self.train_step(b)) for b in ds]
            print(f"JEPA epoch {epoch + 1}/{epochs} - loss: {np.mean(losses):.4f}")


# ------------------------- evaluation helpers -------------------------

def print_class_balance(y_train, y_test, n_classes=10):
    """Checklist item: is the dataset balanced? An imbalanced dataset can
    make training performance look good while the model fails to
    generalize on test data, so we verify counts per class."""
    for name, ys in (("train", y_train), ("test", y_test)):
        counts = np.bincount(ys, minlength=n_classes)
        print(f"{name} class counts:", dict(enumerate(counts)))


def label_indices(n, fraction=1.0, seed=0):
    """Labeled training indices — both arms use the SAME ones, so the
    comparison stays fair when label_fraction < 1."""
    if fraction >= 1.0:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return rng.choice(n, int(n * fraction), replace=False)


def probe_accuracy(train_feats, y_train, test_feats, y_test, emb,
                   label_fraction=1.0, epochs=5, seed=0):
    """Linear probe on frozen features -> test accuracy."""
    idx = label_indices(len(train_feats), label_fraction, seed)
    probe = keras.Sequential([
        keras.layers.Input((emb,)),
        keras.layers.Dense(10),          # raw logits, no softmax
    ])
    probe.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    probe.fit(train_feats[idx], y_train[idx], epochs=epochs, batch_size=256, verbose=0)
    _, acc = probe.evaluate(test_feats, y_test, verbose=0)
    return acc


def dense_baseline(train_images, y_train, test_images, y_test, img, channels,
                   label_fraction=1.0, epochs=5, batch=256, seed=0):
    """Arm 1 (no JEPA): plain Dense network on raw pixels, in the style of the
    original MNIST file: keep it simple; simpler networks generalize better.
    No patches, no embeddings, no encoders — just
    flatten -> Dense(128, relu) -> Dense(10)."""
    idx = label_indices(len(train_images), label_fraction, seed)
    model = keras.Sequential([
        keras.layers.Input((img * img * channels,)),
        keras.layers.Dense(128, activation="relu"),   # as in the original file
        keras.layers.Dense(10),                       # raw logits
    ])
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.fit(train_images.reshape(-1, img * img * channels)[idx], y_train[idx],
              epochs=epochs, batch_size=batch, verbose=0)
    _, acc = model.evaluate(test_images.reshape(-1, img * img * channels), y_test, verbose=0)
    return acc
