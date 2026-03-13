import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import FastICA


# ===============================
# MAE (Masked Autoencoder)
# ===============================

class MAEModel:

    def __init__(self, input_dim, latent_dim=16):

        x_in = keras.Input(shape=(input_dim,))

        x = layers.Dense(128, activation="relu")(x_in)
        x = layers.Dense(64, activation="relu")(x)

        z = layers.Dense(latent_dim)(x)

        self.encoder = keras.Model(x_in, z)

        z_in = keras.Input(shape=(latent_dim,))

        x = layers.Dense(64, activation="relu")(z_in)
        x = layers.Dense(128, activation="relu")(x)

        out = layers.Dense(input_dim)(x)

        self.decoder = keras.Model(z_in, out)

        x_masked = keras.Input(shape=(input_dim,))
        z = self.encoder(x_masked)
        x_hat = self.decoder(z)

        self.model = keras.Model(x_masked, x_hat)

        self.model.compile(
            optimizer="adam",
            loss="mae"
        )


    def fit(self, X, epochs=30, batch_size=256, mask_ratio=0.3):

        mask = (np.random.rand(*X.shape) > mask_ratio)

        X_masked = X * mask

        self.model.fit(
            X_masked,
            X,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )


    def embed(self, X):

        return self.encoder.predict(X, verbose=0)



# ===============================
# VAE
# ===============================

class Sampling(layers.Layer):

    def call(self, inputs):

        z_mean, z_log_var = inputs

        eps = tf.random.normal(shape=tf.shape(z_mean))

        return z_mean + tf.exp(0.5 * z_log_var) * eps


class VAEModel:

    def __init__(self, input_dim, latent_dim=16):

        x_in = keras.Input(shape=(input_dim,))

        x = layers.Dense(128, activation="relu")(x_in)
        x = layers.Dense(64, activation="relu")(x)

        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)

        z = Sampling()([z_mean, z_log_var])

        self.encoder = keras.Model(x_in, [z_mean, z_log_var, z])

        z_in = keras.Input(shape=(latent_dim,))

        x = layers.Dense(64, activation="relu")(z_in)
        x = layers.Dense(128, activation="relu")(x)

        out = layers.Dense(input_dim)(x)

        self.decoder = keras.Model(z_in, out)

        class VAETrainer(keras.Model):

            def __init__(self, encoder, decoder):

                super().__init__()

                self.encoder = encoder
                self.decoder = decoder

            def train_step(self, data):

                if isinstance(data, tuple):
                    data = data[0]

                with tf.GradientTape() as tape:

                    z_mean, z_log_var, z = self.encoder(data)

                    recon = self.decoder(z)

                    recon_loss = tf.reduce_mean(
                        tf.reduce_sum(tf.abs(data - recon), axis=1)
                    )

                    kl = -0.5 * tf.reduce_mean(
                        tf.reduce_sum(
                            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                            axis=1
                        )
                    )

                    loss = recon_loss + 0.001 * kl

                grads = tape.gradient(loss, self.trainable_weights)

                self.optimizer.apply_gradients(
                    zip(grads, self.trainable_weights)
                )

                return {"loss": loss}

        self.vae = VAETrainer(self.encoder, self.decoder)

        self.vae.compile(optimizer="adam")


    def fit(self, X, epochs=40, batch_size=256):

        self.vae.fit(
            X,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )


    def embed(self, X):

        z_mean, _, _ = self.encoder.predict(X, verbose=0)

        return z_mean