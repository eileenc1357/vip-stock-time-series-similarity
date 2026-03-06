# --- models.py ---
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _build_mlp_encoder(input_dim, latent_dim, hidden_dims, name="encoder"):
    inp = keras.Input(shape=(input_dim,), name="x")
    x = inp
    for h in hidden_dims:
        x = layers.Dense(h, activation="relu")(x)
    z = layers.Dense(latent_dim, name="z")(x)
    return keras.Model(inp, z, name=name)

def _build_mlp_decoder(output_dim, latent_dim, hidden_dims, name="decoder"):
    inp = keras.Input(shape=(latent_dim,), name="z")
    x = inp
    for h in hidden_dims:
        x = layers.Dense(h, activation="relu")(x)
    out = layers.Dense(output_dim, activation="linear", name="x_hat")(x)
    return keras.Model(inp, out, name=name)

class MAEModel:
    def __init__(self, input_dim, latent_dim=16, hidden_dims=(128, 64), learning_rate=1e-3):
        self.latent_dim = latent_dim
        self.encoder = _build_mlp_encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = _build_mlp_decoder(input_dim, latent_dim, hidden_dims[::-1])
        
        x_in = keras.Input(shape=(input_dim,), name="x_masked")
        z = self.encoder(x_in)
        x_hat = self.decoder(z)
        
        self.model = keras.Model(x_in, x_hat, name="mae")
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mae")

    def fit(self, X_train, mask_ratio=0.3, batch_size=256, epochs=30, validation_data=None, verbose=0):
        def make_masked(X):
            keep = (np.random.rand(*X.shape) > mask_ratio).astype(np.float32)
            return X * keep

        X_masked = make_masked(X_train)
        val_processed = None
        if validation_data is not None:
             # Validate on reconstruction of masked val data
             val_processed = (make_masked(validation_data), validation_data)

        self.model.fit(
            X_masked, X_train,
            validation_data=val_processed,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

    def embed(self, X):
        return self.encoder.predict(X, verbose=0)

class VAEModel:
    def __init__(self, input_dim, latent_dim=16, hidden_dims=(128, 64), learning_rate=1e-3, kl_weight=1e-3):
        self.kl_weight = kl_weight
        
        # Encoder
        x_in = keras.Input(shape=(input_dim,))
        x = x_in
        for h in hidden_dims:
            x = layers.Dense(h, activation="relu")(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        
        # Sampling
        class Sampling(layers.Layer):
            def call(self, inputs):
                zm, zlv = inputs
                eps = tf.random.normal(shape=tf.shape(zm))
                return zm + tf.exp(0.5 * zlv) * eps
        
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(x_in, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        z_in = keras.Input(shape=(latent_dim,))
        x = z_in
        for h in hidden_dims[::-1]:
             x = layers.Dense(h, activation="relu")(x)
        x_hat = layers.Dense(input_dim, activation="linear")(x)
        self.decoder = keras.Model(z_in, x_hat, name="decoder")

        # VAE Trainer
        class VAE(keras.Model):
            def __init__(self, encoder, decoder, kl_weight, **kwargs):
                super().__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.kl_weight = kl_weight

            def train_step(self, data):
                if isinstance(data, tuple): data = data[0]
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = self.encoder(data)
                    recon = self.decoder(z)
                    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(data - recon), axis=1))
                    kl_loss = -0.5 * tf.reduce_mean(
                        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                    )
                    total_loss = recon_loss + self.kl_weight * kl_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {"loss": total_loss, "recon": recon_loss, "kl": kl_loss}
        
        self.vae = VAE(self.encoder, self.decoder, kl_weight)
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate))

    def fit(self, X_train, batch_size=256, epochs=50, verbose=0):
        self.vae.fit(X_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def embed(self, X):
        # Return z_mean as the embedding
        z_mean, _, _ = self.encoder.predict(X, verbose=0)
        return z_mean
