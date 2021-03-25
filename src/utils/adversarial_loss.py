import tensorflow as tf
import tensorflow.keras.backend as K


class AdversarialLoss:

    def __init__(
            self,
            generator_loss_type=None,
            generator_loss_coef=1,
            discriminator_loss_type=None,
            discriminator_loss_coef=1,
            gradient_penalty_coef=None
    ):
        self._generator_loss_type = generator_loss_type
        self._generator_loss_coef = generator_loss_coef
        self._discriminator_loss_type = discriminator_loss_type
        self._discriminator_loss_coef = discriminator_loss_coef
        self._gradient_penalty_coef = gradient_penalty_coef
        self._set_generator_loss()
        self._set_discriminator_loss()
        self._set_gradient_penalty_loss()
        self._set_discriminator_metrics()

    def _set_generator_loss(self):
        if self._generator_loss_type is None or self._generator_loss_coef == 0:
            self._generator_loss = lambda x: tf.zeros(1, dtype=tf.float32)
        elif self._generator_loss_type == "bce_loss":
            self._generator_loss = generator_bce_loss
        elif self._generator_loss_type == "mse_loss":
            self._generator_loss = generator_mse_loss
        elif self._generator_loss_type == "bce_with_logits_loss":
            self._generator_loss = generator_bce_with_logits_loss
        elif self._generator_loss_type == "wasserstein_loss":
            self._generator_loss = generator_wasserstein_loss

    def _set_discriminator_loss(self):
        if self._discriminator_loss_type is None or self._discriminator_loss_coef == 0:
            self._discriminator_loss = lambda x, y: tf.zeros(1, dtype=tf.float32)
        elif self._discriminator_loss_type == "bce_loss":
            self._discriminator_loss = discriminator_bce_loss
        elif self._discriminator_loss_type == "mse_loss":
            self._discriminator_loss = discriminator_mse_loss
        elif self._discriminator_loss_type == "bce_with_logits_loss":
            self._discriminator_loss = discriminator_bce_with_logits_loss
        elif self._discriminator_loss_type == "wasserstein_loss":
            self._discriminator_loss = discriminator_wasserstein_loss

    def _set_gradient_penalty_loss(self):
        if self._gradient_penalty_coef == 0:
            self._gp_loss = lambda x, y, z: tf.zeros(1, dtype=tf.float32)
        else:
            self._gp_loss = discriminator_gradient_penalty

    def _set_discriminator_metrics(self):
        if self._discriminator_loss_type in ["bce_loss", "mse_loss"]:
            self._discriminator_metrics = self.discriminator_bce_metrics
        elif self._discriminator_loss_type in ["bce_with_logits_loss"]:
            self._discriminator_metrics = self.discriminator_bce_with_logits_metrics
        else:
            self._discriminator_metrics = lambda x, y: tf.zeros(1, dtype=tf.float32)

    def generator_loss(self, fake_discriminator_score):
        return self._generator_loss_coef * tf.reduce_mean(
            self._generator_loss(fake_discriminator_score))

    def discriminator_loss(self, real_discriminator_score, fake_discriminator_score):
        return self._discriminator_loss_coef * tf.reduce_mean(
            self._discriminator_loss(real_discriminator_score, fake_discriminator_score))

    def gradient_penalty_loss(self, discriminator, real_input, fake_input, label=None):
        if self._gradient_penalty_coef == 0:
            return tf.reduce_mean(tf.zeros(1, dtype=tf.float32))
        return self._gradient_penalty_coef * tf.reduce_mean(
            self._gp_loss(discriminator, real_input, fake_input, label=label))

    def discriminator_metrics(self, real_discriminator_score, fake_discriminator_score):
        return self._discriminator_metrics(real_discriminator_score, fake_discriminator_score)

    @staticmethod
    def discriminator_bce_metrics(real_discriminator_score, fake_discriminator_score):
        score_real = tf.reduce_mean(tf.round(real_discriminator_score))
        score_fake = tf.reduce_mean(tf.round(1 - fake_discriminator_score))
        score = 0.5 * (score_real + score_fake)
        return score, score_real, score_fake

    @staticmethod
    def discriminator_bce_with_logits_metrics(real_discriminator_score, fake_discriminator_score):
        score_real = tf.reduce_mean(tf.round(tf.sigmoid(real_discriminator_score)))
        score_fake = tf.reduce_mean(tf.round(1 - tf.sigmoid(fake_discriminator_score)))
        score = 0.5 * (score_real + score_fake)
        return score, score_real, score_fake


def discriminator_bce_loss(real_discriminator_score, fake_discriminator_score):
    y_real_true = tf.ones_like(real_discriminator_score)
    y_fake_true = tf.zeros_like(fake_discriminator_score)
    return tf.keras.losses.BinaryCrossentropy()(y_real_true, real_discriminator_score) + \
           tf.keras.losses.BinaryCrossentropy()(y_fake_true, fake_discriminator_score)


def generator_bce_loss(fake_discriminator_score):
    y_fake_true = tf.ones(fake_discriminator_score.shape)
    return tf.keras.losses.BinaryCrossentropy()(y_fake_true, fake_discriminator_score)


def discriminator_mse_loss(real_discriminator_score, fake_discriminator_score):
    y_real_true = tf.ones_like(real_discriminator_score)
    y_fake_true = tf.zeros_like(fake_discriminator_score)
    return tf.reduce_mean(
        tf.keras.losses.MeanSquaredError()(y_real_true, real_discriminator_score) +
        tf.keras.losses.MeanSquaredError()(y_fake_true, fake_discriminator_score))


def generator_mse_loss(fake_discriminator_score):
    y_fake_true = tf.ones_like(fake_discriminator_score)
    return tf.keras.losses.MeanSquaredError()(y_fake_true, fake_discriminator_score)


def discriminator_bce_with_logits_loss(real_discriminator_score, fake_discriminator_score):
    y_real_true = tf.ones_like(real_discriminator_score)
    y_fake_true = tf.zeros_like(fake_discriminator_score)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_real_true, logits=real_discriminator_score)) + \
           tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
               labels=y_fake_true, logits=fake_discriminator_score))


def generator_bce_with_logits_loss(fake_discriminator_score):
    y_fake_true = tf.ones_like(fake_discriminator_score)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_fake_true, logits=fake_discriminator_score))


def discriminator_wasserstein_loss(real_discriminator_score, fake_discriminator_score):
    real_loss = -tf.reduce_mean(real_discriminator_score)
    fake_loss = tf.reduce_mean(fake_discriminator_score)
    return real_loss + fake_loss


def generator_wasserstein_loss(fake_discriminator_score):
    return - tf.reduce_mean(fake_discriminator_score)


def discriminator_gradient_penalty(
        discriminator, real_input, fake_input, label=None):
    eps = tf.random.uniform(shape=[real_input.shape[0], 1, 1, 1])
    differences = fake_input - real_input
    interpolated_discriminator_input = real_input + (eps * differences)
    with tf.GradientTape() as t:
        t.watch(interpolated_discriminator_input)
        if label is not None:
            interpolated_discriminator_score = discriminator(
                [interpolated_discriminator_input, label])
        else:
            interpolated_discriminator_score = discriminator(interpolated_discriminator_input)
        gradients = t.gradient(
            interpolated_discriminator_score, interpolated_discriminator_input)
        gradient_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(gradient_l2_norm - 1.))
        return K.mean(gradient_penalty)
