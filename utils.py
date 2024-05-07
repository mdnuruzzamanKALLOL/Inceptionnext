import tensorflow as tf

class TensorFlowAMPScaler:
    def __init__(self):
        # Set up mixed precision
        self.policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(self.policy)
        self.optimizer = None
        self.loss_scale_optimizer = None
        self.grad_accum_steps = 1
        self.accumulation_step = 0
        self.accumulated_gradients = []

    def setup_optimizer(self, optimizer, grad_accum_steps=1):
        # Wrap optimizer with a LossScaleOptimizer
        self.optimizer = optimizer
        self.grad_accum_steps = grad_accum_steps
        self.loss_scale_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)

    def accumulate_gradients(self, loss):
        # Accumulate gradients, handling mixed precision
        scaled_loss = self.loss_scale_optimizer.get_scaled_loss(loss)
        scaled_gradients = tf.gradients(scaled_loss, self.optimizer.trainable_variables)
        if self.accumulated_gradients == []:
            self.accumulated_gradients = [tf.zeros_like(g) for g in scaled_gradients]
        self.accumulated_gradients = [acc_g + g for acc_g, g in zip(self.accumulated_gradients, scaled_gradients)]
        self.accumulation_step += 1

    def apply_gradients(self, clip_grad=None, clip_mode='norm'):
        if self.accumulation_step == self.grad_accum_steps:
            unscaled_gradients = [self.loss_scale_optimizer.get_unscaled_gradients(g) for g in self.accumulated_gradients]
            if clip_grad is not None:
                if clip_mode == 'norm':
                    unscaled_gradients = [tf.clip_by_norm(g, clip_grad) for g in unscaled_gradients]
                elif clip_mode == 'global_norm':
                    unscaled_gradients, _ = tf.clip_by_global_norm(unscaled_gradients, clip_grad)
            self.optimizer.apply_gradients(zip(unscaled_gradients, self.optimizer.trainable_variables))
            self.loss_scale_optimizer.update()
            self.accumulation_step = 0
            self.accumulated_gradients = []

    def reset(self):
        # Reset internal state
        self.accumulated_gradients = []
        self.accumulation_step = 0

    def save_state(self):
        # Optionally save the state of the scaler
        return self.loss_scale_optimizer.get_weights()

    def load_state(self, weights):
        # Optionally load the state of the scaler
        self.loss_scale_optimizer.set_weights(weights)
