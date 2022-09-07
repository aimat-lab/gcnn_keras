import tensorflow as tf
ks = tf.keras


class Adan(ks.optimizers.Optimizer):

    def __init__(self, learning_rate=1e-3, name="Adan", betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.0, no_prox=False, **kwargs):
        super(Adan, self).__init__(name=name, **kwargs)

        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("eps", eps)
        self._set_hyper("beta_1", betas[0])
        self._set_hyper("beta_2", betas[1])
        self._set_hyper("beta_3", betas[2])
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("no_prox", no_prox)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "exp_avg")
        for var in var_list:
            self.add_slot(var, "exp_avg_sq")
        for var in var_list:
            self.add_slot(var, "exp_avg_diff")
        for var in var_list:
            self.add_slot(var, "pre_grad")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Adan, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_3_t = tf.identity(self._get_hyper('beta_3', var_dtype))
        weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype))
        bias_correction1 = 1 - tf.pow(beta_1_t, local_step)
        bias_correction2 = 1 - tf.pow(beta_2_t, local_step)
        bias_correction3 = 1 - tf.pow(beta_2_t, local_step)
        eps = tf.convert_to_tensor(self._get_hyper('eps', var_dtype), var_dtype)
        update_dict = dict(
            eps=eps,
            weight_decay=weight_decay,
            beta_1_t=beta_1_t,
            beta_2_t=beta_2_t,
            beta_3_t=beta_3_t,
            bias_correction1=bias_correction1,
            bias_correction2=bias_correction2,
            bias_correction3=bias_correction3
        )
        apply_state[(var_device, var_dtype)].update(update_dict)

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        lr_t = coefficients["lr_t"]  # lr_t = self._decayed_lr(var_dtype)
        weight_decay = coefficients['weight_decay']
        beta1, beta2, beta3 = coefficients["beta_1_t"], coefficients["beta_2_t"], coefficients["beta_3_t"]
        bias_correction1 = coefficients["bias_correction1"]
        bias_correction2 = coefficients["bias_correction2"]
        bias_correction3 = coefficients["bias_correction3"]
        eps = coefficients['eps']

        exp_avg = self.get_slot(var, 'exp_avg')
        exp_avg_sq = self.get_slot(var, 'exp_avg_sq')
        exp_avg_diff = self.get_slot(var, 'exp_avg_diff')
        pre_grad = self.get_slot(var, 'pre_grad')

        diff = grad - pre_grad

        update = grad + beta2 * diff
        exp_avg.assign(beta1*exp_avg + grad*(1-beta1), use_locking=self._use_locking)
        exp_avg_diff.assign(exp_avg_diff*beta2 + diff*(1-beta2))
        exp_avg_sq.assign(exp_avg_sq*beta3 + update*update*(1 - beta3))

        denom = (tf.math.sqrt(exp_avg_sq) / tf.math.sqrt(bias_correction3))+eps
        update = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)/(denom)

        if self._get_hyper['no_prox']:
            var.assign(var*(1 - lr_t * weight_decay))
            var.assign_add(update*(-lr_t))
        else:
            var.assign_add(update*(-lr_t))
            var.assign(var / (1 + lr_t * weight_decay))

        pre_grad.assign(grad)
        return tf.group(*[update, exp_avg, exp_avg_diff, exp_avg_sq, pre_grad])