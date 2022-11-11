import tensorflow as tf

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='Adan')
class Adan(ks.optimizers.Optimizer):
    r"""Optimizer `Adan <https://arxiv.org/abs/2208.06677>`_: Adaptive Nesterov Momentum Algorithm for
    Faster Optimizing Deep Models.

    'Adan develops a Nesterov momentum estimation method to estimate stable and accurate first
    and second momentums of gradient in adaptive gradient algorithms for acceleration'.

    Algorithm of Adan:

    Input: Initialization :math:`θ_0`, step size :math:`\eta`, average parameter
    :math:`(β_1, β_2, β_3) \in [0, 1]^3`, stable parameter :math:`\epsilon > 0`,
    weight decays :math:`\lambda_k > 0`, restart condition.

    Output: some average of :math:`\{\theta_k\}^K_{k=1}`.

    (set :math:`m_0 = g_0` and :math:`v_1 = g_1 - g_0`)


    while :math:`k < K` do:

    .. math::

        m_k &= (1 − \beta_1)m_{k−1} + \beta_1 g_k  \\\\
        v_k &= (1 − \beta_2)v_{k−1} + \beta_2(g_k − g_{k−1}) \\\\
        n_k = (1 − \beta_3)n_{k−1} + \beta_3[g_k + (1 − \beta_2)(g_k − g_{k−1})]^2 \\\\
        \eta_k = \eta / \sqrt{n_k + \epsilon}  \\\\
        θ_{k+1} = (1 + \lambda_k \eta)^{-1} [\theta_k − \eta_k \dot (m_k + (1 − \beta_2) v_k)] \\\\
        \text{if restart condition holds:} \\\\
        \text{   get stochastic gradient estimator } g_0 \text{at } \theta_{k+1} \\\\
        \text{   set } m_0 = g_0, \; v_0 = 0, \; n_0 = g_0^2, \; k = 1 \\\\
        \text{   update } \theta_k
    """

    # Reference pytorch implementations:
    # https://github.com/frgfm/Holocron/blob/main/holocron/optim/functional.py
    # https://github.com/sail-sg/Adan/blob/main/adan.py
    # https://github.com/lucidrains/Adan-pytorch

    def __init__(self, learning_rate: float = 1e-3, name: str = "Adan",
                 beta_1: float = 0.98, beta_2: float = 0.92, beta_3: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0, no_prox: bool = False, amsgrad: bool = False, **kwargs):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate. Default is 1e-3.
            name (str): Name of the optimizer. Defaults to 'Adan'.
            beta_1 (float): Beta 1 parameter. Default is 0.98.
            beta_2 (float): Beta 2 parameter. Default is 0.92.
            beta_3 (float): Beta 3 parameter. Default is 0.99.
            eps (float): Numerical epsilon for denominators. Default is 1e-8.
            weight_decay (float): Decoupled weight decay. Default is 0.0.
            no_prox (bool): How to perform the decoupled weight decay. Default is False.
            amsgrad (bool): Use the maximum of all 2nd moment running averages. Default is False.
        """
        super(Adan, self).__init__(name=name, **kwargs)

        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta_1 parameter: {}".format(beta_1))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta_2 parameter: {}".format(beta_2))
        if not 0.0 <= beta_3 < 1.0:
            raise ValueError("Invalid beta_3 parameter: {}".format(beta_3))

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("eps", eps)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("beta_3", beta_3)
        self._set_hyper("weight_decay", weight_decay)
        self._no_prox = no_prox
        self._use_amsgrad = amsgrad

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "exp_avg")
        for var in var_list:
            self.add_slot(var, "exp_avg_sq")
        for var in var_list:
            self.add_slot(var, "exp_avg_diff")
        for var in var_list:
            self.add_slot(var, "pre_grad")
        if self._use_amsgrad:
            for var in var_list:
                self.add_slot(var, "max_exp_avg_sq")

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

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # Getting coefficients set by `_prepare_local`
        lr_t = coefficients["lr_t"]  # lr_t = self._decayed_lr(var_dtype) done by super
        weight_decay = coefficients["weight_decay"]
        beta1, beta2, beta3 = coefficients["beta_1_t"], coefficients["beta_2_t"], coefficients["beta_3_t"]
        bias_correction1 = coefficients["bias_correction1"]
        bias_correction2 = coefficients["bias_correction2"]
        bias_correction3 = coefficients["bias_correction3"]
        eps = coefficients["eps"]

        exp_avg = self.get_slot(var, 'exp_avg')
        exp_avg_sq = self.get_slot(var, 'exp_avg_sq')
        exp_avg_diff = self.get_slot(var, 'exp_avg_diff')
        pre_grad = self.get_slot(var, 'pre_grad')

        diff = grad - pre_grad
        exp_avg.assign(beta1 * exp_avg + grad * (1 - beta1), use_locking=self._use_locking)
        exp_avg_diff.assign(exp_avg_diff * beta2 + diff * (1 - beta2), use_locking=self._use_locking)
        update = grad + beta2 * diff
        exp_avg_sq.assign(exp_avg_sq * beta3 + update * update * (1 - beta3), use_locking=self._use_locking)

        if self._use_amsgrad:
            max_exp_avg_sq = self.get_slot(var, 'max_exp_avg_sq')
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, exp_avg_sq), use_locking=self._use_locking)
            # Use the max. for normalizing running avg. of gradient
            denom = (tf.math.sqrt(max_exp_avg_sq) / tf.math.sqrt(bias_correction3)) + eps
        else:
            denom = (tf.math.sqrt(exp_avg_sq) / tf.math.sqrt(bias_correction3)) + eps

        update = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2) / (denom)

        if self._no_prox:
            var.assign(var * (1 - lr_t * weight_decay), use_locking=self._use_locking)
            var.assign_add(update * (-lr_t), use_locking=self._use_locking)
        else:
            var.assign_add(update * (-lr_t), use_locking=self._use_locking)
            var.assign(var / (1 + lr_t * weight_decay), use_locking=self._use_locking)

        pre_grad.assign(grad, use_locking=self._use_locking)
        return tf.group(*[update, exp_avg, exp_avg_diff, exp_avg_sq, pre_grad])

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        """Get config dictionary."""
        config = super(Adan, self).get_config()
        config.update(
            {
                "no_prox": bool(self._no_prox),
                "amsgrad": bool(self._use_amsgrad),
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "eps": self._serialize_hyperparameter("eps"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "beta_3": self._serialize_hyperparameter("beta_3"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
            }
        )
        return config
