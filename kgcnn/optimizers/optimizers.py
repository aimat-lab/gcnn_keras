import keras as ks
from keras import ops


@ks.saving.register_keras_serializable(package='kgcnn', name='Adan')
class Adan(ks.optimizers.Optimizer):
    r"""Optimizer `Adan <https://arxiv.org/abs/2208.06677>`__ : Adaptive Nesterov Momentum Algorithm for
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

    def __init__(self,
                 learning_rate: float = 1e-3,
                 name: str = "Adan",
                 beta_1: float = 0.98,
                 beta_2: float = 0.92,
                 beta_3: float = 0.99,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 **kwargs):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate. Default is 1e-3.
            name (str): Name of the optimizer. Defaults to 'Adan'.
            beta_1 (float): Beta 1 parameter. Default is 0.98.
            beta_2 (float): Beta 2 parameter. Default is 0.92.
            beta_3 (float): Beta 3 parameter. Default is 0.99.
            eps (float): Numerical epsilon for denominators. Default is 1e-8.
            weight_decay (float): Decoupled weight decay. Default is 0.0.
            amsgrad (bool): Use the maximum of all 2nd moment running averages. Default is False.
        """
        super(Adan, self).__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            **kwargs)

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

        self._input_learning_rate = float(learning_rate)
        self._eps = float(eps)
        self._beta_1 = float(beta_1)
        self._beta_2 = float(beta_2)
        self._beta_3 = float(beta_3)
        self._input_weight_decay = weight_decay
        self._use_amsgrad = bool(amsgrad)

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._exp_avg = []
        self._exp_avg_sq = []
        self._exp_avg_diff = []
        self._pre_grad = []
        for var in var_list:
            self._exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self._exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sqs"
                )
            )
            self._exp_avg_diff.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_diff"
                )
            )
            self._pre_grad.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="pre_grad"
                )
            )

        if self._use_amsgrad:
            self._max_exp_avg_sq = []
            for var in var_list:
                self._max_exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="max_exp_avg_sq"
                    )
                )

    def update_step(self, grad, var, learning_rate):
        """Update step given gradient and the associated model variable."""
        var_dtype = var.dtype
        lr_t = ops.cast(learning_rate, var_dtype)  # lr_t = self._decayed_lr(var_dtype) done by super
        grad = ops.cast(grad, var_dtype)
        local_step = ops.cast(self.iterations + 1, var_dtype)

        beta1 = ops.cast(self._beta_1, var_dtype)
        beta2 = ops.cast(self._beta_2, var_dtype)
        beta3 = ops.cast(self._beta_3, var_dtype)
        bias_correction1 = 1 - ops.power(beta1, local_step)
        bias_correction2 = 1 - ops.power(beta2, local_step)
        bias_correction3 = 1 - ops.power(beta3, local_step)
        eps = ops.cast(self._eps, var_dtype)

        exp_avg = self._exp_avg[self._get_variable_index(var)]
        exp_avg_sq = self._exp_avg_sq[self._get_variable_index(var)]
        exp_avg_diff = self._exp_avg_diff[self._get_variable_index(var)]
        pre_grad = self._pre_grad[self._get_variable_index(var)]

        diff = grad - pre_grad

        self.assign(exp_avg, beta1 * exp_avg + grad * (1 - beta1))
        self.assign(exp_avg_diff, exp_avg_diff * beta2 + diff * (1 - beta2))
        update = grad + beta2 * diff
        self.assign(exp_avg_sq, exp_avg_sq * beta3 + update * update * (1 - beta3))

        if self._use_amsgrad:
            max_exp_avg_sq = self._max_exp_avg_sq[self._get_variable_index(var)]
            # Maintains the maximum of all 2nd moment running avg. till now
            self.assign(max_exp_avg_sq, ops.maximum(max_exp_avg_sq, exp_avg_sq))
            # Use the max. for normalizing running avg. of gradient
            denominator = (ops.sqrt(max_exp_avg_sq) / ops.sqrt(bias_correction3)) + eps
        else:
            denominator = (ops.sqrt(exp_avg_sq) / ops.sqrt(bias_correction3)) + eps

        update = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2) / denominator

        self.assign_add(var, update * (-lr_t))

        self.assign(pre_grad, grad)

    def get_config(self):
        """Get config dictionary."""
        config = super(Adan, self).get_config()
        config.update(
            {
                "amsgrad": bool(self._use_amsgrad),
                "learning_rate": self._input_learning_rate,
                "eps": self._eps,
                "beta_1": self._beta_1,
                "beta_2": self._beta_2,
                "beta_3": self._beta_3,
                "weight_decay": self._input_weight_decay
            }
        )
        return config
