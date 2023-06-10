import tensorflow as tf
import tensorflow_addons as tfa
class AdaBelief_optimizer:
    @staticmethod
    def get_name():
        return "adabelief"

    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_learning_rate": [1,100],
            "hyperparameter_warmup_proportion_percent": [10, 30, 2],
            "hyperparameter_min_lr": [1,100],
            "hyperparameter_initial_learning_rate": [1,1000],
            "hyperparameter_maximal_learning_rate": [1,90],
            "hyperparameter_step_size": ["1000", "5000", "10000", "50000", "100000", "500000", "1000000", "5000000", "10000000"],
            "hyperparameter_gamma_percent": [80, 99, 1],
            "hyperparameter_sync_period": [2, 10, 1],
            "hyperparameter_slow_step_size": [1, 10, 1]

        }

    @staticmethod
    def init(learning_rate, warmup, min_lr, initial_learning_rate, maximal_learning_rate, step_size,
                 gamma_percent, sync_period, slow_step_size, batch_size=64, size_dataset=1000):


        opt = tfa.optimizers.AdaBelief(
            learning_rate=learning_rate//1*1e-4,
            total_steps=int(batch_size * size_dataset),
            warmup_proportion=warmup // 10 * 0.1,
            min_lr=min_lr//1*1e-6,
            rectify=True,
        )
        lr_schedule = tfa.optimizers.ExponentialCyclicalLearningRate(
            initial_learning_rate=initial_learning_rate//1*1e-4,
            maximal_learning_rate=maximal_learning_rate//1*1e-2,
            step_size=int(step_size),
            gamma=gamma_percent // 10 * 0.1
        )
        opt.learning_rate = lr_schedule
        return tfa.optimizers.Lookahead(opt, sync_period=sync_period,
                                                  slow_step_size=slow_step_size // 10 * 0.1)
