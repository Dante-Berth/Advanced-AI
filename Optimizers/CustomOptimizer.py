import tensorflow as tf
import tensorflow_addons as tfa

BATCH_SIZE = 64
EPOCHS = 10
INIT_LR = 1e-4
MAX_LR = 1e-2

steps_per_epoch = 10000 // BATCH_SIZE
clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
    maximal_learning_rate=MAX_LR,
    scale_fn=lambda x: 1/(2.**(x-1)),
    step_size=2 * steps_per_epoch
)
optimizer = tf.keras.optimizers.SGD(clr)

class CustomAdaBelief(tfa.optimizers.AdaBelief):
    def __init__(self,*args,**kwargs):
        super(tfa.optimizers.AdaBelief).__init__(*args,**kwargs)

initial_learning_rate = 0.001
maximal_learning_rate = 0.01
step_size = 2000
gamma = 0.9


opt = tfa.optimizers.AdaBelief(
    lr=1e-3,
    total_steps=10000,
    warmup_proportion=0.1,
    min_lr=1e-5,
    rectify=True,
)
lr_schedule = tfa.optimizers.ExponentialCyclicalLearningRate(
    initial_learning_rate=initial_learning_rate,
    maximal_learning_rate=maximal_learning_rate,
    step_size=step_size,
    gamma=gamma
)
opt.learning_rate = lr_schedule
ranger = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)




