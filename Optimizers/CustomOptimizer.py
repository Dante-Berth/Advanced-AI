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

opt = tfa.optimizers.AdaBelief(
    lr=1e-3,
    total_steps=10000,
    warmup_proportion=0.1,
    min_lr=1e-5,
    rectify=True,
)
adabelief = tfa.optimizers.AdaBelief()
ranger = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)

initial_learning_rate = 0.001
maximal_learning_rate = 0.01
step_size = 2000
gamma = 0.9

# Define the ExponentialCyclicalLearningRate schedule
lr_schedule = tfa.optimizers.ExponentialCyclicalLearningRate(
    initial_learning_rate=initial_learning_rate,
    maximal_learning_rate=maximal_learning_rate,
    step_size=step_size,
    gamma=gamma
)

# Set the learning rate schedule for the AdaBelief optimizer
adabelief.learning_rate = lr_schedule

opt = tf.keras.optimizers.SGD(learning_rate)
opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
