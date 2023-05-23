import tensorflow as tf
import tensorflow_addons as tfa

class CustomAdaBelief(tfa.optimizers.AdaBelief):
    def __init__(self,*args,**kwargs):
        super(tfa.optimizers.AdaBelief).__init__(*args,**kwargs)

