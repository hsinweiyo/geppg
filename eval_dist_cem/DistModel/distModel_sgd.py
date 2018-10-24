import numpy as np
import tensorflow as tf
import os

# Model
# ========================================
class DistModel_SGD(object):
    def __init__(self, name, inst_dim, skill_dim, n_units=32, learning_rate=0.003):
        
        # hyper-paramter	
        self.n_units = n_units
        self.lr = learning_rate
        
        # inputs placeholder
        self.inst_dim = inst_dim
        self.skill_dim = skill_dim

        # build model
        self._build()

        # make session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
#        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        # writer / saver
        variables = [ v for v in tf.all_variables() if v.name != "input/repre:0" ]
        self.saver = tf.train.Saver(variables)

    def _build(self):
        with tf.variable_scope("input"):
            self.inst = tf.placeholder(tf.float32, shape=[None, self.inst_dim], name='inst')
            self.repre = tf.Variable(tf.zeros([1, self.skill_dim], dtype=tf.float32), name='repre')

        with tf.variable_scope("hidden"):
            h1 = tf.concat([self.inst, self.repre], axis=-1)
            h1 = tf.layers.dense(h1, self.n_units, activation=tf.nn.relu)

        with tf.variable_scope("output"):
            self.y_pred = tf.layers.dense(h1, 1)
            self.y_pred = tf.reshape(self.y_pred, shape=[-1])

    # Save / Load Model
    def save_model(self, save_dir, step):
        self.saver.save(self.sess, save_dir+'model', global_step=step)

    def load_model(self, load_dir):
        ckpt = tf.train.latest_checkpoint(load_dir)
        if ckpt is not None:
            last_step = int(ckpt.split('-')[1])
            self.saver.restore(self.sess, ckpt)
        else:
            last_step = 0
        return last_step
