import tensorflow as tf

class SGD():
    def __init__(self, model_dir, model, instr_dim, repre_dim, num_episode=1000, lr=1e-1):
        # parameters
        self.num_ep = num_episode
        self.model_dir = model_dir
        # init graph
        self.model = model
        # objective function
        self.obj = self.model.y_pred
        # operation to find closest representation
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.obj)

    def eval(self, instr):
        
        self.model.sess.run(tf.global_variables_initializer())
        # load model
        self.model.load_model(self.model_dir)
        for var in tf.global_variables():
            try:
                self.model.sess.run(var)
            except tf.errors.FailedPreconditionError:
                print(var.name)
        for ep in range(self.num_ep):
            _, loss, pred = self.model.sess.run([self.train_op, self.obj, self.model.repre], feed_dict={self.model.inst: [instr]})

        return pred

