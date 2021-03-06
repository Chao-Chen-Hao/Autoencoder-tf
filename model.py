import tensorflow.compat.v1 as tf

############################
#         Parameters       #
############################
W = 160
H = 210
IN = 3
CONV_1 = 32
CONV_2 = 64
CONV_3 = 128

############################
#           Model          #
############################
class Model:
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.95):

        self.input_img = tf.placeholder(tf.float32, [None, H, W, 3])

        self.loss, self.latent, self.output_img = self.forward(is_train=True, reuse=None)
        self.val_loss, self.val_latent, self.val_output_img = self.forward(is_train=False, reuse=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        
        self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_op):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=3, pad_step_number=True)

    def forward(self, is_train, reuse):
        with tf.variable_scope("encoder", reuse=reuse):
            # Encoder
            conv1_e = tf.layers.conv2d(self.input_img, CONV_1, 5, (3, 2), padding='same', name='conv1_e', reuse=reuse)
            bn1_e = tf.layers.batch_normalization(conv1_e, training=is_train)
            relu1_e = tf.nn.relu(bn1_e)
            
            conv2_e = tf.layers.conv2d(relu1_e, CONV_2, 3, (2, 2), padding='same', name='conv2_e', reuse=reuse)
            bn2_e = tf.layers.batch_normalization(conv2_e, training=is_train)
            relu2_e = tf.nn.relu(bn2_e)

        with tf.variable_scope("decoder", reuse=reuse):
            # Decoder
            conv1_d = tf.layers.conv2d_transpose(relu2_e, CONV_1, 3, (2, 2), padding='same', name='conv1_d', reuse=reuse)
            bn1_d = tf.layers.batch_normalization(conv1_d, training=is_train)
            relu1_d = tf.nn.relu(bn1_d)
            
            conv2_d = tf.layers.conv2d_transpose(relu1_d, IN, 5, (3, 2), padding='same', name='conv2_d', reuse=reuse)
            bn2_d = tf.layers.batch_normalization(conv2_d, training=is_train)
            relu2_d = tf.nn.relu(bn2_d)
            
            out_t = tf.clip_by_value(relu2_d, 0.0, 1.0)

        loss = tf.nn.l2_loss(out_t - self.input_img)/ W / H
        return loss, relu2_e, out_t