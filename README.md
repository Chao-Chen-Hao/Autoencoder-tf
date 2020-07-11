Autoencoder
=======

#### Quick start
Run with the default arguments:

```
python main.py
```

change the model settings in model.py:

```

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
```

Requirements
------------
python 3.x, tensorflow 1.x

- OpenCV Python (https://pypi.python.org/pypi/opencv-python)
- matplotlibcv (https://pypi.org/project/matplotlib/)
- tensorflow (https://www.tensorflow.org/)
