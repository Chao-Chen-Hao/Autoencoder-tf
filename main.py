  
import tensorflow.compat.v1 as tf

import cv2
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import sys
import os
import time
import random
random.seed(5487)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model import Model

############################
#           Flags          #
############################
# --------- Mode --------- #
tf.app.flags.DEFINE_boolean("is_train", False, "training mode.")
tf.app.flags.DEFINE_boolean("restore", False, "training mode.")
# ------- Training ------- #
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate.")
tf.app.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
tf.app.flags.DEFINE_integer("batch", 2000, "Number of images per batch.")
# ------ Save Model ------ #
tf.app.flags.DEFINE_string("ckpt_dir", "./checkpoint", "check point directory")
tf.app.flags.DEFINE_string("train_dir", "../Warehouse/atari_data/training", "Dataset directory")
tf.app.flags.DEFINE_string("test_dir", "../Warehouse/atari_data/testing", "Testing directory")
tf.app.flags.DEFINE_integer("inference_version", -1, "The version for inferencing.")
tf.app.flags.DEFINE_integer("val_num", 16000, "number of validation images.")

FLAGS = tf.app.flags.FLAGS

############################
#         Functions        #
############################
W = 160
H = 210

def get_data(is_train=FLAGS.is_train):
    output_img = [] # (4-d tensor) shape : size, w, h, 3
    if is_train:
        directory = FLAGS.train_dir
        st, ed = 1, 50000
    else:
        directory = FLAGS.test_dir
        st, ed = 404306, 405305

    for i in range(st, ed+1):
        if i % 1000 == 0:
            print("loading {:d}-th img".format(i))
        path = directory + '/{:06d}.png'.format(i)
        img = np.reshape(cv2.imread(path), (H, W, 3))
        output_img.append(img/255.0)

    return output_img

def train(model, sess, data):
    print("training...")
    batch_loss = 0
    st, ed, times = 0, 0, 0
    max_len = len(data)
    r = 0

    while st < max_len:
        ed = st + FLAGS.batch if st + FLAGS.batch < max_len else max_len
        feed = {model.input_img: data[st:ed]}
        loss, latent, output_img, _ = sess.run([model.loss, model.latent, model.output_img, model.train_op], feed_dict=feed)
        batch_loss += loss
        times += 1
        st = ed
        print("batch_loss: {:f}".format(loss))
        if r == 0:
            r = 1
            cv2.imwrite('./img/test.png', np.reshape(output_img[0]*255, (H, W, 3)))

    batch_loss /= times
    return batch_loss

def validation(model, sess, data):
    print("validating...")
    batch_loss = 0
    st, ed, times = 0, 0, 0
    max_len = len(data)

    while st < max_len:
        ed = st + FLAGS.batch if st + FLAGS.batch < max_len else max_len
        feed = {model.input_img: data[st:ed]}
        loss, latent, output_img = sess.run([model.val_loss, model.val_latent, model.val_output_img], feed_dict=feed)
        batch_loss += loss
        times += 1
        st = ed
        print("batch_loss: {:f}".format(loss))
    

    batch_loss /= times
    return batch_loss

def test(model, sess, data):
    length = len(data)
    total_loss = 0
    total_time = 0

    for i in range(length):
        if (i%1000 == 999):
            print("processing the "+str(i+1)+"-th image.")
            
        start_time = time.time()
        feed = {model.input_img: [data[i]]}
        loss, latent, output_img = sess.run([model.loss_val, model.latent_val, model.output_img_val], feed_dict=feed)
        total_time += (time.time() - start_time)
        total_loss += loss
                
    print(total_time/length)
    print(total_loss/length)

############################
#           Main           #
############################

# ------- variables ------ #
best_loss = 10000000
best_epoch = 0

# -------- config -------- #
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

# ------- run sess ------- #
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        # building computational graph (model)
        if FLAGS.restore:
            # inference version cannot be -1.
            model = Model(learning_rate=FLAGS.learning_rate)
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
            model.saver.restore(sess, model_path)
        else:    
            model = Model(learning_rate=FLAGS.learning_rate)
            tf.global_variables_initializer().run()
        
        # train list (for documenting loss and acc.)
        #doc = defaultdict(list)
        
        # get training data
        data = get_data()

        # get validation data
        data_val = data[:FLAGS.val_num]
        data = data[FLAGS.val_num:]

        for epoch in range(FLAGS.epoch):
            print("trainig: {:d}-th epoch".format(epoch))
            start_time = time.time()
            train_loss = train(model, sess, data)
            val_loss = validation(model, sess, data_val)

            if val_loss <= best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                model.saver.save(sess, '%s/checkpoint' % FLAGS.ckpt_dir, global_step=model.global_step)

            epoch_time = time.time() - start_time
            print("validation loss: {:f}".format(val_loss))
            print("---------")
    
    else: # testing
        model = Model()
        if FLAGS.inference_version == -1:
            print("Please set the inference version!")
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.ckpt_dir, FLAGS.inference_version)

        model.saver.restore(sess, model_path)
        data = get_data()
        test(model, sess, data)
