import os
import pickle
import tensorflow as tf
from utils import create_model, get_logger
from model import Model
from loader import input_from_line
from train import FLAGS, load_config

from keras import backend as K





os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = load_config(FLAGS.config_file)
logger = get_logger(FLAGS.log_file)
# limit GPU memory
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with open(FLAGS.map_file, "rb") as f:
    tag_to_id, id_to_tag = pickle.load(f)
sess = tf.Session(config=tf_config)
model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)
#ner()


def ner(line):
    #line = input("input sentence, please:")
    result = model.evaluate_line(sess, input_from_line(line, FLAGS.max_seq_len, tag_to_id), id_to_tag)
    all_entity = result['entities']
    #print(all_entity)
    only_entity = [i['word'] for i in all_entity]
    #print(only_entity)
    K.clear_session()
    return only_entity
