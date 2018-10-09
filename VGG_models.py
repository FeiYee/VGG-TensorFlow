import tensorflow as tf
from tensorflow.contrib import slim
'''
VGG系列的复现模型
从VGG11到VGG19的前趋关系
为了能达到更好的效果，在里面添加了BN层算法。
采用Adam优化算法，学习率呈指数衰减。
训练模型时将Batch_norm的is_training设置为True
使用与检测模型时请将Batch_norm的is_train设置为False
'''
WEIGHT_DECAY = 0.001
def VGG11_inference(input_tensor, n_classes,is_train):
    with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None):
        with tf.variable_scope('Block_1'):
            #卷积层，L2正则函数
            data = slim.conv2d(input_tensor,64,3,scope='conv1')
            #批量标准归一化
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_1')
        with tf.variable_scope('Block_2'):
            data = slim.conv2d(data,128,3,scope='conv2')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_2')
        with tf.variable_scope('Block_3'):
            data = slim.conv2d(data,256,3,scope='conv3')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,3,scope='conv4')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_3')
        with tf.variable_scope('Block_4'):
            data = slim.conv2d(data,512,3,scope='conv5')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv6')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_4')
        with tf.variable_scope('Block_5'):
            data = slim.conv2d(data,512,3,scope='conv7')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv8')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_5')
        data = VGG_FC(data,is_train,n_classes)
    return data

def VGG13_inference(input_tensor, n_classes,is_train):
    with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None):
        with tf.variable_scope('Block_1'):
            data = slim.conv2d(input_tensor,64,3,scope='conv1')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,64,3,scope='conv2')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_1')
        with tf.variable_scope('Block_2'):
            data = slim.conv2d(data,128,3,scope='conv3')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,128,3,scope='conv4')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_2')
        with tf.variable_scope('Block_3'):
            data = slim.conv2d(data,256,3,scope='conv5')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,3,scope='conv6')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_3')
        with tf.variable_scope('Block_4'):
            data = slim.conv2d(data,512,3,scope='conv7')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv8')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_4')
        with tf.variable_scope('Block_5'):
            data = slim.conv2d(data,512,3,scope='conv9')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv10')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_5')
        data = VGG_FC(data,is_train,n_classes)
    return data

def VGG16A_inference(input_tensor, n_classes,is_train):
    with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None):
        with tf.variable_scope('Block_1'):
            data = slim.conv2d(input_tensor,64,3,scope='conv1')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,64,3,scope='conv2')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_1')
        with tf.variable_scope('Block_2'):
            data = slim.conv2d(data,128,3,scope='conv3')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,128,3,scope='conv4')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_2')
        with tf.variable_scope('Block_3'):
            data = slim.conv2d(data,256,3,scope='conv5')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,3,scope='conv6')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,1,scope='conv7')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_3')
        with tf.variable_scope('Block_4'):
            data = slim.conv2d(data,512,3,scope='conv8')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv9')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,1,scope='conv10')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_4')
        with tf.variable_scope('Block_5'):
            data = slim.conv2d(data,512,3,scope='conv11')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv12')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,1,scope='conv13')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_5')
        data = VGG_FC(data,is_train,n_classes)
    return data

def VGG16B_inference(input_tensor, n_classes,is_train):
    with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None):
        with tf.variable_scope('Block_1'):
            data = slim.conv2d(input_tensor,64,3,scope='conv1')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,64,3,scope='conv2')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_1')
        with tf.variable_scope('Block_2'):
            data = slim.conv2d(data,128,3,scope='conv3')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,128,3,scope='conv4')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_2')
        with tf.variable_scope('Block_3'):
            data = slim.conv2d(data,256,3,scope='conv5')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,3,scope='conv6')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,3,scope='conv7')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_3')
        with tf.variable_scope('Block_4'):
            data = slim.conv2d(data,512,3,scope='conv8')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv9')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv10')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_4')
        with tf.variable_scope('Block_5'):
            data = slim.conv2d(data,512,3,scope='conv11')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv12')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv13')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_5')
        data = VGG_FC(data,is_train,n_classes)
    return data

def VGG19_inference(input_tensor, n_classes,is_train):
    with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None):
        with tf.variable_scope('Block_1'):
            data = slim.conv2d(input_tensor,64,3,scope='conv1')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,64,3,scope='conv2')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_1')
        with tf.variable_scope('Block_2'):
            data = slim.conv2d(data,128,3,scope='conv3')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,128,3,scope='conv4')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_2')
        with tf.variable_scope('Block_3'):
            data = slim.conv2d(data,256,3,scope='conv5')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,3,scope='conv6')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,3,scope='conv7')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,256,3,scope='conv8')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_3')
        with tf.variable_scope('Block_4'):
            data = slim.conv2d(data,512,3,scope='conv9')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv10')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv11')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv12')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_4')
        with tf.variable_scope('Block_5'):
            data = slim.conv2d(data,512,3,scope='conv13')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv14')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv15')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
            data = slim.conv2d(data,512,3,scope='conv16')
            data = tf.layers.batch_normalization(data, training=is_train)
            data = tf.nn.relu(data)
        data = slim.max_pool2d(data,2,scope='pool_5')
        data = VGG_FC(data,is_train,n_classes)
    return data


def VGG_FC(input_tensor,is_training,n_classes):
    #全连接层
    with tf.variable_scope('Block_FC'):
        toShape = [-1,input_tensor.get_shape().as_list()[1]*input_tensor.get_shape().as_list()[2]*input_tensor.get_shape().as_list()[3]]
        data = tf.reshape(input_tensor,toShape)
        data = slim.fully_connected(data,2048,activation_fn=None,weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),scope='FC1')
        data = tf.layers.batch_normalization(data, training=is_training)
        data = tf.nn.relu(data)
        data = tf.layers.dropout(data,training=is_training)
        data = slim.fully_connected(data,2048,activation_fn=None,weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),scope='FC2')
        data = tf.layers.batch_normalization(data, training=is_training)
        data = tf.nn.relu(data)
        data = tf.layers.dropout(data,training=is_training)
        data = slim.fully_connected(data,n_classes,activation_fn=None,scope='FC3')
    return data