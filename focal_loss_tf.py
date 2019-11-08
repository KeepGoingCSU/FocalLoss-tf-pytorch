# -*- coding: utf-8 -*-
import tensorflow as tf


def focal_loss_softmax(labels, logits, class_num, alpha=None,gamma=2, size_average=True):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    if alpha is None:
        alpha_ = tf.Variable(tf.ones(class_num, 1))
    else:
        alpha_ = tf.Variable(alpha)
    labels = tf.reshape(labels,[-1])
    labels = tf.cast(labels, tf.int32)
    
    N = logits.shape[0]
    C = logits.shape[1]
    P = tf.nn.softmax(logits)

    ids = tf.reshape(labels,[-1, 1])
    class_mask = tf.one_hot(labels,class_num) # one_hot labels

    alpha_ = tf.gather(alpha_,tf.reshape(ids,[-1]))  #取出每个类别对应的权重,shape同ids
    probs = tf.math.reduce_sum(tf.math.multiply(P,class_mask),1, keepdims=True)
    log_p = tf.math.log(probs)
    
    print(alpha_,probs,gamma,log_p)
    batch_loss = -alpha_*(tf.math.pow((1-probs), gamma))*log_p 
    print(batch_loss)

    if size_average:
        loss = tf.math.reduce_mean(batch_loss)
    else:
        loss = tf.math.reduce_sum(batch_loss)
    print(loss)
    return loss

if __name__ == '__main__':
    labels = tf.reshape(tf.Variable([[1,2,0,1]]),[4,1])
    logits = tf.Variable([[3.1,2,1],[-1,2,0],[2,1,-3],[4,1,1]])
    print(labels,logits)
    class_num = 3
    alpha =[2,1,0.5]
    gamma =1
    size_average = True
    focal_loss_softmax(labels, logits, class_num, alpha,gamma, size_average)