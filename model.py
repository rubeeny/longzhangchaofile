# -*- coding:utf-8 -*-
import logging
import os
import numpy as np

import tensorflow as tf
from layers import multihead_attention, ff, get_shape_list,dense_connect
logger = logging.getLogger(__name__)

def reasoning_result(length, probs_):
    """
    根据业务规则，提取答案
    :param length: 实际长度
    :param probs_: 概率
    :return: 顺序
    """
    results=[]
    for i, p in enumerate(probs_):
        p=p[0:length[i],0:length[i]]#取出有效概率长度
        result=[]
        for step in range(length[i]):
            step_number=np.argmax(p[:,step])
            if step_number not in result:
                result.append(step_number)
            else:
                for  j in result:
                    p[j,step]=0
                step_number=np.argmax(p[:,step])
                result.append(step_number)
        results.append(result)
    return results

class Model(object):
    def __init__(self, config):
        self.global_step = tf.Variable(0, trainable=False)
        self.config=config
        self.best = tf.Variable(0.0, trainable=False)
        self.create_placeholders()
        self.enc_seq_length =tf.cast(tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(self.inputs), axis=-1)),axis=-1),  tf.int32)
        self.enc_seq_length_op =tf.add(self.enc_seq_length,0,name="seq_length")

        self.mask_eta = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.inputs), axis=-1)), -1)  #(N, T_q,1)
        self.mask = tf.tile(self.mask_eta, [1, 1, tf.shape(self.inputs)[1]]) ##(N, T_q,T_q)

        def encode(inputs, deep_keep_prob,isTrain):
            '''
            Returns
            memory: encoder outputs. (N, T1, d_model)
            '''
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                #we don't need embedding_lookup
                enc=dense_connect(name="inputs_project",input=inputs,out_dim=config.d_model,l2_scale=config.l2_scale)
                enc *= config.d_model ** 0.5  # scale
                # enc = tf.layers.dropout(enc, self.deep_keep_prob, training=training)
                ## Blocks
                for i in range(config.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # self-attentionc
                        enc = multihead_attention(queries=enc,
                                                  keys=enc,
                                                  values=enc,
                                                  num_heads=config.num_heads,
                                                  dropout_rate=deep_keep_prob,
                                                  l2_scale=config.l2_scale,
                                                  train=isTrain,
                                                  causality=False)
                        # feed forward
                        enc = ff(enc, num_units=[config.d_ff, config.d_model],l2_scale=config.l2_scale)
            memory = enc
            return memory

        def decode(final_hidden,mask):
            padding_num = -2 ** 32 + 1
            final_hidden_shape = get_shape_list(final_hidden, expected_rank=3)
            batch_size = final_hidden_shape[0]
            seq_length = final_hidden_shape[1]
            hidden_size = final_hidden_shape[2]

            output_weights = tf.get_variable(
                "output_weights", [seq_length, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [seq_length], initializer=tf.zeros_initializer())

            final_hidden_matrix = tf.reshape(final_hidden,
                                             [batch_size * seq_length, hidden_size])
            logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            logits = tf.reshape(logits, [batch_size, seq_length, seq_length])

            attention_raw = tf.matmul(final_hidden, tf.transpose(final_hidden, [0, 2, 1]))  # (N, T_q, T_k)

            mask_ =mask*tf.transpose(mask,[0,2,1])
            paddings = tf.ones_like(mask_)*padding_num

            attention_raw = tf.where(tf.equal(mask_, 0), paddings, attention_raw)
            attention_final = tf.nn.softmax(attention_raw)

            logists_final = tf.matmul(attention_final,logits)

            return logists_final,attention_final

        self.memory = encode(self.inputs,self.deep_keep_prob,self.isTrain)#batch_size * seq_len*hidden_size

        mask_ = self.mask * tf.transpose(self.mask, [0, 2, 1])

        with tf.name_scope("rank_layer"):
            padding_num = 2 ** -32
            self.logits,self.attention_final= decode(self.memory,self.mask)  # batch_size*seq_length*seq_lengt
            self.probs = tf.nn.softmax(self.logits)
            # self.pred = tf.argmax(self.probs, axis=2)  # batch_size*seq_length

            # mask = self.mask * tf.transpose(self.mask, [0, 2, 1])
            # paddings = tf.ones_like(self.mask) * padding_num
            # self.logits_ = tf.where(tf.equal(self.logits, 0), paddings, self.logits)

            # paddings = tf.ones_like(self.mask) * padding_num
            #
            # self.logits_ = tf.where(tf.equal(mask_, 0), paddings, self.probs_)
            # self.probs = tf.log(self.logits_)

        with tf.name_scope("eta_layer"):
            self.pre_eta_=tf.concat(self.memory,tf.transpose(self.probs,[0,2,1]),axis=-1)
            self.pre_eta_ =dense_connect('predict', self.pre_eta_
                                         , config.d_model, None,l2_scale=config.l2_scale) #batch_size * seq_len*1
            self.pre_eta_ = tf.nn.dropout(self.pre_eta_, self.deep_keep_prob)
            self.pre_eta_ =dense_connect('predict',  self.pre_eta_, 1, None,l2_scale=config.l2_scale) #batch_size * seq_len*1

            self.pre_eta_op = tf.squeeze(self.pre_eta_,axis=-1)

            # self.pre_eta_ =tf.layers.dense(self.memory,1,use_bias=None)
            self.pre_eta_op = tf.add(self.pre_eta_op, 0, name="predict_eta_op")

        # #计算损失
        # self.loss_eta = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.labels-self.pre_eta_),axis=-1),axis=-1))
        self.loss_eta=tf.losses.mean_squared_error(self.labels,self.pre_eta_*self.mask_eta)
        # #labels batch_size*seq_length
        self.one_hot_labels = tf.one_hot(self.rank_labels,depth=config.max_length,dtype=tf.float32)*mask_

        self.loss_rank_tmp=tf.reduce_sum(tf.reduce_sum(tf.log(self.probs)*self.one_hot_labels,axis=-1),axis=-1)
        self.loss_len = tf.cast(self.enc_seq_length,dtype=tf.float32)+1
        self.loss_rank= -tf.reduce_mean(self.loss_rank_tmp/self.loss_len)

        self.loss_l2=tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # self.loss = self.loss_eta + 0.5*self.loss_rank +self.loss_l2
        self.loss =0.6*self.loss_eta+0.4*self.loss_rank+0.0*self.loss_l2
        logger.info("0.8*self.loss_eta+0.2*self.loss_rank+0.0*self.loss_l2")
        self.opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        grads_and_vars = self.opt.compute_gradients(self.loss)
        capped_grads_vars = [[tf.clip_by_value(g, -config.clip, config.clip), v]
                             for g, v in grads_and_vars]

        self.train_op = self.opt.apply_gradients(capped_grads_vars, global_step=self.global_step)

    def create_feed_dict(self, batch, isTrain=True):
        """
        Create the dictionary of data to feed to tf session during training.
        """
        feed_dict = {
            self.labels: batch[0],
            self.rank_labels: batch[2],
            self.inputs: batch[4],
            self.deep_keep_prob: self.config.dropout_rate if isTrain else 1.0,
            self.isTrain: isTrain
        }
        return feed_dict

    def create_placeholders(self):
        #输入,我们这里输入为三维向量，不需要look_up embeding 最后一维是特征的个数 batch_size*seqlen*feat_size
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.config.max_length,120], name="inputs")
        self.labels = tf.placeholder(tf.float32, shape=[None, self.config.max_length,1], name="label")#batch_size*seqlen
        self.rank_labels = tf.placeholder(tf.int32, shape=[None, self.config.max_length], name="order_label")#batch_size*seqlen
        self.deep_keep_prob = tf.placeholder(tf.float32, name="deep_keep_prob")
        self.isTrain = tf.placeholder(tf.bool, name="isTrain")

    def train_model(self, FLAGS, train_manager, dev_manager):
        """Train the model.
        """
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data
        # 启动会话，执行整个任务
        with tf.Session(config=tf_config) as sess:
            if FLAGS.isrestore and  tf.train.checkpoint_exists(FLAGS.save_parameters):
                # Fit the model
                logger.info("##########Start from trained###########")
                self.tf_saver = tf.train.Saver()
                self.tf_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.save_parameters))
            else:
                # Fit the model
                logger.info("##########Start training###########")
                # Initialize tf stuff
                summary_objs = self.init_tf_ops(sess)
                self.tf_merged_summaries = summary_objs[0]
                self.tf_summary_writer = summary_objs[1]
                self.tf_saver = summary_objs[2]
            for i in range(300):
                for batch in train_manager.iter_batch(shuffle=True):
                    feed_dict = self.create_feed_dict(batch,isTrain=True)
                    step, batch_loss = self.fit(sess, feed_dict, batch)
                    if step % FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, " "train loss:{:>9.6f}".format(
                            iteration, step % steps_per_epoch, steps_per_epoch, np.mean(batch_loss)))
                # whether to be best
                _, best = self.evaluate(sess, dev_manager)
                if best:
                    # Save the model paramenters
                    if FLAGS.save_parameters:
                        self.tf_saver.save(sess, os.path.join(FLAGS.save_parameters, "model"))
                        logger.info("model saved")

    def fit(self, sess, feed_dict, batch):
        """Fit the model to the data.
        Parameters
         ----------
        sess : Tensorflow Session.
        batch : batch labels,text_feats, stat_feats
        feed_dict:
        """
        assert len(batch[0]) == len(batch[1]) == len(batch[2])
        # Train model
        try:
            loss_train, step,length ,_ = sess.run([self.loss, self.global_step, self.enc_seq_length,self.train_op], feed_dict=feed_dict)
        except Exception as e:
            print(e)
            return -999999, -999999

        return step, loss_train

    def init_tf_ops(self,sess):
        """
        Initialize TensorFlow operations.
        """
        summary_merged = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init_op)
        # tensorboard_dir = '/home/longzhangchao/data/model/etaOrderModel/tensorboard'
        tensorboard_dir = self.config.tensorboard_dir
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

        return summary_merged, summary_writer, saver

    def evaluate(self, sess, batch_manager):
        """
        evaluate the model over the eval set.
        """
        total_loss = 0.0
        total_cnt = 0
        log_cnt=0
        correct = 0
        online_correct=0
        correct_10 = 0
        correct_o_10 = 0
        total_order=0

        total_cnt_rank=0
        correct_rank=0
        online_correct_rank=0
        log_cnt_rank=0
        for batch in batch_manager.iter_batch(shuffle=True):
            try:
                batch_len = len(batch[1])
                total_cnt_rank += batch_len
                feed_dict = self.create_feed_dict(batch,isTrain=False)
                # length,probs_,pre_eta_= sess.run([self.enc_seq_length,self.probs_,self.pre_eta_ ], feed_dict=feed_dict)
                # length,pre_eta_,logits,probs,pred,mask,mask_eta= sess.run([self.enc_seq_length,self.pre_eta_,self.logits ,self.probs,self.pred,self.mask,self.mask_eta], feed_dict=feed_dict)
                length,pre_eta_ ,logits,attention_final, one_hot_labels,probs= sess.run([self.enc_seq_length,self.pre_eta_, self.logits,self.attention_final, self.one_hot_labels,self.probs], feed_dict=feed_dict)
                # pred = np.argmax(np.abs(probs),axis=-1)
                pred=reasoning_result(length,probs)
                # 排序准确率
                for pred_rank,true_rank,online_rank, true_length in zip(np.array(pred),np.array(batch[2]), np.array(batch[3]),np.array(length).flatten()):
                    true_rank_str = "".join([str(i) for i in true_rank[0:true_length]])
                    pred_rank_str = "".join([str(i) for i in pred_rank[0:true_length]])
                    online_rank_str = "".join([str(i) for i in online_rank[0:true_length]])
                    if true_rank_str == pred_rank_str:
                        correct_rank += 1
                    if true_rank_str == online_rank_str:
                        online_correct_rank+=1
                    log_cnt_rank += 1
                    if log_cnt_rank % 1000 == 0:
                        logger.info("prediction is:{},true is {},online_rank is {}".format(pred_rank_str, true_rank_str,online_rank_str))
                        # logger.info("probs:{}".format(probs))
                        # logger.info("memory:{}".format(memory))

                for true_label,online_pre,pre_label,true_l in zip(np.array(batch[0]),np.array(batch[1]),np.array(pre_eta_),length):
                    true_label = true_label[0:true_l,:]
                    pre_label = pre_label[0:true_l,:]
                    online_pre = online_pre[0:true_l,:]
                    for t_l,o_l,p_l in zip(true_label,online_pre,pre_label):
                        total_order += 1
                        if np.abs(t_l[0]-p_l[0])<10:
                            correct_10+=1
                        if np.abs(t_l[0]-o_l[0]*100)<10:
                            correct_o_10+=1
                        if total_order%1000==0:
                            logger.info("true eta is {},model_pred_eta is {},online_pred_eta is {}".format(t_l[0],p_l[0],o_l[0]*100))
            except Exception as e:
                logger.info("evalate error pass")
                pass

        acc=0 if correct==0 else correct*1.0/total_cnt
        acc_10= 0 if correct_10==0 else correct_10*1.0/total_order
        acc_O_10= 0 if correct_o_10==0 else correct_o_10*1.0/total_order

        online_acc=0 if online_correct==0 else online_correct*1.0/total_cnt

        acc_online_rank = 0 if online_correct_rank==0 else online_correct_rank*1.0/total_cnt_rank
        acc_pred_rank = 0 if correct_rank==0 else correct_rank*1.0/total_cnt_rank
        best = self.best.eval()
        # tf.assign(self.best, acc_pred_rank).eval()
        is_best = False
        if acc_10> best:
            tf.assign(self.best, acc_10).eval()
            logger.info("best acc_10 : {},best acc_O_10 : {},best acc_total : {}, online_acc_total:{}, ".format(acc_10,acc_O_10,acc,online_acc))
            logger.info("best acc_pred_rank : {}, acc_online_rank:{}, ".format(acc_pred_rank,acc_online_rank))

            is_best = True
        return total_loss, is_best









