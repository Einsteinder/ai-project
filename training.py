import tensorflow as tf
import numpy as np
import evaluation

class Trainer:
    
    def __init__(self,
                 model,
                 sess,
                 summary_path, 
                 batch_size,
                 global_step,
                 X_train, y_train,
                 X_valid, y_valid,):
        
        
        self.model = model
        self.sess = sess
        self.batch_size = batch_size
        self.global_step = global_step
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        
        
        self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)
        init = tf.global_variables_initializer()
        
        self.summary = tf.summary.merge_all()
        self.sess.run(init)
        
     
    def train(self,
              max_iter,
              train_eval_step,
              validation_eval_step,
              display_step):
        
        
        num_batch = np.ceil(float(self.X_train.shape[0]) / float(self.batch_size))
        
        # Keep training until reach max iterations
        for _ in range(max_iter):
            if self.global_step % num_batch == 0:
                self.X_train, self.y_train = self.shuffle(self.X_train, self.y_train) 
        
            
            batch_idx = self.global_step % num_batch
            data_idx = np.arange(batch_idx * self.batch_size,
                                 min((batch_idx+1) * self.batch_size,
                                     self.X_train.shape[0])).astype('int32')
        
            batch_X = self.X_train[data_idx]
            batch_y = self.y_train[data_idx]
            summary_str, loss = self.sess.run((self.summary, self.model.cost), 
                                              feed_dict = {self.model.X : batch_X,
                                                           self.model.y : batch_y})
            
            #print info
            self.summary_writer.add_summary(summary_str, global_step = self.global_step)
            self.summary_writer.flush()
            
            if self.global_step % display_step == 0:
                print "Minibatch Loss = {:.6f}".format(loss)
                
             
            
            if self.global_step % train_eval_step == 0:
                #evaluate
                
                out = self.sess.run(self.model.out,
                                          feed_dict = {self.model.X : batch_X})
                                           
                acc = evaluation.accuracy(out, batch_y)    
                print 'Training accuracy:' , acc
                
                prec, rec, f1 = evaluation.prec_rec(out, batch_y) 
                print 'prec:' , prec, 'rec:', rec, 'f1:',f1
                
                acc_summary_str_train = tf.Summary(value = [
                        tf.Summary.Value(tag = 'Train_accuracy',simple_value = acc),
                        tf.Summary.Value(tag = 'Train_Precision',simple_value = prec),
                        tf.Summary.Value(tag = 'Train_recall',simple_value = rec),
                        tf.Summary.Value(tag = 'Train_F1',simple_value = f1),                        
                    ])
                
                self.summary_writer.add_summary(acc_summary_str_train, global_step = self.global_step)
                self.summary_writer.flush()
                
                
            if self.global_step % validation_eval_step == 0:
                
                out = self.sess.run(self.model.out,
                                          feed_dict = {self.model.X : self.X_valid})
                                           
                acc = evaluation.accuracy(out, self.y_valid) 
                
                print 'Validation accuracy:' , acc
                
                prec, rec, f1 = evaluation.prec_rec(out, self.y_valid) 
                print 'prec:' , prec, 'rec:', rec, 'f1:',f1
                
                acc_summary_str_valid = tf.Summary(value = [
                        tf.Summary.Value(tag = 'validation_accuracy',simple_value = acc),
                        tf.Summary.Value(tag = 'validation_precision',simple_value = prec),
                        tf.Summary.Value(tag = 'validation_recall',simple_value = rec),
                        tf.Summary.Value(tag = 'validation_f1',simple_value = f1),
                    ])
                
                self.summary_writer.add_summary(acc_summary_str_valid, global_step = self.global_step)
                self.summary_writer.flush()
            
            self.sess.run(self.model.training_operation,
                          feed_dict = {self.model.X : batch_X,
                                       self.model.y : batch_y})
                
            self.global_step +=1    
                                   
    def eval_test(self, X, y):
        
        out = self.sess.run(self.model.out,
                                          feed_dict = {self.model.X : X})
                                           
        acc = evaluation.accuracy(out, y) 
                
        print 'Test accuracy:' , acc
        
        prec, rec, f1 = evaluation.prec_rec(out, y) 
        print 'prec:' , prec, 'rec:', rec, 'f1:',f1
                
                
    def shuffle(self, data, gt):

        """
        This function shuffle data and ground truth 
        """
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)

        data = data[idx]
        gt = gt[idx]

        return data,gt 