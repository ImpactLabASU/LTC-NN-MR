import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Run on CPU

import tensorflow.compat.v1 as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse

def cut_in_sequences(x,y,seq_len,inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0,x.shape[0] - seq_len,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x,axis=1),np.stack(sequences_y,axis=1)

#@tf.function(input_signature=[tf.TensorSpec(shape=None,dtype=tf.float32)])
#def compare(x):
#    a = np.float32(np.array([10000.0]))
#    stableEps = tf.constant(a)
#    result = tf.math.greater(x,stableEps)
#    return result




class Custom_CE_Loss(tf.keras.losses.Loss):
    def __init__(self,labels,logits,insulin,meal,basal):
        self.y_true2 = labels
        self.y_pred2 = logits
        self.y_ins = insulin
        self.y_meal = meal
        self.y_basal = basal
        #self.yDummy = tf.tile(self.y_pred2,241)

        
    def lossF(self):        
        
        print("true2 shape: {}".format(self.y_true2))
        xVal = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1])    
        x2Val = self.y_true2[:,:,0]
        x1Val = (6/5)*self.y_true2[:,:,0]
        x3Val = self.y_true2[:,:,0]
        x2Val = tf.expand_dims(x2Val,2)
        x1Val = tf.expand_dims(x1Val,2)     
        x3Val = tf.expand_dims(x3Val,2)

        print("xVal in loop {}".format(xVal))
        
        limitLoop = 200
        tau = 0.2       
        a = np.float32(np.array([100000.0]))
        stableEps = tf.constant(a)
        
        for i in range(1,limitLoop,1):
            dummyX1 = x1Val[:,:,i-1] + (tf.math.multiply((self.y_pred2[:,:,0]),(x2Val[:,:,i-1]-x1Val[:,:,i-1]))+self.y_ins[:,:,i-1])*tau
            dummyX2 = x2Val[:,:,i-1] + (tf.math.multiply(x1Val,(self.y_pred2[:,:,1]-x3Val))-x2Val)*tau
            dummyX3 = x3Val[:,:,i-1] + (tf.math.multiply(x1Val[:,:,i-1],x2Val[:,:,i-1])-tf.math.multiply(self.y_pred2[:,:,2],x3Val))*tau
            
             
            diffDummy = dummyX3 - x3Val[:,:,i-1]
            sumDiff = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.square(diffDummy),axis=0),axis=0)/256-100.0
            dummyX3 = tf.cond(sumDiff > 0.0, lambda: self.y_true2[:,:,i]+10, lambda: dummyX3)
            dummyX1 = tf.expand_dims(dummyX1,2)
            x1Val = tf.concat([x1Val,dummyX1],2)
            dummyX2 = tf.expand_dims(dummyX2,2)
            x2Val = tf.concat([x2Val,dummyX2],2)
            dummyX3 = tf.expand_dims(dummyX3,2)
            x3Val = tf.concat([x3Val,dummyX3],2)
            
        err = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None])
        print("x2Val: {}".format(tf.shape(x2Val)))
        print("true val: {}".format(tf.shape(self.y_true2)))
        #yDummy = tf.reshape(tf.tile(yDummy,[1, 1, 1, 241]),[tf.shape(yDummy)[0],tf.shape(yDummy)[1],tf.shape(yDummy)[2],241])
        #self.yDummy = tf.expand_dims(self.yDummy)
        #xVal = - tf.math.multiply(self.y_pred2,tf.math.square(1-xVal))*0.1 - tf.math.multiply(self.y_pred2,xVal)*0.1
        #yDummy = tf.reshape(tf.tile(self.y_pred2,241),[tf.shape(self.y_pred2)[0],tf.shape(self.y_pred2)[1],tf.shape(self.y_pred2)[2],241]) 
        #print("yDummy {}".format(tf.shape(yDummy))) 
        
        #for i in range(1,241,1):
        #    xVal2 = tf.concat([xVal2,xVal2[:,:,i-1]-tf.math.multiply(self.yDummy[:,:,0],tf.math.square(1-xVal2[:,:,i-1]))*0.1-tf.math.multiply(self.yDummy[:,:,1],xVal2[:,:,i-1])*0.1],2)
        #print("xVal in loop {}".format(xVal))
        #with tf.Session() as sess:
        #    sess.run(yDummy)
        
        #breakpoint()
        #xValD2 = sess.run(xVal)

        #print(xValD2)
        err =  tf.math.reduce_sum(tf.math.square(self.y_true2[:,:,0:limitLoop]-x3Val[:,:,0:limitLoop])/limitLoop,axis=2)
        #err = tf.math.sqrt(err)
            
        print("OverErr: {}".format(err))
        
        #breakpoint()
        #overErr = tf.convert_to_tensor(overErr, dtype=tf.float32)
        print("overErr: {}".format(err))
        
        return err

    
    def lossFV2(self):        
        
        print("true2 shape: {}".format(self.y_true2))
        xVal = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1])    
        x2Val = self.y_true2[:,:,0]
        x1Val = (6/5)*self.y_true2[:,:,0]
        
        x2Val = tf.expand_dims(x2Val,2)
        x1Val = tf.expand_dims(x1Val,2)     


        print("xVal in loop {}".format(xVal))
        
        limitLoop = 1000
        tau = 0.1       
        a = np.float32(np.array([100000.0]))
        stableEps = tf.constant(a)
        breakpoint()
        for i in range(1,limitLoop,1):
            dummyX1 = x1Val[:,:,i-1] + (tf.math.multiply((0.5+0.000001*self.y_pred2[:,:,0]),x1Val[:,:,i-1])-tf.math.multiply((0.025 + 0.0000001*self.y_pred2[:,:,1]),tf.math.multiply(x1Val[:,:,i-1],x2Val[:,:,i-1])))*tau
            dummyX2 = x2Val[:,:,i-1] + (tf.math.multiply((0.005+0.0000001*self.y_pred2[:,:,2]),tf.math.multiply(x1Val[:,:,i-1],x2Val[:,:,i-1]))-tf.math.multiply((0.5+0.0000001*self.y_pred2[:,:,3]),x2Val[:,:,i-1])+self.y_ins[:,:,i-1])*tau 
             
            diffDummy = dummyX2 - x2Val[:,:,i-1]
            sumDiff = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.square(diffDummy),axis=0),axis=0)/256-100000.0
            dummyX2 = tf.cond(sumDiff > 0.0, lambda: x2Val[:,:,i-1], lambda: dummyX2)
            dummyX1 = tf.expand_dims(dummyX1,2)
            x1Val = tf.concat([x1Val,dummyX1],2)
            dummyX2 = tf.expand_dims(dummyX2,2)
            x2Val = tf.concat([x2Val,dummyX2],2)
            
        err = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None])
        
        #yDummy = tf.reshape(tf.tile(yDummy,[1, 1, 1, 241]),[tf.shape(yDummy)[0],tf.shape(yDummy)[1],tf.shape(yDummy)[2],241])
        #self.yDummy = tf.expand_dims(self.yDummy)
        #xVal = - tf.math.multiply(self.y_pred2,tf.math.square(1-xVal))*0.1 - tf.math.multiply(self.y_pred2,xVal)*0.1
        #yDummy = tf.reshape(tf.tile(self.y_pred2,241),[tf.shape(self.y_pred2)[0],tf.shape(self.y_pred2)[1],tf.shape(self.y_pred2)[2],241]) 
        #print("yDummy {}".format(tf.shape(yDummy))) 
        
        #for i in range(1,241,1):
        #    xVal2 = tf.concat([xVal2,xVal2[:,:,i-1]-tf.math.multiply(self.yDummy[:,:,0],tf.math.square(1-xVal2[:,:,i-1]))*0.1-tf.math.multiply(self.yDummy[:,:,1],xVal2[:,:,i-1])*0.1],2)
        #print("xVal in loop {}".format(xVal))
        #with tf.Session() as sess:
        #    sess.run(yDummy)
        
        #breakpoint()
        #xValD2 = sess.run(xVal)

        #print(xValD2)
        err =  tf.math.reduce_sum(tf.math.square(self.y_true2[:,:,0:limitLoop]-x2Val)/limitLoop,axis=2)
        #err = tf.math.sqrt(err)
            
        print("OverErr: {}".format(err))
        
        #breakpoint()
        #overErr = tf.convert_to_tensor(overErr, dtype=tf.float32)
        print("overErr: {}".format(err))
        
        return err

    

        

class HarData:

    def __init__(self,seq_len=16):
        tf.compat.v1.disable_eager_execution()
        tf.disable_v2_behavior()
        train_x = np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataLotkaVR3T2.txt")
        train_y = (np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataLotkaVR3T2.txt")-1)#.astype(np.int32)
        train_ins = np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataLotkaInsulinVR3T2.txt")
        train_meal = np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataLotkaMealVR3T2.txt")
        train_basal = np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataLotkaBasalVR3T2.txt")
        train_meal2 = train_meal

        test_x = np.loadtxt("data/har/UCI HAR Dataset/test/TestDataLotkaVR3T2.txt")
        test_y = (np.loadtxt("data/har/UCI HAR Dataset/test/TestDataLotkaVR3T2.txt")-1)#.astype(np.int32)
        test_ins = np.loadtxt("data/har/UCI HAR Dataset/test/TestDataLotkaInsulinVR3T2.txt")
        test_meal = np.loadtxt("data/har/UCI HAR Dataset/test/TestDataLotkaMealVR3T2.txt")
        test_meal2 = test_meal
        test_basal = np.loadtxt("data/har/UCI HAR Dataset/test/TestDataLotkaBasalVR3T2.txt")
        print("train_x: {}".format(test_x.shape))
        print("train_y: {}".format(test_y.shape)) 
        
        train_x,train_y = cut_in_sequences(train_x,train_y,seq_len)
        train_ins,train_meal = cut_in_sequences(train_ins,train_meal,seq_len)
        train_basal,train_meal2 = cut_in_sequences(train_basal,train_meal2,seq_len)
        test_x,test_y = cut_in_sequences(test_x,test_y,seq_len,inc=8)
        test_ins,test_meal = cut_in_sequences(test_ins,test_meal,seq_len,inc=8)
        test_basal,test_meal2 = cut_in_sequences(test_basal,test_meal2,seq_len,inc=8)
        print("Total number of training sequences: {}".format(train_x.shape[1]))
        #permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
        valid_size = int(0.1*train_x.shape[1])
        print("Validation split: {}, training split: {}".format(valid_size,train_x.shape[1]-valid_size))

        self.valid_x = train_x[:,:valid_size]
        self.valid_ins = train_ins[:,:valid_size]
        self.valid_meal = train_meal[:,:valid_size]
        self.valid_y = train_y[:,:valid_size]
        self.valid_basal = train_basal[:,:valid_size]
        self.train_x = train_x[:,valid_size:]
        self.train_ins = train_ins[:,valid_size:]
        self.train_meal = train_meal[:,valid_size:]
        self.train_y = train_y[:,valid_size:]
        self.train_basal = train_basal[:,valid_size:]

        self.test_x = test_x
        self.test_ins = test_ins
        self.test_meal = test_meal
        self.test_y = test_y
        self.test_basal = test_basal
        print("Total number of test sequences: {}".format(self.test_x.shape[1]))

    def iterate_train(self,batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,start:end]
            batch_ins = self.train_ins[:,start:end]
            batch_mL = self.train_meal[:,start:end]
            batch_y = self.train_x[:,start:end]
            batch_basal = self.train_basal[:,start:end]
            yield (batch_x,batch_y,batch_ins,batch_mL,batch_basal)

class HarModel:

    def __init__(self,model_type,model_size,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1001])
        self.ins = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1001])
        self.mL = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1001])
        self.target_y = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1001])
        self.basal = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1001])

        self.model_size = model_size
        head = self.x
        
        
    
        print("Beginning ")
        
        if(model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            learning_rate = 0.01 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op = self.wm.get_param_constrain_op()
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))


        self.y = tf.compat.v1.layers.Dense(4,activation='sigmoid')(head) # Dense layer output should be same as the number of model parameter 
        print("logit shape: ")
        print(str(self.y.shape))
        print("self.y: ")
        print(self.y)
        #self.loss = tf.reduce_mean(evaluate_loss(self.target_y, self.y))
        # Add model estimation error to the loss
    
        #self.loss = tf.reduce_mean(tf.compat.v1.losses.sparse_softmax_cross_entropy(
        #    labels = self.target_y,
        #    logits = self.y,
        #))
        lossVal = Custom_CE_Loss(labels = self.target_y,logits = self.y,insulin = self.ins,meal=self.mL,basal=self.basal).lossF() 
        print("lossVal {}".format(lossVal))

        self.loss = tf.reduce_mean(Custom_CE_Loss(labels = self.target_y,logits = self.y,insulin=self.ins,meal=self.mL,basal=self.basal).lossF())
        #breakpoint()
        #self.loss = Custom_CE_Loss()
        
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(1.0,tf.int64), tf.cast(1.0,tf.int64)), tf.float32))
        
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results","har","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results/har")):
            os.makedirs("results/har")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","har","{}".format(model_type))
        if(not os.path.exists("tf_sessions/har")):
            os.makedirs("tf_sessions/har")
           
        self.saver = tf.train.Saver()
        

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,gesture_data,epochs,verbose=True,log_period=50):

        best_valid_accuracy = 0
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        for e in range(epochs):
            if(e%log_period == 0):
                print("x data {}".format({self.x: gesture_data.test_x}.keys()))
                print("y data {}".format({self.target_y:gesture_data.valid_y}.keys()))
                
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.test_x,self.target_y: gesture_data.test_x,self.ins: gesture_data.test_ins,self.mL: gesture_data.test_meal,self.basal: gesture_data.test_basal})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.valid_x,self.target_y: gesture_data.valid_x, self.ins: gesture_data.valid_ins, self.mL: gesture_data.valid_meal, self.basal: gesture_data.valid_basal})
                # Accuracy metric -> higher is better
                if(valid_acc > best_valid_accuracy and e > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs)*100,
                        valid_loss,valid_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()

            losses = []
            accs = []
            tY = []
            err = self.loss 
            print("err {}".format(err))
            #breakpoint()
            for batch_x,batch_y,batch_ins,batch_mL,batch_basal in gesture_data.iterate_train(batch_size=16):
                acc,loss,t_step,t_y = self.sess.run([self.accuracy,self.loss,self.train_step,self.y],{self.x:batch_x,self.target_y: batch_y, self.ins: batch_ins, self.mL: batch_mL, self.basal: batch_basal})
                
                print("loss iter: {} ".format(loss))
                #breakpoint()
                if(not self.constrain_op is None):
                    self.sess.run(self.constrain_op)
                tY.append(tf.reduce_mean(tf.reduce_mean(t_y,axis=0),axis=0))
                losses.append(loss)
                accs.append(acc)
            losses2 = losses[:-1]    
            print("loss: {}".format(sum(losses2)/len(losses2)))
            tyMean = tf.reduce_mean(tf.reduce_mean(t_y,axis=0),axis=0)
            tyMean2 = self.sess.run(tyMean)
            print("ty: {}".format(tyMean2))

                

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
                
               
                
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="lstm")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=200,type=int)
    args = parser.parse_args()


    har_data = HarData()
    model = HarModel(model_type = args.model,model_size=args.size)

    model.fit(har_data,epochs=args.epochs,log_period=args.log)

    print(model.y)

    #breakpoint()

