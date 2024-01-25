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
    def __init__(self,labels,logits,insulin,meal):
        self.y_true2 = labels
        self.y_pred2 = logits
        self.y_ins = insulin
        self.y_meal = meal
        #self.yDummy = tf.tile(self.y_pred2,241)

        
    def lossF(self):        
        
        print("true2 shape: {}".format(self.y_true2))
        xVal = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1])    
        GVal = self.y_true2[:,:,0]
        yVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        zVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        insVal = 8.9*tf.ones(tf.shape(GVal),dtype=tf.dtypes.float32)
        xVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        hVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        gVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)	
        print("xHid size {}".format(yVal))
        GVal = tf.expand_dims(GVal,2)
        yVal = tf.expand_dims(yVal,2)
        zVal = tf.expand_dims(zVal,2)
        insVal = tf.expand_dims(insVal,2)
        xVal = tf.expand_dims(xVal,2)
        hVal = tf.expand_dims(hVal,2)
        gVal = tf.expand_dims(gVal,2)
        print("xVal in loop {}".format(xVal))
        
        limitLoop = 200
        tau = 0.5       
        a = np.float32(np.array([100000.0]))
        stableEps = tf.constant(a)
        for i in range(1,limitLoop,1):
            dummyY = yVal[:,:,i-1] + zVal[:,:,i-1]*tau
            dummyZ = zVal[:,:,i-1] + tau*(-2*tf.math.multiply(self.y_pred2[:,:,0]/10,zVal[:,:,i-1]) - tf.math.multiply(tf.math.square(self.y_pred2[:,:,0]/10),yVal[:,:,i-1])+tf.math.multiply(tf.math.square(self.y_pred2[:,:,0]/10),self.y_ins[:,:,i-1]))
            dummyIns = insVal[:,:,i-1] + tau*(-tf.math.multiply(self.y_pred2[:,:,1],self.y_ins[:,:,i-1])+tf.math.multiply(self.y_pred2[:,:,2]/10,(tf.math.add(yVal[:,:,i-1],100*self.y_pred2[:,:,3]))))
            dummyX = xVal[:,:,i-1] + tau*(-tf.math.multiply(self.y_pred2[:,:,4]/10,xVal[:,:,i-1])+tf.math.multiply(self.y_pred2[:,:,5]/1000,insVal[:,:,i-1])-tf.math.divide_no_nan(tf.math.multiply(tf.math.multiply(self.y_pred2[:,:,5],self.y_pred2[:,:,2]/10),100*self.y_pred2[:,:,3]),self.y_pred2[:,:,1]))
            dummyH = hVal[:,:,i-1]+tau*(-2*tf.math.multiply(self.y_pred2[:,:,6],hVal[:,:,i-1])-tf.math.multiply(tf.math.square(self.y_pred2[:,:,6]),gVal[:,:,i-1])+tf.math.multiply(self.y_pred2[:,:,7],self.y_meal[:,:,i-1]))
            dummyg = gVal[:,:,i-1] + tau*(hVal[:,:,i-1])
            dummyG1 = GVal[:,:,i-1] + tau*(-tf.math.multiply(xVal[:,:,i-1],GVal[:,:,i-1])-tf.math.multiply(self.y_pred2[:,:,8]/100,tf.math.subtract(GVal[:,:,i-1],100*self.y_pred2[:,:,9]))+gVal[:,:,i-1])
            diffDummy = dummyG1 - GVal[:,:,i-1]
            sumDiff = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.square(diffDummy),axis=0),axis=0)/256-100000.0
            dummyG1 = tf.cond(sumDiff > 0.0, lambda: GVal[:,:,i-1], lambda: dummyG1)
            dummyY = tf.expand_dims(dummyY,2)
            yVal = tf.concat([yVal,dummyY],2)
            dummyZ = tf.expand_dims(dummyZ,2)
            zVal = tf.concat([zVal,dummyZ],2)
            dummyIns = tf.expand_dims(dummyIns,2)
            insVal = tf.concat([insVal,dummyIns],2)
            dummyX = tf.expand_dims(dummyX,2)
            xVal = tf.concat([xVal,dummyX],2)
            dummyH = tf.expand_dims(dummyH,2)
            hVal = tf.concat([hVal,dummyH],2)
            dummyg = tf.expand_dims(dummyg,2)
            gVal = tf.concat([gVal,dummyg],2)
            dummyG1 = tf.expand_dims(dummyG1,2)
            GVal = tf.concat([GVal,dummyG1],2)
        err = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None])
        
        #yDummy = tf.reshape(tf.tile(yDummy,[1, 1, 1, 241]),[tf.shape(yDummy)[0],tf.shape(yDummy)[1],tf.shape(yDummy)[2],241])
        #self.yDummy = tf.expand_dims(self.yDummy)
        #xVal = - tf.math.multiply(self.y_pred2,tf.math.square(1-xVal))*0.1 - tf.math.multiply(self.y_pred2,xVal)*0.1
        #yDummy = tf.reshape(tf.tile(self.y_pred2,241),[tf.shape(self.y_pred2)[0],tf.shape(self.y_pred2)[1],tf.shape(self.y_pred2)[2],241]) 
        #print("yDummy {}".format(tf.shape(yDummy))) 
        
        #for i in range(1,241,1):
        #    xVal2 = tf.concat([xVal2,xVal2[:,:,i-1]-tf.math.multiply(self.yDummy[:,:,0],tf.math.square(1-xVal2[:,:,i-1]))*0.1-tf.math.multiply(self.yDummy[:,:,1],xVal2[:,:,i-1])*0.1],2)
        print("xVal in loop {}".format(xVal))
        #with tf.Session() as sess:
        #    sess.run(yDummy)
        
        #breakpoint()
        #xValD2 = sess.run(xVal)

        #print(xValD2)
        err =  tf.math.reduce_sum(tf.math.square(self.y_true2[:,:,0:limitLoop]-GVal)/limitLoop,axis=2)
        #err = tf.math.sqrt(err)
            
        print("OverErr: {}".format(err))
        
        #breakpoint()
        #overErr = tf.convert_to_tensor(overErr, dtype=tf.float32)
        print("overErr: {}".format(err))
        
        return err

    

    def lossFV3(self):        
        

        print("true2 shape: {}".format(self.y_true2))
        
        xVal = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1])
        
        #print("true2 value shape: {}".format(t2))
    

        #print("true2 value shape: {}".format(t2))
        xVal = self.y_true2[:,:,0]
        xHid = tf.zeros(tf.shape(xVal),dtype=tf.dtypes.float32)
        print("xHid size {}".format(xHid))
        xVal = tf.expand_dims(xVal,2)
        xHid = tf.expand_dims(xHid,2)
        #sess = tf.Session()
        #assigndone = xVal2.assign(self.y_true2[:,:,0])
        #sess.run(xVal2)
        print("xVal in loop {}".format(xVal))
        #sess = tf.Session()
        #xValD = sess.run(xVal)
        #print(xValD)
        limitLoop = 200
        tau = 0.5
        inputX = np.zeros((1,limitLoop),dtype=float)
        for i in range(40,100,1):
            inputX[0,i] = 2.0
        #print("inputX {}".format(inputX))
        #breakpoint()
        a = np.float32(np.array([100000.0]))

        stableEps = tf.constant(a)
        #print(stableEps)
        for i in range(1,limitLoop,1):
            dummy1 = xHid[:,:,i-1] - tf.math.multiply((0.02+0.000002*self.y_pred2[:,:,0]),xHid[:,:,i-1])*tau+inputX[0,i]*(0.01+0.0000001*self.y_pred2[:,:,1])*tau
            dummy = xVal[:,:,i-1]-tf.math.multiply(0.2 + 0.0000002*self.y_pred2[:,:,2],1-tf.math.square(xVal[:,:,i-1]))*tau - tf.math.multiply(0.1+0.000001*self.y_pred2[:,:,3],xHid[:,:,i-1])*tau
            diffDummy = dummy - xVal[:,:,i-1]
            sumDiff = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.square(diffDummy),axis=0),axis=0)/256-100000.0
            #print("sumDiff {}".format(sumDiff))
            dummy = tf.cond(sumDiff > 0.0, lambda: xVal[:,:,i-1], lambda: dummy)
                
            dummy = tf.expand_dims(dummy,2)
            xVal = tf.concat([xVal,dummy],2)
            dummy1 = tf.expand_dims(dummy1,2)
            xHid = tf.concat([xHid,dummy1],2)
        #xVal2[:,:,0].assign(self.y_true2[:,:,0])
    
        err = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None])
        
        #yDummy = tf.reshape(tf.tile(yDummy,[1, 1, 1, 241]),[tf.shape(yDummy)[0],tf.shape(yDummy)[1],tf.shape(yDummy)[2],241])
        #self.yDummy = tf.expand_dims(self.yDummy)
        #xVal = - tf.math.multiply(self.y_pred2,tf.math.square(1-xVal))*0.1 - tf.math.multiply(self.y_pred2,xVal)*0.1
        #yDummy = tf.reshape(tf.tile(self.y_pred2,241),[tf.shape(self.y_pred2)[0],tf.shape(self.y_pred2)[1],tf.shape(self.y_pred2)[2],241]) 
        #print("yDummy {}".format(tf.shape(yDummy))) 
        
        #for i in range(1,241,1):
        #    xVal2 = tf.concat([xVal2,xVal2[:,:,i-1]-tf.math.multiply(self.yDummy[:,:,0],tf.math.square(1-xVal2[:,:,i-1]))*0.1-tf.math.multiply(self.yDummy[:,:,1],xVal2[:,:,i-1])*0.1],2)
        print("xVal in loop {}".format(xVal))
        #with tf.Session() as sess:
        #    sess.run(yDummy)
        
        #breakpoint()
        #xValD2 = sess.run(xVal)

        #print(xValD2)
        err =  tf.math.reduce_sum(tf.math.square(self.y_true2[:,:,0:limitLoop]-xVal)/limitLoop,axis=2)
        #err = tf.math.sqrt(err)
            
        print("OverErr: {}".format(err))
        
        #breakpoint()
        #overErr = tf.convert_to_tensor(overErr, dtype=tf.float32)
        print("overErr: {}".format(err))
        
        return xVal

	
    def lossFV2(self):        
        

        print("true2 shape: {}".format(self.y_true2))
        
        xVal = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1])
        
        #print("true2 value shape: {}".format(t2))
    

        #print("true2 value shape: {}".format(t2))
        xVal = self.y_true2[:,:,0]
        xHid = tf.zeros(tf.shape(xVal),dtype=tf.dtypes.float32)
        print("xHid size {}".format(xHid))
        xVal = tf.expand_dims(xVal,2)
        xHid = tf.expand_dims(xHid,2)
        #sess = tf.Session()
        #assigndone = xVal2.assign(self.y_true2[:,:,0])
        #sess.run(xVal2)
        print("xVal in loop {}".format(xVal))
        #sess = tf.Session()
        #xValD = sess.run(xVal)
        #print(xValD)
        limitLoop = 300
        tau = 0.5
        inputX = np.zeros((1,limitLoop),dtype=float)
        for i in range(40,100,1):
            inputX[0,i] = 2.0
        #print("inputX {}".format(inputX))
        #breakpoint() 
        for i in range(1,limitLoop,1):
            dummy1 = xHid[:,:,i-1] - tf.math.multiply(self.y_pred2[:,:,0]/10,xHid[:,:,i-1])*tau+inputX[0,i]*self.y_pred2[:,:,1]*tau/10
            dummy = xVal[:,:,i-1]-tf.math.multiply(self.y_pred2[:,:,2],1-tf.math.square(xVal[:,:,i-1]))*tau - tf.math.multiply(self.y_pred2[:,:,3],xHid[:,:,i-1])*tau
            dummy = tf.expand_dims(dummy,2)
            xVal = tf.concat([xVal,dummy],2)
            dummy1 = tf.expand_dims(dummy1,2)
            xHid = tf.concat([xHid,dummy1],2)
        #xVal2[:,:,0].assign(self.y_true2[:,:,0])
    
        err = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None])
        
        #yDummy = tf.reshape(tf.tile(yDummy,[1, 1, 1, 241]),[tf.shape(yDummy)[0],tf.shape(yDummy)[1],tf.shape(yDummy)[2],241])
        #self.yDummy = tf.expand_dims(self.yDummy)
        #xVal = - tf.math.multiply(self.y_pred2,tf.math.square(1-xVal))*0.1 - tf.math.multiply(self.y_pred2,xVal)*0.1
        #yDummy = tf.reshape(tf.tile(self.y_pred2,241),[tf.shape(self.y_pred2)[0],tf.shape(self.y_pred2)[1],tf.shape(self.y_pred2)[2],241]) 
        #print("yDummy {}".format(tf.shape(yDummy))) 
        
        #for i in range(1,241,1):
        #    xVal2 = tf.concat([xVal2,xVal2[:,:,i-1]-tf.math.multiply(self.yDummy[:,:,0],tf.math.square(1-xVal2[:,:,i-1]))*0.1-tf.math.multiply(self.yDummy[:,:,1],xVal2[:,:,i-1])*0.1],2)
        print("xVal in loop {}".format(xVal))
        #with tf.Session() as sess:
        #    sess.run(yDummy)
        
        #breakpoint()
        #xValD2 = sess.run(xVal)

        #print(xValD2)
        err =  tf.math.reduce_sum(tf.math.square(self.y_true2[:,:,0:limitLoop]-xVal)/limitLoop,axis=2)
        #err = tf.math.sqrt(err)
            
        print("OverErr: {}".format(err))
        
        breakpoint()
        #overErr = tf.convert_to_tensor(overErr, dtype=tf.float32)
        print("overErr: {}".format(err))
        
        return xVal


        

class HarData:

    def __init__(self,seq_len=16):
        tf.compat.v1.disable_eager_execution()
        tf.disable_v2_behavior()
        train_x = np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataAPV3.txt")
        train_y = (np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataAPV3.txt")-1)#.astype(np.int32)
        train_ins = np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataAPInsulinV3.txt")
        train_meal = np.loadtxt("data/har/UCI HAR Dataset/train/TrainDataAPMealV3.txt")
        
        test_x = np.loadtxt("data/har/UCI HAR Dataset/test/TestDataAPV3.txt")
        test_y = (np.loadtxt("data/har/UCI HAR Dataset/test/TestDataAPV3.txt")-1)#.astype(np.int32)
        test_ins = np.loadtxt("data/har/UCI HAR Dataset/test/TestDataAPInsulinV3.txt")
        test_meal = np.loadtxt("data/har/UCI HAR Dataset/test/TestDataAPMealV3.txt")
        print("train_x: {}".format(test_x.shape))
        print("train_y: {}".format(test_y.shape)) 
        
        train_x,train_y = cut_in_sequences(train_x,train_y,seq_len)
        train_ins,train_meal = cut_in_sequences(train_ins,train_meal,seq_len)
        test_x,test_y = cut_in_sequences(test_x,test_y,seq_len,inc=8)
        test_ins,test_meal = cut_in_sequences(test_ins,test_meal,seq_len,inc=8)
        print("Total number of training sequences: {}".format(train_x.shape[1]))
        #permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
        valid_size = int(0.1*train_x.shape[1])
        print("Validation split: {}, training split: {}".format(valid_size,train_x.shape[1]-valid_size))

        self.valid_x = train_x[:,:valid_size]
        self.valid_ins = train_ins[:,:valid_size]
        self.valid_meal = train_meal[:,:valid_size]
        self.valid_y = train_y[:,:valid_size]
        self.train_x = train_x[:,valid_size:]
        self.train_ins = train_ins[:,valid_size:]
        self.train_meal = train_meal[:,valid_size:]
        self.train_y = train_y[:,valid_size:]

        self.test_x = test_x
        self.test_ins = test_ins
        self.test_meal = test_meal
        self.test_y = test_y
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
            yield (batch_x,batch_y,batch_ins,batch_mL)

class HarModel:

    def __init__(self,model_type,model_size,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,241])
        self.ins = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,241])
        self.mL = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,241])
        self.target_y = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,241])

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


        self.y = tf.compat.v1.layers.Dense(10,activation='sigmoid')(head) # Dense layer output should be same as the number of model parameter 
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
        lossVal = Custom_CE_Loss(labels = self.target_y,logits = self.y,insulin = self.ins,meal=self.mL).lossF() 
        print("lossVal {}".format(lossVal))

        self.loss = tf.reduce_mean(Custom_CE_Loss(labels = self.target_y,logits = self.y,insulin=self.ins,meal=self.mL).lossF())
        
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
                
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.test_x,self.target_y: gesture_data.test_x,self.ins: gesture_data.test_ins,self.mL: gesture_data.test_meal})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.valid_x,self.target_y: gesture_data.valid_x, self.ins: gesture_data.valid_ins, self.mL: gesture_data.valid_meal})
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
            for batch_x,batch_y,batch_ins,batch_mL in gesture_data.iterate_train(batch_size=16):
                acc,loss,t_step,t_y = self.sess.run([self.accuracy,self.loss,self.train_step,self.y],{self.x:batch_x,self.target_y: batch_y, self.ins: batch_ins, self.mL: batch_mL})
                
                print("loss iter: {} ".format(loss))
                #breakpoint()
                if(not self.constrain_op is None):
                    self.sess.run(self.constrain_op)
                tY.append(tf.reduce_mean(tf.reduce_mean(t_y,axis=0),axis=0))
                losses.append(loss)
                accs.append(acc)
            print("loss: {}".format(loss))
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

    breakpoint()

