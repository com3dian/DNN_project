import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noise_ratio',type=float)
parser.add_argument('--noise_pattern',type=str)
parser.add_argument('--dataset',type=str)
parser.add_argument('--gpu_id',type=int, default=0)
args = parser.parse_args(args=[])

import os
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)

from keras.models import load_model
from my_model import create_model
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import data
import numpy as np
import pandas as pd
import os
import gc

""" parameters """
args.dataset = 'cifar10'
args.noise_ratio = 0.1
args.noise_pattern = 'sym'

noise_ratio = args.noise_ratio
noise_pattern = args.noise_pattern #'sym' or 'asym'
dataset = args.dataset
batch_size = 128
INCV_epochs = 20
INCV_iter = 4
save_dir = os.path.join('results', dataset, noise_pattern, str(noise_ratio))
#h合并路径
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)#把路径搞成save_dir
filepath_INCV = os.path.join(save_dir,'INCV_model.h5')
#弄file_path

####################################################

####################################################
if dataset=='cifar10':
    x_train, y_train, _, _, x_test, y_test = data.prepare_cifar10_data(data_dir='data')
    Num_top = 1 #using top k prediction
elif dataset=='cifar100':
    x_train, y_train, _, _, x_test, y_test = data.prepare_cifar100_data(data_dir='data')
    Num_top = 1 #using top k prediction to select samples
    
    
def modify(y_tr, x_tr, x_tst, y_tst, simple_N, noise_pattern, noise_ratio):
    idx0123 = (np.argmax(y_tr, axis=1)<=3)
    simple_index = range(simple_N)
    x_tr = x_tr[idx0123,:][simple_index,:]
    y_tr = y_tr[idx0123,0:4][simple_index,0:4]
    
    idx0123_test = (np.argmax(y_tst, axis=1)<=3)
    x_tst = x_tst[idx0123_test,:]
    y_tst = y_tst[idx0123_test,0:4]
    y_tr_noisy = data.flip_label(y_tr, pattern=noise_pattern, ratio=noise_ratio, one_hot=True)
    
    return(x_tr, y_tr, x_tst, y_tst, y_tr_noisy)
Modify_N = 50000
#
Modify_N = 1000
x_train, y_train, x_test, y_test, y_train_noisy = modify(y_tr = y_train, 
                                      x_tr = x_train, 
                                      x_tst = x_test, 
                                      y_tst = y_test, 
                                      simple_N = Modify_N, 
                                      noise_pattern = noise_pattern, 
                                      noise_ratio = noise_ratio)



from data import count_nearest_neighbour_graphs
labels_pool_batch = count_nearest_neighbour_graphs(data = x_train, scale = 0.06, labels = ([0]), pca_conponent = 20)
idx_temp = np.array([True for i in range(y_train_noisy.shape[0])])
for i in labels_pool_batch:
    idx_temp[i] = False
x_train = np.concatenate((x_train[labels_pool_batch], x_train[list(idx_temp), :]), axis = 0)
y_train_noisy = data.flip_label(y_train[idx_temp, :], pattern=noise_pattern, ratio=noise_ratio, one_hot=True)
y_train_noisy = np.concatenate((y_train[list(labels_pool_batch), :], y_train_noisy), axis = 0)

pool_batch_size = len(labels_pool_batch)

###################################################

###################################################

datagen = ImageDataGenerator(width_shift_range=4./32,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=4./32,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=True)  # randomly flip images 
class Noisy_acc(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if First_half:
            idx = val2_idx # train on the first half while test on the second half 
        else:
            idx = val1_idx
        
        predict = np.argmax(self.model.predict(x_train[idx,:]),axis=1)
        _acc_mix = accuracy_score(np.argmax(y_train_noisy[idx,:],axis=1), predict)
        _acc_clean = accuracy_score(np.argmax(y_train_noisy[idx,:][clean_index[idx],:],axis=1), predict[clean_index[idx]])
        _acc_noisy = accuracy_score(np.argmax(y_train_noisy[idx,:][noisy_index[idx],:],axis=1), predict[noisy_index[idx]])

        print("- acc_mix: %.4f - acc_clean: %.4f - acc_noisy: %.4f\n" % (_acc_mix, _acc_clean, _acc_noisy))
        return
noisy_acc = Noisy_acc()

def INCV_lr_schedule(epoch):
    # Learning Rate Schedule
    lr = 1e-3
    if epoch > 40:
        lr *= 0.1
    elif epoch > 30:
        lr *= 0.25
    elif epoch > 20:
        lr *= 0.5
    print('Learning rate: ', lr)
    return lr

##############################################################

##############################################################

input_shape = list(x_train.shape[1:])
n_classes = y_train.shape[1]
n_train = x_train.shape[0]
clean_index = np.array([(y_train_noisy[i,:]==y_train[i,:]).all() for i in range(n_train)])
# For tracking only, unused during training
noisy_index = np.array([not i for i in clean_index])
INCV_lr_callback = LearningRateScheduler(INCV_lr_schedule)
# Define optimizer and compile model
optimizer = optimizers.Adam(lr=INCV_lr_schedule(0), beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = create_model(input_shape=input_shape, classes=n_classes, name='INCV_ResNet32', architecture='ResNet32')
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
weights_initial = model.get_weights()
# Print model architecture

###################################################################

###################################################################
train_idx = np.array([False for i in range(n_train)])
val_idx = np.array([True for i in range(n_train)])
INCV_save_best = True
pool_batch_idx = count_nearest_neighbour_graphs(data = x_train, 
                                            scale = 0.1, 
                                            labels = ([0, 43, 51]), 
                                            pca_conponent =20)
for i in pool_batch_idx:
    train_idx[i] = True
    
old_idx = pool_batch_idx
#半监督的这个小集合
########################################################

########################################################
#搞
weight_update = weights_initial#初始化网络
for iter in range(1,INCV_iter+1):
    print('INCV iteration %d including first half and second half  of expansion. In total %d iterations.'
          %(iter,INCV_iter))
    
    append_idx = np.array( count_nearest_neighbour_graphs(data = x_train, 
                                        scale = 0.06, 
                                        labels = old_idx, 
                                        pca_conponent = 20))
    #append_idx 实际上应该是indexs
    #
    
    Iter_train = len(append_idx) - len(old_idx)#扩张搞了多少
    val_idx_int = np.array([append_idx[len(old_idx)+i] for i in range(Iter_train) if val_idx[i]]) # integer index
    #val_验证集
    np.random.shuffle(val_idx_int)
    n_val_half = int(len(val_idx_int)/2)
    val1_idx = val_idx_int[:n_val_half] # integer index
    val2_idx = val_idx_int[n_val_half:] # integer index
    #Train model on the first half of dataset
    First_half = True 
    print('Iteration ' + str(iter) + ' - first half')
    # reset weights
    
    model.set_weights(weights_initial)
    results = model.fit_generator(datagen.flow(np.concatenate([x_train[train_idx,:],x_train[val1_idx,:]]),
                                                  np.concatenate([y_train_noisy[train_idx,:],
                                                             y_train_noisy[val1_idx,:]]),
                                                  batch_size = min(batch_size,
                                                              pool_batch_size)),
                                                  epochs = INCV_epochs,
                                                  validation_data=(x_train[val2_idx,:],                                                                    y_train_noisy[val2_idx,:]),
                                                                                                                                callbacks=[ModelCheckpoint(filepath=filepath_INCV, 
                                                                                                                                          monitor='val_acc', verbose=1, 
                                                                                                                                          save_best_only=INCV_save_best),
                                                          noisy_acc,
                                                          INCV_lr_callback])
    #batch_size
    # Select samples of 'True' prediction
    # datagen
    # concatenate把这些鸡巴连起来
    # epochs是训练网络的迭代次数
    #
      
    y_pred = model.predict(x_train[val2_idx,:])
    cross_entropy = np.sum(-y_train_noisy[val2_idx,:]*np.log(y_pred+1e-8),axis=1)
    top_pred = np.argsort(y_pred, axis=1)[:,-Num_top:]
    y_true_noisy = np.argmax(y_train_noisy[val2_idx,:],axis=1)
    top_True = [y_true_noisy[i] in top_pred[i,:] for i in range(len(y_true_noisy))]
    val2train_idx =  val2_idx[top_True]
    if iter == 1:
        eval_ratio = 0.001
        product = np.sum(top_True)/(n_train/2.)
        while (1-eval_ratio)*(1-eval_ratio)+eval_ratio*eval_ratio/(n_classes/Num_top-1) > product:
            eval_ratio += 0.001
            if eval_ratio>=1:
                break
        print('noisy ratio evaluation: %.4f\n' % eval_ratio)
        discard_ratio = min(2, eval_ratio/(1-eval_ratio))       
        discard_idx = val2_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]] # integer index
    #什么鸡巴
    #discard是说删掉的数据
    #
    else:
        discard_idx = np.concatenate([discard_idx, 
                            val2_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]]])
    print('%d samples selected\n' % (np.sum(train_idx)+val2train_idx.shape[0]))
    
    #Train model on the second half of dataset
    First_half = False
    print('Iteration ' + str(iter) + ' - second half')
    # reset weights
    model.set_weights(weights_initial)
    results = model.fit_generator(datagen.flow(np.concatenate([x_train[train_idx,:],x_train[val2_idx,:]]), 
                                                  np.concatenate([y_train_noisy[train_idx,:],
                                                                                                                                             y_train_noisy[val2_idx,:]]),
                                                  batch_size = batch_size),
                                                  epochs = INCV_epochs,
                                                  validation_data=(x_train[val1_idx,:], 
                                                                                                                                              y_train_noisy[val1_idx,:]),
                                               callbacks=[ModelCheckpoint(filepath=filepath_INCV,
                                                                 monitor='val_acc', 
                                                                 verbose=1, 
                                                             save_best_only=INCV_save_best),                                                noisy_acc,
                                               INCV_lr_callback])
    
    y_pred = model.predict(x_train[val1_idx,:])
    cross_entropy = np.sum(-y_train_noisy[val1_idx,:]*np.log(y_pred+1e-8),axis=1)
    top_pred = np.argsort(y_pred, axis=1)[:,-Num_top:]
    y_true_noisy = np.argmax(y_train_noisy[val1_idx,:],axis=1)
    top_True = [y_true_noisy[i] in top_pred[i,:] for i in range(len(y_true_noisy))]    
    
    val2train_idx =  np.concatenate([val1_idx[top_True],val2train_idx])# integer index
    discard_idx = np.concatenate([discard_idx, 
                                  val1_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]]])
    
    train_idx[val2train_idx]=True
    val_idx[val2train_idx]=False
    if noise_pattern == 'sym':
        val_idx[discard_idx]=False
    print('%d samples selected with noisy ratio %.4f\n' % (np.sum(train_idx),
                                                           (1-np.sum(clean_index[train_idx])/np.sum(train_idx))))
    
    if noise_pattern == 'asym' or eval_ratio > 0.6:
        iter_save_best = 1
    elif eval_ratio > 0.3:
        iter_save_best = 2
    else:
        iter_save_best = 4         
        
    if iter==iter_save_best:
        #什么鸡巴
        INCV_save_best = False
    ####################
    #更新集合
    #更新参数
    #还有啥没更新的
    weights_initial = model.get_weights()
    old_idx = list(append_idx[i] for i in range(len(append_idx)) if val_idx[i])
    
    