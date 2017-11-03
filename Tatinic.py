# -*- coding: utf-8 -*-
# -*- coding: cp936 -*-
# -*- coding: gb18030 -*-


#--------------------------------------------------#
#     Author:guchao
#     mail  :guchaonemo@163.com
#     time  :2017.11.02 15:00
#     USAEG :Tatinic
#--------------------------------------------------#
import numpy as np
import pandas as pd 
import tensorflow as tf

test = pd.read_csv("input/test.csv")
gender_submission = pd.DataFrame(test["PassengerId"],columns=["PassengerId","Survived"])

train = pd.read_csv("input/train.csv")
test_mean_age = test['Age'].median()
train_mean_age = train['Age'].median()
# PassengerId, Survived
#gender_submission_keys = gender_submission.columns.values
#gender_submission = gender_submission.values

#'PassengerId' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare'
#'Cabin' 'Embarked'
test_keys = test.columns.values
test = test.values

#'PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
# 'Ticket' 'Fare' 'Cabin' 'Embarked'
train_keys = train.columns.values
train = train.values
Train_PassengerId = train[:,0]
Train_Survived =np.array([1 if (x)==1 else -1 for x in train[:,1]], dtype=np.float32)
Train_Pclass = np.array(train[:,2], dtype=np.float32)
Train_Sex =   np.array([0 if x=='male' else 1 for x in train[:,4]], dtype=np.float32)
Train_Age =   np.array([train_mean_age if np.isnan(x) else x for x in train[:,5]], dtype=np.float32)
Train_Sibsp = np.array([x for x in train[:,6]], dtype=np.float32)
Train_Parch = np.array([x for x in train[:,7]], dtype=np.float32)
Train_Fare =  np.array([x for x in train[:,9]], dtype=np.float32)

train_data = np.array([Train_Pclass, Train_Sex, Train_Age, Train_Sibsp, Train_Parch, Train_Fare])

test_PassengerId = test[:,0]
test_Pclass = np.array(test[:,1], dtype=np.float32)
test_Sex = np.array([0 if x=='male' else 1 for x in test[:,3]], dtype=np.float32)
test_Age =  np.array([test_mean_age if np.isnan(x) else x for x in test[:,4]], dtype=np.float32)
test_Sibsp = np.array([x for x in test[:,5]], dtype=np.float32)
test_Parch = np.array([x for x in test[:,6]], dtype=np.float32)
meanfare = np.mean(np.array([x if not np.isnan(x) else 0.0 for x in test[:,8]], dtype=np.float32))
test_Fare = np.array([meanfare if np.isnan(x) else x for x in test[:,8]], dtype=np.float32)

test_data = np.array([test_Pclass, test_Sex, test_Age, test_Sibsp, test_Parch, test_Fare])

batch_size = 100
sess = tf.Session()

gamma = tf.constant(-10.0)
x = tf.placeholder(shape=[None, 6], dtype=tf.float32, name = 'x')
pred_x = tf.placeholder(shape=[None, 6], dtype=tf.float32, name ='pred_x')
pred_y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name ='pred_y')
y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name = 'y')
y_y = tf.matmul(y, tf.transpose(y))
dist = tf.reduce_sum( tf.square(x), 1 )
dist = tf.reshape(dist, shape=[-1,1 ])
x_x =tf.matmul(x, tf.transpose(x))
sq_dists = tf.subtract(dist,tf.multiply(2.0, x_x))
sq_dists =tf.add(sq_dists, tf.transpose(dist))
kernel = tf.exp(tf.multiply(gamma, sq_dists), name='exp')

b = tf.Variable( tf.abs(tf.random_normal(shape=[batch_size, 1], dtype=tf.float32) ) )
b_b = tf.matmul(b, tf.transpose(b))
sumOfb = tf.reduce_sum(b)

loss = tf.subtract(tf.reduce_sum(tf.multiply(tf.multiply(y_y, b_b), kernel)), sumOfb)

# prediction 

rA = tf.reshape(tf.reduce_sum(tf.square(x), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(pred_x), 1),[-1,1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x, tf.transpose(pred_x)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
prediction_output = tf.matmul(tf.multiply(tf.transpose(y),b), pred_kernel)
prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(pred_y)), tf.float32))

my_opt = tf.train.GradientDescentOptimizer(0.0005)
train_step = my_opt.minimize(loss)


init = tf.global_variables_initializer()
sess.run(init)

train_data = np.transpose(train_data)
test_data = np.transpose(test_data)
nelements = train_data.shape[0]
count =0
while 1:
    rand_index = np.random.choice(nelements, size=batch_size)
    rand_x = train_data[rand_index,:]
    rand_y = [[ele] for ele in Train_Survived[rand_index]]
    sess.run(train_step,feed_dict={x:rand_x,y:rand_y})
    count+=1
    if count%100==0:
        result=sess.run( accuracy, feed_dict={x:rand_x,y:rand_y,pred_x:train_data,pred_y:[[ele] for ele in Train_Survived]})
        print (result)
        if result > 0.97:
           pred = sess.run(prediction, feed_dict={x:rand_x,y:rand_y,pred_x:test_data})
           break
pred = pred[0, :]
pred = [1 if ele>0 else 0 for ele in pred]
gender_submission['Survived'] = pred
gender_submission.to_csv("input/result.csv",index=False)