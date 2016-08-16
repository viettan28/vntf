import tensorflow as tf
import numpy as np

batch_size = 1
input_layer = 2
hidden_layer = 5
output_layer = 1
learning_rate = 0.01

#Tạo place holder để dựng model, đây chính là nơi chứa dữ liệu cho model.
input=tf.placeholder(tf.float32, shape=(batch_size,input_layer))
target=tf.placeholder(tf.float32, shape=(batch_size, output_layer))

#Khai báo parameter của model
w1=tf.get_variable("w1", [input_layer, hidden_layer], dtype=tf.float32)
w2=tf.get_variable("w2", [hidden_layer, output_layer], dtype=tf.float32)

#Khai báo model
l1=tf.matmul(input, w1)
l2=tf.matmul(l1, w2)

#Tính loss và train
loss=tf.reduce_mean(0.5*tf.square(target - l2))
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train_opt=optimizer.minimize(loss)

#Tạo một session
with tf.Session() as sess:
    #Khởi tạo giá trị của các parameters (w1, w2)
    tf.initialize_all_variables().run()

    for epoch in range(10000):
        #Random giá trị cho input và target
        x=np.random.random_sample((batch_size, input_layer))
        y=np.array([[np.sum(x)]])

        #train
        # lấy ra giá trị predict của model (l2) , loss và thực hiện update parameter (train model(train_opt))
        # truyền vào giá trị input và target
        _output, _loss, _ = sess.run([l2, loss, train_opt], feed_dict={input: x , target:y })
        if epoch % 1000==0:
            print(epoch, x, y, _output, _loss)