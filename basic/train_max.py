import tensorflow as tf
import numpy as np

batch_size = 1
input_layer = 5
hidden_layer1 = 25
hidden_layer2 = 25
output_layer = 5
learning_rate = 0.004

#Tạo place holder để dựng model, đây chính là nơi chứa dữ liệu cho model.
input=tf.placeholder(tf.float32, shape=(batch_size,input_layer))
target=tf.placeholder(tf.float32, shape=(batch_size, output_layer))

#Khai báo parameter của model
w1=tf.get_variable("w1", [input_layer, hidden_layer1], dtype=tf.float32)
w2=tf.get_variable("w2", [hidden_layer1, hidden_layer2], dtype=tf.float32)
w3=tf.get_variable("w3", [hidden_layer2, output_layer], dtype=tf.float32)

#Khai báo model
l1=tf.matmul(input, w1)
l2=tf.matmul(l1, w2)
l3=tf.matmul(l2, w3)
l3_sm=tf.nn.softmax(l3)

#Tính loss và train
loss = tf.nn.softmax_cross_entropy_with_logits(l3_sm, target)
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train_opt=optimizer.minimize(loss)

def max_index(x):
    count=len(x[0])
    index = 0
    max = x[0][0]
    for i in range(count):
        if max < x[0][i]:
            max = x[0][i]
            index = i
    return index

# Hàm tạo dữ liệu mẫu
def get_batch(count):
    x = np.random.random_sample((1,count))
    y=np.zeros((1, count), np.float)
    index=max_index(x)
    y[0][index] = 1
    return x, y

#khởi tạo dữ liệu tì lệ train/valid=9/1
train_data=[]
valid_data=[]
for i in range(3000):
    _i, _o=get_batch(input_layer)
    if i %10==0:
        valid_data.append((_i, _o))
    else:
        train_data.append((_i, _o))


with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(40):
        # training
        for _input, _target in train_data:
            _ = sess.run([train_opt], feed_dict={input: _input , target: _target })

        # validation
        total_loss=0.0
        hit=0
        for _input, _target in valid_data:
            _predict, _loss = sess.run([l3_sm, loss], feed_dict={input: _input , target: _target })
            total_loss+=_loss[0]
            if max_index(_input) == max_index(_predict):
                hit+=1

        print(hit, total_loss)