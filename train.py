import model
import tensorflow as tf
from data_load import read_img_sets
import matplotlib.pyplot as plt

data_dir = '/media/cs/data/BreaKHis_v1/dataset'

img_size=64
colour_channels=3
batch_size=8
training_epochs=5
pltLoss = []
pltError = []
b = []
n = 0

mode = 1        # 0:train   1:predict
if mode == 0:
    data_dir_path = data_dir + '/train'
    max_steps = 1000000
    inner_loop = 10
else:
    data_dir_path = data_dir + '/predict'
    max_steps = 8
    inner_loop = 1

data, category_ref = read_img_sets(data_dir_path, img_size, validation_size=.2)

flat_img_size = model.flat_img_shape(img_size, colour_channels)

num_classes = len(category_ref)

x, y_true, keep_prob = model.variables(flat_img_size, num_classes)
logits = model.model(x, keep_prob, img_size, colour_channels, filter_size=3, neurons=2*img_size, num_classes=num_classes)
cost = model.calulate_cost(logits, y_true)
training_op = model.optimizer(cost)
accuracy = model.calculate_accuracy(logits, y_true)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

    # model.restore_or_initialize(sess, saver, checkpoint_dir)
    if mode == 0:  # train
        sess.run(init)

        # for epoch in range(training_epochs):
        for epoch in range(5):

            batch_count = int(data.train.num_examples / batch_size)
            batch_count = 5
            for i in range(batch_count):

                x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
                x_batch = x_batch.reshape(batch_size, flat_img_size)

                x_test_batch, y_test_batch, _, cls_test_batch = data.test.next_batch(batch_size)
                x_test_batch = x_test_batch.reshape(batch_size, flat_img_size)

                v_opt = sess.run(training_op,
                                      feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 0.5})

                if i % 1 == 0:
                    val_loss = sess.run(cost, feed_dict={x: x_test_batch, y_true: y_test_batch, keep_prob: 1.0})
                    acc = sess.run(accuracy, feed_dict={x: x_test_batch, y_true: y_test_batch, keep_prob: 1.0})
                    print('epoch = %s   total batch = %s    accuracy = %s   loss = %s' % (str(epoch), str(batch_count), str(acc), str(val_loss)))

                    pltLoss.append(val_loss)
                    pltError.append(acc)
                    b.append(n)
                    n = n + 1

        saver.save(sess, './model/model.ckpt')

        plt.plot(b, pltLoss)
        plt.show()
        plt.plot(b, pltError)
        plt.show()
    else:  # predict
        saver.restore(sess, './model/model.ckpt')

        batch_count = int(data.test.num_examples / batch_size)
        for i in range(batch_count):
            x_predict_batch, y_predict_batch, _, cls_predict_batch = data.train.next_batch(batch_size)
            x_predict_batch = x_predict_batch.reshape(batch_size, flat_img_size)

            acc = sess.run(accuracy, feed_dict={x: x_predict_batch, y_true: y_predict_batch, keep_prob: 1.0})

            print('accuracy = %s' % str(acc))

