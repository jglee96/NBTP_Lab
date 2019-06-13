import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import drqn
import layerDBR
from datetime import datetime


minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
tarwave = 400
N_layer = 5
nh = 2.6811  # TiO2 at 400 nm (Siefke)
nl = 1.4701  # SiO2 at 400 nm (Malitson)

# Base data
lbound = int((tarwave/(4*nh))*0.8)
ubound = int((tarwave/(4*nl))*1.2)
print("======== Design Information ========")
print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
print('lbound: {}, ubound: {}'.format(lbound, ubound))

INPUT_SIZE = N_layer
action_size = 2 * N_layer + 1
h_size = 250
batch_size = 4
trace_length = 8
update_freq = 5
learning_rate = 1E-3
discount_factor = 0.99
num_episodes = 500
max_epLength = 100
startE = 1
endE = 0.1
anneling_steps = 5000
pre_train_steps = 5000

laod_model = False
result_path ='D:/NBTP_Lab/DBR/DBR_DRQN/result' 
model_path = 'D:/NBTP_Lab/DBR/DBR_DRQN/model'
train_path = 'D:/NBTP_Lab/DBR/DBR_DRQN/train'


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(
            tfVars[idx+total_vars//2].assign(
                (var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars//2].eval(session=sess)
    if a.all() == b.all():
        print("Targest Set Success")
    else:
        print("Target Set Failed")

def main():
    tf.reset_default_graph()
    load_model = False
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True)
    cellt = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True)
    mainQN = drqn.DRQN(INPUT_SIZE, h_size, cell, learning_rate, 'main')
    targetQN = drqn.DRQN(INPUT_SIZE, h_size, cellt, learning_rate, 'target')

    saver = tf.train.Saver()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, 0.6)
    myBuffer = drqn.Experience_Buffer()

    rList = []

    with tf.Session() as sess:
        if load_model == True:
            print("Loading Model")
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.initializers.global_variables())

        updateTarget(targetOps, sess)

        merged = tf.summary.merge_all()
        train_path_name = train_path + '/' + datetime.now().strftime('%Y%m%d%H%M')
        train_writer = tf.summary.FileWriter(train_path_name, sess.graph)

        total_steps = 0
        e = startE
        stepDrop = (startE - endE) / anneling_steps
        for i in range(num_episodes):
            episodeBuffer = []
            state = int((ubound + lbound) / 2) * np.ones(shape=(1, INPUT_SIZE))
            done = False
            rAll = 0
            rnn_state = (np.zeros(shape=(1, h_size)), np.zeros(shape=(1, h_size)))
            log_update = True
            for j in range(max_epLength):
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    rnn_next_state = sess.run(
                        mainQN.rnn_state,
                        feed_dict={
                            mainQN.X: state, mainQN.trainLength: 1,
                            mainQN.state_in: rnn_state,
                            mainQN.batch_size: 1})
                    if j < trace_length:
                        a = random.randint(0, action_size - 2)
                    else:
                        a = random.randint(0, action_size - 1)
                else:
                    a, rnn_next_state = sess.run(
                        [mainQN.predict, mainQN.rnn_state],
                        feed_dict={
                            mainQN.X: state, mainQN.trainLength: 1,
                            mainQN.state_in: rnn_state,
                            mainQN.batch_size: 1})
                    a = a[0]
                calstate = np.reshape(state, newshape=INPUT_SIZE)
                R = layerDBR.calR(
                    calstate, INPUT_SIZE, wavelength, nh, nl, True)
                next_calstate, reward, done = layerDBR.step(
                    R, tarwave, 50, wavelength, calstate, a,
                    INPUT_SIZE, lbound, ubound)
                next_state = np.reshape(next_calstate, newshape=(1, INPUT_SIZE))
                episodeBuffer.append(np.reshape(np.array([state, a, reward, next_state, done]), [1, 5]))
                total_steps += 1
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop

                    if total_steps % (update_freq*1000) == 0:
                        print("Target network updated")
                        updateTarget(targetOps, sess)
                    
                    if total_steps % (update_freq) == 0:
                        state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))

                        trainBatch = myBuffer.sample(batch_size, trace_length)

                        Q1 = sess.run(
                            mainQN.predict,
                            feed_dict={
                                mainQN.X: np.vstack(trainBatch[:, 3]),
                                mainQN.trainLength: trace_length,
                                mainQN.state_in: state_train,
                                mainQN.batch_size: batch_size})

                        Q2 = sess.run(
                            targetQN.Qout,
                            feed_dict={
                                targetQN.X: np.vstack(trainBatch[:, 3]),
                                targetQN.trainLength: trace_length,
                                targetQN.state_in: state_train,
                                targetQN.batch_size: batch_size})
                        
                        end_multiplier = -(trainBatch[:, 4] - 1)

                        doubleQ = Q2[range(batch_size * trace_length), Q1]
                        targetQ = trainBatch[:, 2] + (discount_factor * doubleQ * end_multiplier)
                        if (i+1) % 50 == 0 and log_update:
                            _, summary = sess.run(
                                [mainQN.updateModel, mainQN.loss_hist],
                                feed_dict={
                                    mainQN.X: np.vstack(trainBatch[:, 0]),
                                    mainQN.targetQ: targetQ,
                                    mainQN.actions: trainBatch[:, 1],
                                    mainQN.trainLength: trace_length,
                                    mainQN.state_in: state_train,
                                    mainQN.batch_size: batch_size})
                            train_writer.add_summary(summary, global_step=i)
                            log_update = False
                        else:
                            sess.run(
                                mainQN.updateModel,
                                feed_dict={
                                    mainQN.X: np.vstack(trainBatch[:, 0]),
                                    mainQN.targetQ: targetQ,
                                    mainQN.actions: trainBatch[:, 1],
                                    mainQN.trainLength: trace_length,
                                    mainQN.state_in: state_train,
                                    mainQN.batch_size: batch_size})

                if not done:
                    rAll = reward
                state = next_state
                rnn_state = rnn_next_state

                if done == True:
                    break

            if (i+1) % 10 == 0:
                print('{}th epidoes, {}th total step done...'.format(i+1, total_steps))

            bufferArray = np.array(episodeBuffer)
            episodeBuffer = bufferArray
            myBuffer.add(episodeBuffer)
            rList.append(rAll)
        saver.save(sess, model_path + '/drqn_model_' + datetime.now().strftime('%Y%m%d%H%M') + '.cptk')

        plt.figure(1)
        plt.plot(range(len(rList)), rList)
        fig1 = plt.gcf()
        plt.show()
        fig1_name = result_path + '/' + datetime.now().strftime("%Y%m%d%H")+'_rList.png'
        fig1.savefig(fig1_name)
if __name__ == "__main__":
    main()