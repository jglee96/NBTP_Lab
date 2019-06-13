import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from collections import deque
import layerDBR
import dqn
from typing import List
from datetime import datetime

# Real world environnment
N_layer = 5
nh = 2.6811  # TiO2 at 400 nm (Siefke)
nl = 1.4701  # SiO2 at 400 nm (Malitson)

minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
tarwave = 400

# Base data
lbound = int((tarwave/(4*nh))*0.5)
ubound = int((tarwave/(4*nl))*1.5)
print("======== Design Information ========")
print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
print('lbound: {}, ubound: {}'.format(lbound, ubound))

# Constants defining our neural network
INPUT_SIZE = N_layer
OUTPUT_SIZE = 2 * N_layer + 1  # 1 for Do nothing

DISCOUNT_RATE = 0.99
BATCH_SIZE = 64
REPLAY_MEMORY = 100000  # usually use 1e6, ideally infinite
TARGET_UPDATE_FREQUENCY = 250
MAX_EPISODES = 500

SAVE_PATH = 'D:/NBTP_Lab/DBR/DBR_QN_slice/result'

# Clear our computational graph
tf.reset_default_graph()


def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list, board: bool) -> float:
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])

    X = states

    Q_target = rewards + DISCOUNT_RATE * np.max(
        targetDQN.predict(next_states), axis=1)

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    if board:
        return mainDQN.updatewTboard(X, y)
    else:
        return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def main():
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    # Create lists to contain total rewards and step count per episode
    rList = []
    sList = []

    with tf.Session() as sess:        
        # Network settings
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())  # Initialize variables
        mainDQN.writer.add_graph(sess.graph)        

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 100) + 1)
            # state = np.random.randint(
            #         low=int(lbound), high=int(ubound),
            #         size=Nslice, dtype=int)
            state = int((ubound + lbound) / 2) * np.ones(INPUT_SIZE)
            step_count = 0
            rAll = 0

            N_learn = 501
            for step_count in range(N_learn):
                if np.random.rand(1) < e:
                    action = np.random.randint(OUTPUT_SIZE)
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                R = layerDBR.calR(state, INPUT_SIZE, wavelength, nh, nl, True)
                reward = layerDBR.reward(R, tarwave, 100, wavelength)
                next_state = layerDBR.step(state, action, INPUT_SIZE)

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    if step_count % (TARGET_UPDATE_FREQUENCY) == 0:
                        summary, _ = replay_train(
                            mainDQN, targetDQN, minibatch, True)
                        mainDQN.writer.add_summary(
                            summary, global_step=episode)
                    else:
                        replay_train(
                            mainDQN, targetDQN, minibatch, False)

                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                rAll += reward
                state = next_state.copy()

            rList.append(rAll)
            sList.append(step_count)
#            print("Episodes: {}({:.2f}%), steps: {}".format(episode,100*(episode+1)/MAX_EPISODES,step_count))
            print("Episodes: {}({:.2f}%)".format(episode, 100 * (episode + 1) / MAX_EPISODES))

        # name for saveing neural network model
        save_file = './model/dqn_'+datetime.now().strftime("%Y%m%d%H")+'.ckpt'
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Save the model
        saver.save(sess, save_file)
        print('*****Trained Model Save(', save_file, ')*****')

    file_name = SAVE_PATH+'Rresult_'+datetime.now().strftime("%Y%m%d%H")+'.txt'
    f = open(file_name, 'w')
    print(R, file=f)
    f.close()

    file_name = SAVE_PATH+'Sresult_'+datetime.now().strftime("%Y%m%d%H")+'.txt'
    f = open(file_name, 'w')
    print(state, file=f)
    f.close()

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(range(len(rList)), rList)

    plt.subplot(2, 1, 2)
    plt.plot(range(len(sList)), sList)
    fig1 = plt.gcf()
    plt.show()
    fig1_name = SAVE_PATH+datetime.now().strftime("%Y-%m-%d-%H")+'_rList_sList.png'
    fig1.savefig(fig1_name)

    x = np.reshape(wavelength, wavelength.shape[1])
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(x, R)

    lx = np.arange(INPUT_SIZE)
    plt.subplot(2, 1, 2)
    plt.bar(lx, state, width=1, color='blue')        

    fig2 = plt.gcf()
    plt.show()
    fig2_name = SAVE_PATH+datetime.now().strftime("%Y%m%d%H")+'_result model.png'
    fig2.savefig(fig2_name)


if __name__ == "__main__":
    main()