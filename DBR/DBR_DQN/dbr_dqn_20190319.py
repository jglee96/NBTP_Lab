import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from collections import deque
import DBR
import dqn
from typing import List

# Real world environnment
Ngrid = 100
N1 = Ngrid
N2 = Ngrid
dx = 5
epsi = 12.25
eps0 = 1.

minwave = 400
maxwave = 1200
wavestep = 10
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 800

R = []
state = []

# Constants defining our neural network
INPUT_SIZE = Ngrid
OUTPUT_SIZE = Ngrid

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 10
MAX_EPISODES = 1000
MAX_ITERATION = 500

# Clear our computational graph
tf.reset_default_graph()

def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
    """Trains `mainDQN` with target Q values given by `targetDQN`

    Args:
        mainDQN (dqn.DQN): Main DQN that will be trained
        targetDQN (dqn.DQN): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]

    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)

def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`

    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)

    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
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

    with tf.Session() as sess:
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 10) + 1)
            state = np.zeros((1,Ngrid))
            prereward = 0
            done = False

            for iteridx in range(MAX_ITERATION):
                if np.random.rand(1) < e:
                    action = np.random.randint(Ngrid)
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                R = DBR.calR(state,Ngrid,wavelength,dx,epsi,eps0)
                rawreward = DBR.reward(Ngrid,wavelength,R,tarwave)
                reward = rawreward - prereward # the Q factor does not belong to action.
                next_state = DBR.step(state,action)
                if iteridx == MAX_ITERATION-1: done = True

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                if iteridx % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                state = next_state.copy()
    
    plt.subplot(2,1,1)
    plt.plot(wavelength,R)
    
    plt.subplot(2,1,2)
    plt.imshow(state,cmap='grays')
    plt.savefig('result model')
    
    plt.show()


if __name__ == "__main__":
    main()