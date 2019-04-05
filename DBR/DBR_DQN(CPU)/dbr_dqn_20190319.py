import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from collections import deque
import DBR
import dqn
from typing import List
from datetime import datetime

# Real world environnment
Ngrid = 150
dx = 10
epsi = 12.25
eps0 = 1.

minwave = 400
maxwave = 1200
wavestep = 10
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 800

# Constants defining our neural network
INPUT_SIZE = Ngrid
OUTPUT_SIZE = 2*Ngrid

DISCOUNT_RATE = 0.99
BATCH_SIZE = 64
REPLAY_MEMORY = 100000 # usually use 1e6, ideally infinite
TARGET_UPDATE_FREQUENCY = 100
MAX_EPISODES = 5000

# Clear our computational graph
tf.reset_default_graph()

def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list, board: bool) -> float:
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
    if board:
        return mainDQN.updatewTboard(X, y)
    else:
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
    # Create lists to contain total rewards and step count per episode
    rList = []
    sList = []
    
    with tf.Session() as sess:        
        # Network settings
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer()) # Initialize Tensorflow variables
        mainDQN.writer.add_graph(sess.graph)        

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 100) + 1)
#            state = np.random.randint(2,size=(1,Ngrid))
#            state = np.zeros((1,Ngrid))
            state = np.ones(Ngrid)
            prereward = 0
            step_count = 0
            rAll = 0
            done = False

            while not done:
                if np.random.rand(1) < e:
                    action = np.random.randint(Ngrid)
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                R = DBR.calR(state,Ngrid,wavelength,dx,epsi,eps0)
                rawreward = DBR.reward(Ngrid,wavelength,R,tarwave)
                reward = rawreward - prereward # the Q factor does not belong to action.
                next_state = DBR.step(state,action,Ngrid)
                if rawreward < 0: done = True

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    if step_count % (TARGET_UPDATE_FREQUENCY/2) == 0:
                        summary, loss, _ = replay_train(mainDQN, targetDQN, minibatch, True)
                        mainDQN.writer.add_summary(summary, global_step=episode)
                    else:
                        loss, _ = replay_train(mainDQN, targetDQN, minibatch, False)

                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                rAll += reward
#                prereward = rawreward
                state = next_state.copy()
                step_count += 1
                
                if step_count > Ngrid:
                    break
                
            rList.append(rAll)
            sList.append(step_count)
            print("Episodes: {}({:.2f}%), steps: {}".format(episode,100*(episode+1)/MAX_EPISODES,step_count))
        
        # name for saveing neural network model
        save_file = './model/dqn_'+datetime.now().strftime("%Y%m%d%H")+'.ckpt'
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Save the model
        saver.save(sess, save_file)
        print('*****Trained Model Save(',save_file,'*****')
    
    file_name = 'Rresult_'+datetime.now().strftime("%Y%m%d%H")+'.txt'
    f = open(file_name,'w')
    print(R,file=f)
    f.close()
    
    file_name = 'Sresult_'+datetime.now().strftime("%Y%m%d%H")+'.txt'
    f = open(file_name,'w')
    print(state,file=f)
    f.close()
    
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(range(len(rList)),rList)
    
    plt.subplot(2,1,2)
    plt.plot(range(len(sList)),sList)
    fig1 = plt.gcf()
    plt.show()
    fig1_name = datetime.now().strftime("%Y-%m-%d-%H")+'_rList_sList.png'
    fig1.savefig(fig1_name)
    
    
    x = np.reshape(wavelength,wavelength.shape[1])
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(x,R)
    
    lx = np.arange(Ngrid)
    plt.subplot(2,1,2)
    plt.bar(lx,state,width=1,color='blue')        
    
    fig2 = plt.gcf()
    plt.show()
    fig2_name = datetime.now().strftime("%Y-%m-%d-%H")+'_result model.png'
    fig2.savefig(fig2_name)


if __name__ == "__main__":
    main()