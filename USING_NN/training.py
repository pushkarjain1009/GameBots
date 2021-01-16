import time
import tensorflow.compat.v1 as tf
import random
import numpy as np
from pathlib import Path
import os
import sys

tf.disable_v2_behavior()


game_rows = rows = 3
game_cols = cols = 3
winning_length = 3
boardSize = rows * cols
actions = rows * cols
Game_W = 0
Game_L = 0
Game_D = 0
LW_1 = 750
LW_2 = 750
LW_3 = 750


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape,  stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01,  shape = shape)
    return tf.Variable(initial)



epsilon = 1.0

GAMMA = 0.9

def InverseBoard(board):
    temp_board = np.copy(board)
    rows,  cols = temp_board.shape
    for r in range(rows):
        for c in range(cols):
            temp_board[r, c] *= -1
    return temp_board.reshape([-1])

def isGameOver(board):
    temp = None
    rows ,  cols = board.shape

    for i in range(rows):
        temp = getRowSum(board,  i)
        if checkValue(temp):
            return True

    for i in range(cols):
        temp = getColSum(board,  i)
        if checkValue(temp):
            return True

    temp = getRightDig(board)
    if checkValue(temp):
        return True

    temp = getLeftDig(board)
    if checkValue(temp):
        return True

    return False

def getRowSum(board ,  r):
    rows ,  cols = board.shape
    sum = 0
    for c in range(cols):
        sum = sum + board[r, c]
    return sum

def getColSum(board ,  c):
    rows ,  cols = board.shape
    sum = 0
    for r in range(rows):
        sum = sum + board[r, c]
    return sum

def getLeftDig(board):
    rows ,  cols = board.shape
    sum = 0
    for i in range(rows):
        sum = sum + board[i, i]
    return sum

def getRightDig(board):
    rows ,  cols = board.shape
    sum = 0
    i = rows - 1
    j = 0
    while i >= 0:
        sum += board[i, j]
        i = i - 1
        j = j + 1
    return sum

def checkValue(sum):
    if sum == -3 or sum == 3:
        return True


def Model():

    L1_W = weight_variable([boardSize,  LW_1])
    L1_B = bias_variable([LW_1])

    L2_W = weight_variable([LW_1,  LW_2])
    L2_B = bias_variable([LW_2])

    L3_W = weight_variable([LW_2,  LW_3])
    L3_B = bias_variable([LW_3])

    OL_W = weight_variable([LW_3,  actions])
    OL_B  = bias_variable([actions])

    # input Layer
    X = tf.placeholder("float",  [None,  boardSize])

    # hidden layers
    HL_1 = tf.nn.relu(tf.matmul(X, L1_W) + L1_B)
    HL_2 = tf.nn.relu(tf.matmul(HL_1, L2_W) + L2_B)
    HL_3 = tf.nn.relu(tf.matmul(HL_2, L3_W) + L3_B)

    # output layer
    Y = tf.matmul(HL_3, OL_W) + OL_B
    pre = tf.argmax(Y[0])

    return X,  Y,  pre


def Train():
    print()

    x ,  y,  pre = Model()

    targety = tf.compat.v1.placeholder("float", [None, actions])
    loss =  tf.reduce_mean(tf.square(tf.subtract(targety,  y)))

    Min = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess = tf.InteractiveSession()

    sav = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    step = 0
    iterations = 0

    checkpoint = tf.train.get_checkpoint_state("model")
    if checkpoint and checkpoint.model_checkpoint_path:
        s = sav.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded the model:",  checkpoint.model_checkpoint_path)
        step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
    else:
        print("Could not find old network weights")
    iterations += step

    print(time.ctime())

    tot_matches = 150000
    number_of_matches_each_episode = 500
    max_iterations = tot_matches / number_of_matches_each_episode
    
    e_downrate = 0.9 / max_iterations

    print("e down rate is ", e_downrate)

    e = epsilon

    print("max iteration = {}".format(max_iterations))
    print()
    
    run_time = 0
    while True:
        start_time = time.time()
        episodes = number_of_matches_each_episode
        global Game_W
        global Game_L
        global Game_D

        total_loss = 0

        epchos = 100
        GamesList = []

        for i in range(episodes):
            completeGame,  victory = Play(e, sess, x,  pre, y)
            GamesList.append(completeGame)
            

        for k in range(epchos):
            random.shuffle(GamesList)
            for i in GamesList:
                len_complete_game = len(i)
                loop_in = 0
                game_reward = 0
                while loop_in < len_complete_game:
                    j = i.pop()
                    currentState = j[0]
                    action = j[1][0]
                    reward = j[2][0]
                    nextState = j[3]

                    if loop_in == 0:
                        game_reward = reward
                    else:
                        nextQ = sess.run(y, feed_dict = {x:[nextState]})
                        maxNextQ = np.max(nextQ)
                        game_reward = GAMMA * ( maxNextQ )

                    
                    targetQ = sess.run(y, feed_dict = {x:[currentState]})

                    for index, item in enumerate(currentState):
                        if item != 0:
                            targetQ[0, index] = -1

                    targetQ[0, action] = game_reward

                    loop_in += 1
                    t_loss = 0
                    
                    t_loss = sess.run([Min, y, loss], feed_dict = {x:[currentState],  targety:targetQ})
                    total_loss += t_loss[2]

        iterations += 1
        time_diff = time.time()-start_time
        run_time += time_diff
        print("iteration {} completed with {} wins,  {} losses {} draws,  out of {} games played,  e is {} \ncost is {} ,  current_time is {},  time taken is {} ,  total time = {} hours \n".format(iterations, 
        Game_W, Game_L, Game_D, episodes, e*100, total_loss, time.ctime(), time_diff, (run_time)/3600))
        start_time = time.time()
        total_loss = 0
        Game_W = 0
        Game_L = 0
        Game_D = 0

        if e > -0.2:
            e -= e_downrate
        else:
             e = random.choice([0.1, 0.05, 0.06, 0.07, 0.15, 0.03, 0.20, 0.25, 0.5, 0.4])

        sav.save(sess,  "./model/model.ckpt", global_step = iterations)


def Play(e, sess, x,  pre,  y):
    global Game_W
    global Game_L
    global Game_D

    win_reward = 10
    loss_reward = -1
    draw_reward = 3

    completeGameMemory = []
    myList = np.array([0]*(rows*cols)).reshape(3, 3)

    turn = random.choice([1, -1])

    if(turn == -1):
        initial_index = random.choice(range(9))
        best_index,  _= sess.run([pre, y],  feed_dict = {x : [np.array(np.copy(myList).reshape(-1))]})
        initial_index = random.choice([best_index, initial_index, best_index])
        myList[int(initial_index/3), initial_index%3] = -1
        turn = turn * -1

    while(True):

        memory = []

        temp_copy = np.array(np.copy(myList).reshape(-1))

        zero_indexes = []
        for index, item in enumerate(temp_copy):
            if item == 0:
                zero_indexes.append(index)

        if len(zero_indexes) == 0:
            reward = draw_reward
            completeGameMemory[-1][2][0] = reward
            Game_D += 1
            break

        selectedRandomIndex = random.choice(zero_indexes)

        pred,  _ = sess.run([pre, y],  feed_dict = {x : [temp_copy]})

        isFalsepre = False if temp_copy[pred] == 0 else True

        memory.append(np.copy(myList).reshape(-1))

        if random.random() > e: 
            action = pred
        else: 
            random_action = random.choice(range(9))
            action = selectedRandomIndex
            
        memory.append([action])

        if action not in zero_indexes:
            reward = loss_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            Game_L +=1
            break
        
        myList[int(action/game_rows), action%game_cols] = 1

        reward = 0

        if isFalsepre == True and action == pred:
            reward = loss_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            Game_L +=1
            break

        if(isGameOver(myList)):
            reward = win_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            Game_W +=1
            break

        temp_copy_inverse = np.array(np.copy(InverseBoard(myList)).reshape(-1))
        temp_copy = np.array(np.copy(myList).reshape(-1))
        zero_indexes = []
        for index, item in enumerate(temp_copy):
            if item == 0:
                zero_indexes.append(index)

        if len(zero_indexes) == 0:
            reward = draw_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            Game_D+=1
            break

        selectedRandomIndex = random.choice(zero_indexes)
        pred,  _ = sess.run([pre, y],  feed_dict={x : [temp_copy_inverse]})
        isFalsepre = False if temp_copy[pred] == 0 else True

        action = None

        if(isFalsepre == True):
            action = random.choice([selectedRandomIndex])
        else:
            action = random.choice([selectedRandomIndex, pred, pred, pred, pred])

        temp_copy2 = np.copy(myList).reshape(-1)
        if temp_copy2[action] != 0:
            print("big time error here ", temp_copy2 ,  action)
            return
        
        myList[int(action/game_rows), action%game_cols] = -1

        if isGameOver(myList) == True:
            reward = loss_reward
            memory.append([reward])

            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            Game_L +=1
            break

        memory.append([0])
        memory.append(np.copy(myList.reshape(-1)))

        completeGameMemory.append(memory)

    return completeGameMemory, reward




Train()

  

