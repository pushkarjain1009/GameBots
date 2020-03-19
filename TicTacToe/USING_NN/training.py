import numpy as np
from pathlib import Path
import os
import sys
import time 
import random
import tensorflow as tf

Board_size = 9
LW = 750
Game_W = Game_L = Game_D = 0

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.01))

def InverseBoard(board):
    temp_board = np.copy(board)
    rows, cols = temp_board.shape
    for r in range(rows):
        for c in range(cols):
            temp_board[r,c] *= -1
    return temp_board.reshape([-1])

def isGameOver(board):
    temp = None
    rows , cols = board.shape


    for i in range(rows):
        temp = getRowSum(board, i)
        if checkValue(temp):
            return True

    for i in range(cols):
        temp = getColSum(board, i)
        if checkValue(temp):
            return True


    temp = getRightDig(board)
    if checkValue(temp):
        return True

    temp = getLeftDig(board)
    if checkValue(temp):
        return True
    return False


def getRowSum(board , r):
    rows , cols = board.shape
    sum = 0
    for c in range(cols):
        sum = sum + board[r,c]
    return sum


def getColSum(board , c):
    rows , cols = board.shape
    sum = 0
    for r in range(rows):
        sum = sum + board[r,c]
    return sum

def getLeftDig(board):
    rows , cols = board.shape
    sum = 0
    for i in range(rows):
        sum = sum + board[i,i]
    return sum

def getRightDig(board):
    rows , cols = board.shape
    sum = 0
    i = rows - 1
    j = 0
    while i >= 0:
        sum += board[i,j]
        i = i - 1
        j = j + 1
    return sum

def checkValue(sum):
    if sum == -3 or sum == 3:
        return True


def Model():
    L1_W = weight_variable([Board_size, LW])
    L1_B = tf.Variable(tf.constant(0.01, [L1_W])

    L2_W = weight_variable([LW, LW])
    L2_B = tf.Variable(tf.constant(0.01, [L2_w]))

    L3_W = weight_variable([LW, LW])
    L3_B = tf.Variable(tf.constant(0.01, [L3_W]))

    OL_W = weight_variable([LW, 9])
    OL_B  = tf.Variable(tf.constant(0.01, [9])


    # input Layer
    X = tf.placeholder("float", [None, Board_size])

    # hidden layers
    HL_1 = tf.nn.relu(tf.matmul(x, LW_1) + LB_1)
    HL_2 = tf.nn.relu(tf.matmul(HL_1, LW_2) + LB_2)
    HL_3 = tf.nn.relu(tf.matmul(HL_2, LW_3) + LB_3)

    # output layer
    Y = tf.matmul(HL_3, OL_W) + OL_B
    prediction = tf.argmax(Y[0])

    return x,y, prediction

def Train():
    global Game_W
    global Game_L
    global Game_D

    x, y, pre = Model()

    target_y = tf.placeholder("float", [None, 9])
    loss = tf.reduce_mean(tf.square(tf.subtract(target_y, y)))

    Min = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess = tf.InteractiveSession()

    save = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    step = 0
    iterations = 0 

    no_of_matches = 70000
    no_of_matches_each_epi = 500
    max_itr = no_of_matches // no_of_matches_each_epoch

    e = 1.0
    e_downrate = 0.9 / max_itr

    run_time = 0
    while True:

        start_time = time.time()
        epi = no_of_matches_each_epi

        total_loss = 0

        epchos = 100
        GamesList = []

        for i in range(epi):
            comp, res = Play(e, sess, x, pre, y)
            GamesList.append(comp)
            

        for k in range(epchos):
            random.shuffle(GamesList)
            for i in GamesList:
                loop_in = 0
                game_reward = 0
                while loop_in < len(i):
                    j = i.pop()
                    currentState = j[0]
                    action = j[1][0]
                    reward = j[2][0]
                    nextState = j[3]

                    if loop_in == 0:
                        game_reward = reward
                    else:
                       
                        nexty = sess.run(y, feed_dict={inputState:[nextState]})
                        maxNexty = np.max(nexty)
                        game_reward = 0.9 * ( maxNexty )

                    
                    targety = sess.run(y, feed_dict={inputState:[currentState]})
                    
                    for index,item in enumerate(currentState):
                        if item != 0:
                            targety[0,index] = -1

                    targety[0,action] = game_reward

                    loop_in += 1
                    t_loss = 0
            
                    
                    t_loss=sess.run([train_step, y, loss],feed_dict={inputState:[currentState], target_y:targety})
                    total_loss += t_loss[2]

        iterations += 1
        time_diff = time.time()-start_time
        run_time += time_diff
        print("iteration {} completed with {} wins, {} losses {} draws, out of {} games played, e is {} \ncost is {} , current_time is {}, time taken is {} , total time = {} hours \n".format(iterations,
        Game_W, Game_L, Game_D, epi, e*100, total_loss, time.ctime(), time_diff, (run_time)/3600))
        start_time = time.time()
        total_loss = 0
        Game_W = 0
        Game_L = 0
        Game_D = 0


        if e > -0.2:
            e -= e_downrate
        else:
             e = random.choice([0.1,0.05,0.06,0.07,0.15,0.03,0.20,0.25,0.5,0.4])

        saver.save(sess, "./model/model.ckpt",global_step=iterations)


def Play (e, sess, x, pre, y):
    global Game_W
    global Game_L
    global Game_D

    win_reward = 10
    loss_reward = -1
    draw_reward = 3

    completeGameMemory = []
    myList = np.array([0]*(rows*cols)).reshape(3,3)

    turn = random.choice([1,-1])


    if(turn == -1):
        initial_index = random.choice(range(9))
        best_index, _= sess.run([pre, y], feed_dict={inputState : [np.array(np.copy(myList).reshape(-1))]})
        initial_index = random.choice([best_index, initial_index, best_index])
        myList[int(initial_index/3), initial_index%3] = -1
        turn = turn * -1

    while True:

        memory = []

        temp_copy = np.array(np.copy(myList).reshape(-1))

        zero_indexes = []
        for index,item in enumerate(temp_copy):
            if item == 0:
                zero_indexes.append(index)

        if len(zero_indexes) == 0:
            reward = draw_reward
            completeGameMemory[-1][2][0] = reward
            draw_games += 1
            break

        selectedRandomIndex = random.choice(zero_indexes)

        pred, _ = sess.run([pre , y], feed_dict={inputState : [temp_copy]})

        isFalsePrediction = False if temp_copy[pred] == 0 else True

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
            lost_games +=1
            break

        myList[int(action/game_rows), action%game_cols] = 1

        reward = 0

        if isFalsePrediction == True and action == pred:
            reward = loss_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            lost_games +=1
            break

        
        if(isGameOver(myList)):
            reward = win_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            won_games +=1
            break




        temp_copy_inverse = np.array(np.copy(InverseBoard(myList)).reshape(-1))
        temp_copy = np.array(np.copy(myList).reshape(-1))
        zero_indexes = []
        for index,item in enumerate(temp_copy):
            if item == 0:
                zero_indexes.append(index)

       
        if len(zero_indexes) == 0:
            reward = draw_reward
            memory.append([reward])
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            draw_games+=1
            break

        selectedRandomIndex = random.choice(zero_indexes)
        pred, _ = sess.run([prediction,Qoutputs], feed_dict={inputState : [temp_copy_inverse]})
        isFalsePrediction = False if temp_copy[pred] == 0 else True

        action = None

        if(isFalsePrediction == True):
            action = random.choice([selectedRandomIndex])
        else:
            action = random.choice([selectedRandomIndex,pred,pred,pred,pred])
        temp_copy2 = np.copy(myList).reshape(-1)
        if temp_copy2[action] != 0:
            print("big time error here ",temp_copy2 , action)
            return

        
        myList[int(action/game_rows),action%game_cols] = -1

        
        if isGameOver(myList) == True:
            reward = loss_reward
            memory.append([reward])
            #final state
            memory.append(np.copy(myList.reshape(-1)))
            completeGameMemory.append(memory)
            lost_games +=1
            break

        
        memory.append([0])
        memory.append(np.copy(myList.reshape(-1)))

        
        completeGameMemory.append(memory)

        return completeGameMemory,reward


Train()
