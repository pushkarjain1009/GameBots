import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

game_rows = rows = 3
game_cols = cols = 3
winning_length = 3
boardSize = rows * cols
actions = rows * cols
LW_1 = 750
LW_2 = 750
LW_3 = 750

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape,  stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01,  shape = shape)
    return tf.Variable(initial)

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


