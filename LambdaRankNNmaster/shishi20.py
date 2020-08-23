import numpy as np
from LambdaRankNN import LambdaRankNN
from keras import backend as K

def lambdarank(entity_info,flag):
    if entity_info:
        if flag == 0:
            X = []
            print(entity_info)
            for i in entity_info:
                X.append(i[2:])
            X = np.array(X)
            print(X)
            ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
            ranker.model.load_weights('LambdaRankNNmaster/Examples/ccks20_entity.h5')
            y_pred = ranker.predict(X)
            K.clear_session()
            return y_pred
        else:
            X = []
            print(entity_info)
            for i in entity_info:
                X.append(i[5:])
            X = np.array(X)
            print(X)
            ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
            ranker.model.load_weights('LambdaRankNNmaster/Examples/ccks20_relation.h5')
            y_pred = ranker.predict(X)
            K.clear_session()
            return y_pred
    else:
        return entity_info

