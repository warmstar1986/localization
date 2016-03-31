import sys
import os
import time
import cPickle as pickle
import itertools

import numpy as np
import math
import pandas as pd
import theano
import theano.tensor as T
from keras.models import Graph, Sequential, model_from_json
from keras.layers import containers
from keras.layers.core import Dense, Dropout, Activation, Flatten, AutoEncoder
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adagrad
from keras.utils.visualize_util import plot

from utils import preprocess, make_vocab, compute_error

def gen_report(true_pts, pred_pts, network_name):
    tot_error = []
    for true_pt, pred_pt in zip(true_pts, pred_pts):
        tot_error.append(compute_error(pred_pt, true_pt))
    f_report = open('report.txt', 'a')
    f_report.write(network_name + '\n')
    f_report.write('Total Test size\t%d\n' % len(tot_error))
    tot_error = sorted(tot_error)
    f_report.write('Total Max error\t%f\n' % np.max(tot_error)) 
    f_report.write('Total Min error\t%f\n' % np.min(tot_error)) 
    f_report.write('Total Mean error\t%f\n' % np.mean(tot_error)) 
    f_report.write('Total Median error\t%f\n' % np.median(tot_error)) 
    f_report.write('Total 67%% error\t%f\n' % tot_error[int(len(tot_error) * 0.67)])
    f_report.write('Total 80%% error\t%f\n' % tot_error[int(len(tot_error) * 0.8)])
    f_report.write('Total 90%% error\t%f\n\n' % tot_error[int(len(tot_error) * 0.9)])

def pre_train(sda_layers, train_data):
    batch_size = 1
    nb_epoch = 15
    sda_layers = [train_data.shape[1]] + sda_layers
    trained_encoders, trained_decoders = [], []
    for i, (n_in, n_out) in enumerate(zip(sda_layers[:-1], sda_layers[1:]), start=1):
        print 'Pre-training Layer %d: %d --> %d' % (i, n_in, n_out)
        encoder = containers.Sequential([Dense(input_dim=n_in, output_dim=n_out, activation='tanh')])
        decoder = Dense(input_dim=n_out, output_dim=n_in, activation='tanh')
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)

        # Train an AutoEncoder
        ae = Sequential()
        ae.add(autoencoder)
        ae.compile(loss='mean_squared_error', optimizer='rmsprop')
        ae.fit(train_data, train_data, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
        trained_encoders.append(ae.layers[0].encoder)
        # Prepare input for next AutoEncoder
        autoencoder.output_reconstruction=False
        ae.compile(loss='mean_squared_error', optimizer='rmsprop')
        train_data = ae.predict(train_data)
    return trained_encoders

def build_mlp(n_con, n_dis, dis_dims, vocab_sizes, trained_encoders):
    emb_size=10
    hidden_size=500
    assert(n_dis == len(dis_dims) == len(vocab_sizes))
    # Define a graph
    network = Graph()

    # Input Layer
    input_layers = []
    network.add_input(name='con_input', input_shape=(n_con,))
    for i, encoder in enumerate(trained_encoders):
        if i == 0:
            network.add_node(encoder, name='encoder%d' % i, input='con_input')
        else:
            network.add_node(encoder, name='encoder%d' % i, input='encoder%d' % (i-1))
    input_layers.append('encoder%d' % (len(trained_encoders)-1))
    for i in range(n_dis):
        network.add_input(name='emb_input%d' % i, input_shape=(dis_dims[i],), dtype=int)
        network.add_node(Embedding(input_dim=vocab_sizes[i], output_dim=emb_size, input_length=dis_dims[i]), name='emb%d' % i, input='emb_input%d' % i)
        network.add_node(Flatten(), name='fla_emb%d' % i, input='emb%d' % i)
        input_layers.append('fla_emb%d' % i)
    
    # Hidden Layer
    network.add_node(layer=Dense(hidden_size, activation='relu'), name='hidden1', inputs=input_layers, merge_mode='concat')

    # Ouput Layer
    network.add_node(Dense(2), name='hidden2', input='hidden1')
    network.add_output(name='output', input='hidden2')

    return network

def permutate_to_augment(df):
    for i in range(len(df)):
        record = df.iloc[i]
        valid_idx = []
        for nei_id in range(1, 7):
            a = record.loc['All-Neighbor LAC (Sorted)[%d]'%nei_id]
            if not math.isnan(a):
                valid_idx.append(nei_id) 
             
def load_dataset(f_name, eng_para, augment=False):
    df = pd.read_csv(f_name)
    # Remove some records with missing values
    df = df[df['All-LAC'].notnull() & df['All-Cell Id'].notnull() & df['All-Longitude'].notnull() & df['All-Latitude'].notnull() & (df['Message Type'] == 'Measurement Report')]
    
    # Augment dataset by permutate the order of lac_ci
#    if augment:
#        df = permutate_to_augment(df)

    # Join with Engineer Parameter data
    df = df.merge(eng_para, left_on=['All-LAC', 'All-Cell Id'], right_on=['LAC', 'CI'], how='left')
    eng_para_ = eng_para.loc[:, ['LAC', 'CI', 'Longitude', 'Latitude']] # For all the neighbor, just join the longitude and latitude of BTSs
    for i in range(1, 7): 
        df = df.merge(eng_para_, left_on=['All-Neighbor LAC (Sorted)[%d]' % i, 'All-Neighbor Cell Id (Sorted)[%d]' % i], right_on=['LAC','CI'], how='left', suffixes=('', '_%d' % i))
        df = df.drop(['LAC_%d' % i, 'CI_%d' % i], axis=1)

    # Fill default number to NULL entries (-999 means very week)
    df = df.fillna(-999)

    # Select LABEL data (longitude and latitude)
    label = df.loc[:, ['All-Longitude', 'All-Latitude']].values
    # Select BTS location data to mesure distance between UE and BTS
    bts = df.loc[:, ['Longitude', 'Latitude']].values 

    # Deal with id features separately 
    # !!!(Here manipulate discrete feature separately, later can write a common version to deal with all the discrete feature by pass discrete feature names)
    # LAC, CI (Location Area Code, Cell Id)
    lacci_vals = [map(lambda x: '%.0f,%.0f' % (x[0], x[1]), df.loc[:, ['All-LAC', 'All-Cell Id']].values)]
    for nei_id in range(1, 7):
        lacci_vals.append(map(lambda x: '%.0f,%.0f' % (x[0], x[1]), 
                    df.loc[:, ['All-Neighbor LAC (Sorted)[%d]'%nei_id, 'All-Neighbor Cell Id (Sorted)[%d]'%nei_id]].values))
        df = df.drop(['All-Neighbor LAC (Sorted)[%d]'%nei_id, 'All-Neighbor Cell Id (Sorted)[%d]'%nei_id], axis=1)
    # BSIC (Base Station Identity Code), value range 0-63
    bsic_vals = [map(lambda x: '%.0f' % x, df.loc[:, 'All-BSIC (Num)'].values)]
    for nei_id in range(1, 7):
        bsic_vals.append(map(lambda x: '%.0f' % x, df.loc[:, 'All-Neighbor BSIC (Num) (Sorted)[%d]'%nei_id]))
        df = df.drop('All-Neighbor BSIC (Num) (Sorted)[%d]'%nei_id, axis=1)
    # ARFCN BCCH (Absolute Radio Frequency Channel Number, Broadcast Control Channel)
    arfcn_vals = [map(lambda x: '%.0f' % x, df.loc[:, 'All-ARFCN BCCH'].values)]
    for nei_id in range(1, 7):
        arfcn_vals.append(map(lambda x: '%.0f' % x, df.loc[:, 'All-Neighbor ARFCN (Sorted)[%d]'%nei_id]))
        df = df.drop('All-Neighbor ARFCN (Sorted)[%d]'%nei_id, axis=1)

    # Drop irrelevant columns
    df = df.drop(['All-Longitude', 'All-Latitude', 'Time', 'MS', 'Frame Number', 'Direction', 'Message Type', 'Event', 'EventInfo', 'All-LAC', 'All-Cell Id', 'LAC', 'CI', 'All-BSIC (Num)', 'All-ARFCN BCCH'], axis=1)

    # Compute distance between UE and BTS
    dist = []
    for bts_pt, true_pt in zip(bts, label):
        if bts_pt[0] > 0:
            dist.append(compute_error(bts_pt, true_pt))
    print 'Mean Distance\t%f' % np.mean(dist)
    print 'Median Distance\t%f' % np.median(dist)
    print 'Max Distance\t%f' % np.max(dist)

    return df.values, label, [lacci_vals, bsic_vals, arfcn_vals]

def main(num_epochs=500):
    # Load the dataset
    print 'Loading dataset ...'
    eng_para = pd.read_csv('data/2g_gongcan.csv')
#eng_para = eng_para.loc[:, ['LAC', 'CI', 'Angle', 'Longitude', 'Latitude', 'Power', 'GSM Neighbor Count', 'TD Neighbor Count']]
    tr_feature, tr_label, tr_ids = load_dataset('data/forward_recovered.csv', eng_para, True) 
    te_feature, te_label, te_ids = load_dataset('data/backward_recovered.csv', eng_para, False)
    ## !!! maybe here need to ensure train data are the same shape as test data
    train_size, n_con = tr_feature.shape
    test_size, n_con = te_feature.shape
    n_dis = len(tr_ids) 

    # Create neural network model
    print 'Preprocessing data ...'
    # Standardize continous input
    tr_feature, te_feature = preprocess(tr_feature, te_feature)
    tr_input = {'con_input' : tr_feature, 'output' : tr_label}
    te_input = {'con_input' : te_feature}
    # Prepare embedding input
    dis_dims, vocab_sizes = [], []
    for ii, tr_ids_, te_ids_ in zip(range(n_dis), tr_ids, te_ids): # make sure tr_ids contain several different discrete features
        vocab_size, vocab_dict = make_vocab(tr_ids_, te_ids_) 
        tr_id_idx_, te_id_idx_ = [], []
        dis_dim = len(tr_ids_)
        for i in range(dis_dim):
            tr_id_idx_ += map(lambda x: vocab_dict[x], tr_ids_[i])
            te_id_idx_ += map(lambda x: vocab_dict[x], te_ids_[i])
        tr_ids = np.array(tr_id_idx_, dtype=np.int32).reshape(dis_dim, train_size).transpose()
        te_ids = np.array(te_id_idx_, dtype=np.int32).reshape(dis_dim, test_size).transpose()

        ## Add discrete feature to dict
        tr_input['emb_input%d' % ii] = tr_ids
        te_input['emb_input%d' % ii] = te_ids

        dis_dims.append(dis_dim)
        vocab_sizes.append(vocab_size)

    print 'Building model and compiling functions ...'
    
    # Pre-training
    sda_layers = [40, 30, 20, 10]
    trained_encoders = pre_train(sda_layers, np.vstack((tr_feature, te_feature)).copy())
    
    network = build_mlp(n_con, n_dis, dis_dims, vocab_sizes, trained_encoders)

    network.compile(loss={'output': 'mean_squared_error'}, optimizer=Adagrad())

    plot(network, to_file='visualize/model_sda.png')

    # Build network

    network_name = 'MLP-Sda-0.1'

    mul_val = 10000.
    lon_offset = np.mean(tr_label[:, 0])
    lon_std = np.mean(tr_label[:, 0])
    lat_offset = np.mean(tr_label[:, 1])
    lat_std = np.mean(tr_label[:, 1])
    ######## Change Target
    tr_label[:, 0] = (tr_label[:, 0] - lon_offset) * mul_val 
    tr_label[:, 1] = (tr_label[:, 1] - lat_offset) * mul_val 
    tr_label = tr_label.astype(np.float32)

    is_train = True
    if is_train:
        build_log = network.fit(tr_input, nb_epoch=500, batch_size=10, verbose=1)
        # Dump Network
        json_string = network.to_json()
        open('model/' + network_name + '.json', 'w').write(json_string)
        network.save_weights('model/' + network_name + '.h5', overwrite=True)
    else:
        # Load Network
        network = model_from_json(open('model/' + network_name + '.json').read())
        network.load_weights('model/' + network_name + '.h5')

    # Make prediction
    te_pred = network.predict(te_input)['output']

    te_pred[:, 0] = te_pred[:, 0] / mul_val + lon_offset
    te_pred[:, 1] = te_pred[:, 1] / mul_val + lat_offset
    f_out = open('pred.csv', 'w')
    for pred_pt, true_pt in zip(te_pred, te_label):
        f_out.write('%f,%f,%f,%f\n' % (pred_pt[0], pred_pt[1], true_pt[0], true_pt[1]))

    # Generate report
    gen_report(te_label, te_pred, network_name)

if __name__ == '__main__':
    main()
