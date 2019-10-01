import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import random
import os
import sys
from itertools import groupby
import argparse

import fifteenmer
import protfactor as pf
import seq2window as sw


model_dir = 'saved_models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

print(model_dir)

class ParseCommands(object):
    print('object : ', object)

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Pretends to be git',
            usage='''git <command> [<args>]
                The most commonly used git commands are:
                commit     Record changes to the repository
                fetch      Download objects and refs from another repository'''
        )

        parser.add_argument('command', help='Subcommand to run')
        '''
        parse_args defaults to [1:] for args, but you need to
        exclude the rest of the args too, or validation will fail
        '''

        args = parser.parse_args(sys.argv[1:2])
        print(args.command)

        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        self.args = parser.parse_args(sys.argv[1:2])
        #
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        parser = argparse.ArgumentParser(
            description='Record changes to the repository'
        )
        # prefixing the argument with -- means it's optional
        parser.add_argument('-pos',action='append',nargs='*')
        parser.add_argument('-neg',action='append',nargs='*')
        parser.add_argument('-window',type = int,default = 15)
        parser.add_argument('-maxLen',type = int,default = 100)
        parser.add_argument('-units',type = int,default = 150)
        parser.add_argument('-epochs',type = int,default = 1)
        parser.add_argument('-lr',type = float,default = 0.01)

        print(parser)
        args = parser.parse_args(sys.argv[2:])
        print('Running toxify train\n positive data:' , args.pos,'\n negative data:' , args.neg)
        self.args = args
        return(self.args)

    def predict(self):
        parser = argparse.ArgumentParser(
            description='Predicts venom probabilities'
        )
        # NOT prefixing the argument with -- means it's not optional
        parser.add_argument('sequences')

        parser.add_argument(
            '-model',
            type = str,
            default = os.path.abspath(__file__).replace("__init__.py",
                                                               "models/max_len_500/window_0/units_270/lr_0.01/epochs_50/models/saved_model"))
        args = parser.parse_args(sys.argv[2:])
        print(' >> ', args)
        print('Running toxify predict\n input data:' , args.sequences)
        self.args = args
        return(self.args)



def main():

    print(' >>> starting main')
    # print(fm.joke())
    # ParseCommands()
    tox_args = ParseCommands().args
    print(' -->', tox_args)
    # print(tox_args)
    if hasattr(tox_args,"sequences"):
        # print(tox_args.sequences)
        """
        HERE needs to be a new way of converting fasta proteins to atchley factors, seq2window funcs
        """
        predictions_dir = tox_args.sequences +"_toxify_predictions"
        model_dir = tox_args.model
        model_len = int(model_dir.split("max_len_")[1].split("/")[0])
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        protein_pd = sw.fa2pd(tox_args.sequences,0,model_len)
        fa_mat = []
        for seq  in protein_pd["sequences"]:
            fa_mat.append(sw.seq2atchley(seq,0,model_len))
        fa_np = np.array(fa_mat)
        # this will produce np array of fifteenmer seqs
        print("saving to ",predictions_dir+"/protein_vectors.npy")
        np.save(predictions_dir+"/protein_vectors.npy",fa_np)
        os.system("saved_model_cli run --dir "+model_dir+"  --tag_set serve --signature_def serving_default --inputs inputs="+predictions_dir+"/protein_vectors.npy  --outdir "+predictions_dir)
        prediction_np = np.load(predictions_dir+"/predictions.npy")
        print(prediction_np.shape,fa_np.shape)
        protein_pd["pred"] = prediction_np[:,0]
        print(list(protein_pd))
        # print(protein_pd.drop())

        protein_pd.drop(["sequences"],axis=1).to_csv(predictions_dir+"/predictions.csv",index=False)
        use15mer = False


        if use15mer:

            proteins = fm.ProteinWindows(tox_args.sequences)
            protein_15mer = proteins.data

            protein_vectors_np = pf.ProteinVectors(protein_15mer).data
            np.save(predictions_dir+"/protein_vectors.npy",protein_vectors_np)

            os.system("saved_model_cli run --dir "+model_dir+"  --tag_set serve --signature_def serving_default --inputs inputs="+predictions_dir+"/protein_vectors.npy  --outdir "+predictions_dir)

            prediction_np = np.load(predictions_dir+"/predictions.npy")
            prediction_15mer = np.hstack((protein_15mer,prediction_np))
            prediction_15mer_df = pd.DataFrame(prediction_15mer).drop(4,1)
            prediction_15mer_df.columns = [ 'header','15mer','sequence','venom_probability']
            columnsTitles=['header','15mer','venom_probability','sequence']
            prediction_15mer_df=prediction_15mer_df.reindex(columns=columnsTitles)
            prediction_15mer_outfile = predictions_dir+"/predictions_15mer.csv"
            prediction_15mer_df.to_csv(prediction_15mer_outfile,index=False)
            prediction_proteins = fm.regenerate(prediction_15mer_df)
            prediction_proteins_outfile = predictions_dir+"/predictions_proteins.csv"
            prediction_proteins.to_csv(prediction_proteins_outfile,index=False)


    # here we are given a list of positive fasta files and a list of negative fasta files
    elif hasattr(tox_args,"pos") and hasattr(tox_args,"neg"):
        max_seq_len = tox_args.maxLen
        window_size = tox_args.window
        N_units = tox_args.units
        lr = tox_args.lr
        epochs = tox_args.epochs
        # here we are given a list of positive fasta files and a list of negative fasta files
        print(tox_args.pos)
        print(tox_args.neg)
        (train_seqs,test_seqs) = sw.seqs2train(tox_args.pos,tox_args.neg,window_size,max_seq_len)
        print("TEST:")
        # print(test_seqs)

        training_dir = "training_data/max_len_" + str(max_seq_len) + "/window_"+str(window_size)+"/units_"+str(N_units)+"/lr_"+str(lr)
        if not os.path.exists(training_dir) or True:
            os.makedirs(training_dir)
            print("writing to: "+training_dir+"testSeqs.csv")
            test_seqs_pd = pd.DataFrame(test_seqs)
            if window_size:
                test_seqs_pd.columns = ['header', 'kmer','sequence','label']
            else:
                test_seqs_pd.columns = ['header', 'sequence','label']
            test_seqs_pd.to_csv(training_dir+"testSeqs.csv",index= False)

            test_mat = []
            test_label_mat = []
            for row in test_seqs:
                seq = row[-2]
                label = float(row[-1])
                # print(label)
                # print(bool(label),label,"row:",row)
                if label:
                    test_label_mat.append([1,0])
                else:
                    test_label_mat.append([0,1])
                test_mat.append(sw.seq2atchley(seq,window_size,max_seq_len))
            test_label_np = np.array(test_label_mat)
            test_np = np.array(test_mat)

            train_mat = []
            train_label_mat = []
            for row in train_seqs:
                seq = row[-2]
                train_mat.append(
                    sw.seq2atchley(seq,window_size,max_seq_len)
                )
                label = float(row[-1])
                if label:
                    train_label_mat.append([1,0])
                else:
                    train_label_mat.append([0,1])
            train_label_np = np.array(train_label_mat)
            train_np = np.array(train_mat)

            np.save(training_dir+"testData.npy",test_np)
            np.save(training_dir+"testLabels.npy",test_label_np)
            np.save(training_dir+"trainData.npy",train_np)
            np.save(training_dir+"trainLabels.npy",train_label_np)


        test_X = np.load(training_dir+"testData.npy")
        test_Y = np.load(training_dir+"testLabels.npy")
        train_X = np.load(training_dir+"trainData.npy")
        train_Y = np.load(training_dir+"trainLabels.npy")

        print("train_X.shape:",train_X.shape)
        print("train_Y.shape:",train_Y.shape)
        # Parameters
        n = train_X.shape[0]  # Number of training sequences
        print('Training samples ', n)
        n_test = train_Y.shape[0]  # Number of test sequences
        print('Testing samples ', n_test) #7352
        m = train_Y.shape[1]  # Output dimension
        print('Output dimension', m) #6
        d = train_X.shape[2]  # Input dimension
        print(' Input dimension;', d) #9
        T = train_X.shape[1]  # Sequence length

        '''
        ----------------- TF Model ---------------------
        '''

        # batch_size = 100
        # Learning rate

        # Placeholders
        inputs = tf.placeholder(tf.float32, [None, None, d])
        target = tf.placeholder(tf.float32, [None, m])

        # Network architecture

        rnn_units = tf.nn.rnn_cell.GRUCell(N_units)
        rnn_output, _ = tf.nn.dynamic_rnn(rnn_units, inputs, dtype=tf.float32)

        # Ignore all but the last timesteps
        last = tf.gather(rnn_output, T - 1, axis=1)

        # Fully connected layer
        logits = tf.layers.dense(last, m, activation=None)
        # Output mapped to probabilities by softmax
        prediction = tf.nn.softmax(logits)
        # Error function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=target, logits=logits))
        # 0-1 loss; compute most likely class and compare with target
        accuracy = tf.equal(tf.argmax(logits, 1), tf.argmax(target, 1))
        # Average 0-1 loss
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        # Optimizer
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", accuracy)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        model_dir = training_dir+"models"


        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            summary_writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
                i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            else:
                print('No checkpoint file found!')
                i_stopped = 0

            #tf.saved_model.simple_save(sess, model_dir+"/saved_model/", inputs={"inputs":inputs,"target":target},outputs={"predictions":prediction})
            #sess.graph.finalize()

            # Do the learning
            for i in range(epochs):
                sess.run(
                    train_step,
                    feed_dict={
                    inputs: train_X,
                    target: train_Y
                })
                _, c, summary = sess.run(
                    [train_step, loss, merged_summary_op],
                    feed_dict={inputs: train_X, target: train_Y}
                )
                summary_writer.add_summary(
                    summary,
                    i
                )

                if (i + 1) % 10 == 0:
                    tmp_loss, tmp_acc = sess.run(
                        [loss, accuracy],
                        feed_dict=
                        {inputs: train_X, target: train_Y}
                    )
                    tmp_acc_test = sess.run(
                        accuracy,
                        feed_dict={
                            inputs: test_X,
                            target: test_Y
                        }
                    )
                    print(i + 1, 'Loss:', tmp_loss, 'Accuracy, train:', tmp_acc, ' Accuracy, test:', tmp_acc_test)

                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(
                        sess,
                        checkpoint_path,
                        global_step=i
                    )
            tf.saved_model.simple_save(
                sess,
                model_dir+"/saved_model/",
                inputs={"inputs":inputs,"target":target},
                outputs={"predictions":prediction}
            )
            sess.graph.finalize()

        # if False:
        #     with tf.Session() as sess:
        #       # Restore variables from disk.
        #       saver.restore(sess, model_dir+"/model.ckpt")
        #       print("Model restored.")
        #       # Check the values of the variables
        #       print("v1 : %s" % v1.eval())
        #       print("v2 : %s" % v2.eval())

main()
