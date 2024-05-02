from __future__ import print_function, division
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import Model
import tensorflow.compat.v1 as tf
from model import Model as Classifier
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_svmlight_file
from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate

from utils.io import load_jsonl

tf.compat.v1.disable_eager_execution()


class FCG():
    def __init__(self, model_name):
        self.apifeature_dims = 3514
        self.z_dims = 100  # noise appended at the end of example
        self.model_name = model_name

        self.hide_layers = 256
        self.generator_layers = [self.apifeature_dims + self.z_dims, self.hide_layers, self.apifeature_dims]
        self.substitute_detector_layers = [self.apifeature_dims, self.hide_layers, 1]
        self.blackbox, self.sess = self.build_blackbox_detector(self.model_name)
        self.optimizer = Adam(lr=0.001)

        # Build and compile the substitute_detector
        self.substitute_detector = self.build_substitute_detector()
        self.substitute_detector.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        input = [example, noise]
        malware_examples = self.generator(input)

        # For the combined model we will only train the generator
        self.substitute_detector.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.substitute_detector(malware_examples)

        # The combined model  (stacked generator and substitute_detector)
        self.combined = Model(input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_generator(self):

        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        x = Concatenate(axis=1)([example, noise])
        for dim in self.generator_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x)
        x = Maximum()([example, x])
        generator = Model([example, noise], x, name='generator')
        generator.summary()
        return generator

    def build_substitute_detector(self):

        input = Input(shape=(self.substitute_detector_layers[0],))
        x = input
        for dim in self.substitute_detector_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x)
        substitute_detector = Model(input, x, name='substitute_detector')
        substitute_detector.summary()
        return substitute_detector

    def build_blackbox_detector(self, model_name):
        PATH = "adv_trained/{}.ckpt".format(model_name)
        # Clear the current graph in each run, to avoid variable duplication
        tf.reset_default_graph()
        model = Classifier()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, PATH)
        print("load model from:", PATH)

        return model, sess

    def train(self, epochs, batch_size):

        model = self.blackbox
        sess = self.sess

        # Load test dataset (all malware)
        seed_dict = pickle.load(open('feat_dict.pickle', 'rb'), encoding='latin1')
        features = []
        sha1 = []
        dist_dict = {}  # [key]: hash [value]: L0 distance
        for key in seed_dict:
            seed_dict[key] = seed_dict[key].toarray()[0]
            features.append(seed_dict[key])
            sha1.append(key)
        feed_feat = np.stack(features)
        xtest_mal, ytest_mal = feed_feat, np.ones(len(feed_feat))

        # Load training dataset
        train_x, train_y = load_svmlight_file("data/train/fcg_dataset.csv",
                                              n_features=15033,
                                              multilabel=False,
                                              zero_based=False,
                                              query_id=False)

        test_x, test_y = load_svmlight_file("data/test/test.csv",
                                            n_features=10000,
                                            multilabel=False,
                                            zero_based=False,
                                            query_id=False)

        train_x = train_x.toarray()
        xtrain_ben = train_x[6896:]
        ytrain_ben = train_y[6896:]
        xtrain_mal = train_x[0:6896]

        # Since the training dataset is unbalanced, we randomly choose sample from benign dataset
        # and add them to the end to make up the gap
        idx = np.random.randint(0, xtrain_ben.shape[0], 6896 - 6294)
        add_on = xtrain_ben[idx]
        add_on_label = ytrain_ben[idx]
        xtrain_ben = np.concatenate((xtrain_ben, add_on), axis=0)
        ytrain_ben = np.concatenate((ytrain_ben, add_on_label), axis=0)

        Test_TPR = []
        d_loss_list, g_loss_list = [], []

        for epoch in range(epochs):

            # Each epoch goes through all the data in the training set
            start = 0

            for step in range(xtrain_mal.shape[0] // batch_size):
                # ---------------------
                #  Train substitute_detector
                # ---------------------

                xmal_batch = xtrain_mal[start: start + batch_size]
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))

                xben_batch = xtrain_ben[start: start + batch_size]
                start = start + batch_size

                # predict using blackbox detector
                yben_batch = sess.run(model.y_pred,
                                      feed_dict={model.x_input: xben_batch})

                # Generate a batch of new malware examples
                gen_examples = self.generator.predict([xmal_batch, noise])
                ymal_batch = sess.run(model.y_pred,
                                      feed_dict={model.x_input: np.ones(gen_examples.shape) * (gen_examples > 0.5)})

                # Train the substitute_detector
                d_loss_real = self.substitute_detector.train_on_batch(xben_batch, yben_batch)
                d_loss_fake = self.substitute_detector.train_on_batch(gen_examples, ymal_batch)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))
                g_loss = self.combined.train_on_batch([xmal_batch, noise], np.zeros((batch_size, 1)))

            # After each epoch, Evaluate evasion performance on the test dataset
            # try different noise for 3 times
            for j in range(3):
                noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
                gen_examples = self.generator.predict([xtest_mal, noise])

                TPR = sess.run(model.accuracy,
                               feed_dict={model.x_input: np.ones(gen_examples.shape) * (gen_examples > 0.5),
                                          model.y_input: np.ones(gen_examples.shape[0], )})

                Test_TPR.append(TPR)

                transformed_to_bin = np.ones(gen_examples.shape) * (gen_examples > 0.5)

                pred_y_label = sess.run(model.y_pred,
                                        feed_dict={model.x_input: np.ones(gen_examples.shape) * (gen_examples > 0.5)})

                # remove successfully evaded malware examples from xtest_mal
                i = 0
                while i < pred_y_label.shape[0]:
                    if pred_y_label[i] == 0:  # should be 1 but predict 0
                        # print(sha1[i], xtrain_mal[i])
                        # calculate L0 distance and put to dictionary
                        L0 = np.sum(transformed_to_bin[i]) - np.sum(xtest_mal[i])  # insertion only
                        dist_dict[sha1[i]] = L0  # [key]: hash [value]: L0 distance
                        xtest_mal = np.delete(xtest_mal, i, 0)
                        pred_y_label = np.delete(pred_y_label, i, 0)
                        sha1 = sha1[:i] + sha1[i + 1:]
                    else:
                        i += 1

                print("remaining malware examples:", xtest_mal.shape[0])
                if xtest_mal.shape[0] == 0:
                    break  # successful evade all

            # Print and record the progress
            print("[FCG] epoch(%d) [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                epoch + 1, d_loss[0], 100 * d_loss[1], g_loss))
            print("[Classifier] Test TPR on remaining test data: %f" % (Test_TPR[-1]))
            d_loss_list.append(d_loss[0])
            g_loss_list.append(g_loss)
            if xtest_mal.shape[0] == 0:
                break  # successful evade all

        sess.close()

        ERA = []
        success_num = 0
        # Calculate ERA for each L0 distance
        for i in range(3515):
            for key in dist_dict:
                if dist_dict[key] == i:
                    success_num += 1
            ERA.append((3435 - success_num) / 3435)

        # report ERA if not completely evaded
        if ERA[-1] != 0:
            print("{} is not completely evaded after {} epochs. ERA = {}".format(self.model_name, epochs, ERA[-1]))

        if self.model_name == 'baseline_checkpoint': curve_name = 'Baseline'
        if self.model_name == "baseline_adv_delete_one": curve_name = 'Adv Retrain A'
        if self.model_name == "robust_delete_one": curve_name = 'Robust A'
        if self.model_name == "baseline_adv_insert_one": curve_name = 'Adv Retrain B'
        if self.model_name == "robust_insert_one": curve_name = 'Robust B'
        if self.model_name == "baseline_adv_delete_two": curve_name = 'Adv Retrain C'
        if self.model_name == "robust_delete_two": curve_name = 'Robust C'
        if self.model_name == "baseline_adv_insert_rootallbutone": curve_name = 'Adv Retrain D'
        if self.model_name == "adv_keep_twocls": curve_name = 'Ensemble D Base Learner'
        if self.model_name == "robust_monotonic": curve_name = 'Robust E'
        if self.model_name == "baseline_adv_combine_two": curve_name = 'Adv Retrain A+B'
        if self.model_name == "adv_del_twocls": curve_name = 'Ensemble A+B Base Learner'
        if self.model_name == "robust_combine_two_v2_e18": curve_name = 'Robust A+B'
        if self.model_name == "robust_insert_allbutone": curve_name = 'Robust D'
        if self.model_name == "robust_combine_three_e17": curve_name = 'Robust A+B+E'

        model_df = pd.DataFrame(dict(ERA=np.asarray(ERA, dtype=np.float32),
                                     model=curve_name, L0=np.arange(3515)))
        return model_df


if __name__ == '__main__':
    model = FCG('adv_keep_twocls')
    df_model = model.train(epochs=50, batch_size=512)

    test_accuracy = df_model['test_accuracy'].value.copy().tolist()

    test_accuracy = np.mean(test_accuracy)

    data_test = load_jsonl('data/test/data.jsonl')

    labels = []

    print(f'Số lượng dữ liệu trên Tập Test là: {len(data_test)}')

    for id, datapoint in tqdm(enumerate(data_test)):
        labels.append(datapoint['label'])

    labels = np.array(labels)

    print(f'Accuracy trên Tập Test Của FCG Model: {np.sum(test_accuracy == labels)}')
