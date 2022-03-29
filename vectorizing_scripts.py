import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
from time import time
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchsummary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def save_json(data, path):
    with open(path, 'w', encoding ='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii = True)


def read_json(path):
    with open(path, 'r', encoding ='utf8') as json_file:
        loader = json.load(json_file)
    return loader


class VectorizeContexts():
    def __init__(self):
        pass

    def load_model(self, model_name, device):
        self.model = SentenceTransformer(model_name, device=device)

    def vectorize_single_text(self, text):
        vector = self.model.encode([text])
        return vector

    def vectorize(self, dataset):
        self.vectorized_dataset = {}
        self.dataset = dataset
        for key, val in tqdm(dataset.items()):
            self.vectorized_dataset[key] = self.model.encode(val)

    def dump_vectorized_dataset(self, path):
        dataset_for_dump = {}
        for key, val in self.vectorized_dataset.items():
            list_of_vectors = [el.tolist() for el in val]
            dataset_for_dump[key] = list_of_vectors

        save_json(dataset_for_dump, path)

    def load_vectorized_dataset(self, path):
        loaded_dataset = read_json(path)
        self.vectorized_dataset = {}
        for key, val in loaded_dataset.items():
            self.vectorized_dataset[key] = np.array(val, dtype="float32")

        return self.vectorized_dataset


class Index:
    def __init__(self):
        self.corpus_text = None
        self.index = None

    def fill_index(self, corpus_text, corpus_embedding, clusters=None):
        self.corpus_text = corpus_text
        if clusters is not None:
            self.index = faiss.index_factory(corpus_embedding.shape[1], "IVF" + str(clusters) + "Flat",
                                             faiss.METRIC_INNER_PRODUCT)
            self.index.train(corpus_embedding)
            self.index.add(corpus_embedding)
        else:
            self.index = faiss.IndexFlatIP(corpus_embedding.shape[1])
            self.index.add(corpus_embedding)

    def retrieval(self, queries_embedding, answers_amount):
        distances, answer_indexes = self.index.search(queries_embedding, answers_amount)
        answers = list(map(lambda row: [self.corpus_text[element] for element in row], answer_indexes))
        return answers


def get_intersections_count(ret_target, ret_anchor, thresholds=[10, 30, 99]):
    cnt_intersections = 0
    res_intersections = []
    for ind, el in enumerate(ret_target):
        if el in ret_anchor:
            cnt_intersections += 1
        if ind in thresholds:
            res_intersections.append(cnt_intersections / ind)

    return np.array(res_intersections)

def create_features(X_train, vect_class_single, processed_contexts, vectorized_dataset, n_of_answers):
    features_matrix = []
    threshold = np.array([0.1, 0.3, 1])

    t1 = time()
    for ind, row in tqdm(X_train.iterrows()):
        single_anchor = row.anchor
        single_target = row.target
        context = row.context
        #y_single_true = y_train.loc[ind]

        vect_of_target = vect_class_single.vectorize_single_text(single_target)
        vect_of_anchor = vect_class_single.vectorize_single_text(single_anchor)

        texts_for_context = processed_contexts[context]
        vectors_for_context = vectorized_dataset[context]
        n_of_contexts = len(texts_for_context)

        feature1 = cosine(vect_of_target, vect_of_anchor)
        feature1 = np.array([feature1])

        index = Index()
        index.fill_index(texts_for_context, vectors_for_context)
        if n_of_contexts > n_of_answers:
            ret_target = index.retrieval(vect_of_target, n_of_answers)[0]
            ret_anchor = index.retrieval(vect_of_anchor, n_of_answers)[0]
            feature2 = get_intersections_count(ret_target, ret_anchor)
        else:
            ret_target = index.retrieval(vect_of_target, n_of_contexts)[0]
            ret_anchor = index.retrieval(vect_of_anchor, n_of_contexts)[0]
            small_threshold = threshold * n_of_contexts
            small_threshold = [int(el) for el in small_threshold]
            feature2 = get_intersections_count(ret_target, ret_anchor, thresholds=small_threshold)

        feature3 = np.array([n_of_contexts])

        features = np.concatenate((feature1, feature2, feature3))
        features_matrix.append(features)

    features_matrix = np.array(features_matrix)
    col = features_matrix[:, -1]
    mean_col = np.mean(col)
    std_col = np.std(col)
    col_red = (col - mean_col) / std_col
    features_matrix[:, -1] = col_red

    t2 = time()
    print("execution time:", t2 - t1)

    return features_matrix


class TinyNeuralNetwork(nn.Module):
    def __init__(self, device, model, need_weights_zeroing=True):
        super().__init__()
        self.device = device
        self.model = model
        self.model.to(self.device)

        if need_weights_zeroing:
            self.init_weights()

        self.init_criterion()
        self.init_optimizer()

    def init_weights(self):
        for ind, layer in enumerate(self.model):
            if ind % 2 == 0:
                torch.nn.init.xavier_uniform(layer.weight)

    def init_criterion(self, criterion=None):
        self.criterion = criterion or nn.CrossEntropyLoss()

    def init_optimizer(self, optimizer=None):
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, inp):
        out = self.model(inp)
        return out

    def train(self, train_loader, test_loader, epoch_number=2):
        self.loss_list = []
        self.epoch_list = []
        self.accuracy_train = []
        self.accuracy_test = []
        self.pearson_train = []
        self.pearson_test = []
        self.learning_time = []
        total_time = time()

        for epoch in range(epoch_number):  # loop over the dataset multiple times
            print("=========================")
            print(f"epoch: {epoch + 1}/{epoch_number}")
            print(f"processing...")
            running_loss = 0.0
            epoch_time = time()
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

            self.epoch_list.append(epoch)
            self.loss_list.append(running_loss)
            acc_train = self.accuracy(train_loader)
            acc_test = self.accuracy(test_loader)
            pearson_train = self.pearson_score(train_loader)
            pearson_test = self.pearson_score(test_loader)
            self.accuracy_train.append(acc_train)
            self.accuracy_test.append(acc_test)
            self.pearson_train.append(pearson_train)
            self.pearson_test.append(pearson_test)
            epoch_time_done = time() - epoch_time
            self.learning_time.append(epoch_time_done)

            print(f"running time: {round(epoch_time_done, 2)}")
            print(f"running loss: {round(running_loss, 2)}")
            print(f"train accuracy: {round(acc_train, 2)}")
            print(f"test accuracy: {round(acc_test, 2)}")
            print(f"train pearson: {round(pearson_train, 2)}")
            print(f"test pearson: {round(pearson_test, 2)}")


        print("=========================")
        print('Finished Training')

    def accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy_score = (100 * correct / total)
        return accuracy_score

    def pearson_score(self, test_loader):
        predicted_list = []
        true_list = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                predicted_list.append(predicted)
                true_list.append(labels)


        joined_predicted = predicted_list[0]
        joined_true = true_list[0]
        for ind, el in enumerate(predicted_list):
            if ind > 0:
                joined_predicted = torch.cat((joined_predicted, el))
        for ind, el in enumerate(true_list):
            if ind > 0:
                joined_true = torch.cat((joined_true, el))

        score = pearsonr(joined_true.to("cpu"), joined_predicted.to("cpu"))[0]
        return 100 * score

    def plot_train_history(self):
        plt.figure(figsize = (16,10))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(20, 12)

        ax1.plot(self.epoch_list, self.loss_list, "b", label="train loss")
        ax1.legend()
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("loss")

        ax2.plot(self.epoch_list, self.accuracy_train, "b", label="train accuracy")
        ax2.plot(self.epoch_list, self.accuracy_test, "g", label="test accuracy")
        ax2.plot(self.epoch_list, self.pearson_train, "r", label="train pearson")
        ax2.plot(self.epoch_list, self.pearson_test, "magenta", label="test pearson")
        ax2.legend()
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("accuracy")

        plt.show()