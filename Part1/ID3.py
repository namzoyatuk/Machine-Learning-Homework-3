import numpy as np


# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}


# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label


class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via a Python array.
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...

    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """

        """
        Entropy calculations
        """

        # detect the labels and count them
        label_count = {}
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1

        entropy_value = 0.0
        total_instances = len(labels)

        # for each label, calculate the probability and
        # entropy of that label finally sum them up
        for label, count in label_count.items():
            probability = count / total_instances
            entropy_value -= probability * np.log2(probability)

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """
        total_instances = len(dataset)
        attribute_values = {}

        attribute_index = self.features.index(attribute)

        # create partitions on the dataset based on the attribute
        for i, data in enumerate(dataset):
            the_attribute_value = data[attribute_index]
            if the_attribute_value not in attribute_values:
                attribute_values[the_attribute_value] = {'data': [], 'labels': []}
            attribute_values[the_attribute_value]['data'].append(data)
            attribute_values[the_attribute_value]['labels'].append(labels[i])

        # calculate the weighted average of the partitions
        for value, partition in attribute_values.items():
            partition_size = len(partition['data'])
            weight = partition_size / total_instances
            entropy_of_partition = self.calculate_entropy__(partition['data'], partition['labels'])
            average_entropy += weight * entropy_of_partition

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        entropy_of_dataset = self.calculate_entropy__(dataset, labels)
        average_entropy = self.calculate_average_entropy__(dataset, labels, attribute)

        information_gain = entropy_of_dataset - average_entropy
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0
        """
            Intrinsic information calculations for a given attribute
        """
        total_instances = len(dataset)
        attribute_values = {}
        attribute_index = self.features.index(attribute)

        # create partitions on the dataset based on the attribute
        for data in dataset:
            the_attribute_value = data[attribute_index]
            if the_attribute_value not in attribute_values:
                attribute_values[the_attribute_value] = 0
            attribute_values[the_attribute_value] += 1

        for value, count in attribute_values.items():
            proportion = count / total_instances
            if proportion > 0:
                intrinsic_info -= proportion + np.log2(proportion)

        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """

        information_gain = self.calculate_information_gain__(dataset, labels, attribute)
        intrinsic_info = self.calculate_intrinsic_information__(dataset, labels, attribute)

        gain_ratio = information_gain / intrinsic_info if intrinsic_info != 0 else 0

        return gain_ratio

    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """

    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")
