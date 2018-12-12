import os
import time

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class InnerNode():

    def __init__(self, depth, args):
        self.args = args
        self.fc = nn.Linear(self.args.input_dim, 1)
        beta = torch.randn(1)
        #beta = beta.expand((self.args.batch_size, 1))
        if self.args.cuda:
            beta = beta.cuda()
        self.beta = nn.Parameter(beta)
        self.leaf = False
        self.prob = None
        self.leaf_accumulator = []
        self.lmbda = self.args.lmbda * 2 ** (-depth)
        self.build_child(depth)
        self.penalties = []

    def reset(self):
        self.leaf_accumulator = []
        self.penalties = []
        self.left.reset()
        self.right.reset()

    def build_child(self, depth):
        if depth < self.args.max_depth:
            self.left = InnerNode(depth+1, self.args)
            self.right = InnerNode(depth+1, self.args)
        else :
            self.left = LeafNode(self.args)
            self.right = LeafNode(self.args)

    def forward(self, x):
        return(F.sigmoid(self.beta*self.fc(x)))
    
    def select_next(self, x):
        prob = self.forward(x)
        if prob < 0.5:
            return(self.left, prob)
        else:
            return(self.right, prob)

    def cal_prob(self, x, path_prob):
        self.prob = self.forward(x) #probability of selecting right node
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1-self.prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)

    def get_penalty(self):
        penalty = (torch.sum(self.prob * self.path_prob) / torch.sum(self.path_prob), self.lmbda)
        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return(self.penalties)


class LeafNode():
    def __init__(self, args):
        self.args = args
        self.param = torch.randn(self.args.output_dim)
        if self.args.cuda:
            self.param = self.param.cuda()
        self.param = nn.Parameter(self.param)
        self.leaf = True
        self.softmax = nn.Softmax()

    def forward(self):
        return(self.softmax(self.param.view(1,-1)))

    def reset(self):
        pass

    def cal_prob(self, x, path_prob):
        Q = self.forward()
        #Q = Q.expand((self.args.batch_size, self.args.output_dim))
        Q = Q.expand((path_prob.size()[0], self.args.output_dim))
        return([[path_prob, Q]])


class SoftDecisionTree(nn.Module):

    def __init__(self, args):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.root = InnerNode(1, self.args)
        self.collect_parameters() ##collect parameters and modules under root node
        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.test_acc = []
        self.define_extras(self.args.batch_size)
        self.best_accuracy = 0.0

    def define_extras(self, batch_size):
        ##define target_onehot and path_prob_init batch size, because these need to be defined according to batch size, which can be differ
        self.target_onehot = torch.FloatTensor(batch_size, self.args.output_dim)
        self.target_onehot = Variable(self.target_onehot)
        self.path_prob_init = Variable(torch.ones(batch_size, 1))
        if self.args.cuda:
            self.target_onehot = self.target_onehot.cuda()
            self.path_prob_init = self.path_prob_init.cuda()
    '''
    def forward(self, x):
        node = self.root
        path_prob = Variable(torch.ones(self.args.batch_size, 1))
        while not node.leaf:
            node, prob = node.select_next(x)
            path_prob *= prob
        return node()
    '''        
    def cal_loss(self, x, y):
        batch_size = y.size()[0]
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = [torch.zeros(self.args.output_dim) for _ in range(batch_size)]
        for (path_prob, Q) in leaf_accumulator:
            TQ = torch.bmm(y.view(batch_size, 1, self.args.output_dim), torch.log(Q).view(batch_size, self.args.output_dim, 1)).view(-1,1)
            loss += path_prob * TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 *(torch.log(penalty) + torch.log(1-penalty))
        output = torch.stack(max_Q)
        self.root.reset() ##reset all stacked calculation
        return(-loss + C, output) ## -log(loss) will always output non, because loss is always below zero. I suspect this is the mistake of the paper?

    def collect_parameters(self):
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        while nodes:
            node = nodes.pop(0)
            if node.leaf:
                param = node.param
                self.param_list.append(param)
            else:
                fc = node.fc
                beta = node.beta
                nodes.append(node.right)
                nodes.append(node.left)
                self.param_list.append(beta)
                self.module_list.append(fc)

    def train_(self, train_loader, epoch):
        self.train()
        self.define_extras(self.args.batch_size)
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.view(self.args.batch_size,-1)
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            self.optimizer.zero_grad()

            loss, output = self.cal_loss(data, self.target_onehot)
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct / len(data)

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0],
                    correct, len(data),
                    accuracy))

    def test_(self, test_loader, epoch):
        self.eval()
        self.define_extras(self.args.batch_size)
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            _, output = self.cal_loss(data, self.target_onehot)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
            correct, len(test_loader.dataset),
            accuracy))
        self.test_acc.append(accuracy)

        if accuracy > self.best_accuracy:
            self.save_best('./result')
            self.best_accuracy = accuracy

    def save_best(self, path):
        try:
            os.makedirs('./result')
        except:
            print('directory ./result already exists')

        with open(os.path.join(path, 'best_model.pkl'), 'wb') as output_file:
            pickle.dump(self, output_file)
