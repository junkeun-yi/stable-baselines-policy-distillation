# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

import gym
from utils.models import *
from torch.optim import Adam, SGD
import torch
from torch.autograd import Variable
import random
import pickle, gzip
from utils.math import get_wasserstein, get_kl
from utils.agent_pd_baselines import AgentCollection,load_env_and_model
import numpy as np
from copy import deepcopy

class Student(object):
    def __init__(self, args, optimizer=None):
        self.env, _ = load_env_and_model(args.env, args.algo, args.folder)

        num_inputs = self.env.observation_space.shape[0]*self.env.observation_space.shape[1]
        # num_actions = self.env.action_space.shape[0]
        num_actions = self.env.action_space.n
        self.training_batch_size = args.student_batch_size
        self.testing_batch_size = args.testing_batch_size
        self.loss_metric = args.loss_metric
        self.policy = Policy(num_inputs, num_actions, hidden_sizes=(args.hidden_size,) * args.num_layers)
        self.agents = AgentCollection([self.env], [self.policy], 'cpu', render=args.render, num_agents=1)
        if not optimizer:
            self.optimizer = Adam(self.policy.parameters(), lr=args.lr)

    def train(self, expert_data):
        batch = random.sample(expert_data, self.training_batch_size)
        # print(batch[0])
        states = torch.stack([torch.from_numpy(x[0]) for x in batch])
        states = states.reshape((self.training_batch_size, states.shape[1]*states.shape[2]))/255
        means_teacher = torch.stack([ torch.tensor([x[1].item(),0,0]) for x in batch]) # FIXME means of actions
        # fake_std = torch.from_numpy(np.array([1e-6]*len(means_teacher))) # for deterministic
        # #fake_std = torch.from_numpy(np.array([1e-6])) # FIXME
        stds_teacher = torch.stack([torch.tensor([1e-6,1e-6,1e-6]) for x in batch]) #FIXME
        

        means_student = self.policy.mean_action(states) # (1000, 3) 
        means_student = torch.mean(means_student) # FIXME a hack, has 3 actions but just averaged
        stds_student = self.policy.get_std(states) # (1000)
        stds_student = torch.mean(stds_student) # FIXME a hack, has 3 actions but just averaged
        if self.loss_metric == 'kl':
            loss = get_kl([means_teacher, stds_teacher], [means_student, stds_student])
        elif self.loss_metric == 'wasserstein':
            loss = get_wasserstein([means_teacher, stds_teacher], [means_student, stds_student])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def test(self):
        memories, logs = self.agents.collect_samples(self.testing_batch_size, exercise=True)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward

    def save(self, ckp_name):
        with gzip.open(ckp_name, 'wb') as f:
            pickle.dump(self.policy, f)

class Teacher(object):
    def __init__(self, envs, policies, args):
        self.envs = envs
        self.policies = policies
        self.expert_batch_size = args.sample_batch_size
        self.agents = AgentCollection(self.envs, self.policies, 'cpu', render=args.render, num_agents=args.agent_count)

    def get_expert_sample(self):
        return self.agents.get_expert_sample(self.expert_batch_size)


class TrainedStudent(object):
    def __init__(self, args, optimizer=None):
        self.env, _ = load_env_and_model(args.env, args.algo, args.folder)
        self.testing_batch_size = args.testing_batch_size

        self.policy = self.load(args.path_to_student)
        self.agents = AgentCollection([self.env], [self.policy], 'cpu', render=args.render, num_agents=1)

    def test(self):
        memories, logs = self.agents.collect_samples(self.testing_batch_size, exercise=True)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward

    def load(self, ckp_name):
        with gzip.open(ckp_name,'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data