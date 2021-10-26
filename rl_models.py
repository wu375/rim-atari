import numpy as np
import gym
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO_rnn(nn.Module):
    def __init__(
        self, 
        num_actions, 
        n_updates, 
        rnn_cell, 
        rnn_cell_args, 
        lr=2.5e-4, 
        device='cuda:0',
    ):
        super(PPO_rnn, self).__init__()
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=LR, rho=0.99, epsilon=1e-5)  # learning_rate=LR, decay=alpha, epsilon=epsilon)
        self.body = DQNBase()

        self.rnn_cell = rnn_cell(**rnn_cell_args)

        self.V_head = CnnV(input_size=6*128, hidden_size=512, num_actions=num_actions)
        self.PI_head = CnnHeadPolicy(input_size=6*128, hidden_size=512, num_actions=num_actions)

        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=1e-6, T_max=n_updates)

    def forward(self, x):
        return 'forward not implemented'

    def V(self, x, h, in_tensor=False):
        x = torch.FloatTensor(x).to(self.device).permute((0, 3, 1, 2)) / 255.
        x = self.body(x)
        x, h, masks = self.rnn_cell.forward_step(x, h)  # (batch, 6*128), tuple (batch, n_rims, hidden_size), (batch, n_rims)

        if in_tensor:
            return self.V_head(x), h
        return self.V_head(x).squeeze().detach().cpu().tolist(), h

    def PI(self, x, h, return_masks=False):
        x = torch.FloatTensor(x).to(self.device).permute((0, 3, 1, 2)) / 255.
        x = self.body(x)
        x, h, masks = self.rnn_cell.forward_step(x, h)

        # x = Dense(a_n, kernel_initializer=kernel_initializer, name='PI_dense')(iinput)
        # x = Softmax()(x) # redundant, you only care about who's the largest
        if return_masks:
            return self.PI_head(x), h, masks
        return self.PI_head(x), h

    def PI_and_V(self, x, h):
        x = x.permute((0, 3, 1, 2)) / 255.
        x = self.body(x)
        x, h, masks = self.rnn_cell.forward_step(x, h)
        PI = self.PI_head(x)
        V = self.V_head(x).squeeze()

        return PI, V, h

    def PI_and_V_sequence(self, x, h):
        x = x.permute((0, 1, 4, 2, 3)) / 255.
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch_size*seq_len, *x.shape[2:])
        x = self.body(x)
        x = x.view(batch_size, seq_len, -1)
        x, h, masks = self.rnn_cell(x, h=h, return_masks=True)
        PI = self.PI_head(x)
        V = self.V_head(x).squeeze()

        return PI, V, h

    def neglogp(self, p, a):
        return nn.CrossEntropyLoss(reduction='none')(p, a).detach().cpu().numpy().squeeze()

    # Mysterious function by openai
    def get_entropy(self, logits):  # (None, a_n)
        # don't know why do this max substraction
        # get one zero, other negatives. Normalize logits before softmax?
        # People on github says its for numerical stability
        a0 = logits - torch.max(logits, 1, keepdims=True)[0]


        ################### softmax on a0 ###################
        ea0 = torch.exp(a0)  # between 0 and 1
        z0 = torch.sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        #######################################################

        # This is mathematically equivalent to -p0*tf.log(p0),
        # don't know why written this way
        return torch.sum(p0 * (torch.log(z0) - a0), dim=1)

    def sample(self, x, get_neglogp=False):
        logits = self.PI(x)
        noise = torch.rand(logits.shape[0], logits.shape[1], device=self.device)
        # Gumbel softmax... But why??????
        action = torch.argmax(logits - torch.log(-torch.log(noise)), axis=1)
        if get_neglogp:
            return action.detach().cpu().tolist(), self.neglogp(logits, action)
        return action.detach().cpu().tolist()

    def sample_and_V(self, x, h, get_neglogp=False):
        x = torch.FloatTensor(x).to(self.device).permute((0, 3, 1, 2)) / 255.
        x = self.body(x)
        x, h, masks = self.rnn_cell.forward_step(x, h)

        logits = self.PI_head(x)
        V = self.V_head(x).squeeze().detach().cpu().tolist()

        noise = torch.rand(logits.shape[0], logits.shape[1], device=self.device)
        # Gumbel softmax... But why??????
        action = torch.argmax(logits - torch.log(-torch.log(noise)), axis=1)
        if get_neglogp:
            return action.detach().cpu().tolist(), self.neglogp(logits, action), h, V
        return action.detach().cpu().tolist(), h, V

    def choose_action(self, s, h):
        logits, h, masks = self.PI(s, h, return_masks=True)
        # actions = nn.Softmax(dim=-1)(logits)
        actions = torch.argmax(logits, dim=1).detach().cpu().numpy().squeeze()
        return actions, h, masks.cpu().numpy().squeeze()

    def get_losses(self, s, adv, a, R, old_neglogp, h, BETA=0.01, CLIPRANGE=0.1):  # (None, 84, 84, 4)
        # h = torch.FloatTensor(h).to(self.device)

        # the original implementation also clips value here
        # 0.5 is applied twice (one more time in total loss)
        policies, vpred, h = self.PI_and_V(s, h)

        # a = torch.LongTensor(a).to(self.device)
        # R = torch.FloatTensor(R).to(self.device)
        # adv = torch.FloatTensor(adv).to(self.device)
        # old_neglogp = torch.FloatTensor(old_neglogp).to(self.device)

        # CrossEntropyLoss does: softmax, negative log. So it's equivalent to tf sparse ce?
        neglogp = nn.CrossEntropyLoss(reduction='none')(policies, a)

        entropy = self.get_entropy(policies).mean(dim=-1)
        V_loss = torch.square(R - vpred).mean(dim=0) # *0.5

        # both are negative logs so this is actually pi/pi_old
        ratio = torch.exp(old_neglogp - neglogp)
        pg_losses1 = -adv * ratio
        pg_losses2 = -adv * torch.clamp(ratio, 1.0-CLIPRANGE, 1.0+CLIPRANGE)
        pg_loss = torch.maximum(pg_losses1, pg_losses2).mean(dim=0)

        total_loss = pg_loss + 0.5 * V_loss - BETA * entropy

        return total_loss, pg_loss, V_loss, entropy

    def optimize(self, S, A, R, masks, V, neglogp, H):
        self.train()
        advs = R - V
        advs = (advs - advs.mean()) / (advs.std() + 1e-8) # Normalize the advantages

        # spinningup implementation: 80 epochs updates with early stopping based on some kl. 
        # Buffer size is 4000, meaning 4000 env steps (in total for multiworkers) per epoch.
        # Looks like A3C style cpu-only implementation?

        # openai baseline implementation: 4 epochs
        # Each epoch is nenvs*nsteps = 16*128 (16 is my guess. nsteps is larger for mujoco at 2048) = 2048
        # A2C style

        # print('S ', S.shape)
        # print('A ', A.shape)
        # print('R ', R.shape)
        # print('V ', V.shape)
        # print('neglogp ', neglogp.shape)
        S = torch.LongTensor(S).to(self.device)
        A = torch.LongTensor(A).to(self.device)
        R = torch.FloatTensor(R).to(self.device)
        advs = torch.FloatTensor(advs).to(self.device)
        neglogp = torch.FloatTensor(neglogp).to(self.device)
        H = (torch.FloatTensor(H[0]).to(self.device), torch.FloatTensor(H[1]).to(self.device))

        n_epochs = 4
        n_total_steps = S.shape[0]
        batch_size = n_total_steps//4
        n_batches = n_total_steps//batch_size
        for _ in range(n_epochs):
            for i in range(n_batches):
                h = (H[0][i*batch_size:(i+1)*batch_size], H[1][i*batch_size:(i+1)*batch_size])
                total_loss, PI_loss, V_loss, entropy = \
                    self.get_losses(
                        S[i*batch_size:(i+1)*batch_size], 
                        advs[i*batch_size:(i+1)*batch_size], 
                        A[i*batch_size:(i+1)*batch_size], 
                        R[i*batch_size:(i+1)*batch_size], 
                        neglogp[i*batch_size:(i+1)*batch_size],
                        h,
                    )
                #######################
                # clip gradients here
                #######################
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
        
        curr_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()

        return PI_loss, V_loss, entropy, curr_lr

    def get_recurrent_losses(self, s, adv, a, R, old_neglogp, h, masks=None, BETA=0.01, CLIPRANGE=0.1):
        masks = masks.view(-1)

        policies, vpred, h = self.PI_and_V_sequence(s, h) # policies: (batch, time, n_a), V: (batch, time)

        policies = policies.view(policies.shape[0]*policies.shape[1], -1)
        a = a.view(a.shape[0]*a.shape[1])
        neglogp = nn.CrossEntropyLoss(reduction='none')(policies, a) # (batch*time, n_a)

        entropy = self.get_entropy(policies) * masks
        entropy = entropy.mean(dim=0)
        V_loss = torch.square(R.view(-1) - vpred.view(-1)) * masks
        V_loss = V_loss.mean(dim=0)

        old_neglogp = old_neglogp.view(-1)
        adv = adv.view(-1)
        # both are negative logs so this is actually pi/pi_old
        ratio = torch.exp(old_neglogp - neglogp)
        pg_losses1 = -adv * ratio
        pg_losses2 = -adv * torch.clamp(ratio, 1.0-CLIPRANGE, 1.0+CLIPRANGE)
        pg_loss = torch.maximum(pg_losses1, pg_losses2) * masks
        pg_loss = pg_loss.mean(dim=0)

        total_loss = pg_loss + 0.5 * V_loss - BETA * entropy

        return total_loss, pg_loss, V_loss, entropy

    def optimize_recurrent(self, sequences):
        self.train()
        advs = sequences['r'] - sequences['v']
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        S = torch.LongTensor(sequences['s']).to(self.device) # (n_seq, time, ...)
        A = torch.LongTensor(sequences['a']).to(self.device)
        R = torch.FloatTensor(sequences['r']).to(self.device)
        advs = torch.FloatTensor(advs).to(self.device)
        neglogp = torch.FloatTensor(sequences['neglogp']).to(self.device)
        HX = torch.FloatTensor(sequences['hx']).to(self.device) # (n_seq, ...)
        CX = torch.FloatTensor(sequences['cx']).to(self.device)
        masks = torch.FloatTensor(sequences['masks']).to(self.device) # (n_seq, time)

        # print(S.shape)
        # print(A.shape)
        # print(R.shape)
        # exit()

        n_epochs = 4
        n_total_seq = S.shape[0]
        batch_size = 64 # n_total_seq//8
        n_batches = int(np.ceil(n_total_seq / batch_size))

        for _ in range(n_epochs):
            for i in range(n_batches):
                batch_end_i = min((i+1)*batch_size, n_total_seq)
                h = (HX[i*batch_size:batch_end_i], CX[i*batch_size:batch_end_i])
                total_loss, PI_loss, V_loss, entropy = \
                    self.get_recurrent_losses(
                        S[i*batch_size:batch_end_i], 
                        advs[i*batch_size:batch_end_i], 
                        A[i*batch_size:batch_end_i], 
                        R[i*batch_size:batch_end_i], 
                        neglogp[i*batch_size:batch_end_i],
                        h,
                        masks[i*batch_size:batch_end_i],
                    )
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
        
        curr_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()

        return PI_loss, V_loss, entropy, curr_lr


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_weights(self):
        return utils.dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        return torch.reshape(x, (x.size(0), -1))

class DQNBase(nn.Module):

    def __init__(self, num_channels=4):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)
    
class CnnHeadPolicy(nn.Module):
    def __init__(self, num_actions, input_size=7*7*64, hidden_size=512):
        super(CnnHeadPolicy, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_actions))

        self.action_dim = num_actions
    def forward(self, x):
        #print('head', x.shape)
        logits = self.head(x)
        return logits
        

    def eval(self, x, deterministic=False, get_raw=False):

        with torch.no_grad():
            action_raw, _, _ = self.forward(x, get_log=False, deterministic=deterministic)
            action = torch.argmax(action_raw, dim=-1).squeeze().cpu().numpy()
            if get_raw:
                return action, action_raw.squeeze().cpu().numpy()
            return action
        
class CnnV(nn.Module):
    def __init__(self, num_actions, input_size=7*7*64, hidden_size=512):
        super(CnnV, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.head(x)