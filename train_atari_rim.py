import os
import numpy as np
import argparse
import gym
from subproc_vec_env import SubprocVecEnv
from atari_env import make_atari, wrap_deepmind
import time
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from rl_models import PPO_rnn
from rim import RIM
from torch.utils.tensorboard import SummaryWriter

nenvs = 16
GAMMA = 0.99
LAM = 0.95
total_timesteps = int(60e6)

DEVICE = 'cuda:0'

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_logging():
    os.makedirs("train_log", exist_ok=True)
    i = 0
    while os.path.isdir('train_log/run'+str(i)):
        i += 1
    log_dir = 'train_log/run'+str(i)+'/'
    return log_dir

def train(env, env_name, num_actions, nsteps, log_dir):
    writer = SummaryWriter(log_dir)
    [print() for _ in range(50)]

    nbatch = nenvs * nsteps
    
    T = total_timesteps // nbatch

    print('num_actions: ', num_actions)

    rim_args = {
        'input_size':7*7*64,
        'hidden_size':6*128,
        'output_size':None,
        'n_rims':6,
        'top_k':4,
        'n_slots':1,
        'n_heads':4,
        'key_size':64,
        'value_size':32,
        'com_n_heads':4,
        'com_key_size':32,
        'com_value_size':32,
        'num_attention_layers':1,
        'attention_mlp_layers':2,
        'dynamics_type':'lstm',
        'com_attention_mlp_layers':2,
        'com_dropout':0.1,
        'input_dropout':0.1,
        'categorical':False,
    }

    model = PPO_rnn(num_actions=num_actions, n_updates=T, rnn_cell=RIM, rnn_cell_args=rim_args)
    model.cuda()

    #T = 1000000 # the sample code train T=1e6 (1000000) with 16 environments
    tstart = time.time()
    is_recurrent = True
    runner = Runner(env=env, model=model, is_recurrent=is_recurrent)


    try:
        for t in tqdm(range(1, T+1)):
            if is_recurrent:
                sequences = runner.run(nsteps=nsteps)
                V = sequences['v'].flatten()
                R = sequences['r'].flatten()
                pi_loss, v_loss, entropy, lr = model.optimize_recurrent(sequences)
            else:
                S, A, R, masks, V, neglogp, H = runner.run(nsteps=nsteps)
                pi_loss, v_loss, entropy, lr = model.optimize(S, A, R, masks, V, neglogp, H)
            nseconds = time.time() - tstart
            if t % 100 == 0 or t == 1:
                ev = explained_variance(V, R)
                test_result = play(model, env_name)

                print(' - - - - - - - ')
                print("n_updates", t)
                print("entropy", float(entropy))
                print("test points", str(test_result))

                writer.add_scalar('lr', float(lr), t)
                writer.add_scalar('explained_variance', ev, t)
                writer.add_scalar('total_timesteps', t*nbatch, t)
                writer.add_scalar('entropy', float(entropy), t)
                writer.add_scalar('value_loss', float(v_loss), t)
                writer.add_scalar('policy_loss', float(pi_loss), t)
                writer.add_scalar('nseconds', float(nseconds), t)
                writer.add_scalar('test_score', float(test_result), t)
        model.save(os.path.join(log_dir, f'model_{t}_{env_name[:-14]}.pth'))
    except KeyboardInterrupt:
        model.save(os.path.join(log_dir, f'model_{t}_{env_name[:-14]}.pth'))

def play(model, env_name):
    env_test = make_atari(env_name)
    env_test = wrap_deepmind(env_test, episode_life=True, frame_stack=True)

    model.eval()

    n_play_times = 3
    all_scores = 0

    for _ in range(n_play_times):
        score = 0
        s = np.asarray(env_test.reset())[np.newaxis, :, :, :]
        h = model.rnn_cell.init_hidden(batch_size=1)
        with torch.no_grad():
            for i in range(5000):
                action, h, mask = model.choose_action(s, h)
                s, r, done, info = env_test.step(action)
                # if i < 30:
                #     print(action, end='')
                score += r
                s = np.asarray(s)[np.newaxis, :, :, :]
                # if info['ale.lives'] < 5:
                #     break
                if done:
                    break
            all_scores += score
    return all_scores / n_play_times

def handle_stacked(stack, s_t):
    stack = np.roll(stack, shift=-1, axis=3)
    stack[:, :, :, -1:] = s_t
    return stack

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary

def discount(rewards, dones):
    res = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + GAMMA * r * (1 - done)
        res.append(r)
    return res[::-1]

class Runner():
    def __init__(self, env, model, is_recurrent=False, sequence_len=32):
        self.s_ts = np.zeros((nenvs, 84, 84, 4), dtype=np.uint8)
        s_t = env.reset()  # (nenv, 84, 84, 1)
        self.s_ts = handle_stacked(self.s_ts, s_t)
        self.dones = [False for i in range(nenvs)]
        self.env = env
        self.model = model
        self.is_recurrent = is_recurrent
        self.sequence_len = sequence_len
        if is_recurrent:
            self.current_episode = []

        self.h = self.model.rnn_cell.init_hidden(batch_size=s_t.shape[0])

    def run(self, nsteps):

        def create_episode(env_i, start_i, end_i):
            ep = {
                's': s_record[start_i:end_i, env_i],
                'a': a_record[start_i:end_i, env_i],
                'r': return_record[start_i:end_i, env_i],
                'v': v_record[start_i:end_i, env_i],
                'neglogp': neglogp_record[start_i:end_i, env_i],
                'hx': hx[start_i:end_i, env_i],
                'cx': cx[start_i:end_i, env_i],
            }
            return ep

        def episode_index(episode, seq_start_i, seq_end_i):
            ep = {
                key: value[seq_start_i:seq_end_i] for key, value in episode.items() if key not in ['hx', 'cx']
            }
            ep['hx'] = episode['hx'][seq_start_i]
            ep['cx'] = episode['cx'][seq_start_i]
            return ep


        self.model.eval()
        s_record = []
        a_record = []
        r_record = []
        v_record = []
        done_record = []
        neglogp_record = []
        h_record = ([], [])
        # nsteps = 5 # exploration horizon?
        for t in range(nsteps):
            h_record[0].append(self.h[0].detach().cpu().numpy())
            h_record[1].append(self.h[1].detach().cpu().numpy())

            with torch.no_grad():
                a_t, neglogp, self.h, values = self.model.sample_and_V(self.s_ts, self.h, get_neglogp=True)

            s_record.append(np.copy(self.s_ts))  # don't think copy is necessary
            a_record.append(a_t)
            v_record.append(values)
            done_record.append(self.dones)

            neglogp_record.append(neglogp)

            s_t, r, self.dones, _ = self.env.step(a_t)
            for i, done in enumerate(self.dones):
                if done:
                    self.s_ts[i] *= 0
                    h_reset = self.model.rnn_cell.init_hidden(batch_size=1)
                    self.h[0][i] = h_reset[0].squeeze(0)
                    self.h[1][i] = h_reset[1].squeeze(0)
            self.s_ts = handle_stacked(self.s_ts, s_t)
            r_record.append(r)

        done_record.append(self.dones)

        # lists to numpy arrays
        # switch axis so that nenv is at axis 0
        s_record = np.asarray(s_record, dtype=np.uint8) #.reshape((nsteps * nenvs, 84, 84, 4))
        r_record = np.asarray(r_record, dtype=np.float32)#.swapaxes(1, 0)
        a_record = np.asarray(a_record, dtype=np.int32)#.swapaxes(1, 0)
        v_record = np.asarray(v_record, dtype=np.float32)#.swapaxes(1, 0)
        done_record = np.asarray(done_record, dtype=np.bool)#.swapaxes(1, 0)
        neglogp_record = np.asarray(neglogp_record, dtype=np.float32)#.swapaxes(1, 0)
        hx = np.asarray(h_record[0], dtype=np.float32)
        cx = np.asarray(h_record[1], dtype=np.float32)
        # masks = done_record[:, :-1]  # exclude the last timestep
        masks = None
        #done_record = done_record[:, 1:]  # exclude the first timestep

        with torch.no_grad():
            last_values, self.h = self.model.V(self.s_ts, self.h)

        # reward discount
        return_record = np.zeros_like(r_record)
        adv_record = np.zeros_like(r_record)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - done_record[t+1]
                nextvalues = v_record[t+1]
            nextvalues = np.array(nextvalues)

            delta = r_record[t] + GAMMA * nextvalues * nextnonterminal - v_record[t]
            adv_record[t] = lastgaelam = delta + GAMMA * LAM * nextnonterminal * lastgaelam
        return_record = adv_record + v_record

        if self.is_recurrent:
            done_indices = []
            done_record = done_record[1:] # remove the first dones
            for env_i in range(done_record.shape[1]):
                done_indices.append(list(done_record[:, env_i].nonzero()[0]))

                if len(done_indices[env_i]) == 0 or done_indices[env_i][-1] != nsteps - 1:
                    done_indices[env_i].append(nsteps - 1)

            sequences = []
            for env_i in range(done_record.shape[1]):
                start_i = 0
                for step_i in done_indices[env_i]:
                    # if done_record[step_i] == 1:
                    episode = create_episode(env_i, start_i, step_i+1)

                    episode_len = step_i+1 - start_i
                    start_i = step_i + 1

                    for seq_start_i in range(0, episode_len, self.sequence_len):
                        seq_end_i = min(seq_start_i + self.sequence_len, episode_len)
                        sequences.append(episode_index(episode, seq_start_i, seq_end_i))
                    # if seq_start_i + self.sequence_len > episode_len:

            sequences = self.pad_and_merge_sequences(sequences) # dict

            return sequences

        else:
            s_record = s_record.reshape((nsteps * nenvs, 84, 84, 4))
            return_record = return_record.flatten()  # (venv*Horizon)
            a_record = a_record.flatten()
            v_record = v_record.flatten()
            neglogp_record = neglogp_record.flatten()
            # masks = masks.flatten()
            hx = hx.reshape((nsteps * nenvs, hx.shape[-2], hx.shape[-1]))
            cx = cx.reshape((nsteps * nenvs, cx.shape[-2], cx.shape[-1]))
            h_record = (hx, cx)
            return s_record, a_record, return_record, masks, v_record, neglogp_record, h_record

    def pad_and_merge_sequences(self, seq_list):
        all_seqs = {
            's': [], # (n_seq, time, ...)
            'a': [],
            'r': [],
            'v': [],
            'neglogp': [],
            'masks': [],
            'hx': [], # (n_seq, ...)
            'cx': [],
        }
        for seq in seq_list:
            if seq['s'].shape[0] < self.sequence_len:
                pad_len = self.sequence_len - seq['s'].shape[0]
                for key in seq:
                    if key not in ['hx', 'cx']:
                        pad = np.zeros((pad_len, *seq[key].shape[1:]))
                        seq[key] = np.concatenate((seq[key], pad), axis=0)
                seq['masks'] = np.array([1 for _ in range(self.sequence_len-pad_len)] + [0 for _ in range(pad_len)])
            else:
                seq['masks'] = np.array([1 for _ in range(self.sequence_len)])
            for key in seq:
                all_seqs[key].append(seq[key])

        for key in all_seqs:
            all_seqs[key] = np.stack(all_seqs[key]) # (n_seq, time, ...)
        return all_seqs


def main(env_name):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_name)
            env.seed(opt.seed + rank)
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk

    set_seeds(opt.seed)
    log_dir = setup_logging()

    # get action space
    temp = make_atari(env_name)
    temp = wrap_deepmind(temp, frame_stack=True, episode_life=True)
    num_actions = temp.action_space.n
    del temp

    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    
    train(env, env_name, num_actions=num_actions, nsteps=opt.horizon_steps, log_dir=log_dir)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--horizon_steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument('--optim', type=str, default='adamw')
    opt = parser.parse_args()
    # main('BreakoutNoFrameskip-v4')
    # main('BeamRiderNoFrameskip-v4')
    main('KungFuMasterNoFrameskip-v4')
    # main('GopherNoFrameskip-v4')
    # main('EnduroNoFrameskip-v4')
    # main('VideoPinballNoFrameskip-v4')
    # main('ZaxxonNoFrameskip-v4')
    # main('NameThisGameNoFrameskip-v4')

