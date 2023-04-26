"""
Functions that are used multiple times
"""

from torch import nn
import torch
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg') # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
       # v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
       pass

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, global_max_avg_r, path,res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
            if global_max_avg_r.value < global_ep_r.value:
                global_max_avg_r.value = global_ep_r.value
    res_queue.put(global_ep_r.value)


    if str(global_ep.value)[-1:] == "0":# much faster than episode % 10
        try:      
            fig, ax = plt.subplots() # fig : figure object, ax : Axes object       
            ax.plot(np.arange(global_ep.value), global_ep_r.value, 'r')
            ax.set_ylabel('Average Score', fontsize=18)
            ax.set_xlabel('Steps', fontsize=18)
            ax.set_title("Episode scores")
            fig.savefig(path+".png", bbox_inches="tight")
            plt.close('all')
        except OSError as e:
            print(e)
        except:
            e = sys.exc_info()[0]
            print("Something else went wrong e: ", e)

    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )