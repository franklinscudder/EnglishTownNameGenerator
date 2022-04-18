
"""
['START', 'END', ' ', "'", '-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
"""

ab = ['$', '£', ' ', "'", '-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

from torch import tensor, empty, randn, argmax, zeros, flatten
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.cuda as cuda
from random import randint

SEQ_LEN = 6
VOCAB = len(ab)

def decode(tens):

    out = ab[int(argmax(tens))]
    
    return out
    
def encode(name):
    out = zeros(SEQ_LEN, VOCAB)
    for pos, letter in enumerate(name):
        index = ab.index(letter)
        out[pos, index] = 1.0
        
    return out
    
def names_loop(names):
    i=0
    while 1:
        yield names[i%len(names)]
        i += 1
        
def accuracy(ys, labels):
    size = ys.shape[0]
    y_indices = [int(argmax(ys[i,:])) for i in range(size)]
    lab_indices = [int(argmax(labels[i,:])) for i in range(size)]
    #print(y_indices, lab_indices)
    correct = sum([1 for i in range(size) if lab_indices[i] == y_indices[i]])
    
    return correct / size
    
def test(g):
    g.eval()
    
    name = "$" * SEQ_LEN
    while name[-1] != "£":
        name += decode(g(encode(name[-5:-1]).unsqueeze(0).to(dev)))
        
        if len(name) == 36:
            break
    
    g.train()
    return name
    

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        
        self.rnn = nn.RNN(input_size=VOCAB, hidden_size=VOCAB, num_layers=5, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inp):
        y, h = self.rnn(inp)
        y = flatten(y[:, -1, :], start_dim=1)
        y = self.softmax(y)
        
        return y
    
if __name__ == "__main__":
    g = Gen()
    BATCH_SIZE = 3000
    
    if cuda.is_available() and 0: # GPU is slower??
        dev = "cuda:0"
        print("USING CUDA!")
        g.cuda()
    else:
        dev = "cpu"
    
    losses = []
    accuracies = []
    
    lossFcn = nn.BCELoss(reduction='mean')
    opt = optim.Adam(g.parameters(), lr=0.001)
    
    # with open("towns.txt", "r") as f:
        # names = f.readlines()
    
    with open("GBPN.csv", "r") as f:
        lines = f.readlines()
        names = list(set([line.split(",")[1].upper().replace("&", "AND") for line in lines]))
        names = [name for name in names if all([letter in ab for letter in name]) and len(name) < 33]
        #print(names)
    
    names = ["$"*SEQ_LEN + name + "£" for name in names]
    #names = [name + " "*(32 - len(name)) for name in names]
    
    names_str = "".join(names)
    total_letters = len(names_str)
    letter_freqs = {letter: sum([1 for x in names_str if x == letter])/total_letters for letter in ab if letter != "$"}
    inv_let_freqs = {letter: 1/letter_freqs[letter] for letter in ab if letter != "$"}
    
    my_names_loop = names_loop(names)
    
    batch_inp = empty(BATCH_SIZE, SEQ_LEN, VOCAB)
    batch_labels = empty(BATCH_SIZE, VOCAB)
    
    b = 0
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title("LOSS")
    ax2.set_title("ACCU")
    fig.show()
    
    while 1:
        opt.zero_grad()
        
        
        sample_index = 0
        i = 0
        name = next(my_names_loop)
        
        while sample_index < BATCH_SIZE:
            if name[i+SEQ_LEN-1] == "£":
                name = next(my_names_loop)
                i = 0
                
            #print(name[i:i+SEQ_LEN], name[i+SEQ_LEN])
                
            inp = encode(name[i:i+SEQ_LEN])
            label = flatten(encode(name[i+SEQ_LEN])[0, :])
            
            batch_inp[sample_index, :, :] = inp
            batch_labels[sample_index, :] = label
            sample_index += 1
            i += 1
                
        
        batch_labels = batch_labels.to(dev)
        batch_inp = batch_inp.to(dev)
            
        #print(["".join([decode(batch_inp[i, j, :]) for j in range(SEQ_LEN)]) for i in range(BATCH_SIZE)])
        #print(batch_labels[3,:])
        y = g(batch_inp)
        loss = lossFcn(y, batch_labels)
        acc = accuracy(y, batch_labels)
        
        #scaled_loss = empty(BATCH_SIZE)
        
        #for i in range(BATCH_SIZE):
            #scaled_loss[i] = loss[i, :].mean() * inv_let_freqs[decode(batch_labels[i, :])]
        
        #scaled_avg_loss = scaled_loss.mean()
        
        print(f"{b} - {loss.item()} - {acc}")
        random_int = randint(0, BATCH_SIZE-1)
        print("".join([decode(batch_inp[random_int, i,:]) for i in range(SEQ_LEN)]), decode(y[random_int, :]))
        
        #scaled_avg_loss.backward()
        loss.backward()
        
        losses.append(loss.item())
        accuracies.append(acc)
        b += 1
        opt.step()
        
        if ((b+1) % 100) == 0:
            print()
            print(">>>TEST>>>>  " + test(g)[3:])
            print()
            
            
            ax1.plot(losses)
            ax2.plot(accuracies)
            plt.draw()
            plt.pause(0.001)
            
            
            