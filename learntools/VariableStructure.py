import Network
import Learning
import numpy as np

def VariableStructure(n_in:int, n_out: int, loss_fun, epochs: int, max_its: int, max_mutations: int, max_depth: int, act_func): #max_mutations should be max pertubations?
    net = Network.network(n_in, n_out)
    net.add_layer(Network.layer_dense(n_in,n_out))

    net.random_initilisation()

    

    for epoch in range(epochs): #go through the epochs
        init_loss = loss_fun(net)

        loss, (its, total_mutations) = Learning.random_learning(net,loss_fun,max_its,max_mutations)

        rem_its = max_its - its
        mutation_prob = rem_its / max_its
        if np.random.uniform(0,1) < mutation_prob:
            # change the structure
            for index in net.mutateable_layers:
                # need a resize function?
                pass
