import numpy
import matplotlib.pyplot as plt
import networkx as nx

class DynamicGraph:

    def __init__(self, state_size, alpha = 0.01):
        self.correlation = numpy.zeros((state_size, state_size))
        self.alpha       = alpha


    def add(self, state):
        state_a = numpy.expand_dims(state, 0)
        state_b = numpy.transpose(state_a)

        correlation = state_a*state_b
        correlation = correlation**2 + 0.0001


        self.correlation = (1.0 - self.alpha)*self.correlation + self.alpha*correlation

        adjacency_matrix = 1.0/self.correlation
        
        self.graph = self._minimum_spanning_tree(adjacency_matrix)

        m0 = numpy.transpose(numpy.matrix(self.graph.edges))
        m1 = numpy.concatenate([m0[1], m0[0]])

        self.edge_index  = numpy.concatenate([m0, m1], axis=1)
        self.adjacency_matrix = numpy.sign(nx.adjacency_matrix(self.graph).todense())
        
        state_ = numpy.repeat(state[numpy.newaxis, :], state.shape[0], 0)
        #self.state_masked = numpy.multiply(self.adjacency_matrix, state_)
        self.state_masked = self.adjacency_matrix


    def process_state(self, state):
        self.add(state)
        return self.state_masked, self.adjacency_matrix, self.edge_index


    def _minimum_spanning_tree(self, adjacency_matrix):
        gr      = nx.from_numpy_matrix(numpy.matrix(adjacency_matrix))
        result  = nx.minimum_spanning_tree(gr, weight="weight")
        return result

    def show_graph(self):
        plt.clf()

        labels = {}
        for i in range(self.correlation.shape[0]):
            labels[i] = str(i)

        nx.draw(self.graph, node_size=500, labels=labels, with_labels=True)
        plt.ion()
        plt.show()
        plt.pause(0.001)
        
    
if __name__ == "__main__":
    state_size = 8
    dg = DynamicGraph(state_size)

    for i in range(1000):
        state = numpy.random.randn(state_size)
        state_masked, adjacency_matrix, edge_index = dg.process_state(state)

        if i%100 == 0:
            print(numpy.round(adjacency_matrix, 3),"\n")
            print(numpy.round(state_masked, 3),"\n\n\n\n")
            

