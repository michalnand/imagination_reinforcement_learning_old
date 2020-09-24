import numpy
import matplotlib.pyplot as plt
import networkx as nx

class DynamicGraph:

    def __init__(self, state_size, alpha = 0.001):
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
        self.m  = numpy.concatenate([m0, m1], axis=1)

        '''
        print(numpy.round(self.correlation, 4))
        print("graph = \n", self.m)
        
        print("\n\n\n")
        '''

    def _minimum_spanning_tree(self, adjacency_matrix):

        gr = nx.from_numpy_matrix(numpy.matrix(adjacency_matrix))
        result = nx.minimum_spanning_tree(gr, weight="weight")
        return result

    def show_graph(self):
        labels = {}
        for i in range(self.correlation.shape[0]):
            labels[i] = str(i)

        nx.draw(self.graph, node_size=500, labels=labels, with_labels=True)
        plt.show()

    
if __name__ == "__main__":
    state_size = 20
    dg = DynamicGraph(state_size)

    for i in range(10):
        state = numpy.random.randn(state_size)
        dg.add(state)

    dg.show_graph()