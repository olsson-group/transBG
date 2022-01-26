# load general packages and functions

import torch
import math

# load program-specific functions
#from parameters.constants import constants as C

'''
class parameters():
    def __init__(self, max_nodes, edge_features, node_features):
        self.enn_depth = 4
        self.enn_dropout_p = 0
        self.enn_hidden_dim = 250
        self.gather_att_depth = 4
        self.gather_att_dropout_p = 0
        self.gather_att_hidden_dim = 250
        self.gather_width = 100
        self.gather_emb_depth = 4
        self.gather_emb_dropout_p = 0
        self.gather_emb_hidden_dim = 250
        self.hidden_node_features = 100
        self.weights_initialization = 'uniform' # or "normal"
        self.message_passes = 3
        self.message_size = 100
        self.max_n_nodes = max_nodes
        self.dim_edges = [max_nodes, max_nodes, edge_features] 
        self.dim_nodes = [max_nodes, node_features]
        self.model = 'GGNN'
'''

# define "BIG" constants (for masks)
BIG_NEGATIVE = -1e6
BIG_POSITIVE = 1e6

class SummationMPNN(torch.nn.Module):
    """ Abstract `SummationMPNN` class. Specific models using this class is GGNN.
    """
    def __init__(self, node_features, hidden_node_features, edge_features, message_size, message_passes):

        super(SummationMPNN, self).__init__()

        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.message_size = message_size
        self.message_passes = message_passes

    def message_terms(self, nodes, node_neighbours, edges):
        """ Message passing function, to be implemented in all `SummationMPNN`
        subclasses.

        Args:
          nodes (torch.Tensor) : Batch of size {total number of nodes in batch,
            number of node features}.
          node_neighbours (torch.Tensor) : Batch of size {total number of nodes
            in batch, max node degree, number of node features}.
          edges (torch.Tensor) : Batch of size {total number of nodes in batch,
            max node degree, number of edge features}.
        """
        raise NotImplementedError

    def update(self, nodes, messages):
        """ Message update function, to be implemented in all `SummationMPNN`
        subclasses.

        Args:
          nodes (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
          messages (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
        """
        raise NotImplementedError

    def readout(self, hidden_nodes, input_nodes, node_mask):
        """ Local readout function, to be implemented in all `SummationMPNN`
        subclasses.

        Args:
          hidden_nodes (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
          input_nodes (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
          node_mask (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}, where elements are 1 if
            corresponding element exists and 0 otherwise.
        """
        raise NotImplementedError

    def forward(self, nodes, edges):
        """ Defines forward pass.

        Args:
          nodes (torch.Tensor) : Batch of size {N, number of nodes,
            number of node features}, where N is the number of subgraphs in each batch.
          edges (torch.Tensor) : Batch of size {N, number of nodes, number of
            nodes, number of edge features}, where N is the number of subgraphs
            in each batch.
        """
        adjacency = torch.sum(edges, dim=3)

        # **note: "idc" == "indices", "nghb{s}" == "neighbour(s)"
        (
            edge_batch_batch_idc,
            edge_batch_node_idc,
            edge_batch_nghb_idc,
        ) = adjacency.nonzero(as_tuple=True)

        (node_batch_batch_idc, node_batch_node_idc) = adjacency.sum(-1).nonzero(as_tuple=True)

        same_batch = node_batch_batch_idc.view(-1, 1) == edge_batch_batch_idc
        same_node = node_batch_node_idc.view(-1, 1) == edge_batch_node_idc

        # element ij of `message_summation_matrix` is 1 if `edge_batch_edges[j]`
        # is connected with `node_batch_nodes[i]`, else 0
        message_summation_matrix = (same_batch * same_node).float()

        edge_batch_edges = edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        # pad up the hidden nodes
        hidden_nodes = torch.zeros(nodes.shape[0], nodes.shape[1], self.hidden_node_features, device="cuda")
        hidden_nodes[:nodes.shape[0], :nodes.shape[1], :nodes.shape[2]] = nodes.clone()
        node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]

        for _ in range(self.message_passes):
            edge_batch_nodes = hidden_nodes[edge_batch_batch_idc, edge_batch_node_idc, :]

            edge_batch_nghbs = hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            message_terms = self.message_terms(edge_batch_nodes,
                                               edge_batch_nghbs,
                                               edge_batch_edges)

            if len(message_terms.size()) == 1:  # if a single graph in batch
                message_terms = message_terms.unsqueeze(0)

            # the summation in eq. 1 of the NMPQC paper happens here
            messages = torch.matmul(message_summation_matrix, message_terms)

            node_batch_nodes = self.update(node_batch_nodes, messages)
            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = node_batch_nodes.clone()

        node_mask = adjacency.sum(-1) != 0

        graph_emb = self.readout(hidden_nodes, nodes, node_mask)

        return graph_emb, hidden_nodes

class GraphGather(torch.nn.Module):
    """ GGNN readout function.
    """
    def __init__(self, node_features, hidden_node_features, out_features,
                 att_depth, att_hidden_dim, att_dropout_p, emb_depth,
                 emb_hidden_dim, emb_dropout_p, init):

        super(GraphGather, self).__init__()

        self.att_nn = MLP(
            in_features=node_features + hidden_node_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=out_features,
            init=init,
            dropout_p=att_dropout_p
        )

        self.emb_nn = MLP(
            in_features=hidden_node_features,
            hidden_layer_sizes=[emb_hidden_dim] * emb_depth,
            out_features=out_features,
            init=init,
            dropout_p=emb_dropout_p
        )

    def forward(self, hidden_nodes, input_nodes, node_mask):
        """ Defines forward pass.
        """
        Softmax = torch.nn.Softmax(dim=1)

        cat = torch.cat((hidden_nodes, input_nodes), dim=2)
        energy_mask = (node_mask == 0).float() * BIG_POSITIVE
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = Softmax(energies)
        embedding = self.emb_nn(hidden_nodes)

        return torch.sum(attention * embedding, dim=1)



class MLP(torch.nn.Module):
    """ Multi-layer perceptron. Applies SELU after every linear layer.

    Args:
      in_features (int) : Size of each input sample.
      hidden_layer_sizes (list) : Hidden layer sizes.
      out_features (int) : Size of each output sample.
      init (str) : Weight initialization ('none', 'normal', or 'uniform').
      dropout_p (float) : Probability of dropping a weight.
    """

    def __init__(self, in_features, hidden_layer_sizes, out_features, init, dropout_p):

        super(MLP, self).__init__()

        activation_function = torch.nn.SELU

        # create list of all layer feature sizes
        fs = [in_features, *hidden_layer_sizes, out_features]

        # create list of linear_blocks
        layers = [self._linear_block(in_f, out_f,
                                     activation_function, init,
                                     dropout_p)
                  for in_f, out_f in zip(fs, fs[1:])]

        # concatenate modules in all sequentials in layers list
        layers = [module for sq in layers for module in sq.children()]

        # add modules to sequential container
        self.seq = torch.nn.Sequential(*layers)

    def _linear_block(self, in_f, out_f, activation, init, dropout_p):
        """ Returns a linear block consisting of a linear layer, an activation
        function (SELU), and dropout (optional) stack.

        Args:
          in_f (int) : Size of each input sample.
          out_f (int) : Size of each output sample.
          activation (torch.nn.Module) : Activation function.
          init (str) : Weight initialization ('none', 'normal', or 'uniform').
          dropout_p (float) : Probability of dropping a weight.
        """
        # bias must be used in most MLPs in our models to learn from empty graphs
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        linear.weight = self.define_weight_initialization(init, linear)

        return torch.nn.Sequential(linear, activation(), torch.nn.AlphaDropout(dropout_p))

    @staticmethod
    def define_weight_initialization(initialization, linear_layer):
        """ Defines the weight initialization scheme to use in `linear_layer`.
        """
        if initialization == "none":
            pass
        elif initialization == "uniform":
            torch.nn.init.xavier_uniform_(linear_layer.weight)
        elif initialization == "normal":
            torch.nn.init.xavier_normal_(linear_layer.weight)
        else:
            raise NotImplementedError

        return linear_layer.weight

    def forward(self, layers_input):
        """ Defines forward pass.
        """
        return self.seq(layers_input)

class GGNN(SummationMPNN):
    """ The "gated-graph neural network" model.

    Args:
      *edge_features (int) : Number of edge features.
      enn_depth (int) : Num layers in 'enn' MLP.
      enn_dropout_p (float) : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int) : Number of weights (layer width) in 'enn' MLP.
      gather_att_depth (int) : Num layers in 'gather_att' MLP in graph gather block.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in
        graph gather block.
      gather_att_hidden_dim (int) : Number of weights (layer width) in
        'gather_att' MLP in graph gather block.
      gather_emb_depth (int) : Num layers in 'gather_emb' MLP in graph gather block.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in
        graph gather block.
      gather_emb_hidden_dim (int) : Number of weights (layer width) in
        'gather_emb' MLP in graph gather block.
      gather_width (int) : Output size of graph gather block block.
      hidden_node_features (int) : Indicates length of node hidden states.
      *initialization (str) : Initialization scheme for weights in feed-forward
        networks ('none', 'uniform', or 'normal').
      message_passes (int) : Number of message passing steps.
      message_size (int) : Size of message passed (output size of all MLPs in
        message aggregation step, input size to `GRU`).
      *n_nodes_largest_graph (int) : Number of nodes in the largest graph.
      *node_features (int) : Number of node features (e.g. `n_atom_types` +
        `n_formal_charge`).
    """
    def __init__(self, edge_features, enn_depth, enn_dropout_p, enn_hidden_dim,
                 gather_att_depth,
                 gather_att_dropout_p, gather_att_hidden_dim, gather_width,
                 gather_emb_depth, gather_emb_dropout_p, gather_emb_hidden_dim,
                 hidden_node_features, initialization, message_passes,
                 message_size, n_nodes_largest_graph, node_features):

        super(GGNN, self).__init__(node_features, hidden_node_features, edge_features, message_size, message_passes)

        self.n_nodes_largest_graph = n_nodes_largest_graph

        self.msg_nns = torch.nn.ModuleList()
        for _ in range(edge_features):
            self.msg_nns.append(
                MLP(
                    in_features=hidden_node_features,
                    hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
                    out_features=message_size,
                    init=initialization,
                    dropout_p=enn_dropout_p,
                )
            )

        self.gru = torch.nn.GRUCell(
            input_size=message_size, hidden_size=hidden_node_features, bias=True
        )

        self.gather = GraphGather(
            node_features=node_features,
            hidden_node_features=hidden_node_features,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            init=initialization,
        )


    def message_terms(self, nodes, node_neighbours, edges):
        edges_v = edges.view(-1, self.edge_features, 1)
        node_neighbours_v = edges_v * node_neighbours.view(-1, 1, self.hidden_node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)

        return graph_embeddings

def initialize_GGNN(C):
    """ Initializes the model to be trained. Possible models: "GGNN".

    Argument:
    C: class containing the parameters of the GGNN model.

    Returns:
      model (modules.SummationMPNN or modules.AggregationMPNN or
        modules.EdgeMPNN) : Neural net model.
    """
    hidden_node_features = C.hidden_node_features

    net = GGNN(
        edge_features=C.dim_edges[2],
        enn_depth=C.enn_depth,
        enn_dropout_p=C.enn_dropout_p,
        enn_hidden_dim=C.enn_hidden_dim,
        gather_att_depth=C.gather_att_depth,
        gather_att_dropout_p=C.gather_att_dropout_p,
        gather_att_hidden_dim=C.gather_att_hidden_dim,
        gather_width=C.gather_width,
        gather_emb_depth=C.gather_emb_depth,
        gather_emb_dropout_p=C.gather_emb_dropout_p,
        gather_emb_hidden_dim=C.gather_emb_hidden_dim,
        hidden_node_features=hidden_node_features,
        initialization=C.weights_initialization,
        message_passes=C.message_passes,
        message_size=C.message_size,
        n_nodes_largest_graph=C.max_n_nodes,
        node_features=C.dim_nodes[1],
    )

    net = net.to(C.device, non_blocking=True)

    return net

'''
C = parameters(max_nodes=50, edge_features=3, node_features=7)
gnn = initialize_model(C)
print(gnn.parameters)
'''
