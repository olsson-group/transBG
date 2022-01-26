# load general packages and functions
import torch

# load program-specific functions
# (none)

# defines MPNN modules and readout functions, and APD readout functions

# define "BIG" constants (for masks)
BIG_NEGATIVE = -1e6
BIG_POSITIVE = 1e6


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


