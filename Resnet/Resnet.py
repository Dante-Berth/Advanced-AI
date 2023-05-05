import tensorflow as tf
class Resnet(tf.keras.layers.Layer):
    """
    Resnet layer allows to know if the residual connexion is relevant, works only if the input_channels==output_channels for the major and minor neural network
    """

    def __init__(self, num_blocks: int = 2, major_neural_network=None,
                 final_activation_layer: torch.nn.Module = torch.nn.Identity(), minor_neural_network=None):
        """

        Args:
            num_blocks: int number of blocks
            major_neural_network: either torch.nn.Module or either torch_geometric.nn major nn is used for the main component
            final_activation_layer: Optional[torch.nn.Module, torch_geometric.nn] activation layer is used at the end of Resnet operations
            minor_neural_network: Optional[torch.nn.Module, torch_geometric.nn] minor nn is used for computing the residuals
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.major_neural_network = major_neural_network
        self.final_activation_layer = final_activation_layer
        self.minor_neural_network = minor_neural_network
        self.weights_list = torch.nn.Parameter(torch.rand(1, num_blocks, 2))
        self.softmax_weights = None

    def forward(self, data):
        """

        Args:
            data: it is a data <==> batch composed of x (set of nodes), edge_attr and edge_index

        Returns:
                x
        """
        self.softmax_weights = F.softmax(self.weights_list, dim=-1)
        for i in range(self.num_blocks):
            data.x = self.major_neural_network(x=data.x,edge_index=data.edge_index,edge_attr=data.edge_attr)
            if self.minor_neural_network is None:
                data_x_ghost = torch.nn.Identity()(data.x)
            else:
                data_x_ghost = self.minor_neural_network(x=data.x,edge_index=data.edge_index,edge_attr=data.edge_attr)
            data.x = data.x * self.softmax_weights.squeeze()[i][0] + data_x_ghost * self.softmax_weights.squeeze()[i][1]
            data.x = self.final_activation_layer(data.x)

        return data.x