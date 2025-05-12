import numpy as np

from .layers import affine_forward, affine_backward, relu_forward, relu_backward, softmax_loss
from .layers_norm import batchnorm_forward, batchnorm_backward, layernorm_forward, layernorm_backward
from .layers_dropout import dropout_forward, dropout_backward


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # IMPORTANT: DO NOT DEFINE ANYTHING MORE than NEEDED! Or it'll raise EXCEPTION.
        # *****START OF YOUR CODE *****

        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, size=(layer_dims[i], layer_dims[i + 1]))
            self.params['b' + str(i + 1)] = np.zeros(layer_dims[i + 1])

        if self.normalization: # == "batchnorm" or "layernorm"
            for i in range(self.num_layers-1):
                self.params['gamma' + str(i + 1)] = np.ones(layer_dims[i + 1])
                self.params['beta' + str(i + 1)] = np.zeros(layer_dims[i + 1])

        # *****END OF YOUR CODE *****

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train", "eps": 1e-5, "momentum": 0.9} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # Recall the architecture: {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        # IMPORTANT: The outputs from the last layer of the network should not be normalized.
        # *****START OF YOUR CODE *****
        
        caches = {}
        out = X
        for i in range(1, self.num_layers):
            out, cache = affine_forward(out, self.params['W'+str(i)], self.params['b'+str(i)])
            caches['affine'+str(i)] = cache
            if self.normalization == "batchnorm":
                out, cache = batchnorm_forward(out, self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i-1])
                caches['batchnorm'+str(i)] = cache
            if self.normalization == "layernorm":
                out, cache = layernorm_forward(out, self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i-1])
                caches['layernorm'+str(i)] = cache
            out, cache = relu_forward(out)
            caches['relu'+str(i)] = cache
            if self.use_dropout:
                out, cache = dropout_forward(out, self.dropout_param)
            caches['dropout'+str(i)] = cache
        i = self.num_layers
        scores, cache = affine_forward(out, self.params['W'+str(i)], self.params['b'+str(i)])
        caches['affine'+str(i)] = cache

        # *****END OF YOUR CODE *****

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Recall the architecture: {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        # *****START OF YOUR CODE *****
        
        caches['affine' + str(i)] = cache # above i = self.num_layers
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)]**2)

        dx, dw, db = affine_backward(dx, caches['affine' + str(i)])
        grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
        grads['b' + str(i)] = db

        for i in range(self.num_layers - 1, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)]**2)
            if self.use_dropout:
                dx = dropout_backward(dx, caches['dropout'+str(i)])
            dx = relu_backward(dx, caches['relu' + str(i)])
            if self.normalization == "batchnorm":
                dx, dgamma, dbeta = batchnorm_backward(dx, caches['batchnorm'+str(i)]) 
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            if self.normalization == "layernorm":
                dx, dgamma, dbeta = layernorm_backward(dx, caches['layernorm'+str(i)]) 
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            dx, dw, db = affine_backward(dx, caches['affine'+str(i)])
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db

        # *****END OF YOUR CODE *****
        return loss, grads
