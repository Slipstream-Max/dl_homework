# Deep Learning Utilities Package

# Import common utilities
from .data_utils import (
    get_CIFAR10_data,
    load_CIFAR10,
    load_tiny_imagenet,
    load_models,
    load_imagenet_val
)

from .gradient_check import (
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    eval_numerical_gradient_blobs
)

from .vis_utils import visualize_grid

# Import layer functions
try:
    from .layers import (
        affine_forward, affine_backward,
        relu_forward, relu_backward,
        softmax_loss
    )
except ImportError:
    pass

# Import CNN functions if available
try:
    from .layers_cnn import (
        conv_forward_naive, conv_backward_naive,
        max_pool_forward_naive, max_pool_backward_naive
    )
    from .fast_layers import (
        conv_forward_fast, conv_backward_fast,
        max_pool_forward_fast, max_pool_backward_fast
    )
except ImportError:
    pass

# Import RNN functions if available
try:
    from .rnn_layers import (
        rnn_step_forward, rnn_step_backward,
        rnn_forward, rnn_backward,
        word_embedding_forward, word_embedding_backward,
        temporal_affine_forward, temporal_affine_backward,
        temporal_softmax_loss
    )
except ImportError:
    pass

# Import solver
try:
    from .solver import Solver
except ImportError:
    pass

# Import optimizers
try:
    from .optim import (
        sgd,
        sgd_momentum,
        rmsprop,
        adam
    )
except ImportError:
    pass
