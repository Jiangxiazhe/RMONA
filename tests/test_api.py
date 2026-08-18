"""Import / public-API surface tests for the rmona package."""
import rmona
from rmona import FlowOptimizer, Rmona
from rmona.manifold import (
    cayley_retraction,
    msign,
    msign_skew,
    orthogonality_error,
    proj_tangent,
    qr_retraction,
    retract,
    solve_smp,
)
from rmona.models import OrthogonalRNN, ParamRNN


def test_version():
    assert rmona.__version__ == "0.1.0"


def test_public_api():
    assert FlowOptimizer is Rmona
    for fn in [proj_tangent, retract, solve_smp, msign, msign_skew,
               qr_retraction, cayley_retraction, orthogonality_error]:
        assert callable(fn)


def test_model_forward():
    import torch
    model = OrthogonalRNN(input_size=784, hidden=32, num_classes=10)
    x = torch.randn(2, 784)
    out = model(x, model.W_hh)
    assert out.shape == (2, 10)

    p_model = ParamRNN(hidden=32, num_classes=10, param='exp')
    out = p_model(torch.randn(2, 784))
    assert out.shape == (2, 10)
