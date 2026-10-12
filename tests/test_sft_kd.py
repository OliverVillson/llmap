import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from stages import sft  # noqa: E402


def test_chunked_kd_matches_the_plain_kl_and_its_gradient():
    torch.manual_seed(0)
    s = torch.randn(37, 50, requires_grad=True)
    t = torch.randn(37, 50).to(torch.bfloat16)
    ref = F.kl_div(F.log_softmax(s / 2.0, -1), F.log_softmax(t.float() / 2.0, -1),
                   log_target=True, reduction="batchmean") * 4.0
    (g_ref,) = torch.autograd.grad(ref, s)
    got = sft.kd_kl(s, t, temp=2.0, chunk=8)
    (g,) = torch.autograd.grad(got, s)
    assert torch.allclose(got, ref, atol=1e-5) and torch.allclose(g, g_ref, atol=1e-6)
