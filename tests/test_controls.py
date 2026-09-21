"""Small coordinate/interface checks; these are not experimental results."""
import pytest
import torch

from utils.controls import clock_label, conjugate_editor, coordinate, wrapped_step
from utils.probes import features
from test_sampler import Counter, tiny_sampler


def test_coordinate_inverse_endpoints_and_before_pooling():
    x = torch.linspace(-3, 2, 4 * 6 * 6).reshape(1, 4, 6, 6)
    for k in range(6):
        w = coordinate(x, k, 5, 2.0)
        recovered = coordinate(w, k, 5, 2.0, inverse=True)
        torch.testing.assert_close(recovered, x, rtol=0, atol=5e-7)
        torch.testing.assert_close(features(recovered, 0, 2, 2), features(x, 0, 2, 2),
                                   rtol=0, atol=5e-7)
    assert torch.equal(coordinate(x, 0, 5, 2.0), x)
    assert torch.equal(coordinate(x, 5, 5, 2.0), x)
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, 2)
    pooled_wrapped = torch.nn.functional.adaptive_avg_pool2d(coordinate(x, 2, 5, 2.0), 2)
    assert not torch.allclose(coordinate(pooled_wrapped, 2, 5, 2.0, inverse=True), pooled)
    with pytest.raises(ValueError, match="even channels"):
        coordinate(x[:, :3], 1, 5, 2.0)


def test_wrapped_endpoint_and_transported_edit_recovery(tiny_sampler):
    sampler = tiny_sampler
    conditioning = sampler.encode("object", "", Counter())
    x = sampler.initial(91, 8, 8)
    native, wrapped = x.clone(), x.clone()
    with torch.no_grad():
        for k in range(sampler.num_steps):
            native, _ = sampler.step(native, k, conditioning, Counter())
            wrapped, _ = wrapped_step(sampler, wrapped, k, conditioning, 1.5, Counter())
    torch.testing.assert_close(wrapped, native, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(sampler.decode(wrapped, Counter()),
                               sampler.decode(native, Counter()), rtol=1e-4, atol=2e-6)
    k = 1
    paused = sampler.continue_from(x, 0, conditioning, Counter(), stop=k)
    w = coordinate(paused, k, sampler.num_steps, 1.5)
    bounded = lambda state: (state + 0.01, {"delta_rms": 0.01})
    edited_w, info = conjugate_editor(w, k, sampler.num_steps, 1.5, bounded)
    recovered = coordinate(edited_w, k, sampler.num_steps, 1.5, inverse=True)
    native_edited, _ = bounded(paused)
    assert info["delta_rms"] == 0.01
    torch.testing.assert_close(recovered, native_edited, rtol=0, atol=3e-7)
    torch.testing.assert_close(sampler.continue_from(recovered, k, conditioning, Counter()),
                               sampler.continue_from(native_edited, k, conditioning, Counter()),
                               rtol=1e-5, atol=2e-6)


def test_clock_relabels_only():
    work = [0, 2, 4, 6]
    assert [clock_label(k, 3, 2) for k in range(4)] != [k / 3 for k in range(4)]
    assert work == [0, 2, 4, 6]
