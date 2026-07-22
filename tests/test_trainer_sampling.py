import torch

from engine.trainer import Trainer


def test_crop_sampling_returns_every_pixel_in_each_crop(monkeypatch):
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.H = 8
    trainer.W = 8
    trainer.crop_size = 3
    trainer.crops_per_batch = 1
    trainer.lod_sampling = "fixed0"
    trainer.num_lods = 1
    trainer.mip_target_mode = "discrete"

    # Make the crop origin deterministic: (y=2, x=2).
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([2]))

    _, _, ys, xs, _, _ = trainer._sample_coords()

    assert len(ys) == 9
    assert set(zip(ys.tolist(), xs.tolist())) == {
        (y, x) for y in range(2, 5) for x in range(2, 5)
    }
    assert trainer._get_sample_count() == 9
