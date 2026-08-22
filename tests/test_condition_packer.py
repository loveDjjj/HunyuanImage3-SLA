import torch

from sampling.condition_packer import decode_rope_image_info, encode_rope_image_info, pack_condition


class _Output:
    tokens = torch.tensor([1, 2, 3])
    gen_image_mask = torch.tensor([False, True, True])
    gen_timestep_scatter_index = torch.tensor([0])
    guidance_scatter_index = None
    gen_timestep_r_scatter_index = None


class _Model:
    def preprocess_inputs(self, **kwargs):
        return {"output": _Output(), "sections": []}

    def build_batch_rope_image_info(self, output, sections):
        return [[(slice(1, 3), (2, 1))]]


def test_packer_serializes_static_condition_and_rope():
    tensors, metadata = pack_condition(_Model(), "caption", 64, 64)
    assert tensors["input_ids"].shape == (1, 3)
    assert tensors["image_mask"].shape == (1, 3)
    assert metadata["rope_image_info"] == [[[1, 3, 2, 1]]]
    assert decode_rope_image_info(metadata["rope_image_info"])[0][0][0] == slice(1, 3)
