from typing import Any

import torch
import triton_python_backend_utils as pb_utils
from torchvision import transforms  # type: ignore


class TritonPythonModel:
    def initialize(self, args):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def execute(self, requests) -> list[Any]:
        responses: list[pb_utils.InferenceResponse] = []

        for request in requests:
            images_batch = pb_utils.get_input_tensor_by_name(
                request, "IMAGE"
            ).as_numpy()

            # cringe: https://github.com/triton-inference-server/server/issues/4743
            images_batch_t = torch.stack([self.transform(img) for img in images_batch])
            # numpy array is required
            images_preprocessed = pb_utils.Tensor(
                "IMAGE_PREPROCESSED", images_batch_t.numpy()
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[images_preprocessed]
            )
            responses.append(inference_response)

        return responses
