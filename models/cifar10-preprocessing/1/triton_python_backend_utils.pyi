# I had to implement this stub for mypy to stop crying
# because of backend utils
# ref: https://github.com/triton-inference-server/server/issues/4743
from typing import Any, List, Optional

class Tensor:
    def __init__(self, name: str, shape: Any) -> None: ...

class InferenceResponse:
    def __init__(
        self, output_tensors: List[Tensor], error: Optional[str] = None
    ) -> None: ...

def get_input_tensor_by_name(inference_request: Any, name: Any): ...
