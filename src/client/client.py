import sys
from io import BytesIO

import numpy as np
import requests  # type: ignore
import tritonclient.grpc as grpcclient  # type: ignore
from PIL import Image


def get_client() -> grpcclient.InferenceServerClient:
    """
    Creates and returns a Triton Inference Server client connection.
    Sets up keepalive options to maintain the connection.
    """
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,  # Maximum time between pings
            keepalive_timeout_ms=20000,  # Time to wait for ping response
            keepalive_permit_without_calls=False,  # Don't send pings without active calls
            http2_max_pings_without_data=2,  # Maximum pings without data
        )
        # Create client connection to local Triton server
        # Port is forwarded is docker-compose
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:19091", verbose=False, keepalive_options=keepalive_options
        )
        return triton_client
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()


def load_img(
    url: str = "https://cdn.britannica.com/10/250610-050-BC5CCDAF/Zebra-finch-Taeniopygia-guttata-bird.jpg",
) -> np.ndarray:
    """
    Downloads and loads an image from a URL into a numpy array.
    Returns: numpy array with shape (1, height, width, channels)
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    img = Image.open(BytesIO(response.content))
    img_array = np.array(img, dtype=np.uint8)[None, :]  # Add batch dimension

    return img_array


def inference_cnn(triton_client: grpcclient.InferenceServerClient, model_name: str):
    """
    Performs inference using a CNN model on Triton server.
    Uses random test data of shape (1, 3, 32, 32) as input.
    """
    print(f"Model {model_name} is ready: {triton_client.is_model_ready(model_name)}")
    inputs: list[grpcclient.InferInput] = []
    outputs: list[grpcclient.InferRequestedOutput] = []

    # Set up input tensor with random test data
    inputs.append(grpcclient.InferInput("IMAGE", [1, 3, 32, 32], "FP32"))
    inputs[0].set_data_from_numpy(np.random.randn(1, 3, 32, 32).astype(np.float32))

    # Request output logits
    outputs.append(grpcclient.InferRequestedOutput("LOGITS"))

    # Perform inference and get results
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    output0_data = results.as_numpy("LOGITS")
    print(output0_data)


def inference_preprocessing(
    triton_client: grpcclient.InferenceServerClient, model_name: str
):
    """
    Performs image preprocessing inference on Triton server.
    Takes a raw image as input and returns the preprocessed version.
    """
    print(f"Model {model_name} is ready: {triton_client.is_model_ready(model_name)}")
    inputs: list[grpcclient.InferInput] = []
    outputs: list[grpcclient.InferRequestedOutput] = []

    # Load and prepare input image
    img = load_img()
    img_input = grpcclient.InferInput("IMAGE", img.shape, "UINT8")  # 1 x h x w x 3
    img_input.set_data_from_numpy(img)
    inputs.append(img_input)

    # Request preprocessed image output
    output = grpcclient.InferRequestedOutput("IMAGE_PREPROCESSED")
    outputs.append(output)

    # Perform inference and get results
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    output0_data = results.as_numpy("IMAGE_PREPROCESSED")
    print(output0_data.shape)


def inference_ensemble(
    triton_client: grpcclient.InferenceServerClient, model_name: str
):
    """
    Performs inference using an ensemble model that combines preprocessing and CNN.
    Takes a raw image as input and returns the final model predictions.
    """
    print(f"Model {model_name} is ready: {triton_client.is_model_ready(model_name)}")
    inputs: list[grpcclient.InferInput] = []
    outputs: list[grpcclient.InferRequestedOutput] = []

    # Load and prepare input image
    img = load_img()
    img_input = grpcclient.InferInput("IMAGE", img.shape, "UINT8")
    img_input.set_data_from_numpy(img)
    inputs.append(img_input)

    # Request output logits
    output = grpcclient.InferRequestedOutput("LOGITS")
    outputs.append(output)

    # Perform inference and get results
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    output0_data = results.as_numpy("LOGITS")
    print(output0_data)


if __name__ == "__main__":
    # Create Triton client connection
    triton_client = get_client()

    # Test all three inference paths:
    # 1. Image preprocessing only
    # 2. CNN inference with random data
    # 3. Full ensemble (preprocessing + CNN)
    inference_preprocessing(triton_client, model_name="cifar10-preprocessing")
    inference_cnn(triton_client, model_name="cifar10-cnn")
    inference_ensemble(triton_client, model_name="cifar10-ensemble")
