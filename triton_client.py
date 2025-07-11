import tritonclient.grpc as grpcclient
import numpy as np

class TritonClient:
    def __init__(self, url: str, model_name: str):
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)

    def infer(self, prompt: str) -> str:
        """
        Send prompt to Triton LLM model and get generated text response.
        Assumes the model accepts a string input tensor named 'PROMPT' and returns
        output tensor 'OUTPUT' as bytes.

        Modify input/output names per your model config.
        """
        from tritonclient.utils import InferenceInput, InferenceRequestedOutput

        # Prepare input tensor
        input_data = np.array([prompt.encode('utf-8')], dtype=object)
        inputs = []
        inputs.append(InferenceInput(name="PROMPT", shape=[1], datatype="BYTES"))
        inputs[0].set_data_from_numpy(input_data)

        outputs = []
        outputs.append(InferenceRequestedOutput(name="OUTPUT"))

        response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        output_data = response.as_numpy("OUTPUT")
        # output_data is an array of bytes, decode first element
        return output_data[0].decode("utf-8")
