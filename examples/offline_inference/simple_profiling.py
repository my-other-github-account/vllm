# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import os

os.environ["VLLM_USE_SPECIALIZED_MODELS"] = "1"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    [0] * 10_000,
    [1] * 10_000,
    [2] * 10_000,
    [3] * 10_000,
    [4] * 10_000,
    [5] * 10_000,
    [6] * 10_000,
    [7] * 10_000,
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)


def main():
    # Create an LLM.
    llm = LLM(
        model="nvidia/DeepSeek-V3.2-NVFP4",
        tensor_parallel_size=4,
        kernel_config={"enable_flashinfer_autotune": False},
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": f"./vllm_profile/bsz{len(prompts)}/",
        },
        enable_prefix_caching=False,
        load_format="dummy",
        compilation_config={"max_cudagraph_capture_size": 64},
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        max_num_batched_tokens=32768,
    )

    outputs = llm.generate(prompts, sampling_params)
    llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()
