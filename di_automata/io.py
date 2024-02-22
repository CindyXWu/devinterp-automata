import torch
import io

from di_automata.config_setup import MainConfig


def append_tensor_to_file(tensor: torch.Tensor, file_path: str):
    """Serialize tensor to from buffer to bytes for repeated appending.
    Use to iteratively save logits to avoid excessive RAM use during training.
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer) 
    with open(file_path, "ab") as f:
        f.write(buffer.getvalue())


def read_tensors_from_file(file_path: str, config: MainConfig) -> list[torch.Tensor]:
    """Assume tensors are float32."""
    tensors = []
    num_elements = config.rlct_config.ed_config.batches_per_checkpoint * config.dataloader_config.train_bs * config.task_config.output_vocab_size * config.task_config.length
    itemsize = torch.FloatTensor().element_size()  # float32 - if float64, then DoubleTensor()
    tensor_size = itemsize * num_elements  # Total size of one tensor in bytes

    with open(file_path, "rb") as f:
        while True:
            buffer = io.BytesIO(f.read(tensor_size))  # Read a segment of the file into a buffer
            if buffer.getbuffer().nbytes == 0:
                break  # If no bytes were read, end loop
            buffer.seek(0)  # Ensure the buffer's read pointer is at the start
            tensor = torch.load(buffer)  # Deserialize the tensor from the buffer
            tensors.append(tensor)
    return tensors