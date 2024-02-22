import torch
import io
import boto3

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
            buffer = io.BytesIO(f.read(tensor_size))
            if buffer.getbuffer().nbytes == 0:
                break 
            buffer.seek(0)
            tensor = torch.load(buffer)
            tensors.append(tensor)
    return tensors


def delete_s3_subfolder(bucket_name: str, prefix: str):
    """Usage:
    bucket = s3.Bucket('your_bucket_name')
    prefix = 'path/to/your/subfolder/' 
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.objects.filter(Prefix=prefix).delete()