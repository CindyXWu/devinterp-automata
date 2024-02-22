import glob
import io
import logging
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ( Any, Callable, Generic, List, 
                    Optional,  Set, Tuple,  TypeVar, Union)

import boto3
import torch
from pydantic import BaseModel, Field, validator

from devinfra.monitoring import process_steps

logger = logging.getLogger(__name__)


T = TypeVar("T")

IDType = TypeVar("IDType", bound=Union[int, str, Tuple[Union[int, str], ...]])


class BaseStorageProvider(Generic[IDType], ABC):
    """Base class for storage providers.

    Args:
        id_to_key (Callable, optional): Function to map a file ID to a storage key.
                                        Defaults to None.
        key_to_id (Callable, optional): Function to map a storage key to a file ID.
                                        Defaults to None.
        device (str, optional): Device to use for loading files. Defaults to "cpu".
        root_dir (str, optional): Root directory for storage. Defaults to "data".

    Attributes:
        file_ids (List[IDType]): List of file IDs in the storage provider.

    """

    def __init__(
        self,
        id_to_key: Optional[Callable[[IDType], str]] = None,
        key_to_id: Optional[Callable[[str], IDType]] = None,
        device: Optional[str] = None,
        root_dir: str = "data",
    ):
        self._id_to_key = id_to_key or self.default_id_to_key
        self._key_to_id = key_to_id or self.default_key_to_id
        self.device = torch.device(device) if device else None
        self.root_dir = Path(root_dir)
        self.file_ids: List[IDType] = []

        self.sync()

    @abstractmethod
    def save_file(self, file_id: IDType, file: Any):
        """Abstract method to save a file."""
        raise NotImplementedError

    @abstractmethod
    def load_file(self, file_id: IDType):
        """Abstract method to load a file."""
        raise NotImplementedError

    @abstractmethod
    def get_file_ids(self) -> List[IDType]:
        """Abstract method to get a list of file IDs."""
        raise NotImplementedError

    def id_to_key(self, file_id: IDType) -> str:
        """Map a file ID to a storage key."""
        return str(self.root_dir / self._id_to_key(file_id))

    def key_to_id(self, key: str) -> IDType:
        """Map a storage key to a file ID."""
        root_dir = str(self.root_dir)

        if not key.startswith(root_dir):
            raise ValueError(f"Key `{key}` does not start with root_dir `{root_dir}`")
        else:
            key = key[len(root_dir) + 1 :]

        return self._key_to_id(key)

    @staticmethod
    def default_id_to_key(file_id: IDType) -> str:
        """Default method to map a file ID to a storage key."""
        return f"{file_id}.pt"

    @staticmethod
    def default_key_to_id(key: str) -> IDType:
        """Default method to map a storage key to a file ID."""
        warnings.warn(
            "Using default key_to_id. This yields a string, which may not be what you want."
        )
        return key.split("/")[-1].replace(".pt", "")  # type: ignore

    def sync(self):
        self.file_ids = self.get_file_ids()
        return self

    def __iter__(self):
        for file_id in self.file_ids:
            yield self.load_file(file_id)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx: Union[int, IDType]):
        if isinstance(idx, int):
            return self.load_file(self.file_ids[idx])

        elif idx not in self.file_ids:
            warnings.warn(f"File with id `{idx}` not found.")
            return self.load_file(idx)

        raise TypeError(f"Invalid argument `{idx}` of type `{type(idx)}`")

    def __contains__(self, file_id):
        return file_id in self.file_ids


class LocalStorageProvider(BaseStorageProvider[IDType]):
    """Local storage provider.

    Args:
        id_to_key (Callable): Function to map a file ID to a storage key.
        key_to_id (Callable): Function to map a storage key to a file ID.
        device (str): Device to use for loading files.
        root_dir (str): Base directory in which to save files locally.
    """

    def __init__(
        self,
        id_to_key: Optional[Callable[[IDType], str]] = None,
        key_to_id: Optional[Callable[[str], IDType]] = None,
        device: Optional[str] = None,
        root_dir: str = "data",
    ):
        super().__init__(id_to_key, key_to_id, device=device, root_dir=root_dir)

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def save_file(self, file_id: IDType, file: Any):
        """Save a file locally."""
        torch.save(file, self.id_to_key(file_id))

    def load_file(self, file_id: IDType):
        """Load a file locally."""
        return torch.load(self.id_to_key(file_id))

    def get_file_ids(self) -> List[IDType]:
        """
        Returns a list of tuples of all files in the local directory.
        """
        files = glob.glob(f"{self.root_dir}/*")
        return sorted(
            [self.key_to_id(str(self.root_dir / os.path.basename(f))) for f in files]
        )

    def __repr__(self):
        return f"LocalStorageProvider({self.root_dir})"


class S3StorageProvider(BaseStorageProvider[IDType]):
    """AWS S3 Storage Provider.

    Args:
        bucket_name (str): Name of the S3 bucket.
        id_to_key (Callable): Function to map a file ID to a storage key.
        key_to_id (Callable): Function to map a storage key to a file ID.
        device (str): Device to use for loading files.
        root_dir (str, optional): Root directory for storage. Defaults to "data".
    """

    def __init__(
        self,
        bucket_name: str,
        id_to_key: Optional[Callable[[IDType], str]] = None,
        key_to_id: Optional[Callable[[str], IDType]] = None,
        root_dir: str = "data",
        device: Optional[str] = None,
    ):
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            raise EnvironmentError("AWS environment variables not set.")

        self.client = boto3.client("s3")
        self.bucket_name = bucket_name
        super().__init__(id_to_key, key_to_id, device=device, root_dir=root_dir)

    def save_file(self, file_id: IDType, file: Any):
        """Save a file to an S3 bucket."""
        key = self.id_to_key(file_id)
        buffer = io.BytesIO()
        torch.save(file, buffer)
        self.client.put_object(Bucket=self.bucket_name, Key=key, Body=buffer.getvalue())

    def load_file(self, file_id: IDType):
        """Load a file from an S3 bucket."""
        key = self.id_to_key(file_id)
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        buffer = io.BytesIO(response["Body"].read())

        if self.device:
            return torch.load(buffer, map_location=self.device)

        return torch.load(buffer)

    def get_file_ids(self) -> List[IDType]:
        """
        Returns a list of tuples of all files in the bucket directory.
        """
        logger.info("Retrieving file IDs from S3 bucket...")
        response = self.client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=str(self.root_dir)
        )

        if "Contents" in response:
            logger.info("Found %d files in S3 bucket.", len(response["Contents"]))

            return sorted(
                [self.key_to_id(item["Key"]) for item in response["Contents"]]
            )

        warnings.warn(f"No files found in bucket {self.bucket_name}.")
        logger.info(response)
        return []

    def __repr__(self):
        return f"S3StorageProvider(s3://{self.bucket_name}/{self.root_dir})"


class CompositeStorageProvider(BaseStorageProvider[IDType]):
    """Composite storage provider that can use multiple providers.

    Args:
        providers (List[BaseStorageProvider]): List of storage providers to use.

    # TODO: Create a StorageProvider Protocol and use that instead of BaseStorageProvider.
    """

    def __init__(self, providers: List[BaseStorageProvider[IDType]]):
        self.providers = providers
        super().__init__()

    def save_file(self, file_id: IDType, file: Any):
        """Save a file using all the underlying storage providers."""
        for provider in self.providers:
            provider.save_file(file_id, file)

    def load_file(self, file_id: IDType):
        """Load a file from one of the underlying storage providers."""
        for provider in self.providers:
            try:
                return provider.load_file(file_id)
            except FileNotFoundError:
                continue
        raise FileNotFoundError("File not found in any provider")

    def get_file_ids(self) -> List[IDType]:
        """
        Returns a list of tuples of all files in the bucket directory.
        """

        file_ids = set()

        for provider in self.providers:
            file_ids |= set(provider.get_file_ids())

        return sorted(list(file_ids))

    def __repr__(self):
        return f"CompositeStorageProvider({self.providers})"

    def sync(self):
        for provider in self.providers:
            provider.sync()

        self.file_ids = self.get_file_ids()


def create_storage_provider(
    bucket_name: Optional[str] = None,
    local_root: Optional[str] = None,
    root_dir: str = "data",
    device: Optional[str] = None,
    id_to_key: Optional[Callable[[IDType], str]] = None,
    key_to_id: Optional[Callable[[str], IDType]] = None,
):
    """Factory for creating a composite storage provider."""

    def create_provider(
        provider_type: str, **kwargs
    ) -> Union[S3StorageProvider, LocalStorageProvider]:
        if provider_type == "s3":
            return S3StorageProvider(**kwargs)
        elif provider_type == "local":
            return LocalStorageProvider(**kwargs)

        raise ValueError("Invalid provider_type.")

    def create_composite_provider(
        types_and_configs: List[Tuple[str, dict]]
    ) -> CompositeStorageProvider:
        providers = [
            create_provider(
                t, device=device, id_to_key=id_to_key, key_to_id=key_to_id, **c
            )
            for t, c in types_and_configs
        ]
        return CompositeStorageProvider(providers)

    providers = []

    if local_root:
        providers.append(("local", {"root_dir": local_root + "/" + root_dir}))

    if bucket_name:
        providers.append(
            (
                "s3",
                {"bucket_name": bucket_name, "root_dir": root_dir},
            )
        )

    return create_composite_provider(providers)


def int_id_to_key(file_id: int) -> str:
    """Map an integer file ID to a storage key."""
    return f"{file_id}.pt"


def key_to_int_id(key: str) -> int:
    """Map a storage key to an integer file ID."""
    file_name = key.split("/")[-1]
    file_id = file_name.split(".")[0]
    return int(file_id)


class CheckpointerConfig(BaseModel):
    checkpoint_steps: Set[int] = Field(default_factory=set)
    project_dir: str = "checkpoints"
    bucket_name: Optional[str] = None
    local_root: Optional[str] = None
    device: Optional[str] = None

    class Config:
        frozen = True

    @validator("checkpoint_steps", pre=True, always=True)
    @classmethod
    def validate_checkpoint_steps(cls, v):
        """Validate `checkpoint_steps`."""
        return process_steps(v)

    def factory(self):
        return create_storage_provider(
            bucket_name=self.bucket_name,
            local_root=self.local_root,
            root_dir=f"checkpoints/{self.project_dir}",
            device=self.device,
            id_to_key=int_id_to_key,
            key_to_id=key_to_int_id,
        )

    def __repr_args__(self):
        return [
            ("bucket_name", self.bucket_name),
            ("project_dir", self.project_dir),
            ("local_root", self.local_root),
            (
                "checkpoint_steps",
                f"({min(self.checkpoint_steps)}...{max(self.checkpoint_steps)}) {len(self.checkpoint_steps)} steps",
            ),
        ]

    def __repr__(self):
        return f"CheckpointerConfig({', '.join(f'{k}={v}' for k, v in self.__repr_args__())})"
