from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar
import platform

import h5py
import numpy as np
import torch
from torch import nn

from emg2qwerty.charset import CharacterSet, charset
from emg2qwerty.transforms import ToTensor, Transform


# adapted from https://github.com/facebookresearch/emg2qwerty/blob/main/emg2qwerty/data.py


@dataclass
class EMGSessionData:
    """A read-only interface to a single emg2qwerty session file stored in
    HDF5 format.

    A session here refers to a span of a few minutes during which two-handed
    EMG signals were recorded while a user typed out a series of prompted
    words/sentences. This class encapsulates the EMG timeseries, ground-truth,
    and additional metadata corresponding to a single session.

    ``self.timeseries`` is a `h5py.Dataset` instance with a compound datatype
    as in a numpy structured array containing three fields - EMG data from the
    left and right wrists, and their corresponding timestamps.
    The sampling rate of EMG is 2kHz, each EMG device has 16 electrode
    channels, and the signal has been high-pass filtered. Therefore, the fields
    corresponding to left and right EMG are 2D arrays of shape ``(T, 16)`` each
    and ``timestamps`` is a 1D array of length ``T``.

    ``self.metadata`` contains two kinds of ground-truth:
      1. A sequence of ``prompts`` displayed to the user (where each prompt
         is a handful of words) along with their start and end timestamps.
         This offers less accurate ground-truth as there is no guarantee that
         the user actually typed out the prompted words accurately without typos.
         It also lacks time alignment of each key-press within the prompt window.
      2. A sequence of ``keystrokes`` indicating the key-presses on a keyboard
         as recorded by a keylogger, along with the timestamps corresponding to
         individual key-presses and the key-releases. This offers high-quality
         and accurate ground-truth as well as temporal alignment of EMG window
         with each key character.

    NOTE: Only the metadata and ground-truth are loaded into memory while the
    EMG data is accesssed directly from disk. When wrapping this interface
    within a PyTorch Dataset, use multiple dataloading workers to mask the
    disk seek and read latencies."""

    HDF5_GROUP: ClassVar[str] = "emg2qwerty"
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG_LEFT: ClassVar[str] = "emg_left"
    EMG_RIGHT: ClassVar[str] = "emg_right"
    TIMESTAMPS: ClassVar[str] = "time"
    SESSION_NAME: ClassVar[str] = "session_name"
    USER: ClassVar[str] = "user"
    CONDITION: ClassVar[str] = "condition"
    DURATION_MINS: ClassVar[str] = "duration_mins"
    KEYSTROKES: ClassVar[str] = "keystrokes"
    PROMPTS: ClassVar[str] = "prompts"

    hdf5_path: Path

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        emg2qwerty_group: h5py.Group = self._file[self.HDF5_GROUP]

        # ``timeseries`` is a compound HDF5 Dataset with aligned left and right EMG
        # along with corresponding timestamps. Avoid loading the entire timeseries
        # into memory here - users should instead rely on PyTorch DataLoader workers
        # to mask the disk seek/read latency.
        self.timeseries: h5py.Dataset = emg2qwerty_group[self.TIMESERIES]
        assert self.timeseries.dtype.fields is not None
        assert self.EMG_LEFT in self.timeseries.dtype.fields
        assert self.EMG_RIGHT in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields

        # Load the metadata entirely into memory as it's rather small.
        self.metadata: dict[str, Any] = {}
        for key, val in emg2qwerty_group.attrs.items():
            if key in {self.KEYSTROKES, self.PROMPTS}:
                self.metadata[key] = json.loads(val)
            else:
                self.metadata[key] = val

    def __enter__(self) -> EMGSessionData:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()

    def __len__(self) -> int:
        return len(self.timeseries)

    def __getitem__(self, key: slice | str) -> np.ndarray:
        return self.timeseries[key]

    def slice(self, start_t: float = -np.inf, end_t: float = np.inf) -> np.ndarray:
        """Load and return a contiguous slice of the timeseries windowed by the
        provided start and end timestamps.

        Args:
            start_t (float): The start time of the window to grab
                (in absolute unix time). Defaults to selecting from the
                beginning of the session. (default: ``-np.inf``).
            end_t (float): The end time of the window to grab
                (in absolute unix time). Defaults to selecting until the
                end of the session. (default: ``np.inf``)
        """

        start_idx, end_idx = self.timestamps.searchsorted([start_t, end_t])
        return self[start_idx:end_idx]

    def ground_truth(
        self, start_t: float = -np.inf, end_t: float = np.inf, reference_time='start',
    ) -> LabelData:
        if self.condition == "on_keyboard":
            return LabelData.from_keystrokes(
                self.keystrokes, start_t=start_t, end_t=end_t, reference_time=reference_time
            )
        else:
            return LabelData.from_prompts(self.prompts, start_t=start_t, end_t=end_t)
        
    @property
    def fields(self) -> list[str]:
        """The names of the fields in the ``timeseries`` compound HDF5 Dataset."""
        return list(self.timeseries.dtype.fields.keys())

    @property
    def timestamps(self) -> np.ndarray:
        """EMG timestamps.

        NOTE: This reads the entire sequence of timesetamps from the underlying
        HDF5 file and therefore incurs disk latency. Avoid this in the critical
        path."""
        timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(timestamps) >= 0).all(), "Timestamps are not monotonic"
        return timestamps

    @property
    def session_name(self) -> str:
        """Unique name of the session."""
        return self.metadata[self.SESSION_NAME]  # type: ignore

    @property
    def user(self) -> str:
        """Unique ID of the user this session corresponds to."""
        return self.metadata[self.USER]  # type: ignore

    @property
    def condition(self) -> str:
        return self.metadata[self.CONDITION]  # type: ignore

    @property
    def duration_mins(self) -> float:
        """The duration of the EMG session in minutes."""
        return self.metadata[self.DURATION_MINS]  # type: ignore

    @property
    def keystrokes(self) -> list[dict[str, Any]]:
        """Sequence of keys recorded by the keylogger during the
        data-collection session along with the press and release timestamps
        for each key."""
        return self.metadata[self.KEYSTROKES]  # type: ignore

    @property
    def prompts(self) -> list[dict[str, Any]]:
        """Sequence of sentences prompted to the user during the
        data-collection session along with the start and end timestamps
        for each prompt."""
        return self.metadata[self.PROMPTS]  # type: ignore

    def __str__(self) -> str:
        """Human-readable string representation for display."""
        return (
            f"{self.__class__.__name__} {self.session_name}: "
            f"({len(self)} EMG samples, "
            f"{len(self.keystrokes)} keystrokes, "
            f"{self.duration_mins:.2f} mins)"
        )


@dataclass
class LabelData:
    """Canonical representation for text/label data together with optional
    character-level timestamps. Supports standarization from keylogger keystrokes,
    prompts, and pynput key representations.

    NOTE: Avoid calling ``LabelData`` constructor directly and instead
    use the provided factory classmethods as much as possible."""

    text: str
    _timestamps: InitVar[Sequence[float] | None] = None
    _charset: CharacterSet = field(default_factory=charset)

    def __post_init__(self, _timestamps: Sequence[float] | None) -> None:
        self.timestamps: np.ndarray | None = None
        if _timestamps is not None:
            self.timestamps = np.array(_timestamps)
            assert self.timestamps.ndim == 1
            assert len(self.timestamps) == len(self.text)
            if not (np.diff(self.timestamps) >= 0).all():
                print('NON MONOTONIC INPUT: ', f"{self.timestamps}, {np.diff(self.timestamps)}")

    @classmethod
    def from_keystrokes(
        cls,
        keystrokes: Sequence[Mapping[str, Any]],
        start_t: float = -np.inf,
        end_t: float = np.inf,
        _charset: CharacterSet | None = None,
        reference_time: str = 'start'
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a sequence of
        keystrokes between the provided start and end timestamps, after
        normalizing and cleaning up as per ``CharacterSet``. The returned
        object also includes the key-press timestamps corresponding to each
        character in ``self.text``.

        Args:
            keystrokes (list): Each keystroke entry in the list should be a
                dict in the format of ``EMGSessionData.keystrokes``.
            start_t (float): The start timestamp of the window in absolute
                unix time. (default: ``-np.inf``)
            end_t (float): The end timestamp of the window in absolute
                unix time. (default: ``np.inf``)
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        label_data = cls(text="", _timestamps=[], _charset=_charset)
        for key in keystrokes:
            if key["start"] > end_t:
                break
            if key["start"] >= start_t:
                label_data += cls.from_key(key, _charset=_charset, reference_time=reference_time)
        return label_data

    @classmethod
    def from_key(
        cls,
        key: str | Mapping[str, Any],
        timestamp: float | None = None,
        _charset: CharacterSet | None = None,
        reference_time: str = 'start'
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a single keystroke,
        after normalizing and cleaning up as per ``CharacterSet``.

        Args:
            key (str or dict): A single pynput.Key string or a keystroke
                dict in the format of ``EMGSessionData.keystrokes``.
            timestamp (float): Optional timestamp corresponding to the key.
                If not provided and if ``key`` is a dict, this will be set to the
                key-press time available in the dict. (default: ``None``)
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        if isinstance(key, str):
            _key = key
        else:
            _key = key["key"]
            timestamp = key[reference_time] if timestamp is None else timestamp

        _key = _charset._normalize_keys([_key])[0]
        if _key not in _charset:  # Out of vocabulary
            return cls(text="", _timestamps=[], _charset=_charset)

        text = _charset.keys_to_str([_key])
        timestamps = [timestamp] if timestamp is not None else None
        # print(key, type(key), timestamps, timestamp)
        return cls(text, timestamps, _charset=_charset)

    @classmethod
    def from_prompts(
        cls,
        prompts: Sequence[Mapping[str, Any]],
        enforce_newline: bool = True,
        start_t: float = -np.inf,
        end_t: float = np.inf,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a sequence of prompts
        between the provided start and end timestamps, after normalizing and
        cleaning up as per ``CharacterSet``. The returned object does not
        include character-level timestamps.

        Assumes the input prompt sequence is sorted by time.

        Args:
            prompts (list): Each prompt entry in the list should be a dict in
                the format of ``EMGSessionData.prompts``.
            enforce_newline (bool): If set, end each prompt with a newline
                if not present already. (default: ``True``)
            start_t (float): The start timestamp of the window in absolute
                unix time. (default: ``-np.inf``)
            end_t (float): The end timestamp of the window in absolute
                unix time. (default: ``np.inf``)
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        label_data = cls(text="", _charset=_charset)
        for prompt in prompts:
            if prompt["start"] > end_t:
                break
            if prompt["start"] >= start_t:
                label_data += cls.from_prompt(
                    prompt,
                    enforce_newline=enforce_newline,
                    _charset=_charset,
                )
        return label_data

    @classmethod
    def from_prompt(
        cls,
        prompt: str | Mapping[str, Any],
        enforce_newline: bool = True,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a single prompt, after
        normalizing and cleaning up as per ``CharacterSet``. The returned
        object does not include character-level timestamps.

        Args:
            prompt (str or dict): A single prompt, either as raw text or a
                dict in the format of ``EMGSessionData.prompts``.
            enforce_newline (bool): If set, end the prompt with a newline
                if not present already. (default: ``True``)
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        if isinstance(prompt, str):
            text = prompt
        else:
            payload = prompt["payload"]
            text = payload["text"] if payload is not None else None

        # Do not add terminal newline if there was no prompt payload
        if text is None:
            return cls(text="", _charset=charset)

        text = _charset.clean_str(text)
        if enforce_newline and (len(text) == 0 or text[-1] != "⏎"):
            text += "⏎"
        return cls(text, _charset=_charset)

    @classmethod
    def from_str(
        cls,
        text: str,
        timestamps: Sequence[float] | None = None,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a raw string, after
        normalizing and cleaning up as per ``CharacterSet``.

        Args:
            text (str): Raw text string to normalize and wrap into ``LabelData``.
            timestamps (list): Optional list of character-level timestamps of the
                same length as ``text``.
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        text = _charset.clean_str(text)
        return cls(text, timestamps, _charset=_charset)

    @classmethod
    def from_labels(
        cls,
        labels: Sequence[int],
        timestamps: Sequence[float] | None = None,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from integer labels
        and optionally together with its corresponding timestamps.

        Args:
            labels (list): Sequene of integer labels belonging to CharacterSet.
            timestamps (list): Optional list of timestamps of the
                same length as ``labels``.
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        text = _charset.labels_to_str(labels)
        return cls(text, timestamps, _charset=_charset)

    @property
    def labels(self) -> np.ndarray:
        """Integer labels corresponding to the label string."""
        labels = self._charset.str_to_labels(self.text)
        return np.asarray(labels, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.text)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LabelData):
            return NotImplemented
        return self.text == other.text

    def __add__(self, other: LabelData) -> LabelData:
        text = self.text + other.text
        if self.timestamps is not None and other.timestamps is not None:
            timestamps = np.append(self.timestamps, other.timestamps)
        else:
            timestamps = None

        return LabelData(text, timestamps, _charset=self._charset)

    def __str__(self) -> str:
        """Human-readable string representation for display."""
        return self.text.replace("⏎", "\n")


@dataclass
class WindowedEMGDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` corresponding to an instance of `EMGSessionData`
    that iterates over EMG windows of configurable length and stride.

    Args:
        hdf5_path (str): Path to the session file in hdf5 format.
        window_length (int): Size of each window. Specify None for no windowing
            in which case this will be a dataset of length 1 containing the
            entire session. (default: ``None``)
        stride (int): Stride between consecutive windows. Specify None to set
            this to window_length, in which case there will be no overlap
            between consecutive windows. (default: ``window_length``)
        padding (tuple[int, int]): Left and right contextual padding for
            windows in terms of number of raw EMG samples.
        jitter (bool): If True, randomly jitter the offset of each window.
            Use this for training time variability. (default: ``False``)
        transform (Callable): A composed sequence of transforms that takes
            a window/slice of `EMGSessionData` in the form of a numpy
            structured array and returns a `torch.Tensor` instance.
            (default: ``emg2qwerty.transforms.ToTensor()``)
    """

    hdf5_path: Path
    loss: str
    window_length: InitVar[int | None] = None
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(default_factory=ToTensor)
    output_metadata: bool = False,

    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ) -> None:
        with EMGSessionData(self.hdf5_path) as session:
            assert (
                session.condition == "on_keyboard"
            ), f"Unsupported condition {self.session.condition}"
            self.session_length = len(session)

        self.window_length = (
            window_length if window_length is not None else self.session_length
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0


    def __len__(self) -> int:
        return int(max(self.session_length - self.window_length, 0) // self.stride + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        # Lazy init `EMGSessionData` per dataloading worker
        # since `h5py.File` objects can't be picked.
        if not hasattr(self, "session"):
            self.session = EMGSessionData(self.hdf5_path)

        offset = idx * self.stride

        # Randomly jitter the window offset.
        leftover = len(self.session) - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Expand window to include contextual padding and fetch.
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]
        # window.dtype.names  are ('emg_left', 'emg_right', 'time')

        # Extract EMG tensor corresponding to the window.
        emg = self.transform(window)
        assert torch.is_tensor(emg)

        # Extract labels corresponding to the original (un-padded) window.
        timestamps = window[EMGSessionData.TIMESTAMPS]                          # timestamps of the window, in unix time:
        start_t = timestamps[offset - window_start]                             # in unix time
        end_t = timestamps[(offset + self.window_length - 1) - window_start]    # in unix time
        if self.loss == 'ctc_loss':
            label_data = self.session.ground_truth(start_t, end_t)
            labels = torch.as_tensor(label_data.labels)
            if self.output_metadata:
                timestamps_indexes = np.searchsorted(timestamps, label_data.timestamps) # index of the labels in the window
                metadata_timestamps_start = timestamps_indexes  # (np.array) timestamps of the labels in timesteps units
                
            return emg, labels, (metadata_timestamps_start if self.output_metadata else None)
        
        elif self.loss == 'cross_entropy_loss':
            label_data = self.session.ground_truth(start_t, end_t, 'end')
            labels = torch.as_tensor(label_data.labels)
            timestamps_releases = label_data.timestamps  # timestamps of the labels in unix time: seconds from 1970-01-01

            if not (np.diff(timestamps_releases) >= 0).all():
                print(f'non monotonic timestamps_releases: {timestamps_releases} {np.diff(timestamps_releases)}')

            if len(timestamps_releases) == 0:
                # if there are no labels, return empty tensor
                return emg, torch.full(timestamps.shape, charset().null_class)

            indexes = np.searchsorted(timestamps, timestamps_releases) # index of the labels in the window

            # resort indexes and labels:
            indexes, labels = zip(*sorted(zip(indexes, labels.tolist())))
            if not (np.diff(indexes) >= 0).all():
                print(f'non monotonic indexes: {indexes} {np.diff(indexes)}')

            indexes = np.insert(indexes, 0, 0) # add the first index (0) at the beginning of the array
            indexes[-1] = len(timestamps) # last index is the last timestamp of the window

            indexes_diff = np.diff(indexes) # number of times each label is present in the ground truth

            outputs = torch.cat( [torch.tensor([lab] * lenght) for lab, lenght in zip(labels, indexes_diff)] ).long()
            return emg, outputs


    @staticmethod
    def collate(
        samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
        continuous_val = False
    ) -> dict[str, torch.Tensor]:
        """Collates a list of samples into a padded batch of inputs and targets.
        Each input sample in the list should be a tuple of (input, target) tensors.
        Also returns the lengths of unpadded inputs and targets for use in loss
        functions such as CTC or RNN-T.

        Follows time-first format. That is, the retured batch is of shape (T, N, ...).
        """

        # samples has lenght n_batch, 2 (input with shape T,bands,C and output with shape number of keys pressed)

        inputs = [sample[0] for sample in samples]  # [(T, ...)] actually not: here it is n_batch, T, bancds, C
        targets = [sample[1] for sample in samples]  # [(T,)]    nope: n_batch, number of keys pressed
        metadata = [sample[2] for sample in samples]

        # Batch of inputs and targets padded along time  --> this function puts time as first dimension and pads so that every input has the same duration
        input_batch = nn.utils.rnn.pad_sequence(inputs)  # (T, N, ...)
        target_batch = nn.utils.rnn.pad_sequence(targets)  # (T, N)

        # Lengths of unpadded input and target sequences for each batch entry
        input_lengths = torch.as_tensor(
            [len(_input) for _input in inputs], dtype=torch.int32
        )
        target_lengths = torch.as_tensor(
            [len(target) for target in targets], dtype=torch.int32
        )

        if metadata[0] is None:
            metadata = None
        else:
            metadata = [torch.as_tensor(m, dtype=torch.int32) for m in metadata]
            metadata = nn.utils.rnn.pad_sequence(metadata)  # the relevant part has length target_lengths

        return {
            "inputs": input_batch,
            "targets": target_batch,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
            "continuous_val": continuous_val,
            "metadata": metadata,
        }


    @staticmethod
    def collateTrue(
        samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return WindowedEMGDataset.collate(samples, True)

    @staticmethod
    def collateFalse(
        samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return WindowedEMGDataset.collate(samples, False)
