import math
import os
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
import torchaudio
from hydra.utils import instantiate
from sentencepiece import SentencePieceProcessor
from torch import Tensor, nn
from torch.jit import TracerWarning
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import cached_file

DIR_NAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR_NAME)  # enable using modules through modeling_gigaam.<module_name>


IMPORT_FLASH = False
SAMPLE_RATE = 16000
LONGFORM_THRESHOLD = 25 * SAMPLE_RATE
_PIPELINE = None


### preprocess ###


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> Tensor:
    """
    Load an audio file and resample it to the specified sample rate.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        audio = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError("Failed to load audio") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return torch.frombuffer(audio, dtype=torch.int16).float() / 32768.0


class SpecScaler(nn.Module):
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(nn.Module):
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """

    def __init__(self, sample_rate: int, features: int, **kwargs):
        super().__init__()
        self.hop_length = kwargs.get("hop_length", sample_rate // 100)
        self.win_length = kwargs.get("win_length", sample_rate // 40)
        self.n_fft = kwargs.get("n_fft", sample_rate // 40)
        self.center = kwargs.get("center", True)
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=features,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                center=self.center,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        if self.center:
            return (
                input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()
            )
        else:
            return (
                (input_lengths - self.win_length)
                .div(self.hop_length, rounding_mode="floor")
                .add(1)
                .long()
            )

    def forward(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)


### utils ###


def onnx_converter(
    model_name: str,
    module: torch.nn.Module,
    out_dir: str,
    inputs: Optional[Tuple[Tensor, ...]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[
        Union[Dict[str, List[int]], Dict[str, Dict[int, str]]]
    ] = None,
    opset_version: int = 17,
):
    if inputs is None:
        inputs = module.input_example()  # type: ignore[operator]
    if input_names is None:
        input_names = module.input_names()  # type: ignore[operator]
    if output_names is None:
        output_names = module.output_names()  # type: ignore[operator]

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    out_path = str(Path(out_dir) / f"{model_name}.onnx")
    saved_dtype = next(module.parameters()).dtype
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=TracerWarning)
        torch.onnx.export(
            module.to(torch.float32),
            inputs,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
    print(f"Succesfully ported onnx {model_name} to {out_path}.")
    module.to(saved_dtype)


def format_time(seconds: float) -> str:
    """
    Formats time in seconds to HH:MM:SS:mm format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    full_seconds = int(seconds)
    milliseconds = int((seconds - full_seconds) * 100)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
    return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"


def rtt_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=x1.ndim - 1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, offset: int = 0
) -> Tuple[Tensor, Tensor]:
    """
    Applies Rotary Position Embeddings to query and key tensors.
    """
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rtt_half(q) * sin), (k * cos) + (rtt_half(k) * sin)


def _normalize_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Normalize device parameter to torch.device."""
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_str)
    if isinstance(device, str):
        return torch.device(device)
    return device


def download_short_audio():
    """Download test audio file if not exists"""
    audio_file = "example.wav"
    if not os.path.exists(audio_file):
        os.system(
            'wget -O example.wav "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav"'
        )
    assert os.path.exists(audio_file), "Short audio file not found"
    return audio_file


def download_long_audio():
    """Download test audio file if not exists"""
    audio_file = "long_example.wav"
    if not os.path.exists(audio_file):
        os.system(
            'wget -O long_example.wav "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/long_example.wav"'
        )
    assert os.path.exists(audio_file), "Long audio file not found"
    return audio_file


class AudioDataset(torch.utils.data.Dataset):
    """
    Helper class for creating batched inputs
    """

    def __init__(self, lst: List[Union[str, np.ndarray, torch.Tensor]]):
        assert isinstance(lst[0], (str, np.ndarray, torch.Tensor)), (
            f"Unexpected dtype: {type(lst[0])}"
        )
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        item = self.lst[idx]
        if isinstance(item, str):
            wav_tns = load_audio(item)
        elif isinstance(item, np.ndarray):
            wav_tns = torch.from_numpy(item)
        elif isinstance(item, torch.Tensor):
            wav_tns = item
        else:
            raise RuntimeError(f"Unexpected sample type: {type(item)} at idx={idx}")
        return wav_tns

    @staticmethod
    def collate(wavs):
        lengths = torch.tensor([len(wav) for wav in wavs])
        max_len = lengths.max().item()
        wav_tns = torch.zeros(len(wavs), max_len, dtype=wavs[0].dtype)
        for idx, wav in enumerate(wavs):
            wav_tns[idx, : wav.shape[-1]] = wav.squeeze()
        return wav_tns, lengths


### vad utils ###


def get_pipeline(device: torch.device):
    """
    Retrieves a PyAnnote voice activity detection pipeline and move it to the specified device.
    The pipeline is loaded only once and reused across subsequent calls.
    It requires the Hugging Face API token to be set in the HF_TOKEN environment variable.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE.to(device)

    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    try:
        hf_token = os.environ["HF_TOKEN"]
    except KeyError as exc:
        raise ValueError("HF_TOKEN environment variable is not set") from exc
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    _PIPELINE = VoiceActivityDetection(segmentation=model)
    _PIPELINE.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

    return _PIPELINE.to(device)


def segment_audio_file(
    wav_file: str,
    sr: int,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """
    Segments an audio waveform into smaller chunks based on speech activity.
    The segmentation is performed using a PyAnnote voice activity detection pipeline.
    """

    audio = load_audio(wav_file)
    pipeline = get_pipeline(device)
    sad_segments = pipeline(wav_file)

    segments: List[torch.Tensor] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    boundaries: List[Tuple[float, float]] = []

    def _update_segments(curr_start: float, curr_end: float, curr_duration: float):
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
        boundaries.append((curr_start, curr_end))

    # Concat segments from pipeline into chunks for asr according to max/min duration
    # Segments longer than strict_limit_duration are splitted manually
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(audio.shape[0] / sr, segment.end)
        if curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

    return segments, boundaries


### encoder ###


class StridingSubsampling(nn.Module):
    """
    Strided Subsampling layer used to reduce the sequence length.
    """

    def __init__(
        self,
        subsampling: str,
        kernel_size: int,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
    ):
        super().__init__()
        self.subsampling_type = subsampling
        assert self.subsampling_type in ["conv1d", "conv2d"]
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = kernel_size
        self._padding = (self._kernel_size - 1) // 2

        layers: List[nn.Module] = []
        in_channels = 1 if self.subsampling_type == "conv2d" else feat_in
        subs_conv_class = (
            torch.nn.Conv2d if self.subsampling_type == "conv2d" else torch.nn.Conv1d
        )
        for _ in range(self._sampling_num):
            layers.append(
                subs_conv_class(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = conv_channels

        out_length = self.calc_output_length(torch.tensor(feat_in))
        if self.subsampling_type == "conv2d":
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def calc_output_length(self, lengths: Tensor) -> Tensor:
        """
        Calculates the output length after applying the subsampling.
        """
        lengths = lengths.to(torch.float)
        add_pad = 2 * self._padding - self._kernel_size
        for _ in range(self._sampling_num):
            lengths = torch.div(lengths + add_pad, self._stride) + 1.0
            lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        if self.subsampling_type == "conv2d":
            x = self.conv(x.unsqueeze(1))
            b, _, t, _ = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        else:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x, self.calc_output_length(lengths)


class MultiHeadAttention(nn.Module, ABC):
    """
    Base class of Multi-Head Attention Mechanisms.
    """

    def __init__(
        self, n_head: int, n_feat: int, flash_attn=False, torch_sdpa_attn=False
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.flash_attn = flash_attn
        self.torch_sdpa_attn = torch_sdpa_attn
        if self.flash_attn and not IMPORT_FLASH:
            raise RuntimeError(
                f"flash_attn_func was imported with err {IMPORT_FLASH_ERR}. "
                "Please install flash_attn or use --no_flash flag. "
                "If you have already done this, "
                "--force-reinstall flag might be useful"
            )

    def forward_qkv(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Projects the inputs into queries, keys, and values for multi-head attention.
        """
        b = query.size(0)
        q = self.linear_q(query).view(b, -1, self.h, self.d_k)
        k = self.linear_k(key).view(b, -1, self.h, self.d_k)
        v = self.linear_v(value).view(b, -1, self.h, self.d_k)
        if self.flash_attn:
            return q, k, v
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward_attention(
        self, value: Tensor, scores: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
        """
        Computes the scaled dot-product attention given the projected values and scores.
        """
        b = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).reshape(b, -1, self.h * self.d_k)
        return self.linear_out(x)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """
    Relative Position Multi-Head Attention module.
    """

    def __init__(self, n_head: int, n_feat: int):
        super().__init__(n_head, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))

    def rel_shift(self, x: Tensor) -> Tensor:
        b, h, qlen, pos_len = x.size()
        x = torch.nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)
        p = self.linear_pos(pos_emb)
        p = p.view(pos_emb.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RotaryPositionMultiHeadAttention(MultiHeadAttention):
    """
    Rotary Position Multi-Head Attention module.
    """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: List[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, t, _ = value.size()
        query = query.transpose(0, 1).view(t, b, self.h, self.d_k)
        key = key.transpose(0, 1).view(t, b, self.h, self.d_k)
        value = value.transpose(0, 1).view(t, b, self.h, self.d_k)

        cos, sin = pos_emb
        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=0)

        q, k, v = self.forward_qkv(
            query.view(t, b, self.h * self.d_k).transpose(0, 1),
            key.view(t, b, self.h * self.d_k).transpose(0, 1),
            value.view(t, b, self.h * self.d_k).transpose(0, 1),
        )

        if not self.flash_attn and not self.torch_sdpa_attn:
            scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.d_k))
            return self.forward_attention(v, scores, mask)
        elif self.flash_attn:
            if mask is None:
                scores = flash_attn_func(q, k, v)
            else:
                scores = apply_masked_flash_attn(q, k, v, mask, self.h, self.d_k)
            scores = scores.view(b, -1, self.h * self.d_k)
            return self.linear_out(scores)
        else:
            attn_mask = None if mask is None else ~mask.unsqueeze(1)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
            )
            attn_output = attn_output.transpose(1, 2).reshape(b, t, self.h * self.d_k)
            return self.linear_out(attn_output)


class PositionalEncoding(nn.Module, ABC):
    """
    Base class of Positional Encodings.
    """

    def __init__(self, dim: int, base: int):
        super().__init__()
        self.dim = dim
        self.base = base

    @abstractmethod
    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        pass

    def extend_pe(self, length: int, device: torch.device):
        """
        Extends the positional encoding buffer to process longer sequences.
        """
        pe = self.create_pe(length, device)
        if pe is None:
            return
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)


class RelPositionalEmbedding(PositionalEncoding):
    """
    Relative Positional Embedding module.
    """

    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        """
        Creates the relative positional encoding matrix.
        """
        if hasattr(self, "pe") and self.pe.shape[1] >= 2 * length - 1:
            return None
        positions = torch.arange(length - 1, -length, -1, device=device).unsqueeze(1)
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.dim, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=pe.device)
            * -(math.log(10000.0) / self.dim)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        input_len = x.size(1)
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        return x, self.pe[:, start_pos:end_pos]


class RotaryPositionalEmbedding(PositionalEncoding):
    """
    Rotary Positional Embedding module.
    """

    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        """
        Creates or extends the rotary positional encoding matrix.
        """
        if hasattr(self, "pe") and self.pe.size(0) >= 2 * length:
            return None
        positions = torch.arange(0, length, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        t = torch.arange(length, device=positions.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(positions.device)
        return torch.cat([emb.cos()[:, None, None, :], emb.sin()[:, None, None, :]])

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, List[Tensor]]:
        cos_emb = self.pe[0 : x.shape[1]]
        half_pe = self.pe.shape[0] // 2
        sin_emb = self.pe[half_pe : half_pe + x.shape[1]]
        return x, [cos_emb, sin_emb]


class ConformerConvolution(nn.Module):
    """
    Conformer Convolution module.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        norm_type: str,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        assert norm_type in ["batch_norm", "layer_norm"]
        self.norm_type = norm_type
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        self.batch_norm = (
            nn.BatchNorm1d(d_model)
            if norm_type == "batch_norm"
            else nn.LayerNorm(d_model)
        )
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        if self.norm_type == "batch_norm":
            x = self.batch_norm(x)
        else:
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerFeedForward(nn.Module):
    """
    Conformer Feed Forward module.
    """

    def __init__(self, d_model: int, d_ff: int, use_bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class ConformerLayer(nn.Module):
    """
    Conformer Layer module.
    This module combines several submodules including feed forward networks,
    depthwise separable convolution, and multi-head self-attention
    to form a single Conformer block.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        self_attention_model: str,
        n_heads: int = 16,
        conv_norm_type: str = "batch_norm",
        conv_kernel_size: int = 31,
        flash_attn: bool = False,
    ):
        super().__init__()
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
        )
        self.norm_self_att = nn.LayerNorm(d_model)
        if self_attention_model == "rotary":
            self.self_attn: nn.Module = RotaryPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                flash_attn=flash_attn,
                torch_sdpa_attn=not flash_attn,
            )
        else:
            assert not flash_attn, "Not supported flash_attn for rel_pos"
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
            )
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        pos_emb: Union[Tensor, List[Tensor]],
        att_mask: Optional[Tensor] = None,
        pad_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(x, x, x, pos_emb, mask=att_mask)
        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        x = self.norm_out(residual)
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder module.
    This module encapsulates the entire Conformer encoder architecture,
    consisting of a StridingSubsampling layer, positional embeddings, and
    a stack of Conformer Layers.
    It serves as the main component responsible for processing speech features.
    """

    def __init__(
        self,
        feat_in: int = 64,
        n_layers: int = 16,
        d_model: int = 768,
        subsampling: str = "conv2d",
        subs_kernel_size: int = 3,
        subsampling_factor: int = 4,
        ff_expansion_factor: int = 4,
        self_attention_model: str = "rotary",
        n_heads: int = 16,
        pos_emb_max_len: int = 5000,
        conv_norm_type: str = "batch_norm",
        conv_kernel_size: int = 31,
        flash_attn: bool = False,
    ):
        super().__init__()
        self.feat_in = feat_in
        assert self_attention_model in [
            "rotary",
            "rel_pos",
        ], f"Not supported attn = {self_attention_model}"

        self.pre_encode = StridingSubsampling(
            subsampling=subsampling,
            kernel_size=subs_kernel_size,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=d_model,
        )

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rotary":
            self.pos_enc: PositionalEncoding = RotaryPositionalEmbedding(
                d_model // n_heads, pos_emb_max_len
            )
        else:
            self.pos_enc = RelPositionalEmbedding(d_model, pos_emb_max_len)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_model * ff_expansion_factor,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_norm_type=conv_norm_type,
                conv_kernel_size=conv_kernel_size,
                flash_attn=flash_attn,
            )
            self.layers.append(layer)

    def input_example(
        self,
        batch_size: int = 1,
        seqlen: int = 200,
    ) -> Tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        features = torch.zeros(batch_size, self.feat_in, seqlen)
        feature_lengths = torch.full([batch_size], features.shape[-1])
        return features.float().to(device), feature_lengths.to(device)

    def input_names(self) -> List[str]:
        return ["audio_signal", "length"]

    def output_names(self) -> List[str]:
        return ["encoded", "encoded_len"]

    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "audio_signal": {0: "batch_size", 2: "seq_len"},
            "length": {0: "batch_size"},
            "encoded": {0: "batch_size", 1: "seq_len"},
            "encoded_len": {0: "batch_size"},
        }

    def forward(self, audio_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        if not hasattr(self.pos_enc, "pe"):
            self.pos_enc.extend_pe(self.pos_emb_max_len, audio_signal.device)

        audio_signal, length = self.pre_encode(
            x=audio_signal.transpose(1, 2), lengths=length
        )

        max_len = audio_signal.size(1)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)

        att_mask = None
        if audio_signal.shape[0] > 1:
            att_mask = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
            att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
            att_mask = ~att_mask

        pad_mask = ~pad_mask

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                pos_emb=pos_emb,
                att_mask=att_mask,
                pad_mask=pad_mask,
            )

        return audio_signal.transpose(1, 2), length


### decoders ###


class RNNTJoint(nn.Module):
    """
    RNN-Transducer Joint Network Module.
    This module combines the outputs of the encoder and the prediction network using
    a linear transformation followed by ReLU activation and another linear projection.
    """

    def __init__(
        self, enc_hidden: int, pred_hidden: int, joint_hidden: int, num_classes: int
    ):
        super().__init__()
        self.enc_hidden = enc_hidden
        self.pred_hidden = pred_hidden
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.enc = nn.Linear(enc_hidden, joint_hidden)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(joint_hidden, num_classes))

    def joint(self, encoder_out: Tensor, decoder_out: Tensor) -> Tensor:
        """
        Combine the encoder and prediction network outputs into a joint representation.
        """
        enc = self.enc(encoder_out).unsqueeze(2)
        pred = self.pred(decoder_out).unsqueeze(1)
        return self.joint_net(enc + pred).log_softmax(-1)

    def input_example(self) -> Tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        enc = torch.zeros(1, self.enc_hidden, 1)
        dec = torch.zeros(1, self.pred_hidden, 1)
        return enc.float().to(device), dec.float().to(device)

    def input_names(self) -> List[str]:
        return ["enc", "dec"]

    def output_names(self) -> List[str]:
        return ["joint"]

    def forward(self, enc: Tensor, dec: Tensor) -> Tensor:
        return self.joint(enc.transpose(1, 2), dec.transpose(1, 2))


class RNNTDecoder(nn.Module):
    """
    RNN-Transducer Decoder Module.
    This module handles the prediction network part of the RNN-Transducer architecture.
    """

    def __init__(self, pred_hidden: int, pred_rnn_layers: int, num_classes: int):
        super().__init__()
        self.blank_id = num_classes - 1
        self.pred_hidden = pred_hidden
        self.embed = nn.Embedding(num_classes, pred_hidden, padding_idx=self.blank_id)
        self.lstm = nn.LSTM(pred_hidden, pred_hidden, pred_rnn_layers)

    def predict(
        self,
        x: Optional[Tensor],
        state: Optional[Tensor],
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Make predictions based on the current input and previous states.
        If no input is provided, use zeros as the initial input.
        """
        if x is not None:
            emb: Tensor = self.embed(x)
        else:
            emb = torch.zeros(
                (batch_size, 1, self.pred_hidden), device=next(self.parameters()).device
            )
        g, hid = self.lstm(emb.transpose(0, 1), state)
        return g.transpose(0, 1), hid

    def input_example(self) -> Tuple[Tensor, Tensor, Tensor]:
        device = next(self.parameters()).device
        label = torch.tensor([[0]]).to(device)
        hidden_h = torch.zeros(1, 1, self.pred_hidden).to(device)
        hidden_c = torch.zeros(1, 1, self.pred_hidden).to(device)
        return label, hidden_h, hidden_c

    def input_names(self) -> List[str]:
        return ["x", "h", "c"]

    def output_names(self) -> List[str]:
        return ["dec", "h", "c"]

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        ONNX-specific forward with x, state = (h, c) -> x, h, c.
        """
        emb = self.embed(x)
        g, (h, c) = self.lstm(emb.transpose(0, 1), (h, c))
        return g.transpose(0, 1), h, c


class RNNTHead(nn.Module):
    """
    RNN-Transducer Head Module.
    This module combines the decoder and joint network components of the RNN-Transducer architecture.
    """

    def __init__(self, decoder: Dict[str, int], joint: Dict[str, int]):
        super().__init__()
        self.decoder = RNNTDecoder(**decoder)
        self.joint = RNNTJoint(**joint)


### decoding ###


class Tokenizer:
    """
    Tokenizer for converting between text and token IDs.
    The tokenizer can operate either character-wise or using a pre-trained SentencePiece model.
    """

    def __init__(self, vocab: List[str], model_path: Optional[str] = None):
        self.charwise = model_path is None
        if self.charwise:
            self.vocab = vocab
        else:
            self.model = SentencePieceProcessor()
            self.model.load(model_path)

    def decode(self, tokens: List[int]) -> str:
        """
        Convert a list of token IDs back to a string.
        """
        if self.charwise:
            return "".join(self.vocab[tok] for tok in tokens)
        return self.model.decode(tokens)

    def __len__(self):
        """
        Get the total number of tokens in the vocabulary.
        """
        return len(self.vocab) if self.charwise else len(self.model)


class RNNTGreedyDecoding:
    def __init__(
        self,
        vocabulary: List[str],
        model_path: Optional[str] = None,
        max_symbols_per_step: int = 10,
    ):
        """
        Class for performing greedy decoding of RNN-T outputs.
        """
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)
        self.max_symbols = max_symbols_per_step

    def _greedy_decode(self, head: RNNTHead, x: Tensor, seqlen: Tensor) -> str:
        """
        Internal helper function for performing greedy decoding on a single sequence.
        """
        hyp: List[int] = []
        dec_state: Optional[Tensor] = None
        last_label: Optional[Tensor] = None
        for t in range(seqlen):
            f = x[t, :, :].unsqueeze(1)
            not_blank = True
            new_symbols = 0
            while not_blank and new_symbols < self.max_symbols:
                g, hidden = head.decoder.predict(last_label, dec_state)
                k = head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item()
                if k == self.blank_id:
                    not_blank = False
                else:
                    hyp.append(int(k))
                    dec_state = hidden
                    last_label = torch.tensor([[hyp[-1]]]).to(x.device)
                    new_symbols += 1

        return self.tokenizer.decode(hyp)

    @torch.inference_mode()
    def decode(self, head: RNNTHead, encoded: Tensor, enc_len: Tensor) -> List[str]:
        """
        Decode the output of an RNN-T model into a list of hypotheses.
        """
        b = encoded.shape[0]
        pred_texts = []
        encoded = encoded.transpose(1, 2)
        for i in range(b):
            inseq = encoded[i, :, :].unsqueeze(1)
            pred_texts.append(self._greedy_decode(head, inseq, enc_len[i]))
        return pred_texts


### models ###


class GigaAM(nn.Module):
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)

    def forward(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform forward pass through the preprocessor and encoder.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        if self._device.type == "cpu":
            return self.encoder(features, feature_lengths)
        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            return self.encoder(features, feature_lengths)

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_wav(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Prepare an audio file for processing by loading it onto
        the correct device and converting its format.
        """
        wav = load_audio(wav_file)
        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)
        return wav, length

    def embed_audio(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Extract audio representations using the GigaAM model.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        self._to_onnx(dir_path)
        omegaconf.OmegaConf.save(self.cfg, f"{dir_path}/{self.cfg.model_name}.yaml")

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        onnx_converter(
            model_name=f"{self.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.encoder,
            dynamic_axes=self.encoder.dynamic_axes(),
        )


class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model for Speech Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    @torch.inference_mode()
    def transcribe(self, wav_file: str) -> str:
        """
        Transcribes a short audio file into text.
        """
        wav, length = self.prepare_wav(wav_file)
        if length.item() > LONGFORM_THRESHOLD:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

        encoded, encoded_len = self.forward(wav, length)
        return self.decoding.decode(self.head, encoded, encoded_len)[0]

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        return self.head(self.encoder(features, feature_lengths)[0])

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx ASR model.
        `ctc`:  exported entirely in encoder-decoder format.
        `rnnt`: exported in encoder/decoder/joint parts separately.
        """
        if "ctc" in self.cfg.model_name:
            saved_forward = self.forward
            self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
            onnx_converter(
                model_name=self.cfg.model_name,
                out_dir=dir_path,
                module=self,
                inputs=self.encoder.input_example(),
                input_names=["features", "feature_lengths"],
                output_names=["log_probs"],
                dynamic_axes={
                    "features": {0: "batch_size", 2: "seq_len"},
                    "feature_lengths": {0: "batch_size"},
                    "log_probs": {0: "batch_size", 1: "seq_len"},
                },
            )
            self.forward = saved_forward  # type: ignore[assignment, method-assign]
        else:
            super()._to_onnx(dir_path)  # export encoder
            onnx_converter(
                model_name=f"{self.cfg.model_name}_decoder",
                out_dir=dir_path,
                module=self.head.decoder,
            )
            onnx_converter(
                model_name=f"{self.cfg.model_name}_joint",
                out_dir=dir_path,
                module=self.head.joint,
            )

    @torch.inference_mode()
    def transcribe_longform(
        self, wav_file: str, **kwargs
    ) -> List[Dict[str, Union[str, Tuple[float, float]]]]:
        """
        Transcribes a long audio file by splitting it into segments and
        then transcribing each segment.
        """
        transcribed_segments = []
        segments, boundaries = segment_audio_file(
            wav_file, SAMPLE_RATE, device=self._device, **kwargs
        )
        for segment, segment_boundaries in zip(segments, boundaries):
            wav = segment.to(self._device).unsqueeze(0).to(self._dtype)
            length = torch.full([1], wav.shape[-1], device=self._device)
            encoded, encoded_len = self.forward(wav, length)
            result = self.decoding.decode(self.head, encoded, encoded_len)[0]
            transcribed_segments.append(
                {"transcription": result, "boundaries": segment_boundaries}
            )
        return transcribed_segments


class GigaAMEmo(GigaAM):
    """
    Giga Acoustic Model for Emotion Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.id2name = cfg.id2name

    def get_probs(self, wav_file: str) -> Dict[str, float]:
        """
        Calculate probabilities for each emotion class based on the provided audio file.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, _ = self.forward(wav, length)
        encoded_pooled = nn.functional.avg_pool1d(
            encoded, kernel_size=encoded.shape[-1]
        ).squeeze(-1)

        logits = self.head(encoded_pooled)[0]
        probs = nn.functional.softmax(logits, dim=-1).detach().tolist()

        return {self.id2name[i]: probs[i] for i in range(len(self.id2name))}

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        encoded, _ = self.encoder(features, feature_lengths)
        enc_pooled = encoded.mean(dim=-1)
        return nn.functional.softmax(self.head(enc_pooled), dim=-1)

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx Emo model.
        """
        saved_forward = self.forward
        self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
        onnx_converter(
            model_name=self.cfg.model_name,
            out_dir=dir_path,
            module=self,
            inputs=self.encoder.input_example(),
            input_names=["features", "feature_lengths"],
            output_names=["probs"],
            dynamic_axes={
                "features": {0: "batch_size", 2: "seq_len"},
                "feature_lengths": {0: "batch_size"},
                "probs": {0: "batch_size", 1: "seq_len"},
            },
        )
        self.forward = saved_forward  # type: ignore[assignment, method-assign]


### transformers ###


class GigaAMConfig(PretrainedConfig):
    model_type = "gigaam"

    def __init__(self, cfg: omegaconf.DictConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg


class GigaAMModel(PreTrainedModel):
    config_class = GigaAMConfig
    base_model_prefix = "gigaam"

    def __init__(self, config: GigaAMConfig):
        super().__init__(config)
        self.config = config
        if (
            "decoding" in self.config.cfg["model"]["cfg"]
            and "model_path" in self.config.cfg["model"]["cfg"]["decoding"]
        ):
            resolved_tokenizer_path = cached_file(
                config.name_or_path,
                "tokenizer.model",
                revision=getattr(config, "_commit_hash", None),
                cache_dir=getattr(config, "cache_dir", None),
                use_auth_token=getattr(config, "use_auth_token", None),
            )
            self.config.cfg["model"]["cfg"]["decoding"]["model_path"] = (
                resolved_tokenizer_path
            )

        self.model = instantiate(config.cfg["model"], _recursive_=False)

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor):
        return self.model(features, feature_lengths)

    def embed_audio(self, wav_file: str) -> torch.Tensor:
        return self.model.embed_audio(wav_file)

    def transcribe(self, wav_file: str) -> str:
        return self.model.transcribe(wav_file)

    def transcribe_longform(
        self, wav_file: str
    ) -> List[Dict[str, Union[str, Tuple[float, float]]]]:
        return self.model.transcribe_longform(wav_file)

    def get_probs(self, wav_file: str) -> Dict[str, float]:
        return self.model.get_probs(wav_file)

    @torch.no_grad()
    def to_onnx(self, dir_path: str = ".") -> None:
        self.model.to_onnx(dir_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
