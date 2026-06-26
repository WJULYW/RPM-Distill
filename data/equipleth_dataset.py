import os
import pickle
from typing import List, Tuple

import numpy as np

from data.organizer import Organizer
from data.rf_processing import create_fast_slow_matrix, find_range

try:
    import cv2
except Exception:
    cv2 = None


def _read_image(path: str):
    if cv2 is not None:
        return cv2.imread(path)
    try:
        import imageio.v2 as iio

        image = iio.imread(path)
    except Exception:
        return None
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] > 3:
        image = image[:, :, :3]
    return image


def _resize_image(image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(image, image_size)
    target_w, target_h = image_size
    src_h, src_w = image.shape[:2]
    if src_h == target_h and src_w == target_w:
        return image
    y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int32)
    return image[y_idx][:, x_idx]


class PairedVideoRFWindowDataset:
    """
    EquiPleth paired RGB/RF/PPG dataset with strict window alignment.

    Each sample returns:
    - RGB window: (C, T, H, W)
    - RF window: (10, T)
    - PPG label: (T,)
    """

    ppg_offset = 25

    def __init__(
        self,
        datapath: str,
        datapaths: List[str],
        recording_str: str = "rgbd_rgb",
        ppg_str: str = "rgbd",
        video_length: int = 900,
        frame_length: int = 256,
        step: int = 30,
        sampling_ratio: int = 4,
        window_size: int = 5,
        samp_f: float = 5e6,
        freq_slope: float = 60.012e12,
        samples: int = 256,
        image_size: Tuple[int, int] = (128, 128),
        **kwargs,
    ):
        del kwargs
        self.datapath = datapath
        self.video_list = datapaths.copy()

        self.recording_str = recording_str
        self.ppg_str = ppg_str
        self.video_length = video_length
        self.frame_length = frame_length
        self.step = step
        self.sampling_ratio = sampling_ratio
        self.window_size = window_size
        self.samp_f = samp_f
        self.freq_slope = freq_slope
        self.samples = samples
        self.image_size = image_size

        self.signal_list: List[np.ndarray] = []
        self.rf_data_list: List[np.ndarray] = []
        self.all_idxs: List[Tuple[int, int]] = []

        self._initialize_dataset()

    def _initialize_dataset(self) -> None:
        self._load_signals()
        self._normalize_signals()
        self._load_rf_data()
        self._generate_indices()

    @staticmethod
    def _resolve_rf_folder(video_name: str) -> str:
        if video_name.startswith("v_"):
            parts = video_name.split("_")
            if len(parts) >= 3:
                return f"{parts[1]}_{parts[2]}"
        return video_name

    def _load_signals(self) -> None:
        keep_videos: List[str] = []
        keep_signals: List[np.ndarray] = []

        for video_name in self.video_list:
            rgb_path = os.path.join(self.datapath, "rgb_files", video_name)
            ppg_file = os.path.join(rgb_path, f"{self.ppg_str}_ppg.npy")
            signal = None

            if os.path.exists(ppg_file):
                signal = np.load(ppg_file)
            else:
                rf_folder = self._resolve_rf_folder(video_name)
                rf_path = os.path.join(self.datapath, "rf_files", rf_folder)
                vital_file = os.path.join(rf_path, "vital_dict.npy")
                if os.path.exists(vital_file):
                    vital_dict = np.load(vital_file, allow_pickle=True).item()
                    signal = vital_dict["rgbd"]["NOM_PLETHWaveExport"]

            if signal is None or len(signal) <= self.ppg_offset:
                print(f"[WARN] Skip {video_name}: PPG signal not found or too short.")
                continue

            keep_videos.append(video_name)
            keep_signals.append(np.asarray(signal[self.ppg_offset:], dtype=np.float32))

        self.video_list = keep_videos
        self.signal_list = keep_signals

    def _normalize_signals(self) -> None:
        if not self.signal_list:
            return
        concat = np.concatenate(self.signal_list, axis=0)
        mean = float(np.mean(concat))
        std = float(np.std(concat)) + 1e-8
        self.signal_list = [(sig - mean) / std for sig in self.signal_list]

    def _build_rf_window(self, video_name: str) -> np.ndarray:
        rf_folder = self._resolve_rf_folder(video_name)
        rf_file = os.path.join(self.datapath, "rf_files", rf_folder, "rf.pkl")
        if not os.path.exists(rf_file):
            raise FileNotFoundError(rf_file)

        with open(rf_file, "rb") as fp:
            raw = pickle.load(fp)

        organizer = Organizer(raw, 1, 1, 1, 2 * self.samples)
        frames = organizer.organize()
        frames = frames[:, :, :, 0::2]

        data_f = create_fast_slow_matrix(frames)  # (T_rf, range_bins)
        range_index = find_range(data_f, self.samp_f, self.freq_slope, self.samples)

        half = self.window_size // 2
        start = max(0, range_index - half)
        end = min(data_f.shape[1], range_index + half + 1)
        window = data_f[:, start:end]

        if window.shape[1] < self.window_size:
            pad_cols = self.window_size - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad_cols)), mode="constant")

        # (T_rf, range) -> (2, range, T_rf)
        iq = np.array([np.real(window), np.imag(window)], dtype=np.float32)
        iq = np.transpose(iq, (0, 2, 1))
        return iq

    def _load_rf_data(self) -> None:
        keep_videos: List[str] = []
        keep_signals: List[np.ndarray] = []
        keep_rf: List[np.ndarray] = []

        for video_name, signal in zip(self.video_list, self.signal_list):
            try:
                rf_data = self._build_rf_window(video_name)
            except Exception as exc:
                print(f"[WARN] Skip {video_name}: cannot load RF data ({exc}).")
                continue
            keep_videos.append(video_name)
            keep_signals.append(signal)
            keep_rf.append(rf_data)

        self.video_list = keep_videos
        self.signal_list = keep_signals
        self.rf_data_list = keep_rf

    def _generate_indices(self) -> None:
        self.all_idxs = []
        for vidx, (signal, rf_data) in enumerate(zip(self.signal_list, self.rf_data_list)):
            signal_len = len(signal)
            rf_len = rf_data.shape[2]
            max_available = min(signal_len, rf_len // self.sampling_ratio)
            max_start = max_available - self.frame_length
            if max_start < 0:
                continue
            for frame_start in range(0, max_start + 1, self.step):
                self.all_idxs.append((vidx, frame_start))

    def __len__(self) -> int:
        return len(self.all_idxs)

    def __getitem__(self, idx: int):
        video_idx, frame_start = self.all_idxs[idx]
        video_name = self.video_list[video_idx]

        rgb_dir = os.path.join(self.datapath, "rgb_files", video_name)
        rgb_frames = []
        for t in range(self.frame_length):
            frame_path = os.path.join(rgb_dir, f"{self.recording_str}_{frame_start + t}.png")
            if os.path.exists(frame_path):
                frame = _read_image(frame_path)
                if frame is not None:
                    frame = _resize_image(frame, self.image_size)
                    rgb_frames.append(frame)
                    continue
            rgb_frames.append(np.zeros((*self.image_size, 3), dtype=np.uint8))

        rgb = np.asarray(rgb_frames, dtype=np.float32) / 255.0
        rgb = np.transpose(rgb, (3, 0, 1, 2))  # (C, T, H, W)

        rf_data = self.rf_data_list[video_idx]
        rf_start = frame_start * self.sampling_ratio
        rf_stop = rf_start + self.frame_length * self.sampling_ratio
        if rf_stop <= rf_data.shape[2]:
            rf_window = rf_data[:, :, rf_start:rf_stop]
        else:
            rf_window = rf_data[:, :, rf_start:]
            pad_len = rf_stop - rf_data.shape[2]
            if pad_len > 0:
                rf_window = np.concatenate(
                    [rf_window, np.zeros((2, self.window_size, pad_len), dtype=rf_window.dtype)],
                    axis=2,
                )

        rf_window = rf_window[:, :, :: self.sampling_ratio]
        if rf_window.shape[2] < self.frame_length:
            pad_len = self.frame_length - rf_window.shape[2]
            rf_window = np.concatenate(
                [rf_window, np.zeros((2, self.window_size, pad_len), dtype=rf_window.dtype)],
                axis=2,
            )
        rf_window = rf_window[:, :, : self.frame_length]
        rf = rf_window.reshape(-1, self.frame_length).astype(np.float32) / 1.255e5

        label = self.signal_list[video_idx][frame_start: frame_start + self.frame_length]
        if len(label) < self.frame_length:
            label = np.concatenate(
                [label, np.zeros(self.frame_length - len(label), dtype=np.float32)],
                axis=0,
            )
        label = label[: self.frame_length].astype(np.float32)

        return rgb.astype(np.float32), rf, label
