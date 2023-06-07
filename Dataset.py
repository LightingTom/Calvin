import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import random
from episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)


def get_validation_window_size(idx: int, min_window_size: int, max_window_size: int) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + random.randint(0, window_range)

class CalvinDataset(Dataset):
    def __init__(self,
                 train: bool = True,
                 skip_frames: int = 1,
                 pretrain: bool = False,
                 aux_lang_loss_window: int = 1,
                 lang: bool = False,
                 pad: bool = True):
        super().__init__()
        self.max_window_size = 32
        self.min_window_size = 16
        self.aux_lang_loss_window = aux_lang_loss_window
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.lang = lang
        self.train = train
        self.pad = pad
        self.observation_space = {"rgb_obs":['rgb_static'],
               "depth_obs":[],
               "state_obs":['robot_obs'],
               "actions":['actions'],
               "language":['language']}
        self.proprio_state = {"n_state_obs":15,
                   "keep_indices":[[0,15]],
                   "robot_orientation_idx":[3,6],
                   "normalize":True,
                   "normalize_robot_orientation":True}
        self.transforms = {}
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        if train:
            self.dir = Path('calvin_debug_dataset/training/')
        else:
            self.dir = Path('calvin_debug_dataset/validation/')
        if lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann = self.load_lang_idx()
        else:
            self.episode_lookup = self.load_vis_idx()

    def load_vis_idx(self):
        episode_lookup = []
        ep_start_end_ids = np.load(self.dir / "ep_start_end_ids.npy")
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def load_lang_idx(self):
        episode_lookup = []
        lang_data = np.load(self.dir / 'lang_annotations' / "auto_lang_ann.npy", allow_pickle=True).item()
        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.pretrain:
                start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann

    def _get_window_size(self, idx: int) -> int:
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(self.max_window_size, (self.min_window_size + steps_to_next_episode - 1))
        else:
            max_window = self.max_window_size

        if not self.train:
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def load_episode(self, idx, window_size):
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = ['rgb_static', 'robot_obs', 'actions', 'scene_obs']
        episodes = [np.load(self.dir / f"{'episode_'}{file_idx:0{7}d}{'.npz'}") for file_idx in range(start_idx, end_idx)]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]
        # In validation set, there are some sequence longer than max_window_size, so truncate
        if episode['actions'].shape[0] > 32:
            episode['actions'] = episode['actions'][0:32]
        if episode['robot_obs'].shape[0] > 32:
            episode['robot_obs'] = episode['robot_obs'][0:32]
        if episode['rgb_static'].shape[0] > 32:
            episode['rgb_static'] = episode['rgb_static'][0:32]
        if episode['scene_obs'].shape[0] > 32:
            episode['scene_obs'] = episode['scene_obs'][0:32]
        return episode

    def _add_language_info(self, info, idx: int):
        if not self.lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

    def _get_sequences(self, idx: int, window_size: int):
        episode = self.load_episode(idx, window_size)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.lang)
        info = self._add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence) -> int:
        ret = self.max_window_size - len(sequence["actions"])
        return ret

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _pad_sequence(self, seq, pad_size: int):
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update({"rgb_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["rgb_obs"].items()}})
        seq.update({"depth_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["depth_obs"].items()}})
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update({"state_info": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    def __getitem__(self, idx):
        window_size = self._get_window_size(idx)
        sequence = self._get_sequences(idx, window_size)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)
        return sequence

