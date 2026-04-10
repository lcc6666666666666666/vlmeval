import json
import os
import os.path as osp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_intermediate_file_path, load,
                         modelscope_flag_set)
from .utils.yorn import YOrN_Extraction
from .video_concat_dataset import ConcatVideoDataset
from .video_base import VideoBaseDataset


class VideoHallucerDataset(VideoBaseDataset):
    TYPE = 'VideoHallucination'
    HF_REPO_ID = 'ColorfulAI/VideoHallucer'
    ROOT_ENV_VARS = ('VIDEOHALLUCER_ROOT', 'VIDEOHALLUCER_DATA_ROOT')

    DATASET_SPECS = {
        'VideoHallucer_ObjectRelation': dict(
            subset='object_relation',
            json_file='object_relation.json',
            category='object_relation',
        ),
        'VideoHallucer_Temporal': dict(
            subset='temporal',
            json_file='temporal.json',
            category='temporal',
        ),
        'VideoHallucer_SemanticDetail': dict(
            subset='semantic_detail',
            json_file='semantic_detail.json',
            category='semantic_detail',
        ),
        'VideoHallucer_Interaction': dict(
            subset='interaction',
            json_file='interaction.json',
            category='interaction',
        ),
        'VideoHallucer_ExternalFactual': dict(
            subset='external_factual',
            json_file='external_factual.json',
            category='external_factual',
        ),
        'VideoHallucer_ExternalNonFactual': dict(
            subset='external_nonfactual',
            json_file='external_nonfactual.json',
            category='external_nonfactual',
        ),
        'VideoHallucer_FactDetect': dict(
            subset='fact_detect',
            json_file='fact_detect.json',
            category='fact_detect',
        ),
        'VideoHallucer_FactDetectYN': dict(
            subset='fact_detect',
            json_file='fact_detect_yn.json',
            category='fact_detect_yn',
        ),
    }

    def __init__(self, dataset='VideoHallucer_ObjectRelation', nframe=0, fps=-1):
        if dataset not in self.DATASET_SPECS:
            raise ValueError(f'Unsupported VideoHallucer dataset: {dataset}')
        self.dataset_spec = self.DATASET_SPECS[dataset]
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SPECS)

    @classmethod
    def _normalize_dataset_root(cls, root):
        root = Path(root).expanduser().resolve()
        if (root / 'videohallucer_datasets').exists():
            return root / 'videohallucer_datasets'
        return root

    @classmethod
    def _candidate_dataset_roots(cls):
        candidates = []
        for env_key in cls.ROOT_ENV_VARS:
            env_value = os.environ.get(env_key)
            if env_value:
                candidates.append(Path(env_value))

        current = Path(__file__).resolve()
        candidates.extend([
            current.parents[2] / 'VideoHallucer-main',
            current.parents[2] / 'videohallucer_datasets',
            current.parents[3] / 'VideoHallucer-main',
            current.parents[3] / 'videohallucer_datasets',
            Path.cwd() / 'VideoHallucer-main',
            Path.cwd() / 'videohallucer_datasets',
        ])

        seen = set()
        normalized = []
        for candidate in candidates:
            try:
                root = cls._normalize_dataset_root(candidate)
            except Exception:
                continue
            if root in seen:
                continue
            seen.add(root)
            normalized.append(root)
        return normalized

    @classmethod
    def _resolve_dataset_root(cls, subset, json_file):
        for root in cls._candidate_dataset_roots():
            if (root / subset / json_file).exists():
                return root

        cache_path = get_cache_path(cls.HF_REPO_ID)
        if cache_path is not None:
            cache_root = cls._normalize_dataset_root(cache_path)
            if (cache_root / subset / json_file).exists():
                return cache_root

        if modelscope_flag_set():
            raise FileNotFoundError(
                'VideoHallucer data not found locally and ModelScope fallback is not configured. '
                f'Please set one of {cls.ROOT_ENV_VARS} to the VideoHallucer dataset root.'
            )

        dataset_path = snapshot_download(repo_id=cls.HF_REPO_ID, repo_type='dataset')
        dataset_root = cls._normalize_dataset_root(dataset_path)
        if (dataset_root / subset / json_file).exists():
            return dataset_root

        raise FileNotFoundError(
            f'Unable to locate VideoHallucer subset `{subset}` with annotation file `{json_file}`. '
            f'Please set one of {cls.ROOT_ENV_VARS} to the VideoHallucer dataset root.'
        )

    def prepare_dataset(self, dataset_name):
        subset = self.dataset_spec['subset']
        json_file = self.dataset_spec['json_file']
        dataset_root = self._resolve_dataset_root(subset, json_file)

        json_path = dataset_root / subset / json_file
        video_dir = dataset_root / subset / 'videos'
        if not video_dir.exists():
            warnings.warn(
                f'VideoHallucer annotations were found at {json_path}, but the video directory '
                f'{video_dir} is missing. Inference will fail until videos are available.'
            )

        data_file = dataset_root / f'{dataset_name}.tsv'
        if not data_file.exists():
            with open(json_path, 'r', encoding='utf-8') as fin:
                pairs = json.load(fin)

            rows = []
            for pair_id, pair in enumerate(pairs):
                pair_type = pair.get('type', self.dataset_spec['category'])
                for variant in ('basic', 'hallucination'):
                    sample = pair[variant]
                    video_name = sample['video']
                    video_stem = Path(video_name).stem
                    rows.append({
                        'index': len(rows),
                        'pair_id': pair_id,
                        'variant': variant,
                        'category': pair_type,
                        'subset': subset,
                        'video': video_stem,
                        'video_file': video_name,
                        'video_path': osp.join(subset, 'videos', video_name).replace('\\', '/'),
                        'question': str(sample['question']).strip(),
                        'answer': str(sample['answer']).strip().lower(),
                    })

            pd.DataFrame(rows).to_csv(data_file, sep='\t', index=False)

        return dict(root=str(dataset_root), data_file=str(data_file))

    def save_video_frames(self, line):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        vid_path = osp.join(self.data_root, line['video_path'])
        if not osp.exists(vid_path):
            raise FileNotFoundError(
                f'Video file not found: {vid_path}. '
                f'Please make sure {self.dataset_name} videos are available under {self.data_root}.'
            )

        import decord

        vid = decord.VideoReader(vid_path)
        video_fps = vid.get_avg_fps()
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            total_duration = len(vid) / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(line['video'], len(indices))
        else:
            raise ValueError(
                f'{self.dataset_name} requires either nframe or fps to be set when using frame-based input.'
            )

        if not np.all([osp.exists(p) for p in frame_paths]):
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for image, path in zip(images, frame_paths):
                        if not osp.exists(path):
                            image.save(path)
        return frame_paths

    def save_video_into_images(self, line):
        return self.save_video_frames(line)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        prompt = f"{line['question']}\nAnswer the question using 'yes' or 'no'."
        video_path = osp.join(self.data_root, line['video_path'])

        message = []
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            for image_path in self.save_video_into_images(line):
                message.append(dict(type='image', value=image_path))
        message.append(dict(type='text', value=prompt))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x).strip().lower() for x in data['answer']]
        data['extracted'] = [YOrN_Extraction(x).lower() for x in data['prediction']]
        data['score'] = (data['extracted'] == data['answer']).astype(int)

        detail_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(data, detail_file)

        pair_rows = []
        for pair_id, subset in data.groupby('pair_id', sort=True):
            row = {
                'pair_id': pair_id,
                'category': subset['category'].iloc[0],
                'basic_correct': 0,
                'halluc_correct': 0,
            }
            for variant in ('basic', 'hallucination'):
                variant_rows = subset[subset['variant'] == variant]
                if len(variant_rows):
                    row[f'{variant}_correct'] = int(variant_rows['score'].iloc[0])
            row['pair_correct'] = int(row['basic_correct'] and row['halluc_correct'])
            pair_rows.append(row)

        pair_df = pd.DataFrame(pair_rows)
        pair_file = get_intermediate_file_path(eval_file, '_pair_score', 'csv')
        dump(pair_df, pair_file)

        metrics = {
            'basic_accuracy': float(pair_df['basic_correct'].mean() * 100),
            'halluc_accuracy': float(pair_df['halluc_correct'].mean() * 100),
            'accuracy': float(pair_df['pair_correct'].mean() * 100),
            'num_pairs': int(len(pair_df)),
        }

        by_category = {}
        for category, subset in pair_df.groupby('category', sort=True):
            by_category[str(category)] = {
                'basic_accuracy': float(subset['basic_correct'].mean() * 100),
                'halluc_accuracy': float(subset['halluc_correct'].mean() * 100),
                'accuracy': float(subset['pair_correct'].mean() * 100),
                'num_pairs': int(len(subset)),
            }
        if by_category:
            metrics['by_category'] = by_category

        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(metrics, rating_file)
        return metrics


class VideoHallucer(ConcatVideoDataset):
    DATASET_SETS = {
        'VideoHallucer': [
            'VideoHallucer_ObjectRelation',
            'VideoHallucer_Temporal',
            'VideoHallucer_SemanticDetail',
            'VideoHallucer_Interaction',
            'VideoHallucer_ExternalFactual',
            'VideoHallucer_ExternalNonFactual',
        ]
    }

    @classmethod
    def supported_datasets(cls):
        return ['VideoHallucer']

    def evaluate(self, eval_file, **judge_kwargs):
        data_all = load(eval_file)
        results = {}

        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            data_sub = data_all[data_all['SUB_DATASET'] == dname].copy()
            data_sub.drop(columns=['index', 'SUB_DATASET'], inplace=True)
            data_sub.rename(columns={'original_index': 'index'}, inplace=True)
            dump(data_sub, tgt)
            results[dname] = self.dataset_map[dname].evaluate(tgt, **judge_kwargs)

        summary = dict(results)
        if results:
            summary['all'] = {
                'basic_accuracy': float(np.mean([res['basic_accuracy'] for res in results.values()])),
                'halluc_accuracy': float(np.mean([res['halluc_accuracy'] for res in results.values()])),
                'accuracy': float(np.mean([res['accuracy'] for res in results.values()])),
                'num_subsets': int(len(results)),
            }

        score_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(summary, score_file)
        return summary
