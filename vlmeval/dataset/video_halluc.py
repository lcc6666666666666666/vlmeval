import ast
import json
import os
import os.path as osp
import re
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import dump, get_intermediate_file_path, load
from .utils.yorn import YOrN_Extraction
from .video_base import VideoBaseDataset
from .video_concat_dataset import ConcatVideoDataset


class VideoHallucDataset(VideoBaseDataset):
    TYPE = 'VideoHalluc'
    HF_REPO_ID = 'chaoyuli/VidHalluc'
    _SIMCSE_TOKENIZER = None
    _SIMCSE_MODEL = None
    STH_SIMILARITY_THRESHOLD_HIGH = 0.8
    STH_SIMILARITY_THRESHOLD_LOW = 0.5
    ROOT_ENV_VARS = (
        'VIDEOHALLUC_ROOT',
        'VIDEOHALLUC_DATA_ROOT',
        'VIDHALLUC_ROOT',
        'VIDHALLUC_DATA_ROOT',
    )
    STH_PROMPT = (
        "Watch the given video and determine if a scene change occurs. "
        "If no change occurs, respond: 'Scene change: No, Locations: None'. "
        "If there is a scene change, respond in the format: "
        "'Scene change: Yes, Locations: from [location1] to [location2].'"
    )
    DATASET_SPECS = {
        'VideoHalluc_BQA': dict(
            task='bqa',
            annotation_file='ach_binaryqa.json',
            archives=('data/ACH_videos.zip', ),
        ),
        'VideoHalluc_MCQ': dict(
            task='mcq',
            annotation_file='ach_mcq.json',
            archives=('data/ACH_videos.zip', ),
        ),
        'VideoHalluc_TSH': dict(
            task='tsh',
            annotation_file='tsh.json',
            archives=('data/TSH_videos.zip', ),
        ),
        'VideoHalluc_STH': dict(
            task='sth',
            annotation_file='sth.json',
            archives=('data/STH_videos.zip', ),
        ),
    }

    def __init__(self, dataset='VideoHalluc_BQA', nframe=0, fps=-1):
        if dataset not in self.DATASET_SPECS:
            raise ValueError(f'Unsupported VideoHalluc dataset: {dataset}')
        self.dataset_spec = self.DATASET_SPECS[dataset]
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SPECS)

    @classmethod
    def _candidate_dataset_roots(cls):
        candidates = []
        for env_key in cls.ROOT_ENV_VARS:
            env_value = os.environ.get(env_key)
            if env_value:
                candidates.append(Path(env_value))

        current = Path(__file__).resolve()
        candidates.extend([
            current.parents[2] / 'VidHalluc',
            current.parents[2] / 'VidHalluc-main',
            current.parents[2] / 'videohalluc_datasets',
            current.parents[3] / 'VidHalluc',
            current.parents[3] / 'VidHalluc-main',
            current.parents[3] / 'videohalluc_datasets',
            Path.cwd() / 'VidHalluc',
            Path.cwd() / 'VidHalluc-main',
            Path.cwd() / 'videohalluc_datasets',
            Path.cwd().parent / 'VidHalluc',
            Path.cwd().parent / 'VidHalluc-main',
            Path.cwd().parent / 'videohalluc_datasets',
        ])

        seen = set()
        normalized = []
        for candidate in candidates:
            try:
                root = Path(candidate).expanduser().resolve()
            except Exception:
                continue
            if root in seen:
                continue
            seen.add(root)
            normalized.append(root)
        return normalized

    @classmethod
    def _find_annotation_root(cls, root, annotation_file):
        direct = root / annotation_file
        if direct.exists():
            return root

        try:
            for matched in root.rglob(annotation_file):
                return matched.parent
        except Exception:
            return None
        return None

    @classmethod
    def _write_sentinel(cls, sentinel_path):
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = sentinel_path.with_suffix(sentinel_path.suffix + '.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as fout:
            fout.write('done')
        os.replace(tmp_path, sentinel_path)

    @classmethod
    def _extract_zip(cls, archive_path):
        import zipfile

        sentinel_path = archive_path.parent / f'.{archive_path.stem}_extracted'
        if sentinel_path.exists():
            return

        with zipfile.ZipFile(archive_path, 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                rel = os.path.normpath(info.filename).lstrip('/\\')
                dst = archive_path.parent / rel

                abs_root = archive_path.parent.resolve()
                abs_dst = dst.resolve()
                if not str(abs_dst).startswith(str(abs_root) + os.sep):
                    raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                    out.write(src.read())

        cls._write_sentinel(sentinel_path)

    @classmethod
    def _ensure_archives_extracted(cls, dataset_root, spec):
        for rel_path in spec['archives']:
            archive_name = Path(rel_path).name
            candidates = [
                dataset_root / rel_path,
                dataset_root / archive_name,
                dataset_root / 'data' / archive_name,
            ]
            for archive_path in candidates:
                if archive_path.exists():
                    cls._extract_zip(archive_path)
                    break

    @classmethod
    def _resolve_dataset_root(cls, spec):
        for root in cls._candidate_dataset_roots():
            if not root.exists():
                continue
            annotation_root = cls._find_annotation_root(root, spec['annotation_file'])
            if annotation_root is None:
                continue
            cls._ensure_archives_extracted(annotation_root, spec)
            return annotation_root

        allow_patterns = [spec['annotation_file'], *spec['archives']]
        dataset_root = Path(
            snapshot_download(
                repo_id=cls.HF_REPO_ID,
                repo_type='dataset',
                allow_patterns=allow_patterns,
            )
        ).resolve()
        cls._ensure_archives_extracted(dataset_root, spec)
        annotation_root = cls._find_annotation_root(dataset_root, spec['annotation_file'])
        if annotation_root is None:
            raise FileNotFoundError(
                f"Unable to locate {spec['annotation_file']} in the downloaded VidHalluc dataset."
            )
        return annotation_root

    @classmethod
    def _build_video_index(cls, dataset_root):
        cache_file = dataset_root / '.videohalluc_video_index.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as fin:
                    return json.load(fin)
            except Exception:
                pass

        video_index = {}
        for video_path in dataset_root.rglob('*.mp4'):
            try:
                rel_path = video_path.relative_to(dataset_root).as_posix()
            except Exception:
                rel_path = video_path.as_posix()
            for key in {video_path.name.lower(), video_path.stem.lower()}:
                prev = video_index.get(key)
                if prev is None or len(rel_path) < len(prev):
                    video_index[key] = rel_path

        with open(cache_file, 'w', encoding='utf-8') as fout:
            json.dump(video_index, fout, ensure_ascii=False, indent=2)
        return video_index

    @classmethod
    def _resolve_video_relpath(cls, video_index, video_name):
        video_name = str(video_name).strip()
        path = Path(video_name)
        candidates = [
            video_name.lower(),
            path.name.lower(),
            path.stem.lower(),
            f'{path.stem.lower()}.mp4',
        ]
        for key in candidates:
            if key in video_index:
                return video_index[key]
        return f'{path.stem}.mp4'

    @classmethod
    def _load_json(cls, json_path):
        with open(json_path, 'r', encoding='utf-8') as fin:
            return json.load(fin)

    @classmethod
    def _build_bqa_rows(cls, payload, video_index):
        rows = []
        for sample_id, question_list in payload.items():
            for question_id, question_data in enumerate(question_list):
                question = str(question_data['q']).strip()
                for clip_name, answer in question_data['a'].items():
                    rows.append({
                        'index': len(rows),
                        'sample_id': str(sample_id),
                        'question_id': int(question_id),
                        'clip_id': str(clip_name),
                        'task_type': 'BQA',
                        'video': str(clip_name),
                        'video_path': cls._resolve_video_relpath(video_index, clip_name),
                        'question': question,
                        'answer': str(answer).strip().lower(),
                    })
        return rows

    @classmethod
    def _build_mcq_rows(cls, payload, video_index):
        rows = []
        for sample_id, clip_map in payload.items():
            for clip_name, question_data in clip_map.items():
                rows.append({
                    'index': len(rows),
                    'sample_id': str(sample_id),
                    'clip_id': str(clip_name),
                    'task_type': 'MCQ',
                    'video': str(clip_name),
                    'video_path': cls._resolve_video_relpath(video_index, clip_name),
                    'question': str(question_data['Question']).strip(),
                    'choices': json.dumps(question_data['Choices'], ensure_ascii=False),
                    'answer': str(question_data['Correct Answer']).strip().upper(),
                })
        return rows

    @classmethod
    def _build_tsh_rows(cls, payload, video_index):
        rows = []
        for sample_id, question_data in payload.items():
            video_name = str(question_data['video']).strip()
            rows.append({
                'index': len(rows),
                'sample_id': str(sample_id),
                'task_type': 'TSH',
                'video': video_name,
                'video_path': cls._resolve_video_relpath(video_index, video_name),
                'question': str(question_data['Question']).strip(),
                'answer': str(question_data['Correct Answer']).strip().upper(),
            })
        return rows

    @classmethod
    def _build_sth_rows(cls, payload, video_index):
        rows = []
        for video_name, question_data in payload.items():
            rows.append({
                'index': len(rows),
                'sample_id': str(video_name),
                'task_type': 'STH',
                'video': str(video_name),
                'video_path': cls._resolve_video_relpath(video_index, video_name),
                'question': cls.STH_PROMPT,
                'answer': str(question_data['Scene change']).strip().lower(),
                'answer_scene_change': str(question_data['Scene change']).strip().lower(),
                'answer_locations': str(question_data['Locations']).strip(),
            })
        return rows

    def prepare_dataset(self, dataset_name):
        spec = self.dataset_spec
        dataset_root = self._resolve_dataset_root(spec)
        annotation_path = dataset_root / spec['annotation_file']
        if not annotation_path.exists():
            raise FileNotFoundError(f'VidHalluc annotation file not found: {annotation_path}')

        data_file = dataset_root / f'{dataset_name}.tsv'
        if not data_file.exists():
            payload = self._load_json(annotation_path)
            video_index = self._build_video_index(dataset_root)
            if spec['task'] == 'bqa':
                rows = self._build_bqa_rows(payload, video_index)
            elif spec['task'] == 'mcq':
                rows = self._build_mcq_rows(payload, video_index)
            elif spec['task'] == 'tsh':
                rows = self._build_tsh_rows(payload, video_index)
            elif spec['task'] == 'sth':
                rows = self._build_sth_rows(payload, video_index)
            else:
                raise ValueError(f"Unsupported VidHalluc task: {spec['task']}")
            pd.DataFrame(rows).to_csv(data_file, sep='\t', index=False)

        if not any(dataset_root.rglob('*.mp4')):
            warnings.warn(
                f'VidHalluc annotations were found at {annotation_path}, but no local video files were found '
                f'under {dataset_root}. Inference will fail until the videos are extracted or downloaded.'
            )

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

    @classmethod
    def _extract_preferred_answer_text(cls, text):
        if pd.isna(text):
            return ''
        text = str(text)
        text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
        matches = re.findall(r'<answer>(.*?)</answer>', text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            text = matches[-1]
        return re.sub(r'\s+', ' ', text).strip()

    @classmethod
    def _load_choices(cls, raw_choices):
        if isinstance(raw_choices, dict):
            return raw_choices
        if isinstance(raw_choices, str):
            try:
                return json.loads(raw_choices)
            except Exception:
                return ast.literal_eval(raw_choices)
        raise TypeError(f'Invalid choices type: {type(raw_choices)}')

    @classmethod
    def _normalize_mcq_response(cls, response):
        response = cls._extract_preferred_answer_text(response)
        prefixes = [
            'The best answer is',
            'The correct answer is',
            'The answer is',
            'The answer',
            'The best option is',
            'The correct option is',
            'Best answer:',
            'Best option:',
            'Answer:',
            'Option:',
            'Correct answer',
            'Correct option',
            '<|im_end|>',
        ]
        for prefix in prefixes:
            response = response.replace(prefix, ' ')
        return response.strip()

    @classmethod
    def _extract_mcq_answer_set(cls, response, choices):
        response = cls._normalize_mcq_response(response)
        valid_choices = {str(key).upper() for key in choices}

        token_text = re.sub(r'[^A-Za-z]', ' ', response.upper())
        for token in token_text.split():
            if token and all(ch in valid_choices for ch in token):
                return set(token), 'direct'

        matched = set()
        lowered = response.lower()
        for key, value in choices.items():
            if str(value).strip().lower() in lowered:
                matched.add(str(key).upper())
        if matched:
            return matched, 'text'

        return set(), 'none'

    @classmethod
    def _extract_tsh_answer(cls, response):
        answer = cls._extract_preferred_answer_text(response).lower()
        compact = re.sub(r'[\s\.,;:!?\-_/\'"\(\)\[\]\{\}]', '', answer)
        if compact in {'ab', 'ba', 'a', 'b'}:
            return compact.upper()
        if 'not clear' in answer or 'no clear' in answer:
            return 'None'

        has_action_a = 'action a' in answer
        has_action_b = 'action b' in answer

        if has_action_a and has_action_b:
            if 'before' in answer:
                before_phrase = re.search(
                    r'(action [ab]|[ab][\.\)])[^before]+before[^action]*?(action [ab]|[ab][\.\)])',
                    answer
                )
                if before_phrase:
                    return before_phrase.group(1)[-1].upper() + before_phrase.group(2)[-1].upper()
            elif 'then' in answer:
                then_phrase = re.search(
                    r'(action [ab]|[ab][\.\)])[^then]+then[^action]*?(action [ab]|[ab][\.\)])',
                    answer
                )
                if then_phrase:
                    return then_phrase.group(1)[-1].upper() + then_phrase.group(2)[-1].upper()
            elif 'after' in answer:
                after_phrase = re.search(
                    r'(action [ab]|[ab][\.\)])[^after]+after[^action]*?(action [ab]|[ab][\.\)])',
                    answer
                )
                if after_phrase:
                    return after_phrase.group(2)[-1].upper() + after_phrase.group(1)[-1].upper()

            positions = [
                (match.start(), match.group(1)[-1].upper())
                for match in re.finditer(r'(action [ab]|[ab][\.\)])', answer)
            ]
            positions.sort()
            ordered = ''.join(action for _, action in positions)
            return ordered if ordered else 'None'
        if has_action_a:
            return 'A'
        if has_action_b:
            return 'B'
        return 'None'

    @classmethod
    def _parse_sth_prediction(cls, response):
        answer = cls._extract_preferred_answer_text(response)
        scene_change_match = re.search(r'scene\s*change\s*:\s*(yes|no)', answer, re.IGNORECASE)
        if scene_change_match:
            scene_change = scene_change_match.group(1).lower()
        else:
            scene_change = YOrN_Extraction(answer).lower()

        locations_match = re.search(r'locations?\s*:\s*(.*)', answer, re.IGNORECASE | re.DOTALL)
        if locations_match:
            locations = locations_match.group(1).strip()
        else:
            span_match = re.search(r'from\s+.+?\s+to\s+.+?(?:\.|$)', answer, re.IGNORECASE | re.DOTALL)
            locations = span_match.group(0).strip() if span_match else ''
        return scene_change, locations

    @staticmethod
    def _extract_scenes(description):
        pattern = r'from (.+?) to (.+?)(?:\.|$)'
        match = re.search(pattern, str(description).strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + np.exp(-value))

    @classmethod
    def _get_simcse_components(cls):
        from transformers import AutoModel, AutoTokenizer

        if cls._SIMCSE_TOKENIZER is None or cls._SIMCSE_MODEL is None:
            model_name = 'princeton-nlp/sup-simcse-roberta-large'
            cls._SIMCSE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            cls._SIMCSE_MODEL = AutoModel.from_pretrained(model_name)
        return cls._SIMCSE_TOKENIZER, cls._SIMCSE_MODEL

    @classmethod
    def _evaluate_scene_descriptions(cls, gt_locations, pred_locations):
        from sklearn.metrics.pairwise import cosine_similarity
        import torch
        tokenizer, model = cls._get_simcse_components()

        @lru_cache(maxsize=None)
        def get_embedding(sentence):
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

        gt_from, gt_to = cls._extract_scenes(gt_locations)
        pred_from, pred_to = cls._extract_scenes(pred_locations)
        if not all([gt_from, gt_to, pred_from, pred_to]):
            return 0.0

        def score_pair(
            pred_scene,
            gt_scene,
            low_threshold=cls.STH_SIMILARITY_THRESHOLD_LOW,
            high_threshold=cls.STH_SIMILARITY_THRESHOLD_HIGH,
        ):
            del high_threshold
            pred_emb = get_embedding(pred_scene)
            gt_emb = get_embedding(gt_scene)
            similarity = cosine_similarity(pred_emb, gt_emb)[0][0]
            if similarity <= low_threshold:
                return 0.0
            numerator = cls._sigmoid(similarity) - cls._sigmoid(low_threshold)
            denominator = cls._sigmoid(1.0) - cls._sigmoid(low_threshold)
            return float(numerator / denominator)

        return score_pair(pred_from, gt_from) + score_pair(pred_to, gt_to)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        task = self.dataset_spec['task']
        question = str(line['question']).strip()
        if task == 'bqa':
            prompt = f"{question}\nOnly answer with a single word 'Yes' or 'No'."
        elif task == 'mcq':
            choices = self._load_choices(line['choices'])
            option_text = '\n'.join(f'{key}. {value}' for key, value in choices.items())
            prompt = (
                f"{question}\n"
                "Please select the correct answer (one or more options) and return only the option letter(s). "
                "(e.g., ABCD)\n"
                f"Choices:\n{option_text}"
            )
        elif task == 'tsh':
            prompt = (
                f"{question}\n"
                "Sort these two actions in the order they occur in the video, and return which action happens first. "
                "If you only detect one action, return that action."
            )
        elif task == 'sth':
            prompt = question
        else:
            raise ValueError(f'Unsupported VidHalluc task: {task}')

        video_path = osp.join(self.data_root, line['video_path'])

        message = []
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            for image_path in self.save_video_into_images(line):
                message.append(dict(type='image', value=image_path))
        message.append(dict(type='text', value=prompt))
        return message

    def _evaluate_bqa(self, data, eval_file):
        data = data.copy()
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x).strip().lower() for x in data['answer']]
        data['extracted'] = [YOrN_Extraction(x).lower() for x in data['prediction']]
        data['score'] = (data['extracted'] == data['answer']).astype(int)

        detail_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(data, detail_file)

        question_df = (
            data.groupby(['sample_id', 'question_id'], sort=True)
            .agg(
                question_correct=('score', lambda x: int(bool(np.all(x)))),
                num_clips=('score', 'size'),
            )
            .reset_index()
        )
        question_file = get_intermediate_file_path(eval_file, '_question_score', 'csv')
        dump(question_df, question_file)

        question_accuracy = float(question_df['question_correct'].mean() * 100) if len(question_df) else 0.0
        clip_accuracy = float(data['score'].mean() * 100) if len(data) else 0.0
        metrics = {
            'score': question_accuracy,
            'question_accuracy': question_accuracy,
            'clip_accuracy': clip_accuracy,
            'num_questions': int(len(question_df)),
            'num_clips': int(len(data)),
        }
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(metrics, rating_file)
        return metrics

    def _evaluate_mcq(self, data, eval_file):
        data = data.copy()
        data['prediction'] = [str(x) for x in data['prediction']]

        extracted = []
        match_modes = []
        scores = []
        for _, row in data.iterrows():
            choices = self._load_choices(row['choices'])
            pred_set, match_mode = self._extract_mcq_answer_set(row['prediction'], choices)
            gt_set = set(re.findall(r'[A-Z]', str(row['answer']).upper()))
            extracted.append(''.join(sorted(pred_set)) if pred_set else 'None')
            match_modes.append(match_mode)
            scores.append(int(pred_set == gt_set))

        data['extracted'] = extracted
        data['match_mode'] = match_modes
        data['score'] = scores

        detail_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(data, detail_file)

        accuracy = float(data['score'].mean() * 100) if len(data) else 0.0
        metrics = {
            'score': accuracy,
            'accuracy': accuracy,
            'num_questions': int(len(data)),
        }
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(metrics, rating_file)
        return metrics

    def _evaluate_tsh(self, data, eval_file):
        data = data.copy()
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x).strip().upper() for x in data['answer']]
        data['extracted'] = [self._extract_tsh_answer(x) for x in data['prediction']]
        data['score'] = (data['extracted'] == data['answer']).astype(int)

        detail_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(data, detail_file)

        accuracy = float(data['score'].mean() * 100) if len(data) else 0.0
        metrics = {
            'score': accuracy,
            'accuracy': accuracy,
            'num_questions': int(len(data)),
        }
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(metrics, rating_file)
        return metrics

    def _evaluate_sth(self, data, eval_file):
        from sklearn.metrics import confusion_matrix, matthews_corrcoef

        data = data.copy()
        data['prediction'] = [str(x) for x in data['prediction']]

        pred_scene_change = []
        pred_locations = []
        description_scores = []
        scene_change_correct = []
        gt_labels = []
        pred_labels = []
        total_description_score = 0.0
        max_description_score = 0.0

        for _, row in data.iterrows():
            pred_change, pred_location = self._parse_sth_prediction(row['prediction'])
            gt_change = str(row['answer_scene_change']).strip().lower()
            gt_location = str(row['answer_locations']).strip()

            pred_scene_change.append(pred_change)
            pred_locations.append(pred_location)
            scene_change_correct.append(int(pred_change == gt_change))

            gt_label = int(gt_change == 'yes')
            pred_label = int(pred_change == 'yes')
            gt_labels.append(gt_label)
            pred_labels.append(pred_label)

            desc_score = 0.0
            if gt_change == 'yes' and pred_change == 'yes':
                desc_score = self._evaluate_scene_descriptions(gt_location, pred_location)
                total_description_score += desc_score
                max_description_score += 2.0
            description_scores.append(desc_score)

        data['extracted'] = pred_scene_change
        data['pred_locations'] = pred_locations
        data['description_score'] = description_scores
        data['score'] = scene_change_correct

        detail_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(data, detail_file)

        cm = confusion_matrix(gt_labels, pred_labels, labels=[1, 0])
        if cm.shape == (2, 2):
            tp, fn = cm[0]
            fp, tn = cm[1]
        else:
            tp = fp = tn = fn = 0
            if gt_labels.count(1) == 0:
                tn = int(cm[0][0])
            else:
                tp = int(cm[0][0])

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = float(matthews_corrcoef(gt_labels, pred_labels))
        classification_score = float(((mcc + 1) / 2) ** 2)
        description_accuracy = float(total_description_score / max_description_score) if max_description_score > 0 else 0.0
        overall_score = float((classification_score * 0.6) + (description_accuracy * 0.4))

        metrics = {
            'score': overall_score * 100,
            'overall_score': overall_score * 100,
            'mcc': mcc,
            'classification_score': classification_score * 100,
            'description_accuracy': description_accuracy * 100,
            'scene_change_accuracy': float(np.mean(scene_change_correct) * 100) if scene_change_correct else 0.0,
            'recall': float(recall * 100),
            'specificity': float(specificity * 100),
            'num_videos': int(len(data)),
        }
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(metrics, rating_file)
        return metrics

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        task = self.dataset_spec['task']
        if task == 'bqa':
            return self._evaluate_bqa(data, eval_file)
        if task == 'mcq':
            return self._evaluate_mcq(data, eval_file)
        if task == 'tsh':
            return self._evaluate_tsh(data, eval_file)
        if task == 'sth':
            return self._evaluate_sth(data, eval_file)
        raise ValueError(f'Unsupported VidHalluc task: {task}')


class VideoHalluc(ConcatVideoDataset):
    DATASET_SETS = {
        'VideoHalluc': [
            'VideoHalluc_BQA',
            'VideoHalluc_MCQ',
            'VideoHalluc_TSH',
            'VideoHalluc_STH',
        ]
    }

    @classmethod
    def supported_datasets(cls):
        return ['VideoHalluc']

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
                'score': float(np.mean([res['score'] for res in results.values()])),
                'num_subsets': int(len(results)),
            }

        score_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(summary, score_file)
        return summary
