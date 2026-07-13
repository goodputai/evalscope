import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock, patch

from flask import Flask, jsonify

from evalscope.benchmarks.aime.aime_adapter import (
    AIME26Adapter,
    GLM52_AIME_SYSTEM_PROMPT,
    JUDGE_MAX_ATTEMPTS,
)
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.evaluator.evaluator import DefaultEvaluator, _PoolContext, _WorkItem
from evalscope.service.blueprints.eval import bp_eval
from evalscope.utils.tqdm_utils.progress_tracker import ProgressTracker


class AIME26JudgeContractTests(unittest.TestCase):

    def test_glm52_prompt_is_scoped_to_aime26(self) -> None:
        aime26 = BENCHMARK_REGISTRY['aime26']
        aime24 = BENCHMARK_REGISTRY['aime24']
        self.assertEqual(aime26.prompt_template, '{question}')
        self.assertEqual(aime26.system_prompt, GLM52_AIME_SYSTEM_PROMPT)
        self.assertNotEqual(aime24.prompt_template, '{question}')
        self.assertIsNone(aime24.system_prompt)

    def test_extract_answer_accepts_glm52_exact_answer_format(self) -> None:
        adapter = object.__new__(AIME26Adapter)
        prediction = 'Explanation: work\nExact Answer: 042\nConfidence: 95%'
        self.assertEqual(adapter.extract_answer(prediction, Mock()), '42')

    def test_judge_retries_unparseable_responses_and_records_audit(self) -> None:
        adapter = object.__new__(AIME26Adapter)
        judge = Mock(model_id='relay-gpt-5.5')
        judge.judge.side_effect = ['', 'Yes, because they match', 'Yes']
        adapter._llm_judge = judge
        adapter._task_config = Mock(judge_strategy='llm')

        with patch('evalscope.benchmarks.aime.aime_adapter.time.sleep'):
            score = adapter.llm_match_score('42', '42', '42', Mock())

        self.assertEqual(judge.judge.call_count, JUDGE_MAX_ATTEMPTS)
        self.assertEqual(score.main_value, 1.0)
        self.assertEqual(score.metadata['parsed_verdict'], 'Yes')
        self.assertEqual(score.metadata['attempt_count'], 3)
        self.assertIn('Expression 1: 42', score.metadata['judge_prompt'])

    def test_judge_failure_never_becomes_incorrect_answer(self) -> None:
        adapter = object.__new__(AIME26Adapter)
        judge = Mock(model_id='relay-gpt-5.5')
        judge.judge.return_value = '[ERROR] relay timeout'
        adapter._llm_judge = judge
        adapter._task_config = Mock(judge_strategy='llm')

        with patch('evalscope.benchmarks.aime.aime_adapter.time.sleep'), self.assertRaises(RuntimeError):
            adapter.llm_match_score('42', '42', '42', Mock())
        self.assertEqual(judge.judge.call_count, JUDGE_MAX_ATTEMPTS)

    def test_progress_snapshot_includes_phase_and_stage_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tracker = ProgressTracker(temporary, pipeline='eval', total_count=3, write_interval=0)
            tracker.initialize_stage_counts(1, 1)
            tracker.set_phase('generating')
            tracker.record_prediction()
            tracker.set_phase('judging')
            tracker.record_review()

            payload = json.loads(Path(temporary, 'progress.json').read_text())

        self.assertEqual(payload['phase'], 'judging')
        self.assertEqual(payload['prediction_processed_count'], 2)
        self.assertEqual(payload['review_processed_count'], 2)
        self.assertEqual(GLM52_AIME_SYSTEM_PROMPT.splitlines()[1], 'Explanation: {your explanation for your final answer}')

    def test_review_failure_persists_prediction_before_raising(self) -> None:
        evaluator = object.__new__(DefaultEvaluator)
        task_state = Mock(sample_id='sample-1')
        evaluator.model = Mock()
        evaluator.benchmark_name = 'aime26'
        evaluator.benchmark = Mock(use_batch_scoring=False, save_metadata=True)
        evaluator.benchmark.run_inference.return_value = task_state
        evaluator.benchmark.calculate_metrics.side_effect = RuntimeError('judge unavailable')
        evaluator.cache_manager = Mock()
        evaluator.cache_manager.save_prediction_cache.return_value.pretty_print.return_value = 'prediction'
        evaluator.task_config = Mock(eval_batch_size=1, ignore_errors=False)
        evaluator._record_perf = Mock()
        context = _PoolContext(
            work_items=[_WorkItem(subset='default', sample=Mock())],
            cached_scores_by_subset=defaultdict(list),
            review_pending_by_subset=defaultdict(list),
            model_prediction_dir='/tmp/predictions',
            total_cached=0,
            prediction_cached=0,
            review_cached=0,
        )

        def run_inline(items, worker, on_result, on_error, **kwargs):
            for item in items:
                try:
                    on_result(item, worker(item))
                except Exception as exc:
                    on_error(item, exc)

        with (
            patch('evalscope.evaluator.evaluator.run_in_threads_with_progress', side_effect=run_inline),
            self.assertRaisesRegex(RuntimeError, 'judge unavailable'),
        ):
            evaluator._run_pool(context)

        evaluator.cache_manager.save_prediction_cache.assert_called_once_with('default', task_state, True)
        evaluator.cache_manager.save_review_cache.assert_not_called()

    def test_service_resume_preserves_successful_review_cache(self) -> None:
        app = Flask(__name__)
        app.register_blueprint(bp_eval)

        def response(task_id, task_config, label):
            return jsonify({'rerun_review': task_config.rerun_review, 'use_cache': task_config.use_cache})

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch('evalscope.service.blueprints.eval.OUTPUT_DIR', temporary),
                patch('evalscope.service.blueprints.eval._execute_task', side_effect=response),
            ):
                Path(temporary, 'platform-task').mkdir()
                client = app.test_client()
                result = client.post(
                    '/api/v1/eval/resume/invoke',
                    headers={'EvalScope-Task-Id': 'platform-task'},
                    json={'model': 'glm', 'datasets': ['aime26'], 'api_url': 'https://model.example/v1'},
                )

        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.get_json()['rerun_review'])
        self.assertTrue(result.get_json()['use_cache'].endswith('platform-task'))


if __name__ == '__main__':
    unittest.main()
