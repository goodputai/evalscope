import unittest
import tempfile

from evalscope.service.utils import process as process_registry
from evalscope.utils.runtime_liveness import (
    configure_runtime_liveness,
    read_runtime_liveness,
    record_request_started,
    record_sample_started,
    record_stream_event,
)


class FakeProcess:
    def __init__(self, pid: int, alive: bool = True) -> None:
        self.pid = pid
        self.alive = alive

    def is_alive(self) -> bool:
        return self.alive


class EvalProcessOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        with process_registry._active_lock:
            process_registry._active_processes.clear()
            process_registry._active_attempts.clear()

    def tearDown(self) -> None:
        with process_registry._active_lock:
            process_registry._active_processes.clear()
            process_registry._active_attempts.clear()

    def test_task_slot_is_reserved_before_process_start(self) -> None:
        self.assertTrue(process_registry._reserve_process("task-1"))
        self.assertFalse(process_registry._reserve_process("task-1"))
        self.assertEqual(process_registry.process_status("task-1"), {
            "task_id": "task-1",
            "running": True,
            "pid": None,
            "attempt_id": None,
        })

    def test_old_process_cannot_unregister_replacement(self) -> None:
        first = FakeProcess(101)
        replacement = FakeProcess(202)
        self.assertTrue(process_registry._reserve_process("task-1"))
        process_registry.register_process("task-1", first)
        with process_registry._active_lock:
            process_registry._active_processes["task-1"] = replacement

        process_registry.unregister_process("task-1", first)

        self.assertEqual(process_registry.process_status("task-1")["pid"], 202)

    def test_stream_events_update_true_liveness_fields(self) -> None:
        class Message:
            id = "provider-request-1"

        class Event:
            message = Message()

            @staticmethod
            def model_dump_json() -> str:
                return '{"chunk":"hello"}'

        with tempfile.TemporaryDirectory() as temporary:
            configure_runtime_liveness(temporary, "attempt-1")
            record_sample_started("aime26:test:29")
            record_request_started()
            record_stream_event(Event())
            payload = read_runtime_liveness(temporary)

        self.assertEqual(payload["attempt_id"], "attempt-1")
        self.assertEqual(payload["provider_request_id"], "provider-request-1")
        self.assertGreater(payload["bytes_received"], 0)
        self.assertTrue(payload["last_request_started_at"].endswith("Z"))
        self.assertTrue(payload["last_chunk_at"].endswith("Z"))
        self.assertEqual(payload["current_sample_uid"], "aime26:test:29")


if __name__ == "__main__":
    unittest.main()
