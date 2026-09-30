import unittest

from agent.tool_engine import CallStatus, DuplicateStateChangingCallError, ToolEngineError
from agent.tool_manifest import (
    ManifestError,
    make_mock_handler,
    parse_manifest,
    register_manifest_tools,
)
from agent.tools_builtin import ToolRunner


FLIGHT_MANIFEST = [
    {
        "name": "search_flights",
        "description": "Search flights to a destination.",
        "state_changing": False,
        "parameters": [{"name": "destination", "type": "string", "required": True}],
    },
    {
        "name": "book_flight",
        "description": "Book a flight.",
        "state_changing": True,
        "parameters": [
            {"name": "destination", "type": "string", "required": True},
            {"name": "seat", "type": "string", "required": False},
        ],
    },
]


class TestParseManifest(unittest.TestCase):
    def test_parses_valid_entries(self):
        result = parse_manifest(FLIGHT_MANIFEST)
        self.assertEqual(result.errors, [])
        names = {s.name for s in result.specs}
        self.assertEqual(names, {"search_flights", "book_flight"})
        book = next(s for s in result.specs if s.name == "book_flight")
        self.assertTrue(book.state_changing)
        search = next(s for s in result.specs if s.name == "search_flights")
        self.assertFalse(search.state_changing)

    def test_type_strings_map_to_python_types(self):
        manifest = [
            {
                "name": "t",
                "parameters": [
                    {"name": "a", "type": "string"},
                    {"name": "b", "type": "integer", "required": False},
                    {"name": "c", "type": "boolean", "required": False},
                    {"name": "d", "type": "number", "required": False},
                ],
            }
        ]
        spec = parse_manifest(manifest).specs[0]
        types = {p.name: p.type for p in spec.parameters}
        self.assertEqual(types, {"a": str, "b": int, "c": bool, "d": float})

    def test_unknown_type_is_a_recoverable_error_not_a_crash(self):
        manifest = [{"name": "t", "parameters": [{"name": "a", "type": "spaceship"}]}]
        result = parse_manifest(manifest)
        self.assertEqual(result.specs, [])
        self.assertEqual(len(result.errors), 1)
        self.assertIn("spaceship", result.errors[0])

    def test_missing_name_is_a_recoverable_error(self):
        result = parse_manifest([{"description": "no name"}])
        self.assertEqual(result.specs, [])
        self.assertIn("name", result.errors[0])

    def test_one_bad_entry_does_not_block_the_rest_of_the_manifest(self):
        manifest = [{"description": "bad, no name"}, FLIGHT_MANIFEST[0]]
        result = parse_manifest(manifest)
        self.assertEqual(len(result.specs), 1)
        self.assertEqual(result.specs[0].name, "search_flights")
        self.assertEqual(len(result.errors), 1)

    def test_duplicate_tool_name_keeps_first_and_reports_error(self):
        manifest = [FLIGHT_MANIFEST[0], FLIGHT_MANIFEST[0]]
        result = parse_manifest(manifest)
        self.assertEqual(len(result.specs), 1)
        self.assertEqual(len(result.errors), 1)

    def test_non_list_manifest_is_rejected_gracefully(self):
        result = parse_manifest({"not": "a list"})
        self.assertEqual(result.specs, [])
        self.assertEqual(len(result.errors), 1)

    def test_default_required_is_true(self):
        manifest = [{"name": "t", "parameters": [{"name": "a", "type": "string"}]}]
        spec = parse_manifest(manifest).specs[0]
        self.assertTrue(spec.parameters[0].required)


class TestMockHandler(unittest.TestCase):
    def test_deterministic_across_calls(self):
        h = make_mock_handler("weird_tool")
        self.assertEqual(h({"x": 1}), h({"x": 1}))

    def test_reflects_tool_name_and_args(self):
        h = make_mock_handler("frobnicate")
        out = h({"z": 2, "a": 1})
        self.assertIn("frobnicate", out)
        self.assertIn("a=1", out)
        self.assertIn("z=2", out)


class TestRegisterManifestTools(unittest.TestCase):
    def test_unseen_tool_is_callable_end_to_end_via_mock_handler(self):
        runner = ToolRunner()
        result = register_manifest_tools(runner, FLIGHT_MANIFEST)
        self.assertEqual(result.errors, [])
        self.assertEqual(set(runner.tool_names()), {"search_flights", "book_flight"})

        outcome = runner.run("search_flights", {"destination": "Goa"})
        self.assertTrue(outcome.ok)
        self.assertIn("search_flights", outcome.text)
        self.assertIn("Goa", outcome.text)

    def test_state_changing_manifest_tool_gets_duplicate_protection(self):
        runner = ToolRunner()
        register_manifest_tools(runner, FLIGHT_MANIFEST)

        # Directly exercise the engine's fingerprint guard, the same one
        # built-in tools use — book_flight is state_changing per the
        # manifest, so a second identical pending call must be rejected.
        call = runner.engine.create_call("book_flight", {"destination": "Pune"})
        with self.assertRaises(DuplicateStateChangingCallError):
            runner.engine.create_call("book_flight", {"destination": "Pune"})
        runner.engine.complete_call(call.call_id, {"ok": True})
        # Now that the first has completed, a new identical call is fine.
        runner.engine.create_call("book_flight", {"destination": "Pune"})

    def test_read_only_manifest_tool_is_never_deduplicated(self):
        runner = ToolRunner()
        register_manifest_tools(runner, FLIGHT_MANIFEST)
        runner.engine.create_call("search_flights", {"destination": "Pune"})
        # Must NOT raise — read-only calls are repeatable.
        runner.engine.create_call("search_flights", {"destination": "Pune"})

    def test_real_handler_overrides_mock(self):
        runner = ToolRunner()
        calls = []

        def real_handler(args):
            calls.append(args)
            return "REAL RESULT"

        register_manifest_tools(
            runner, FLIGHT_MANIFEST, handlers={"search_flights": real_handler}
        )
        outcome = runner.run("search_flights", {"destination": "Goa"})
        self.assertEqual(outcome.text, "REAL RESULT")
        self.assertEqual(calls, [{"destination": "Goa"}])

    def test_argument_validation_still_applies_to_manifest_tools(self):
        runner = ToolRunner()
        register_manifest_tools(runner, FLIGHT_MANIFEST)
        outcome = runner.run("search_flights", {})  # missing required 'destination'
        self.assertFalse(outcome.ok)

    def test_reregistering_same_manifest_is_idempotent(self):
        runner = ToolRunner()
        register_manifest_tools(runner, FLIGHT_MANIFEST)
        register_manifest_tools(runner, FLIGHT_MANIFEST)  # must not raise
        self.assertEqual(set(runner.tool_names()), {"search_flights", "book_flight"})


if __name__ == "__main__":
    unittest.main()
