import math
import unittest

from agent.action_schema import ActionSchemaError, is_valid_action, validate_action


def snapshot(intent=None, slots=None, last_updated=1.0):
    return {"intent": intent, "slots": slots or {}, "last_updated": last_updated}


class TestValidActions(unittest.TestCase):
    def test_filler_is_valid(self):
        validate_action({"action": "filler", "call_id": "filler_1", "text": "okay"})

    def test_tool_call_is_valid(self):
        validate_action(
            {
                "action": "tool_call",
                "call_id": "call_1",
                "tool_name": "search_flights",
                "args": {"destination": "Goa"},
            }
        )

    def test_cancel_is_valid(self):
        validate_action({"action": "cancel", "call_id": "turn_1"})

    def test_clarification_request_is_valid_without_call_id(self):
        # call_id is not required for clarification_request — several real
        # call sites (raw_input, tool_manifest errors) emit it with none.
        validate_action({"action": "clarification_request", "reason": "missing destination"})

    def test_clarification_request_is_valid_with_call_id(self):
        validate_action(
            {"action": "clarification_request", "call_id": "turn_1", "reason": "ambiguous"}
        )

    def test_final_is_valid(self):
        validate_action(
            {
                "action": "final",
                "call_id": "turn_1",
                "response": "Booked!",
                "state_snapshot": snapshot(intent="booking", slots={"destination": "Goa"}),
            }
        )

    def test_final_with_extra_fields_is_still_valid(self):
        # Extra fields (tool_trace, error) are allowed — the schema is a
        # required-fields contract, not a closed/exhaustive shape.
        validate_action(
            {
                "action": "final",
                "call_id": "turn_1",
                "response": "timed out",
                "error": "timeout",
                "tool_trace": [],
                "state_snapshot": snapshot(),
            }
        )

    def test_is_valid_action_true_for_good_action(self):
        self.assertTrue(is_valid_action({"action": "cancel", "call_id": "x"}))


class TestInvalidActions(unittest.TestCase):
    def test_non_dict_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action("not a dict")

    def test_missing_action_field_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"call_id": "x"})

    def test_empty_action_field_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "", "call_id": "x"})

    def test_unknown_action_type_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "self_destruct", "call_id": "x"})

    def test_filler_missing_call_id_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "filler", "text": "hi"})

    def test_filler_empty_call_id_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "filler", "call_id": "", "text": "hi"})

    def test_tool_call_missing_tool_name_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "tool_call", "call_id": "c1", "args": {}})

    def test_tool_call_args_must_be_object(self):
        with self.assertRaises(ActionSchemaError):
            validate_action(
                {"action": "tool_call", "call_id": "c1", "tool_name": "x", "args": "not a dict"}
            )

    def test_cancel_missing_call_id_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "cancel"})

    def test_clarification_request_missing_reason_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "clarification_request"})

    def test_clarification_request_none_reason_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "clarification_request", "reason": None})

    def test_final_missing_state_snapshot_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "final", "call_id": "t1", "response": "ok"})

    def test_final_snapshot_missing_intent_rejected(self):
        bad = {"slots": {}, "last_updated": 1.0}
        with self.assertRaises(ActionSchemaError):
            validate_action(
                {"action": "final", "call_id": "t1", "response": "ok", "state_snapshot": bad}
            )

    def test_final_snapshot_slots_must_be_object(self):
        bad = {"intent": None, "slots": "not a dict", "last_updated": 1.0}
        with self.assertRaises(ActionSchemaError):
            validate_action(
                {"action": "final", "call_id": "t1", "response": "ok", "state_snapshot": bad}
            )

    def test_final_snapshot_last_updated_must_be_a_number_not_bool(self):
        bad = {"intent": None, "slots": {}, "last_updated": True}
        with self.assertRaises(ActionSchemaError):
            validate_action(
                {"action": "final", "call_id": "t1", "response": "ok", "state_snapshot": bad}
            )

    def test_nan_is_rejected_even_though_json_dumps_would_normally_allow_it(self):
        with self.assertRaises(ActionSchemaError):
            validate_action(
                {
                    "action": "final",
                    "call_id": "t1",
                    "response": "ok",
                    "state_snapshot": snapshot(last_updated=math.nan),
                }
            )

    def test_non_json_serializable_value_rejected(self):
        with self.assertRaises(ActionSchemaError):
            validate_action({"action": "cancel", "call_id": "x", "blob": b"raw bytes"})

    def test_is_valid_action_false_for_bad_action(self):
        self.assertFalse(is_valid_action({"action": "cancel"}))


if __name__ == "__main__":
    unittest.main()
