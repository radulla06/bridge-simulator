import pytest

from bridge_simulator.contracts import Contract, calculate_contract_score


def test_contract_parser_supports_long_and_short_forms():
    assert Contract.from_string("4H N") == Contract.from_string("4H by N")
    contract = Contract.from_string("3NT S xx")
    assert contract.doubled is True
    assert contract.redoubled is True


def test_contract_parser_reports_bad_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        Contract.from_string("")
    with pytest.raises(ValueError, match="Unknown contract suit"):
        Contract.from_string("4Z S")


def test_standard_contract_scores():
    assert calculate_contract_score(Contract.from_string("3NT S"), 9) == 400
    assert calculate_contract_score(Contract.from_string("4H S"), 9) == -50
    assert (
        calculate_contract_score(Contract.from_string("6S S"), 12, vulnerable=True)
        == 1430
    )


def test_redoubled_insult_bonus_is_applied():
    assert calculate_contract_score(Contract.from_string("1C S xx"), 7) == 230
