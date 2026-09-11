"""Regression test: config fields nothing reads must say so instead of silently doing nothing."""

import dataclasses

import pytest

from thinkrl.config.base import UNHONOURED, DataConfig, LoggingConfig, ModelConfig, PeftConfig, ThinkRLConfig


def test_trust_remote_code_is_off_by_default():
    """It executes code from the model repository, so the safe value is the default."""
    assert ModelConfig().trust_remote_code is False


def test_defaults_report_nothing():
    assert ThinkRLConfig().unhonoured_fields() == []


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "load_in_8bit", True),
        ("data", "streaming", True),
        ("data", "num_workers", 8),
        ("logging", "save_every_n_steps", 50),
        ("logging", "wandb_entity", "some-team"),
        ("distributed", "backend", "gloo"),
    ],
)
def test_setting_an_unhonoured_field_is_reported(section, field, value):
    config = ThinkRLConfig()
    setattr(getattr(config, section), field, value)

    messages = config.unhonoured_fields()

    assert any(f"{section}.{field}" in message for message in messages), messages


def test_report_explains_why_rather_than_only_naming_the_field():
    config = ThinkRLConfig(model=ModelConfig(load_in_8bit=True))

    (message,) = config.unhonoured_fields()

    assert "full precision" in message


def test_validate_still_returns_real_errors_only():
    """The report is a warning, not a validation failure, so a run is not blocked by it."""
    config = ThinkRLConfig(data=DataConfig(streaming=True))

    assert config.validate() == []


def test_every_listed_field_exists():
    """Guards the list against drift: a renamed or removed field must not sit here unnoticed."""
    # peft is optional and defaults to None, so it has to be supplied to be inspected.
    config = ThinkRLConfig(peft=PeftConfig())
    for section_name, field_name, _reason in UNHONOURED:
        section = getattr(config, section_name, None)
        assert section is not None, f"unknown section {section_name}"
        names = {f.name for f in dataclasses.fields(section)}
        assert field_name in names, f"{section_name}.{field_name} is listed but does not exist"


def test_list_covers_the_logging_fields_that_have_no_reader():
    listed = {field for section, field, _ in UNHONOURED if section == "logging"}
    declared = {f.name for f in dataclasses.fields(LoggingConfig())}

    assert listed <= declared
    assert "save_every_n_steps" in listed


def test_optional_peft_section_is_reported_when_present():
    config = ThinkRLConfig(peft=PeftConfig(auto_target_modules=False))

    assert any("peft.auto_target_modules" in message for message in config.unhonoured_fields())


def test_absent_peft_section_is_skipped_rather_than_crashing():
    assert ThinkRLConfig(peft=None).unhonoured_fields() == []
