"""Tests for the programmatic inverter."""

from __future__ import annotations

from pathlib import Path

import pytest

from cloudgym.inverter._tf_injectors import TF_INJECTOR_REGISTRY
from cloudgym.inverter._cf_injectors import CF_INJECTOR_REGISTRY

FIXTURES = Path(__file__).parent / "fixtures"
TF_FIXTURE = FIXTURES / "gold_main.tf"
CF_FIXTURE = FIXTURES / "gold_template.yaml"


@pytest.fixture
def tf_text():
    return TF_FIXTURE.read_text()


@pytest.fixture
def tf_parsed(tf_text):
    try:
        import hcl2
        import io
        return hcl2.load(io.StringIO(tf_text))
    except Exception:
        return {}


@pytest.fixture
def cf_text():
    return CF_FIXTURE.read_text()


@pytest.fixture
def cf_parsed(cf_text):
    from cloudgym.inverter._yaml_cf import cf_load
    return cf_load(cf_text)


# Collect all TF fault IDs
TF_FAULT_IDS = list(TF_INJECTOR_REGISTRY.keys())
CF_FAULT_IDS = list(CF_INJECTOR_REGISTRY.keys())


class TestTerraformInjectors:
    """Test each Terraform injector produces a different output."""

    @pytest.mark.parametrize("fault_id", TF_FAULT_IDS)
    def test_tf_injector_produces_change(self, fault_id, tf_text, tf_parsed):
        injector = TF_INJECTOR_REGISTRY[fault_id]
        result = injector(tf_text, tf_parsed)

        if result is None:
            pytest.skip(f"Fault {fault_id} not applicable to fixture")

        broken_text, injection = result
        assert broken_text != tf_text, f"Injector {fault_id} produced no change"
        assert injection is not None
        assert injection.description

    @pytest.mark.parametrize("fault_id", TF_FAULT_IDS)
    def test_tf_injector_returns_valid_types(self, fault_id, tf_text, tf_parsed):
        injector = TF_INJECTOR_REGISTRY[fault_id]
        result = injector(tf_text, tf_parsed)

        if result is None:
            pytest.skip(f"Fault {fault_id} not applicable to fixture")

        broken_text, injection = result
        assert isinstance(broken_text, str)
        assert len(broken_text) > 0


class TestCloudFormationInjectors:
    """Test each CloudFormation injector produces a different output."""

    @pytest.mark.parametrize("fault_id", CF_FAULT_IDS)
    def test_cf_injector_produces_change(self, fault_id, cf_text, cf_parsed):
        injector = CF_INJECTOR_REGISTRY[fault_id]
        result = injector(cf_text, cf_parsed)

        if result is None:
            pytest.skip(f"Fault {fault_id} not applicable to fixture")

        broken_text, injection = result
        assert broken_text != cf_text, f"Injector {fault_id} produced no change"
        assert injection is not None
        assert injection.description

    @pytest.mark.parametrize("fault_id", CF_FAULT_IDS)
    def test_cf_injector_returns_valid_types(self, fault_id, cf_text, cf_parsed):
        injector = CF_INJECTOR_REGISTRY[fault_id]
        result = injector(cf_text, cf_parsed)

        if result is None:
            pytest.skip(f"Fault {fault_id} not applicable to fixture")

        broken_text, injection = result
        assert isinstance(broken_text, str)
        assert len(broken_text) > 0


class TestProgrammaticInjectFault:
    """Test the top-level inject_fault dispatcher."""

    @pytest.mark.asyncio
    async def test_inject_fault_terraform(self, tf_text):
        from cloudgym.inverter.programmatic import inject_fault
        from cloudgym.taxonomy.terraform import TF_MISSING_CLOSING_BRACE

        result = await inject_fault(tf_text, TF_MISSING_CLOSING_BRACE, "terraform")
        assert result is not None
        broken, injection = result
        assert broken != tf_text
        assert injection.fault_type == TF_MISSING_CLOSING_BRACE

    @pytest.mark.asyncio
    async def test_inject_fault_cloudformation(self, cf_text):
        from cloudgym.inverter.programmatic import inject_fault
        from cloudgym.taxonomy.cloudformation import CF_INVALID_YAML

        result = await inject_fault(cf_text, CF_INVALID_YAML, "cloudformation")
        assert result is not None
        broken, injection = result
        assert broken != cf_text

    @pytest.mark.asyncio
    async def test_inject_fault_unknown_format(self, tf_text):
        from cloudgym.inverter.programmatic import inject_fault
        from cloudgym.taxonomy.terraform import TF_MISSING_CLOSING_BRACE

        result = await inject_fault(tf_text, TF_MISSING_CLOSING_BRACE, "unknown_format")
        assert result is None


class TestHclUtils:
    """Test HCL text manipulation utilities."""

    def test_find_resource_blocks(self, tf_text):
        from cloudgym.inverter._hcl_utils import find_resource_blocks

        blocks = find_resource_blocks(tf_text)
        assert len(blocks) >= 4  # vpc, subnet, sg, instance, s3
        resource_types = [b[0] for b in blocks]
        assert "aws_vpc" in resource_types
        assert "aws_instance" in resource_types

    def test_find_variable_refs(self, tf_text):
        from cloudgym.inverter._hcl_utils import find_variable_refs

        refs = find_variable_refs(tf_text)
        var_names = [r[0] for r in refs]
        assert "instance_type" in var_names
        assert "environment" in var_names

    def test_find_block_boundaries(self, tf_text):
        from cloudgym.inverter._hcl_utils import find_block_boundaries

        blocks = find_block_boundaries(tf_text, "terraform")
        assert len(blocks) >= 1
        start, end = blocks[0]
        block_text = tf_text[start:end]
        assert "required_providers" in block_text


class TestCfUtils:
    """Test CloudFormation dict manipulation utilities."""

    def test_find_refs(self, cf_parsed):
        from cloudgym.inverter._cf_utils import find_refs

        refs = find_refs(cf_parsed)
        targets = [t for t, _ in refs]
        assert "VPC" in targets
        assert "InstanceType" in targets

    def test_get_resource_logical_ids(self, cf_parsed):
        from cloudgym.inverter._cf_utils import get_resource_logical_ids

        ids = get_resource_logical_ids(cf_parsed)
        assert "VPC" in ids
        assert "WebServer" in ids
        assert "LogsBucket" in ids

    def test_get_parameter_names(self, cf_parsed):
        from cloudgym.inverter._cf_utils import get_parameter_names

        params = get_parameter_names(cf_parsed)
        assert "EnvironmentType" in params
        assert "InstanceType" in params

    def test_get_condition_names(self, cf_parsed):
        from cloudgym.inverter._cf_utils import get_condition_names

        conditions = get_condition_names(cf_parsed)
        assert "IsProduction" in conditions

    def test_find_getatt(self, cf_parsed):
        from cloudgym.inverter._cf_utils import find_getatt

        getatt_refs = find_getatt(cf_parsed)
        assert len(getatt_refs) >= 1  # LogsBucket.Arn

    def test_find_subs(self, cf_parsed):
        from cloudgym.inverter._cf_utils import find_subs

        subs = find_subs(cf_parsed)
        assert len(subs) >= 1  # "${EnvironmentType}-vpc"

    def test_find_selects(self, cf_parsed):
        from cloudgym.inverter._cf_utils import find_selects

        selects = find_selects(cf_parsed)
        assert len(selects) >= 1  # Select in PublicSubnet AZ

    def test_find_joins(self, cf_parsed):
        from cloudgym.inverter._cf_utils import find_joins

        joins = find_joins(cf_parsed)
        assert len(joins) >= 1  # Join in WebServer tags
