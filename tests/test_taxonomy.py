"""Tests for fault taxonomy definitions."""

from __future__ import annotations

from cloudgym.taxonomy.base import (
    REGISTRY,
    FaultCategory,
    FaultType,
    IaCFormat,
    Severity,
)

# Import to trigger registration
import cloudgym.taxonomy.terraform as tf_faults
import cloudgym.taxonomy.cloudformation as cf_faults


class TestFaultRegistry:
    def test_registry_has_faults(self):
        assert len(REGISTRY) > 0

    def test_all_faults_have_required_fields(self):
        for fault in REGISTRY.all():
            assert fault.name, f"Fault missing name: {fault}"
            assert fault.category is not None
            assert fault.description
            assert fault.severity is not None
            assert len(fault.applicable_formats) > 0

    def test_fault_ids_are_unique(self):
        ids = [f.id for f in REGISTRY.all()]
        assert len(ids) == len(set(ids)), f"Duplicate fault IDs: {ids}"

    def test_terraform_faults_exist(self):
        tf = REGISTRY.list_by_format(IaCFormat.TERRAFORM)
        assert len(tf) >= 14, f"Expected >= 14 TF faults, got {len(tf)}"

    def test_cloudformation_faults_exist(self):
        cf = REGISTRY.list_by_format(IaCFormat.CLOUDFORMATION)
        assert len(cf) >= 14, f"Expected >= 14 CF faults, got {len(cf)}"

    def test_all_categories_represented_in_terraform(self):
        tf_faults_list = REGISTRY.list_by_format(IaCFormat.TERRAFORM)
        categories = {f.category for f in tf_faults_list}
        expected = {
            FaultCategory.SYNTACTIC,
            FaultCategory.REFERENCE,
            FaultCategory.SEMANTIC,
            FaultCategory.DEPENDENCY,
            FaultCategory.PROVIDER,
            FaultCategory.SECURITY,
            FaultCategory.CROSS_RESOURCE,
        }
        assert expected.issubset(categories), f"Missing TF categories: {expected - categories}"

    def test_all_cf_categories_represented(self):
        cf = REGISTRY.list_by_format(IaCFormat.CLOUDFORMATION)
        categories = {f.category for f in cf}
        expected = {
            FaultCategory.SYNTACTIC,
            FaultCategory.REFERENCE,
            FaultCategory.SEMANTIC,
            FaultCategory.DEPENDENCY,
            FaultCategory.INTRINSIC,
            FaultCategory.SECURITY,
        }
        assert expected.issubset(categories), f"Missing CF categories: {expected - categories}"

    def test_severity_distribution(self):
        all_faults = REGISTRY.all()
        severities = {s: 0 for s in Severity}
        for f in all_faults:
            severities[f.severity] += 1
        # Should have at least one fault at each severity
        for sev, count in severities.items():
            assert count > 0, f"No faults at severity {sev}"

    def test_list_by_category(self):
        syntactic = REGISTRY.list_by_category(FaultCategory.SYNTACTIC)
        assert len(syntactic) >= 2

    def test_fault_id_format(self):
        for fault in REGISTRY.all():
            assert "." in fault.id
            parts = fault.id.split(".")
            assert len(parts) == 2
            assert parts[0] in [c.name for c in FaultCategory]


class TestTerraformFaults:
    def test_missing_brace_fault(self):
        fault = tf_faults.TF_MISSING_CLOSING_BRACE
        assert fault.category == FaultCategory.SYNTACTIC
        assert fault.severity == Severity.LOW
        assert IaCFormat.TERRAFORM in fault.applicable_formats
        assert IaCFormat.OPENTOFU in fault.applicable_formats

    def test_opentofu_shares_terraform_faults(self):
        tf = set(REGISTRY.list_by_format(IaCFormat.TERRAFORM))
        tofu = set(REGISTRY.list_by_format(IaCFormat.OPENTOFU))
        assert tf == tofu, "OpenTofu should share all Terraform fault types"


class TestCloudFormationFaults:
    def test_intrinsic_faults_are_cf_only(self):
        intrinsic = REGISTRY.list_by_category(FaultCategory.INTRINSIC)
        for fault in intrinsic:
            assert IaCFormat.CLOUDFORMATION in fault.applicable_formats
            assert IaCFormat.TERRAFORM not in fault.applicable_formats

    def test_broken_ref_fault(self):
        fault = cf_faults.CF_BROKEN_REF
        assert fault.category == FaultCategory.REFERENCE
        assert IaCFormat.CLOUDFORMATION in fault.applicable_formats
