"""File utilities for loading reference data."""

import logging
import torch
import yaml

from src.normalization.element_refs import ElementReferences

log = logging.getLogger(__name__)


def load_reference(reference_path: str) -> ElementReferences:
    """Load element reference energies from YAML and return helper module."""

    with open(reference_path, "r") as f:
        try:
            all_elem_refs = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            log.error(f"Failed to parse element references at {reference_path}: {exc!r}")
            raise

    omol_elem_refs = torch.tensor(all_elem_refs["omol_elem_refs"], dtype=torch.float)
    return ElementReferences(omol_elem_refs)
