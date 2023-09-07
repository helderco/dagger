import logging

from cattrs.preconf.json import make_converter as make_json_converter

from ._utils import syncify

logger = logging.getLogger(__name__)


def make_converter():
    import dagger
    from dagger.client._guards import is_id_type, is_id_type_subclass

    conv = make_json_converter(
        omit_if_default=True,
        detailed_validation=True,
    )

    # TODO: register cache volume for custom handling since it's different
    # than the other types.

    def dagger_type_structure(id_, cls):
        """Get dagger object type from id."""
        return dagger.default_client()._get_object_instance(id_, cls)  # noqa: SLF001

    def dagger_type_unstructure(obj):
        """Get id from dagger object."""
        if not is_id_type(obj):
            msg = f"Expected dagger Type object, got `{type(obj)}`"
            raise TypeError(msg)
        return syncify(obj.id)

    conv.register_structure_hook_func(
        is_id_type_subclass,
        dagger_type_structure,
    )

    conv.register_unstructure_hook_func(
        is_id_type_subclass,
        dagger_type_unstructure,
    )

    return conv
