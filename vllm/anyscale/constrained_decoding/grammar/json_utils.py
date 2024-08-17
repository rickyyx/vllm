import json
from typing import Any, Dict, Union

from .json_schema_to_grammar import SchemaConverter


class JSONGrammarError(Exception):
    """Raised when an exception is encountered during JSON to GBNF
        conversion.
    """
    pass


def json_schema_to_gbnf(schema: Union[str, Dict[str, Any]], ) -> str:
    """Converts an input json schema to a string GBNF representation."""
    if isinstance(schema, str):
        schema = json.loads(schema)

    converter = SchemaConverter(
        prop_order={},
        allow_fetch=True,
        dotall=False,
        raw_pattern=False,
    )
    try:
        converter.resolve_refs(schema, "")
        converter.visit(schema, "")
        return converter.format_grammar()
    except Exception as e:
        raise JSONGrammarError(
            f"Error converting JSON schema to GBNF: {e}") from e
