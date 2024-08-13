import pytest
from anyguide import JSONGrammarError, json_schema_to_gbnf

from .utils import (gen_all_invalid_json_schemas,
                    gen_all_json_schema_grammar_pairs)


class TestJSONGrammar:

    @pytest.mark.parametrize("test_name, schema, expected_gbnf",
                             gen_all_json_schema_grammar_pairs())
    def test_grammar_valid(self, test_name, schema, expected_gbnf):
        print(f"[{test_name}] Testing schema\n`{schema}`")
        grammar_str = json_schema_to_gbnf(schema)
        assert grammar_str == expected_gbnf

    @pytest.mark.parametrize("test_name, schema",
                             gen_all_invalid_json_schemas())
    def test_grammar_invalid(self, test_name, schema):
        print(f"[{test_name}] Testing invalid schema\n`{schema}`")
        with pytest.raises(JSONGrammarError):
            json_schema_to_gbnf(schema)
