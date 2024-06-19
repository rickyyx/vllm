# ruff: noqa
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from tabulate import tabulate

from vllm.anyscale.constrained_decoding.grammar.json_schema_to_grammar import (
    ARRAY, BOOLEAN, HEX_DIGIT, INTEGER, NULL, NUMBER, OBJECT, SPACE_END,
    SPACE_RULE, STRING, VALUE)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_stats(metrics: Iterable[float], name=""):
    metrics = np.array(metrics)
    mean = np.mean(metrics)
    ste = np.std(metrics) / np.sqrt(len(metrics))
    p50 = np.percentile(metrics, 50)
    p90 = np.percentile(metrics, 90)
    max_val = np.max(metrics)
    min_val = np.min(metrics)

    # Print a nice table with name on the rows and these metrics on the columns
    # Create a table data
    table_data = [[name, mean, ste, p50, p90, max_val, min_val]]

    # Create a table header
    headers = ["Metric", "Mean", "Standard Error", "P50", "P90", "Max", "Min"]

    # Print the table using tabulate
    print(tabulate(table_data, headers, tablefmt="pipe", floatfmt=".2f"))


@dataclass
class TestCase:

    schema: Optional[Union[Dict[str, Any], str]] = None
    grammar: Optional[str] = None
    valid_examples: Optional[List[Union[str, Dict[str, Any]]]] = None
    invalid_examples: Optional[List[Union[str, Dict[str, Any]]]] = None
    test_name: Optional[str] = None


def get_all_valid_test_cases():
    return [
        TestCase(
            test_name="string-basic",
            schema={"type": "string"},
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\nroot ::= {STRING}\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '"a"',
                '"a\\nb"',
                '"a b"',
                '"a 😀 b"',
            ],
            invalid_examples=[
                '\\',
                '"""',
            ]),
        TestCase(
            test_name="integer-basic",
            schema={"type": "integer"},
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\nroot ::= {INTEGER}\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '1',
                '123',
                '0',
                '-1',
            ],
            invalid_examples=[
                '01',
                '1.0',
                '-1.0',
                '1.0e1',
                '1e10',
                '"foo"',
                'true',
            ]),
        TestCase(
            test_name="number-basic",
            schema={"type": "number"},
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\nroot ::= {NUMBER}\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=['1', '123', '0.1', '1.0', '-1.0'
                            '1e10', '1.0e10'],
            invalid_examples=[
                '01',
                '"foo"',
                'true',
            ]),
        TestCase(
            test_name="bool-basic",
            schema={"type": "boolean"},
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\nroot ::= {BOOLEAN}\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                'true',
                'false',
            ],
            invalid_examples=[
                '1',
                '0',
                'null',
                '"foo"',
            ]),
        TestCase(
            test_name="const-string",
            schema={"const": "foo"},
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\nroot ::= "\\"foo\\""\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '"foo"',
            ],
            invalid_examples=[
                '"bar"',
                '1',
                'true',
            ]),
        TestCase(
            test_name="const-integer",
            schema={"const": 123},
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\nroot ::= "123"\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '123',
                '1',  # half baked is fine
            ],
            invalid_examples=[
                '321',
                'true',
            ]),
        TestCase(
            test_name="string-pattern-basic",
            schema={
                "type": "string",
                "pattern": "^abc?d*efg+(hij)?kl$"
            },
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\nroot ::= "\\"" "ab" "c"? "d"* "ef" "g"+ ("hij")? "kl" "\\"" space_e\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '"abefgkl"',
                '"abcefgkl"',
                '"abdefgkl"',
                '"abcdddefgkl"',
                '"abefggkl"',
                '"abefghijkl"',
            ],
            invalid_examples=[
                '""',
                '"abefg"',
                '"abcefkl"',
                '"abcdefgkhijl"',
            ]),
        TestCase(
            test_name="free-object",
            schema={},
            grammar=
            f'array ::= {ARRAY}\nboolean ::= {BOOLEAN}\nhexdigit ::= {HEX_DIGIT}\nnull ::= {NULL}\nnumber ::= {NUMBER}\nobject ::= {OBJECT}\nroot ::= object\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}\nstring ::= {STRING}\nvalue ::= {VALUE}',
            valid_examples=[
                '{}',
                '{"a": 1}',
                '{"a": 1, "b": [1, 2]}',
                '{"a": true}',
                '{"a": "null"}',
                '{"a"',  # half-baked json is ok
            ],
            invalid_examples=[
                '{"a": 1, "b": [1 2]}',
                '{"a": 1,}',
                '{{}}',
                '{"a": True}',
                '{"a":}',
                '{"a": null}',
            ]),
        TestCase(
            test_name="array-with-expected-types",
            schema={
                "items": [{
                    "type": "string"
                }, {
                    "type": "integer"
                }, {
                    "type": "boolean"
                }, {
                    "type": "number"
                }]
            },
            # TODO (Kourosh), add grammar
            grammar=None,
            valid_examples=[
                '["a", 1, true, 1.0]',
            ],
            invalid_examples=[
                '["a", 1, true, 1.0, "foo"]',
                '[1, "a", true, 1.0]',
                '[]',
            ]),
        TestCase(
            test_name="enum",
            schema={"enum": ["red", "amber", "green", None, 42, ["foo"]]
                    },
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\nroot ::= "\\"red\\"" | "\\"amber\\"" | "\\"green\\"" | "null" | "42" | "[\\"foo\\"]"\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '"red"',
                '"amber"',
                '"green"',
                'null',
                '42',
                '["foo"]',
            ],
            invalid_examples=[
                '"blue"',
                '0',
                '["bar"]',
            ]),
        TestCase(
            test_name="prefix-items-address",
            schema={
                "type":
                "array",
                "prefixItems": [{
                    "type": "number"
                }, {
                    "type": "string"
                }, {
                    "enum": ["Street", "Avenue", "Boulevard"]
                }, {
                    "enum": ["NW", "NE", "SW", "SE"]
                }]
            },
            # TODO (Kourosh): What's the grammar?
            grammar=None,
            valid_examples=[
                '[1600, "Pennsylvania", "Avenue", "NW"]',
            ],
            invalid_examples=[
                '[]',
                '["Pennsylvania", "Avenue", "NW"]',
                '[1600, "Pennsylvania", "LN", "NW"]',
            ]),
        TestCase(
            test_name="minitems-basic",
            schema={
                "items": {
                    "type": "boolean"
                },
                "minItems": 2
            },
            grammar=
            f'boolean ::= {BOOLEAN}\nhexdigit ::= {HEX_DIGIT}\nroot ::= "[" space boolean ( "," space boolean )( "," space boolean )* "]" space_e\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '[true, false]',
                '[true, false, true]',
            ],
            invalid_examples=[
                '[]',
                '[true]',
                '[1, 2]',
            ]),
        TestCase(
            test_name="maxitems-basic",
            schema={
                "items": {
                    "type": "boolean"
                },
                "maxItems": 1
            },
            grammar=
            f'boolean ::= {BOOLEAN}\nhexdigit ::= {HEX_DIGIT}\nroot ::= "[" space ( boolean  )? "]" space_e\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '[]',
                '[true]',
            ],
            invalid_examples=[
                '[true, false]',
                '[1]',
            ]),
        TestCase(
            test_name="maxitems-2",
            schema={
                "items": {
                    "type": "boolean"
                },
                "maxItems": 2
            },
            grammar=
            f'boolean ::= {BOOLEAN}\nhexdigit ::= {HEX_DIGIT}\nroot ::= "[" space ( boolean ( "," space boolean )? )? "]" space_e\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '[]',
                '[true]',
                '[true, false]',
            ],
            invalid_examples=[
                '[true, false, true]',
                '[1]',
            ]),
        TestCase(
            test_name="minitems-maxitems",
            schema={
                "items": {
                    "type": ["number", "integer"]
                },
                "minItems": 3,
                "maxItems": 5
            },
            grammar=
            f'hexdigit ::= {HEX_DIGIT}\ninteger ::= {INTEGER}\nitem ::= number | integer\nnumber ::= {NUMBER}\nroot ::= "[" space item ( "," space item )( "," space item )( "," space item )?( "," space item )? "]" space_e\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '[1, 2, 3]',
                '[1, 2, 3, 4]',
                '[1, 2, 3, 4, 5]',
            ],
            invalid_examples=[
                '[]',
                '[1, 2]',
                '[1, 2, 3, 4, 5, 6]',
                '["a", "b", "c"]',
            ]),
        TestCase(
            test_name="required-props-basic",
            schema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string"
                    },
                    "b": {
                        "type": "string"
                    },
                },
                "required": ["a", "b", "c"],
                "additionalProperties": False,
            },
            grammar=
            f'a-kv ::= "\\"a\\"" space ":" space string\nb-kv ::= "\\"b\\"" space ":" space string\nhexdigit ::= {HEX_DIGIT}\n'
            + 'root ::= "{" space a-kv "," space b-kv "}" space_e\n' +
            f'space ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}\nstring ::= {STRING}',
            valid_examples=[
                '{"a":"foo", "b":"bar"}',
            ],
            invalid_examples=[
                '{}',
                '{"a":"foo"}',
                '{"b":"foo", "a":"bar"}',  # out of order is invalid
                '{"a":"foo", "b":"bar", "c":"baz"}',
            ]),
        TestCase(
            test_name="additionalProperties-true",
            schema={
                "properties": {
                    "a": {
                        "type": "string"
                    }
                },
                "additionalProperties": True
            },
            grammar=
            f'a-kv ::= "\\"a\\"" space ":" space string\na-rest ::= additional-kvs\nadditional-kv ::= string ":" space additional-value\nadditional-kvs ::= additional-kv ( "," space additional-kv )*\nadditional-value ::= object\narray ::= {ARRAY}\nboolean ::= {BOOLEAN}\nhexdigit ::= {HEX_DIGIT}\nnull ::= {NULL}\nnumber ::= {NUMBER}\nobject ::= {OBJECT}\n'
            +
            'root ::= "{" space  (a-kv a-rest | additional-kvs )? "}" space_e\n'
            +
            f'space ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}\nstring ::= {STRING}\nvalue ::= {VALUE}',
            # TODO (Kourosh): Check what the proper behavior of
            # additionalProperties=True should be?
            valid_examples=[
                # '{}',
                # '{"a":"foo"}',
                # '{"a":"foo", "b":"bar"}',
                # '{"b":"foo"}',
            ],
            invalid_examples=[]),
        TestCase(
            test_name="required-additional-props",
            schema={
                "properties": {
                    "b":
                    {
                        "type": "string"
                    },
                    "a":
                    {
                        "type": "string"
                    },
                    "d":
                    {
                        "type": "string"
                    },
                    "c":
                    {
                        "type": "string"
                    },
                },
                "required": ["a",
                             "b"],
                "additionalProperties": False
            },
            grammar=
            f'a-kv ::= "\\"a\\"" space ":" space string\nb-kv ::= "\\"b\\"" space ":" space string\nc-kv ::= "\\"c\\"" space ":" space string\nd-kv ::= "\\"d\\"" space ":" space string\nd-rest ::= ( "," space c-kv )?\nhexdigit ::= {HEX_DIGIT}\n'
            +
            'root ::= "{" space b-kv "," space a-kv ( "," space ( d-kv d-rest | c-kv ) )? "}" space_e\n'
            +
            f'space ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}\nstring ::= {STRING}',
            valid_examples=[
                '{"b":"foo", "a":"bar"}',
                '{"b":"foo", "a":"bar", "c":"baz"}',
                '{"b":"foo", "a":"bar", "d":"qux"}',
                '{"b":"foo", "a":"bar", "d":"qux", "c":"baz"}',
            ],
            invalid_examples=[
                '{}',
                '{"a":"foo"}',
                '{"b":"foo"}',
                '{"a":"foo", "c":"bar"}',
                '{"b":"foo", "a":"bar", "e":"baz"}',
            ]),
        TestCase(
            test_name="additional-props-any-array",
            schema={
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    }
                }
            },
            grammar=
            f'additional-kv ::= string ":" space additional-value\nadditional-kvs ::= additional-kv ( "," space additional-kv )*\nadditional-value ::= "[" space ( number ( "," space number )* )? "]" space_e\nhexdigit ::= {HEX_DIGIT}\nnumber ::= {NUMBER}\n'
            + 'root ::= "{" space  (additional-kvs )? "}" space_e\n' +
            f'space ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}\nstring ::= {STRING}',
            valid_examples=[
                '{}',
                '{"a": []}',
                '{"a": [1]}',
                '{"a": [1, 2]}',
                '{"a": [1], "b": [2, 3]}',
            ],
            invalid_examples=[
                '{"a": 1}',
                '{"a": [true]}',
                '{"a": ["1"]}',
                '{"a": {"b": [1]}}',
            ]),
        TestCase(
            test_name="type-object",
            schema={"type": "object"},
            grammar=
            f'array ::= {ARRAY}\nboolean ::= {BOOLEAN}\nhexdigit ::= {HEX_DIGIT}\nnull ::= {NULL}\nnumber ::= {NUMBER}\nobject ::= {OBJECT}\nroot ::= object\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}\nstring ::= {STRING}\nvalue ::= {VALUE}',
            valid_examples=[
                '{}',
                '{"a": 1}',
                '{"a": {"b": [1, 2]}}',
            ],
            invalid_examples=[
                'null',
                '1',
                '"foo"',
                '[]',
            ]),
        TestCase(test_name="empty-object",
                 schema={
                     "type": "object",
                     "additionalProperties": False
                 },
                 grammar=f'hexdigit ::= {HEX_DIGIT}\n' +
                 'root ::= "{" space  "}" space_e\n' +
                 f'space ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
                 valid_examples=[
                     '{}',
                 ],
                 invalid_examples=[
                     '{"a": 1}',
                     '[1]',
                 ]),
        TestCase(
            test_name="required-additional-props-type",
            schema={
                "type": "object",
                "properties": {
                    "a":
                    {
                        "type": "number"
                    }
                },
                "required": ["a"],
                "additionalProperties": {
                    "type": "string"
                }
            },
            grammar=
            f'a-kv ::= "\\"a\\"" space ":" space number\nadditional-kv ::= string ":" space string\nadditional-kvs ::= additional-kv ( "," space additional-kv )*\nhexdigit ::= {HEX_DIGIT}\nnumber ::= {NUMBER}\n'
            +
            'root ::= "{" space a-kv ( "," space ( additional-kvs ) )? "}" space_e\n'
            +
            f'space ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}\nstring ::= {STRING}',
            valid_examples=[
                '{"a": 1}',
                '{"a": 1, "b": "foo"}',
                '{"a": 1, "b": "foo", "c": "bar"}',
            ],
            invalid_examples=[
                '{}',
                '{"b": "foo"}',
                '{"a": 1, "b": 2}',
                '{"a": 1, "b": true}',
            ]),
        TestCase(
            test_name="ref-basic",
            schema={
                "$ref": "#/definitions/MyType",
                "definitions": {
                    "MyType": {
                        "type": "object",
                        "properties": {
                            "a":
                            {
                                "type": "string"
                            }
                        },
                        "required": ["a"],
                        "additionalProperties": False
                    }
                }
            },
            grammar=
            ('MyType ::= "{" space MyType-a-kv "}" space_e\nMyType-a-kv ::= "\\"a\\"" space ":" space string\n'
             +
             f'hexdigit ::= {HEX_DIGIT}\nroot ::= MyType\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}\nstring ::= {STRING}'
             ),
            valid_examples=[
                '{"a": "foo"}',
            ],
            invalid_examples=[
                '{}',
                '{"b": "foo"}',
            ]),
        TestCase(
            test_name="any-of",
            schema={
                "anyOf": [{
                    "$ref": "#/definitions/foo"
                }, {
                    "$ref": "#/definitions/bar"
                }],
                "definitions": {
                    "foo": {
                        "properties": {
                            "a":
                            {
                                "type": "number"
                            }
                        }
                    },
                    "bar": {
                        "properties": {
                            "b":
                            {
                                "type": "number"
                            }
                        }
                    }
                },
                "type":
                "object"
            },
            grammar=f'alternative-0 ::= foo\nalternative-1 ::= bar\n' +
            'bar ::= "{" space  (bar-b-kv )? "}" space_e\nbar-b-kv ::= "\\"b\\"" space ":" space number\nfoo ::= "{" space  (foo-a-kv )? "}" space_e\nfoo-a-kv ::= "\\"a\\"" space ":" space number\n'
            +
            f'hexdigit ::= {HEX_DIGIT}\nnumber ::= {NUMBER}\nroot ::= alternative-0 | alternative-1\nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '{"a": 1}',
                '{"b": 2}',
                '{}',
            ],
            invalid_examples=[
                '{"a": "foo"}',
                '{"b": "bar"}',
                '{"a": 1, "b": 2}',
            ]),
        TestCase(
            test_name="all-of-nested-any-of",
            schema={
                "allOf": [{
                    "$ref": "#/definitions/foo"
                }, {
                    "$ref": "#/definitions/bar"
                }, {
                    "anyOf": [{
                        "$ref": "#/definitions/baz"
                    }, {
                        "$ref": "#/definitions/bam"
                    }]
                }],
                "definitions": {
                    "foo": {
                        "properties": {
                            "a":
                            {
                                "type": "number"
                            }
                        }
                    },
                    "bar": {
                        "properties": {
                            "b":
                            {
                                "type": "number"
                            }
                        }
                    },
                    "bam": {
                        "properties": {
                            "c":
                            {
                                "type": "number"
                            }
                        }
                    },
                    "baz": {
                        "properties": {
                            "d":
                            {
                                "type": "number"
                            }
                        }
                    }
                },
                "type":
                "object"
            },
            grammar=
            f'a-kv ::= "\\"a\\"" space ":" space number\nb-kv ::= "\\"b\\"" space ":" space number\nc-kv ::= "\\"c\\"" space ":" space number\nd-kv ::= "\\"d\\"" space ":" space number\nd-rest ::= ( "," space c-kv )?\nhexdigit ::= {HEX_DIGIT}\nnumber ::= {NUMBER}\n'
            +
            'root ::= "{" space a-kv "," space b-kv ( "," space ( d-kv d-rest | c-kv ) )? "}" space_e\n'
            + f'space ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '{"a": 1, "b": 2, "c": 3}',
                '{"a": 1, "b": 2, "d": 4}',
                '{"a": 1, "b": 2}',
            ],
            invalid_examples=[
                '{}',
                '{"a": 1}',
                '{"b": 2}',
                '{"a": 1, "c": 3}',
                '{"a": 1, "d": 4}',
                '{"b": 2, "c": 3}',
                '{"b": 2, "d": 4}',
                '{"a": 1, "b": 2, "c": 3, "d": 4}',
            ]),
        TestCase(
            test_name="object-properties-nesting",
            schema={
                "type": "object",
                "properties": {
                    "number": {
                        "type": "object",
                        "properties": {
                            "number": {
                                "type": "object",
                                "properties": {
                                    "root": {
                                        "type": "number"
                                    }
                                },
                                "required": ["root"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["number"],
                        "additionalProperties": False
                    }
                },
                "required": ["number"],
                "additionalProperties": False,
            },
            grammar=f'hexdigit ::= {HEX_DIGIT}\nnumber ::= {NUMBER}\n' +
            'number- ::= "{" space number-number-kv "}" space_e\nnumber-kv ::= "\\"number\\"" space ":" space number-\nnumber-number ::= "{" space number-number-root-kv "}" space_e\nnumber-number-kv ::= "\\"number\\"" space ":" space number-number\nnumber-number-root-kv ::= "\\"root\\"" space ":" space number\nroot ::= "{" space number-kv "}" space_e\n'
            + f'space ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}',
            valid_examples=[
                '{"number": {"number": {"root": 42}}}',
            ],
            invalid_examples=[
                '{}',
                '{"number": {}}',
                '{"number": {"number": {}}}',
                '{"number": {"number": {"root": "foo"}}}',
            ]),
        TestCase(test_name="object-with-string-type",
                 schema={
                     "type": "object",
                     "properties": {
                         "title": {
                             "type": "string"
                         },
                     },
                     "required": ["title"]
                 }),
        TestCase(test_name="nested-arrays-of-strings",
                 schema={
                     "type": "object",
                     "properties": {
                         "title": {
                             "type": "string"
                         },
                         "authors": {
                             "type": "array",
                             "items": {
                                 "type": "string"
                             },
                         },
                         "publication_date": {
                             "type": "string",
                             "format": "date",
                         },
                         "population": {
                             "type": "string",
                         },
                         "sample_size": {
                             "type": "integer"
                         },
                         "study_duration": {
                             "type": "string"
                         },
                         "exercise_intervention": {
                             "type": "string",
                         },
                         "cognitive_assessment": {
                             "type": "string",
                         },
                         "main_findings": {
                             "type": "array",
                             "items": {
                                 "type": "string"
                             },
                         }
                     },
                     "required": ["sample_size"]
                 }),
        TestCase(test_name="ingredients",
                 schema={
                     'properties': {
                         'ingredients': {
                             'items': {
                                 'properties': {
                                     'quantity': {
                                         'title': 'Quantity',
                                         'type': 'number'
                                     },
                                     'unit': {
                                         'title': 'Unit',
                                         'type': 'string'
                                     },
                                     'name': {
                                         'title': 'Name',
                                         'type': 'string'
                                     }
                                 },
                                 'required': ['quantity', 'unit', 'name'],
                                 'title': 'Ingredient',
                                 'type': 'object'
                             },
                             'type': 'array'
                         },
                         'instructions': {
                             'items': {
                                 'title': 'Instruction',
                                 'type': 'string'
                             },
                             'type': 'array'
                         },
                         'cooking_time': {
                             'title': 'Cooking Time',
                             'type': 'number'
                         }
                     },
                     'required':
                     ['ingredients', 'instructions', 'cooking_time'],
                     'title': 'Recipe',
                     'type': 'object'
                 }),
        TestCase(test_name="AnyOfRedundantString",
                 schema={
                     "properties": {
                         "groceryList": {
                             "items": {
                                 "anyOf": [
                                     {
                                         "title": "Produce",
                                         "type": "string"
                                     },
                                     {
                                         "title": "Dairy",
                                         "type": "string"
                                     },
                                 ]
                             },
                             "title": "GroceryList",
                             "type": "array"
                         }
                     },
                     "required": ["groceryList"],
                     "title": "GroceryListResponse",
                     "type": "object"
                 }),
    ]


def get_all_invalid_cases():
    return [
        TestCase(
            test_name="type-kaboom",
            schema={"type": "kaboom"},
        ),
        TestCase(
            test_name="type-int",
            schema={"type": 123},
        ),
    ]


def gen_all_json_schema_example_pairs():
    """Generates all schema-example pairs.

    Yielded object is of type:
    (test_name, schema, example, is_valid)
    """
    for case in get_all_valid_test_cases():
        if case.schema is None:
            continue

        if case.valid_examples is not None:
            for valid_example in case.valid_examples:
                yield case.test_name, case.schema, valid_example, True

        if case.invalid_examples is not None:
            for invalid_example in case.invalid_examples:
                yield case.test_name, case.schema, invalid_example, False


def gen_all_json_schemas():
    """Generates all schemas.

    Yielded object is of type:
    (test_name, schema)
    """
    for case in get_all_valid_test_cases():
        if case.schema is not None:
            yield case.test_name, case.schema


def gen_all_json_schema_grammar_pairs():
    """Generates all schema-grammar pairs.

    Yielded object is of type:
    (test_name, schema, grammar)
    """
    for case in get_all_valid_test_cases():
        if case.grammar is not None and case.schema is not None:
            yield case.test_name, case.schema, case.grammar


def gen_all_invalid_json_schemas():
    """Generates all invalid schemas.

    Yielded object is of type:
    (test_name, schema)
    """
    for case in get_all_invalid_cases():
        yield case.test_name, case.schema
