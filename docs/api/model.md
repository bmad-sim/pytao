# Model Base Classes

The model base classes provide serialization (JSON, JSON.GZ, msgpack, YAML),
equality checking, and (for settable models) the ability to generate
and apply Tao `set` commands.

## TaoBaseModel

Common base for all PyTao Pydantic models. Provides `write()` and `from_file()`
for serialization to JSON, compressed JSON (`.json.gz`), msgpack, and YAML.

::: pytao.model.TaoBaseModel

## TaoModel

Read-only model populated by querying a Tao instance. Each subclass defines a
`_tao_command_attr_` that maps to a specific Tao pipe command.

::: pytao.model.TaoModel

## TaoSettableModel

Extends `TaoModel` with the ability to generate `set` commands and apply changes
back to Tao. Supports differential updates (only changed fields), context-manager
usage, and error-tolerant application.

::: pytao.model.TaoSettableModel

## Serialization Helpers

::: pytao.model.load_model_data

::: pytao.model.load_model

::: pytao.model.dump_model
