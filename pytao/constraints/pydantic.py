from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)


class ConstraintsBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def handle_exclude_defaults(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        # Run the standard serialization logic
        serialized_data = handler(self)
        type_manually_excluded = info.exclude and "type" in info.exclude

        # Re-inject the field manually if stripped by exclude_defaults,
        # but not excluded explicitly
        if (
            hasattr(self, "type")
            and ("type" not in serialized_data)
            and not type_manually_excluded
        ):
            serialized_data["type"] = self.type

        return serialized_data
