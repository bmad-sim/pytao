from pydantic import BaseModel, ConfigDict


class ConstraintsBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
