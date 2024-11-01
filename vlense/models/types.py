from pydantic import BaseModel

class CompletionResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
