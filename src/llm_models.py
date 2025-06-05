from enum import Enum

class OpenAIModel(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    
class AnthropicModel(Enum):
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
    CLAUDE_3_7_HAIKU = "claude-3-7-haiku"

