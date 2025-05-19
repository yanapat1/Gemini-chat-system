from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from typing import Dict, List
from pydantic import BaseModel

class ModelResponse(BaseModel):
    content: str
    usage_metadata: dict
    model_version: str = None
    avg_logprobs: float = None
    cost: float = 0.

class ModelGemini:
    def __init__(self, token: str, model_name: str = "gemini-1.5-flash-8b"):
        genai.configure(api_key=token)

        self.model_name = model_name

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.generation_config = genai.types.GenerationConfig(
            temperature = 1
        )

        self.system_instruction = None
        self.tools_calling = None
        self.tasks = list()
        self.history: List[Dict] = list()

        self.model = genai.GenerativeModel(
            model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
            )
    
    def __repr__(self):
        return f'Gemini -> {self.model_name}'

    def add_system_prompt(self, system_instruction):
        self.system_instruction = system_instruction
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config = self.generation_config,
            safety_settings = self.safety_settings,
            system_instruction = self.system_instruction,
            tools = self.tools_calling
            )

    def add_tool_call(self, tools_calling: List[object]):
        self.tools_calling = tools_calling
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config = self.generation_config,
            safety_settings = self.safety_settings,
            system_instruction = self.system_instruction,
            tools = self.tools_calling
            )
    
    def add_generation_config(self, **kwargs):
        ''' 
            temperature = 1 
            candidate_count = 1 
            stop_sequences = ['assistant']
            max_output_tokens = 2048
            top_p= 1
            top_k= 1
        '''
        self.generation_config = genai.types.GenerationConfig(
            temperature = 1,
            **kwargs
        )
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config = self.generation_config,
            safety_settings = self.safety_settings,
            system_instruction = self.system_instruction,
            tools = self.tools_calling
        )
    
    def add_safety_settings(self, hate_speech: int = 4, harassent: int = 4, sexually_expicit: int = 4, dangerous_content: int = 4):
        """
            Values:

            BLOCK_LOW_AND_ABOVE (1):
            Content with NEGLIGIBLE will be allowed.

            BLOCK_MEDIUM_AND_ABOVE (2):
            Content with NEGLIGIBLE and LOW will be allowed.

            BLOCK_ONLY_HIGH (3):
            Content with NEGLIGIBLE, LOW, and MEDIUM will
            be allowed.

            BLOCK_NONE (4):
            All content will be allowed.
        """
        hbt = {
            1 : HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            2 : HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            3 : HarmBlockThreshold.BLOCK_ONLY_HIGH,
            4 : HarmBlockThreshold.BLOCK_NONE,
        }
        hate_speech = hbt.get(hate_speech, HarmBlockThreshold.BLOCK_NONE)
        harassent = hbt.get(harassent, HarmBlockThreshold.BLOCK_NONE)
        sexually_expicit = hbt.get(sexually_expicit, HarmBlockThreshold.BLOCK_NONE)
        dangerous_content = hbt.get(dangerous_content, HarmBlockThreshold.BLOCK_NONE)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: hate_speech,
            HarmCategory.HARM_CATEGORY_HARASSMENT: harassent,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: sexually_expicit,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: dangerous_content,
        }
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config = self.generation_config,
            safety_settings = self.safety_settings,
            system_instruction = self.system_instruction,
            tools = self.tools_calling
        )

    def create_tasks(self, content):
        if 'text' in content:
            self.history.append({'role':'model', 'parts': [content]})
        if 'function_call' in content:
            function_name = content.function_call.name
            self.tasks.append({'agent': function_name, 'content': dict(content.function_call.args).get('quesiton')})

    async def generate_text(self, content: str | list[Dict]):
        if isinstance(content, str):
            self.history.append({'role':'user', 'parts': [content]})
            chat = self.model.start_chat()
        if isinstance(content, list):
            self.history = content[:-1]
            chat = self.model.start_chat(history=self.history)
            content = content[-1]['parts'][0]
        global responses
        responses = await chat.send_message_async(content)

        for response in responses.parts:
            self.create_tasks(response)

        return responses

    def gemini_1p5_flash_8b(self, input_token: int, output_token: int) -> float:
        if input_token <= 128_000:
            in_cost = ( input_token / 1_000_000 ) * 0.0375
        else:
            in_cost = ( input_token / 1_000_000 ) * 0.0750
        if output_token <= 128_000:
            op_cost = ( output_token / 1_000_000 ) * 0.150
        else:
            op_cost = ( output_token / 1_000_000 ) * 0.300
        
        cost = in_cost + op_cost
        return cost

    def calculated_cost(self, input_token: int, output_token: int) -> float:
        match self.model_name:
            case 'gemini-1.5-flash-8b': cost = self.gemini_1p5_flash_8b(input_token, output_token)
            case _: cost = 0
                
        return cost

    async def ainvoke(self, content: str | List[Dict], **kwargs):
        if kwargs:
            self.add_generation_config(**kwargs)
        response = self.generate_text(content)
        
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        cost = self.calculated_cost(input_tokens, output_tokens)

        info_response = ModelResponse(
            content = response.candidates[0].content.parts[0].text,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": response.usage_metadata.total_token_count
            },
            model_version = response.model_version,
            avg_logprobs = response.candidates[0].avg_logprobs,
            cost = cost
        )
        return info_response
