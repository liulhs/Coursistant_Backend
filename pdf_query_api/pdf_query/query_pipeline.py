# query_pipeline.py
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from typing import Any, Dict, List, Optional
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_pipeline import QueryPipeline, InputComponent, ArgPackComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.llms import ChatMessage
from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel
from llama_index.core.bridge.pydantic import Field
from llama_index.core.output_parsers import PydanticOutputParser

def extract_page_as_image(file_path: str, page_number: int) -> str:
    document = fitz.open(file_path)
    page = document.load_page(page_number - 1)  # page_number is 1-based
    pix = page.get_pixmap()

    # Convert to PIL Image
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save image to a bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode image to base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str

class AnswerFormat(BaseModel):
    """Object representing a single knowledge pdf file."""
    answer: str = "None"
    file_name: str = "None"
    page_number: int = 0
    image: str = None

    @classmethod
    def schema(cls, by_alias: bool = True) -> Dict[str, Any]:
        schema = super().model_json_schema(by_alias)
        properties = schema.get("properties", {})

        # Manually adding descriptions
        properties["answer"]["description"] = "Your Answer to the given query"
        properties["file_name"]["description"] = "PDF file's file name where the answer can be found, fill in with empty string if you couldn't find it"
        properties["page_number"]["description"] = "Page number where the answer can be found, fill in with 0 if you couldn't find it"
        properties["image"]["description"] = "Base64 encoded image of the PDF page"

        return schema

output_parser = PydanticOutputParser(AnswerFormat)
prompt_str = """\
You are given a context with course materials in pdf files format, and you have access to these file's names, page numbers, and the content of the files. 
The file name is displayed at the beginning of every context chunk, and the page number is displayed as a integer number at the end of every page, and pages are separated by a dashed line.
Answer the following questions: {query_str}
And then find the file name and page number where the answer is mentioned.
"""
json_prompt_str = output_parser.format(prompt_str)
DEFAULT_CONTEXT_PROMPT = json_prompt_str + (
    "Here is some context that may be relevant:\n"
    "-----\n"
    "{node_context}\n"
    "-----\n"
)


class Response(CustomQueryComponent):
    llm: OpenAI = Field(..., description="OpenAI LLM")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use for the LLM"
    )
    context_prompt: str = Field(
        default=DEFAULT_CONTEXT_PROMPT,
        description="Context prompt to use for the LLM",
    )

    def _validate_component_inputs(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return input

    @property
    def _input_keys(self) -> set:
        return {"nodes", "query_str"}

    @property
    def _output_keys(self) -> set:
        return {"response"}

    def _prepare_context(
        self,
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[ChatMessage]:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"Context Chunk {idx}:\n{node_text}\n\n"

        formatted_context = self.context_prompt.format(
            node_context=node_context, query_str=query_str
        )
        user_message = ChatMessage(role="user", content=formatted_context)

        context = [user_message]

        if self.system_prompt is not None:
            context = [
                ChatMessage(role="system", content=self.system_prompt)
            ] + context

        return context

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(
            nodes, query_str
        )
        
        response = self.llm.chat(prepared_context)
        return {"response": response}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(
            nodes, query_str
        )

        response = await self.llm.achat(prepared_context)

        return {"response": response}

def create_pipeline(persist_dir):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    input_component = InputComponent()
    llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
    )
    retriever = index.as_retriever(similarity_top_k=4)
    reranker = ColbertRerank(top_n=3)
    response_component = Response(
        llm=llm,
        system_prompt=(
            "You are a Q&A system. You will be provided with the previous chat history, "
            "as well as possibly relevant context, to assist in answering a user message."
        )
    )
    output_parser = PydanticOutputParser(AnswerFormat)

    pipeline = QueryPipeline(
        modules={
            "input": input_component,
            "query_retriever": retriever,
            "reranker": reranker,
            "response_component": response_component,
            "output_parser": output_parser
        },
        verbose=False,
    )

    pipeline.add_link("input", "query_retriever", src_key="query_str")
    pipeline.add_link("query_retriever", "reranker", dest_key="nodes")
    pipeline.add_link(
        "input", "reranker", src_key="query_str", dest_key="query_str"
    )
    pipeline.add_link("reranker", "response_component", dest_key="nodes")
    pipeline.add_link("input", "response_component", dest_key="query_str")
    pipeline.add_link("response_component", "output_parser")

    return pipeline

def query_llm(user_input, pipeline):
    """
    Query the LLM with user input, using the provided pipeline.

    Args:
        user_input (str): The user's query.
        pipeline (QueryPipeline): The pipeline to process the query.

    Returns:
        response (AnswerFormat): The response from the pipeline.
    """
    # Run pipeline
    response = pipeline.run(query_str=user_input)

    # Set default values if not found
    response.answer = response.answer if response.answer else 'Not Found'
    response.file_name = response.file_name if response.file_name else 'Not Found'
    response.page_number = response.page_number if response.page_number > 0 else 1

    # Add image extraction logic if the answer is found
    if response.answer != 'Not Found' and response.file_name != 'Not Found':
        file_path = f"/home/jason/coursistant/Coursistant_Backend/llama_api/pdf_query/data/EE542_Slides/{response.file_name}"
        image_base64 = extract_page_as_image(file_path, response.page_number)
        response.image = image_base64

    return response

