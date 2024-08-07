# query_pipeline_piazza.py
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

class AnswerFormat(BaseModel):
    """Object representing a Piazza post response."""
    answer: str = "None"

    @classmethod
    def schema(cls, by_alias: bool = True) -> Dict[str, Any]:
        schema = super().model_json_schema(by_alias)
        properties = schema.get("properties", {})

        # Manually adding descriptions
        properties["answer"]["description"] = "Your Answer to the given query"

        return schema

output_parser = PydanticOutputParser(AnswerFormat)
prompt_str = """\
You are a Virtual Teaching Assistant. Answer questions in the style of a human TA. 
If you can find the answer in the context, don't modify it too much, but if there are multiple relevant answers, structure them and summarize.
Answer the following questions: {query_str}
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
        model="gpt-4",
        temperature=0.2,
    )
    retriever = index.as_retriever(similarity_top_k=4)
    reranker = ColbertRerank(top_n=3)
    response_component = Response(
        llm=llm,
        system_prompt=(
            "You are a Virtual Teaching Assistant. Answer questions in the style of a human TA."
            "If you can find the answer in the context, don't modify it too much, but if there are multiple relevant answers, structure them and summarize."
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

    return response
