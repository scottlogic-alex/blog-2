---
title: The LangChain Dream is Overstated
date: 2024-01-15 00:00:00 Z
categories:
- abirch
- Tech
author: abirch
layout: default_post
summary: LangChain promises easy swappability. It didn't go so well for us.
---

<style>
  details {
    font-weight: 300;
  }
  summary {
    font-weight: 300;
    display: block;
  }
  summary::after {
    cursor: pointer;
    content: '[+more]';
    text-decoration: underline;
    text-decoration-style: dotted;
    padding-left: 0.5em;
    font-size: 0.8em;
  }
  details[open] > summary::after {
    content: ' [−less]';
  }
  dl.mydl {
    margin-left: 0.5em;
  }
  dl.mydl > dd {
    margin-left: 0.5em;
    font-weight: 300;
  }
</style>

[LangChain](https://www.langchain.com/) is a framework which assists with sampling from LLMs (Large Language Models). It provides capabilities such as:


<dl class="mydl">
  <dt><a href="https://arxiv.org/abs/2005.11401">RAG (Retrieval-Augmented Generation)</a></dt>
  <dd><details><summary>Include useful documents in the prompt to help the <abbr title="Large Language Model">LLM</abbr> answer with information outside of its training dataset.</summary>Relevant documents are found via embedding similarity search. The query can be based on the user input (e.g. a summary of their request), or the <abbr title="Large Language Model">LLM</abbr> can be trained to recognise retrieval opportunities and output them as part of its own self-talk.</details></dd>

  <!-- <dt><a href="https://arxiv.org/abs/2309.11392">Evidenced question answering</a></dt>
  <dd></dd> -->

  <dt><a href="https://arxiv.org/abs/2210.03629">ReAct</a></dt>
  <dd>Prompt <abbr title="Large Language Model">LLM</abbr> to provide a reasoned response. An evolution of <a href="https://arxiv.org/abs/2201.11903">CoT (Chain-of-Thought)</a> prompting.</dd>

  <dt><a href="https://arxiv.org/abs/2311.08719">TiM (Think-in-Memory)</a></dt>
  <dd>Compresses conversation history by extracting entities and storing only a summary of their most recent status.</dd>

  <dt><a href="https://arxiv.org/abs/2302.04761">Tool usage</a></dt>
  <dd>Prompt <abbr title="Large Language Model">LLM</abbr> to invoke "tools", using external functionality (e.g. calculator, web search) to query for solutions that it cannot predict with its training.</dd>

  <!-- <dt><a href="https://arxiv.org/abs/2005.14165">Few-shot prompting</a></dt>
  <dd></dd> -->
</dl>

<!-- <details><summary><em>Click</em></summary>
<em>(yes, like that. there's a few of these, so keep an eye out)</em>
</details>
<p>
<em>to reveal more detail. We hide deep-dives in these collapsible sections to keep the article snappy.</em>
</p> -->

It [promises](https://www.langchain.com/) "flexible abstractions", "easily-swappable components" and "pre-built chains for any use-case".

So we can replace its proprietary API usage (e.g. ChatGPT) with a locally-run LLM? Let's give it a try.

### Local LLM Clients (Might) Cost VRAM

LangChain provides an abstract interface [`BaseLanguageModel`](https://api.python.langchain.com/en/stable/language_models/langchain_core.language_models.base.BaseLanguageModel.html) for text generation.

We had been using an [`OpenAI`](https://api.python.langchain.com/en/stable/llms/langchain_community.llms.openai.OpenAI.html) client as the concrete implementation throughout our codebase.

To support local LLMs, we generalize all such client construction into a `get_base_language_model()` factory, and configure it to return a local LLM client (e.g. [`LlamaCpp`](https://api.python.langchain.com/en/stable/llms/langchain_community.llms.llamacpp.LlamaCpp.html)).

This plan doesn't work out. We run out of VRAM.

`LlamaCpp` instances each spin up their own [`llama.cpp`](https://github.com/ggerganov/llama.cpp) instance. With their own resources and startup associated with them.

So, we need to re-use `LlamaCpp` instances. Memoizing the Llama client construction with an `@lru_cache()` decorator resolves this problem:

```python
from langchain.llms.llamacpp import LlamaCpp
from langchain_core.language_models.base import BaseLanguageModel
from functools import lru_cache

@lru_cache(maxsize=8)
def get_llamacpp_llm(model_path: str, n_ctx: int, cache=False) -> LlamaCpp:
    """Memoized function which returns a LlamaCpp handle"""
    return LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=35,
        cache=cache,
        verbose=False,
        seed=42,
    )

# returns a LlamaCpp client (via get_llamacpp_llm) or OpenAI client, based on your config
def get_base_language_model() -> BaseLanguageModel: ...
```

### Local LLM Clients (Might) Have Non-Serializable State

To persist conversations into the session of our webserver (Flask), we had been relying on saving the chat memory (in our case [`ConversationEntityMemory`](https://api.python.langchain.com/en/latest/memory/langchain.memory.entity.ConversationEntityMemory.html)).

When we switched to local LLMs, we found that loading chat memory caused the webserver to terminate without an error message.

This is because `ConversationEntityMemory` holds a reference to your `BaseLanguageModel`. API clients such as `OpenAI` are no problem to serialize, but clients such as `LlamaCpp` own resources, handles and state.

To fix this, we looked at the properties of `ConversationEntityMemory` to determine which properties constitute its conversation history (`chat_memory` and `entity_cache`), and persisted just those. We then determined how to hydrate empty `ConversationEntityMemory` instances with this loaded state.

```python
from langchain.memory import ConversationEntityMemory
from langchain.memory.entity import InMemoryEntityStore
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from typing import List, Dict, TypedDict, Optional

class BaseMemoryState(TypedDict):
    messages: List[BaseMessage]

class ConversationEntityMemoryState(BaseMemoryState):
    entity_cache: List[str]
    entity_store: Dict[str, Optional[str]]

def set_state_chat(memory: BaseChatMemory, state: BaseMemoryState) -> None:
    assert isinstance(memory.chat_memory, ChatMessageHistory)
    memory.chat_memory.messages.extend(state["messages"])

def get_state_conversation_entity_memory(memory: ConversationEntityMemory) -> ConversationEntityMemoryState:
    assert isinstance(memory.entity_store, InMemoryEntityStore)
    base_state: BaseMemoryState = get_state_chat(memory)
    return ConversationEntityMemoryState(
        **base_state,
        entity_cache=memory.entity_cache,
        entity_store=memory.entity_store.store,
    )
```

### LLM Client Installation is Non-Trivial

An additional dependency, [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) is needed in order to use Langchain's LlamaCpp client.

`llama-cpp-python` relies on platform-specific code, so installing it isn't currently as simple as `pip install llama-cpp-python`. Doing so will give you a CPU-only binary.

To compile it from source for our platform, Linux/CUDA, we used:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DLLAMA_CUDA_F16=ON" FORCE_CMAKE=1 pip install llama-cpp-python
```

> We enable [`LLAMA_CUDA_F16`](https://github.com/ggerganov/llama.cpp/blob/2b3a665d3917edf393761a24c4835447894df74a/README.md?plain=1#L445) as a speed-boost for [supported GPUs](https://github.com/ggerganov/llama.cpp/issues/2844) (CUDA [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) 6.0/Pascal or higher).

If you had already installed it the naïve way (CPU-only wheel distribution) and want to reinstall it with platform-specific functionality: you can use `--upgrade --force-reinstall --no-cache-dir --no-deps` options to force a rebuild:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DLLAMA_CUDA_F16=ON" FORCE_CMAKE=1 pip install --upgrade --force-reinstall --no-cache-dir --no-deps llama-cpp-python
```

Installation for other platforms is detailed [in the `llama-cpp-python` README](https://github.com/abetlen/llama-cpp-python#installation-with-specific-hardware-acceleration-blas-cuda-metal-etc).