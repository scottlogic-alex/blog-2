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

It promises "flexible abstractions", "easily-swappable components" and "pre-built chains for any use-case".

So we can replace its proprietary API usage (e.g. ChatGPT) with a locally-run LLM? Let's give it a try.

