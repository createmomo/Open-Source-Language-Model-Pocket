# 开源语言模型百宝袋 (Ver. 1.6)
Open-Source Language Model Pocket

**Github**: https://github.com/createmomo/Open-Source-Language-Model-Pocket

相关文章：
- [练习场：穷穷穷孩子如何体验ColossalAI SFT（Kaggle篇）](https://mp.weixin.qq.com/s/Q29uSNxvPMy0rC-QxHiGZA)
- [练习场：穷穷穷孩子如何体验ColossalAI SFT（Colab篇）](https://mp.weixin.qq.com/s/NS4yySeYd7QUYb7CB9V0lA)

## 1 工具箱（Tools）
### 高效对齐算法RAFT「木筏」
- https://github.com/OptimalScale/LMFlow
- https://arxiv.org/abs/2304.06767
- https://optimalscale.github.io/LMFlow/examples/raft.html

An extensible, convenient, and efficient toolbox for finetuning large machine learning models, designed to be user-friendly, speedy and reliable, and accessible to the entire community.

### Alpaca-LoRA
- https://github.com/tloen/alpaca-lora

Low-Rank LLaMA Instruct-Tuning

This repository contains code for reproducing the Stanford Alpaca results using low-rank adaptation (LoRA). We provide an Instruct model of similar quality to text-davinci-003 that can run on a Raspberry Pi (for research), and the code can be easily extended to the 13b, 30b, and 65b models.

In addition to the training code, which runs within five hours on a single RTX 4090, we publish a script for downloading and inference on the foundation model and LoRA, as well as the resulting LoRA weights themselves. To fine-tune cheaply and efficiently, we use Hugging Face's PEFT as well as Tim Dettmers' bitsandbytes.

Without hyperparameter tuning or validation-based checkpointing, the LoRA model produces outputs comparable to the Stanford Alpaca model. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.

### Alpaca-CoT
- https://github.com/PhoebusSi/Alpaca-CoT
- https://mp.weixin.qq.com/s/Q5Q3RpQ80XmpbfhSxq2R1Q

An Instruction Fine-Tuning Platform with Instruction Data Collection and Unified Large Language Models Interface

Alpaca-CoT项目旨在探究如何更好地通过instruction-tuning的方式来诱导LLM具备类似ChatGPT的交互和instruction-following能力。为此，我们广泛收集了不同类型的instruction（尤其是Chain-of-Thought数据集），并基于LLaMA给出了深入细致的实证研究，以供未来工作参考。据我们所知，我们是首个将CoT拓展进Alpaca的工作，因此简称为"Alpaca-CoT"。

### Auto-GPT
- https://github.com/torantulino/auto-gpt

Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. This program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, Auto-GPT pushes the boundaries of what is possible with AI.

### AlpacaFarm
- https://mp.weixin.qq.com/s/CIF2F5Vx_RSN1-LwU_ppOQ
- https://tatsu-lab.github.io/alpaca_farm_paper.pdf
- https://github.com/tatsu-lab/alpaca_farm

主流的大型语言模型训练都离不开RLHF(人工反馈强化学习)，其主要思想是使用人类专家提供的反馈示例来指导模型的学习过程，它可以加速强化学习过程，提高大模型的性能，但「目前RLHF这个过程既复杂又昂贵」。

 针对RLHF这个问题，学术界目前主要有两种解决方法：「1）避开RLHF」，比如Meta最近研究的“Meta最新模型：LIMA-65B，没有RLHF，模型效果远胜Alpaca！！”，验证了精心制作的少量标注数据同样能达到不错的效果。2）「简化RLHF」，就是今天给大家分享的这篇文章：斯坦福发布了一个名为AlpacaFarm（羊驼农场）的模拟器，旨在降低训练语言模型的成本，且比人工成本低45倍，并表现出与人类反馈的高度一致性，同时也为RLHF的研究开辟了新的道路。

### BELLE: Bloom-Enhanced Large Language model Engine
- https://github.com/LianjiaTech/BELLE
- https://zhuanlan.zhihu.com/p/616079388

本项目目标是促进中文对话大模型开源社区的发展，愿景做能帮到每一个人的LLM Engine。现阶段本项目基于一些开源预训练大语言模型（如BLOOM），针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。

### ColossalAI
- https://github.com/hpcaitech/ColossalAI

Colossal-AI: Making large AI models cheaper, faster and more accessible

Colossal-AI provides a collection of parallel components for you. We aim to support you to write your distributed deep learning models just like how you write your model on your laptop. We provide user-friendly tools to kickstart distributed training and inference in a few lines.

### Cerebras
- https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/
- https://huggingface.co/cerebras

开源7个可商用GPT模型，含数据集和可直接下载的预训练模型权重: Cerebras 开源 7 个 GPT 模型，均可商用，参数量分别达到 1.11 亿、2.56 亿、5.9 亿、13 亿、27 亿、67 亿和 130 亿。其中最大的模型参数量达到 130 亿，与 Meta 最近开源的 LLaMA-13B 相当。该项目开源数据集和预训练模型权重，其中预训练模型权重文件大小近50G可直接下载，并且可用于商业和研究用途。与此前的 GPT-3 模型相比，Cerebras 开源的模型具有更高的可用性和透明度，研究人员和开发者可以使用少量数据对其进行微调，构建出高质量的自然语言处理应用。

### ChatPiXiu
- https://github.com/catqaq/ChatPiXiu

我们是羡鱼智能【xianyu.ai】，主要成员是一群来自老和山下、西湖边上的咸鱼们，塘主叫作羡鱼，想在LLMs时代做点有意义的事！我们的口号是：做OpenNLP和OpenX！希望在CloseAI卷死我们之前退出江湖！

也许有一天，等到GPT-X发布的时候，有人会说NLP不存在了，但是我们想证明有人曾经来过、热爱过！在以ChatGPT/GPT4为代表的LLMs时代，在被CloseAI卷死之前，我们发起了OpenNLP计划，宗旨是OpenNLP for everyone!

ChatPiXiu项目为OpenNLP计划的第2个正式的开源项目，旨在Open ChatGPT for everyone！在以ChatGPT/GPT4为代表的LLMs时代，在被OpenAI卷死之前，做一点有意义的事情！未来有一天，等到GPT-X发布的时候，或许有人会说NLP不存在了，但是我们想证明有人曾来过！

### ChatRWKV
- https://github.com/BlinkDL/ChatRWKV

ChatRWKV is like ChatGPT but powered by my RWKV (100% RNN) language model, which is the only RNN (as of now) that can match transformers in quality and scaling, while being faster and saves VRAM. Training sponsored by Stability EleutherAI :)

### ChatLLaMA
- https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama

ChatLLaMA 🦙 has been designed to help developers with various use cases, all related to RLHF training and optimized inference.

ChatLLaMA is a library that allows you to create hyper-personalized ChatGPT-like assistants using your own data and the least amount of compute possible. Instead of depending on one large assistant that “rules us all”, we envision a future where each of us can create our own personalized version of ChatGPT-like assistants. Imagine a future where many ChatLLaMAs at the "edge" will support a variety of human's needs. But creating a personalized assistant at the "edge" requires huge optimization efforts on many fronts: dataset creation, efficient training with RLHF, and inference optimization.

### DeepSpeed-Chat
- https://mp.weixin.qq.com/s/t3HA4Hu61LLDC3h2Njmo_Q
- https://github.com/microsoft/DeepSpeed

微软宣布开源 DeepSpeed-Chat，帮助用户轻松训练类 ChatGPT 等大语言模型。

据悉，Deep Speed Chat 是基于微软 Deep Speed 深度学习优化库开发而成，具备训练、强化推理等功能，还使用了 RLHF（基于人类反馈的强化学习）技术，可将训练速度提升 15 倍以上，而成本却大大降低。

### Dolly 1&2
- https://github.com/databrickslabs/dolly
- https://huggingface.co/databricks/dolly-v2-12b
- https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html

We show that anyone can take a dated off-the-shelf open source large language model (LLM) and give it magical ChatGPT-like instruction following ability by training it in 30 minutes on one machine, using high-quality training data. Surprisingly, instruction-following does not seem to require the latest or largest models: our model is only 6 billion parameters, compared to 175 billion for GPT-3. We open source the code for our model (Dolly) and show how it can be re-created on Databricks. We believe models like Dolly will help democratize LLMs, transforming them from something very few companies can afford into a commodity every company can own and customize to improve their products.

### FlexGen
- https://github.com/FMInference/FlexGen

FlexGen is a high-throughput generation engine for running large language models with limited GPU memory. FlexGen allows high-throughput generation by IO-efficient offloading, compression, and large effective batch sizes.

Limitation. As an offloading-based system running on weak GPUs, FlexGen also has its limitations. FlexGen can be significantly slower than the case when you have enough powerful GPUs to hold the whole model, especially for small-batch cases. FlexGen is mostly optimized for throughput-oriented batch processing settings (e.g., classifying or extracting information from many documents in batches), on single GPUs.

### FlagAI and FlagData

- https://github.com/FlagAI-Open/FlagAI

FlagAI (Fast LArge-scale General AI models) is a fast, easy-to-use and extensible toolkit for large-scale model. Our goal is to support training, fine-tuning, and deployment of large-scale models on various downstream tasks with multi-modality.

- https://github.com/FlagOpen/FlagData

FlagData, a data processing toolkit that is easy to use and expand. FlagData integrates the tools and algorithms of multi-step data processing, including cleaning, condensation, annotation and analysis, providing powerful data processing support for model training and deployment in multiple fields, including natural language processing and computer vision. 

### Facebook LLaMA
- https://github.com/facebookresearch/llama

LLaMA: Open and Efficient Foundation Language Models

We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community.

### * 【Gorilla】
- https://mp.weixin.qq.com/s/p9tx3q3Lpr4fNqdyxWhzyA
- gorilla.cs.berkeley.edu
- arxiv.org/abs/2305.15334
- https://github.com/ShishirPatil/gorilla/

大型语言模型性能强大，但为了更好地用于解决实际问题，各式各样的 API 是必不可少的。

加利福尼亚大学伯克利分校和微软研究院造出了一只「大猩猩」Gorilla，该模型能根据用户输入的自然语言为用户选择合适的 API 来执行对应任务。理论上讲，这个模型可以根据用户需求调用其它各种 AI 模型，因此 Gorilla 有望成为一个统御其它 AI 的 AI 模型。该项目的代码、模型、数据和演示都已发布。

### GPT4All
- https://github.com/nomic-ai/gpt4all

Demo, data and code to train an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa

### HuggingChat
- https://huggingface.co/chat/

Making the community's best AI chat models available to everyone.

### HuggingGPT
- https://mp.weixin.qq.com/s/o51CmLt2JViJ4nsKfBJfwg
- https://arxiv.org/pdf/2303.17580.pdf

HuggingGPT利用ChatGPT作为控制器，连接HuggingFace社区中的各种AI模型，来完成多模态复杂任务。

这意味着，你将拥有一种超魔法，通过HuggingGPT，便可拥有多模态能力，文生图、文生视频、语音全能拿捏了。

### HugNLP
- https://mp.weixin.qq.com/s/IpgOQJ8vrIvnjdrmGCT2FA
- https://github.com/HugAILab/HugNLP
- https://arxiv.org/abs/2302.14286

华师大HugAILab团队研发了HugNLP框架，这是一个面向研究者和开发者的全面统一的NLP训练框架，可支持包括文本分类、文本匹配、问答、信息抽取、文本生成、小样本学习等多种NLP任务模型搭建和训练。

HugNLP还集成了大量最新的Prompt技术，例如Prompt-Tuning、In-Context Learning、Instruction-tuning，未来还将引入Chain-of-thought

HugAILab团队还研发了一系列的应用，例如CLUE&GLUE刷榜工具，可支持ChatGPT类模型训练和部署产品HugChat，以及统一信息抽取产品HugIE等。

HugNLP是一个分层式框架，遵循“高内聚低耦合”的开发模式，其核心包括模型层（Models）、处理器层（Processors）、评估器层（Evaluators）和应用层（Applications）四部分。

### Koala: A Dialogue Model for Academic Research
- https://bair.berkeley.edu/blog/2023/04/03/koala/

In this post, we introduce Koala, a chatbot trained by fine-tuning Meta’s LLaMA on dialogue data gathered from the web. We describe the dataset curation and training process of our model, and also present the results of a user study that compares our model to ChatGPT and Stanford’s Alpaca. Our results show that Koala can effectively respond to a variety of user queries, generating responses that are often preferred over Alpaca, and at least tied with ChatGPT in over half of the cases.

### LLaMA复刻版OpenLLaMA
- https://github.com/openlm-research/open_llama

In this repo, we release a permissively licensed open source reproduction of Meta AI's LLaMA large language model. In this release, we're releasing a public preview of the 7B OpenLLaMA model that has been trained with 200 billion tokens. We provide PyTorch and Jax weights of pre-trained OpenLLaMA models, as well as evaluation results and comparison against the original LLaMA models. Stay tuned for our updates.

### LLMPruner：大语言模型裁剪工具
- https://mp.weixin.qq.com/s/u0UcCxzJOkF4fO_JI6ToQA
- https://github.com/yangjianxin1/LLMPruner

在许多下游任务中，我们往往只需要使用到一两种语言，例如在中文场景中，一般只会用到中英文。 所以我们可以对大语言模型的词表进行裁剪，只留下所需的部分词表，这样不仅能够充分保留模型的预训练知识，并且减少模型参数量，降低显存占用，提升训练速度，使用更少的显卡进行下游任务的finetune训练。

基于上述原因，笔者开发了LLMPruner项目，目前主要包含裁剪后的各种参数规模的Bloom模型。对Bloom进行词表裁剪，保留常用的中英文token，词表由250880将至46145，缩减为原来的18.39%。

### llama.cpp
- https://github.com/ggerganov/llama.cpp

Inference of LLaMA model in pure C/C++

The main goal is to run the model using 4-bit quantization on a MacBook
- Plain C/C++ implementation without dependencies
- Apple silicon first-class citizen - optimized via ARM NEON
- AVX2 support for x86 architectures
- Mixed F16 / F32 precision
- 4-bit quantization support
- Runs on the CPU

### Llama-X: Open Academic Research on Improving LLaMA to SOTA LLM
- https://github.com/AetherCortex/Llama-X

This is the repo for the Llama-X, which aims to:
- Progressively improve the performance of LLaMA to SOTA LLM with open-source community.
- Conduct Llama-X as an open academic research which is long-term, systematic and rigorous.
- Save the repetitive work of community and we work together to create more and faster increment.

### Lit-LLaMA ️
- https://github.com/Lightning-AI/lit-llama

Lit-LLaMA is:
- Simple: Single-file implementation without boilerplate.
- Correct: Numerically equivalent to the original model.
- Optimized: Runs on consumer hardware or at scale.
- Open-source: No strings attached.

### MLC LLM
- https://github.com/mlc-ai/mlc-llm

MLC LLM is a universal solution that allows any language models to be deployed natively on a diverse set of hardware backends and native applications, plus a productive framework for everyone to further optimize model performance for their own use cases.

Our mission is to enable everyone to develop, optimize and deploy AI models natively on everyone's devices.

Everything runs locally with no server support and accelerated with local GPUs on your phone and laptops. Supported platforms include:
- iPhone, iPad
- Metal GPUs and Intel/ARM MacBooks;
- AMD, Intel and NVIDIA GPUs via Vulkan on Windows and Linux;
- NVIDIA GPUs via CUDA on Windows and Linux;
- WebGPU on browsers (through companion project WebLLM).

### MPT-7B
- https://www.mosaicml.com/blog/mpt-7b
- https://huggingface.co/mosaicml/mpt-7b

MPT-7B is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. This model was trained by MosaicML.

MPT-7B is part of the family of MosaicPretrainedTransformer (MPT) models, which use a modified transformer architecture optimized for efficient training and inference.

Introducing MPT-7B, the latest entry in our MosaicML Foundation Series. MPT-7B is a transformer trained from scratch on 1T tokens of text and code. It is open source, available for commercial use, and matches the quality of LLaMA-7B. MPT-7B was trained on the MosaicML platform in 9.5 days with zero human intervention at a cost of ~$200k. Starting today, you can train, finetune, and deploy your own private MPT models, either starting from one of our checkpoints or training from scratch. For inspiration, we are also releasing three finetuned models in addition to the base MPT-7B: MPT-7B-Instruct, MPT-7B-Chat, and MPT-7B-StoryWriter-65k+, the last of which uses a context length of 65k tokens!

### OpenChatKit
- https://www.together.xyz/blog/openchatkit 
- https://huggingface.co/spaces/togethercomputer/OpenChatKit
- https://github.com/togethercomputer/OpenChatKit

OpenChatKit uses a 20 billion parameter chat model trained on 43 million instructions and supports reasoning, multi-turn conversation, knowledge and generative answers.

OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications. The kit includes an instruction-tuned 20 billion parameter language model, a 6 billion parameter moderation model, and an extensible retrieval system for including up-to-date responses from custom repositories. It was trained on the OIG-43M training dataset, which was a collaboration between Together, LAION, and Ontocord.ai. Much more than a model release, this is the beginning of an open source project. We are releasing a set of tools and processes for ongoing improvement with community contributions.

### Open-Assistant
- https://github.com/LAION-AI/Open-Assistant
- https://open-assistant.io/zh

Open Assistant is a project meant to give everyone access to a great chat based large language model.

We believe that by doing this we will create a revolution in innovation in language. In the same way that stable-diffusion helped the world make art and images in new ways we hope Open Assistant can help improve the world by improving language itself.

### PandaLM
- https://github.com/WeOpenML/PandaLM
- https://zhuanlan.zhihu.com/p/630173415
- https://mp.weixin.qq.com/s/HE6jez3G9aEO5qLkvwtKXg

This is the official repository for PandaLM: ReProducible and Automated Language Model Assessment.

PandaLM aims to provide reproducible and automated comparisons between different large language models (LLMs). By giving PandaLM the same context, it can compare the responses of different LLMs and provide a reason for the decision, along with a reference answer. The target audience for PandaLM may be organizations that have confidential data and research labs with limited funds that seek reproducibility. These organizations may not want to disclose their data to third parties or may not be able to afford the high costs of secret data leakage using third-party APIs or hiring human annotators. With PandaLM, they can perform evaluations without compromising data security or incurring high costs, and obtain reproducible results. To demonstrate the reliability and consistency of our tool, we have created a diverse human-annotated test dataset of approximately 1,000 samples, where the contexts and the labels are all created by humans. On our test dataset, PandaLM-7B has achieved 94% ChatGPT's evaluation ability in terms of accuracy. The papers and more features are coming soon.

### PKU-Beaver 河狸 (Safe RLHF)
- https://github.com/PKU-Alignment/safe-rlhf
- https://mp.weixin.qq.com/s/ZpkgszXbisl5xf63EfTNjQ

北京大学团队开源了名为 PKU-Beaver（河狸）项目，其开源地址为：https://github.com/PKU-Alignment/safe-rlhf。该项目首次公开了 RLHF 所需的数据集、训练和验证代码，是目前首个开源的可复现的 RLHF 基准。同时，为解决人类标注产生的偏见和歧视等不安全因素，北京大学团队首次提出了带有约束的价值对齐技术 CVA（Constrained Value Alignment）。该技术通过对标注信息进行细粒度划分，并结合带约束的安全强化学习方法，显著降低了模型的偏见和歧视，提高了模型的安全性。Beaver使用GPT4进行Evaluation，结果表明，在原有性能保持不变的情况下，Beaver回复的安全性大幅度提升。

### PaLM + RLHF (Pytorch)
- https://github.com/lucidrains/PaLM-rlhf-pytorch

Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the PaLM architecture. Maybe I'll add retrieval functionality too, à la RETRO

### * 【Guanaco & QloRA】
- https://mp.weixin.qq.com/s/SGJQHsEJTNB6hiVqdc87sg
- https://arxiv.org/abs/2305.14314
- https://github.com/artidoro/qlora

We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. We release all of our models and code, including CUDA kernels for 4-bit training.


### RL4LMs
- https://github.com/allenai/RL4LMs
- https://rl4lms.apps.allenai.org/

A modular RL library to fine-tune language models to human preferences

We provide easily customizable building blocks for training language models including implementations of on-policy algorithms, reward functions, metrics, datasets and LM based actor-critic policies

### Reinforcement Learning with Language Model
- https://github.com/HarderThenHarder/transformers_tasks/tree/main/RLHF

在这个项目中，我们将通过开源项目 trl 搭建一个通过强化学习算法（PPO）来更新语言模型（GPT-2）的几个示例，包括：
- 基于中文情感识别模型的正向评论生成机器人（No Human Reward）
- 基于人工打分的正向评论生成机器人（With Human Reward）
- 基于排序序列（Rank List）训练一个奖励模型（Reward Model）
- 排序序列（Rank List）标注平台

### StableLM
- https://zhuanlan.zhihu.com/p/623542189
- https://github.com/Stability-AI/StableLM

StableLM: Stability AI Language Models

This repository contains Stability AI's ongoing development of the StableLM series of language models and will be continuously updated with new checkpoints. The following provides an overview of all currently available models. More coming soon.

### StableVicuna
- https://github.com/Stability-AI/StableLM

StableVicuna基于小羊驼Vicuna-13B的进一步指令微调和RLHF训练的版本。Vicuna-13B是LLaMA-13B的一个指令微调模型。

### Stanford Alpaca
- https://crfm.stanford.edu/2023/03/13/alpaca.html
- https://alpaca-ai.ngrok.io/
- https://github.com/tatsu-lab/stanford_alpaca

Alpaca: A Strong, Replicable Instruction-Following ModelAl

We introduce Alpaca 7B, a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations. On our preliminary evaluation of single-turn instruction following, Alpaca behaves qualitatively similarly to OpenAI’s text-davinci-003, while being surprisingly small and easy/cheap to reproduce (<600$).

### Transformer Reinforcement Learning
- https://github.com/lvwerra/trl

With trl you can train transformer language models with Proximal Policy Optimization (PPO). The library is built on top of the transformers library by 🤗 Hugging Face. Therefore, pre-trained language models can be directly loaded via transformers. At this point most of decoder architectures and encoder-decoder architectures are supported.

### Transformer Reinforcement Learning X
- https://github.com/CarperAI/trlx

trlX is a distributed training framework designed from the ground up to focus on fine-tuning large language models with reinforcement learning using either a provided reward function or a reward-labeled dataset.

Training support for 🤗 Hugging Face models is provided by Accelerate-backed trainers, allowing users to fine-tune causal and T5-based language models of up to 20B parameters, such as facebook/opt-6.7b, EleutherAI/gpt-neox-20b, and google/flan-t5-xxl. For models beyond 20B parameters, trlX provides NVIDIA NeMo-backed trainers that leverage efficient parallelism techniques to scale effectively.

### Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality
- https://chat.lmsys.org/
- https://vicuna.lmsys.org/
- https://github.com/lm-sys/FastChat

An open platform for training, serving, and evaluating large language model based chatbots.

### Wombat
- https://mp.weixin.qq.com/s/xoPKmOzjlNZ2qGdcKeGARw
- https://mp.weixin.qq.com/s/UI-ij5o43ct1efYoNVdQDg
- https://arxiv.org/abs/2304.05302v1
- https://github.com/GanjinZero/RRHF

This is the repository for RRHF (Rank Response to align Human Feedback) and open-sourced language models Wombat. RRHF helps align large language models with human perference easier.

Reinforcement Learning from Human Feedback (RLHF) enables the alignment of large language models with human preference, improving the quality of interactions between humans and language models. Recent practice of RLHF uses PPO to enable the large language model optimization of such alignment. However, implementing PPO is non-trivial (where the training procedure requires interactive between policy, behavior policy, reward, value model) and it is also tedious to tuning many hyper-parameters. Our motivation is to simplify the alignment between language models with human preference, and our proposed paradigm RRHF (Rank Response from Human Feedback) can achieve such alignment as easily as conventional fine-tuning. It is simpler than PPO from the aspects of coding, model counts, and hyperparameters.

## 2 中文开源模型（Chinese Open Source Language Models）

### 华佗
- https://zhuanlan.zhihu.com/p/626536996
- https://github.com/scir-hi/huatuo-llama-med-chinese

华佗: 基于中文医学知识的LLaMa指令微调模型

在生物医学领域，LLM模型（如LLaMa，ChatGLM）因为缺乏一定的医学专业知识语料而表现不佳。该项目通过医学知识图谱和GPT3.5API构建了中文医学指令数据集，并对LLaMa模型进行了指令微调得到了一个针对医学领域的智能问诊模型HuaTuo，相比于未经过医学数据指令微调的原LLaMa而言，HuaTuo模型在智能问诊层面表现出色，可生成一些更为可靠的医学知识回答；与此同时，基于相同医学数据，该项目还训练了医疗版本的ChatGLM模型: ChatGLM-6B-Med，

除了华佗模型，该团队还即将发布扁鹊模型PienChueh(同为基于医学数据训练的大模型)，欢迎大家届时使用体验。

### 中文Alpaca模型Luotuo
- https://sota.jiqizhixin.com/project/luotuo
- https://github.com/LC1332/Luotuo-Chinese-LLM

Alpaca 是斯坦福团队基于 LLaMA 7B 在 52k 指令上微调得到的模型，能出色适应多种自然语言应用场景。近日来自商汤科技和华中科技大学开源中文语言模型 Luotuo，基于 ChatGPT API 翻译 Alpaca 微调指令数据，并使用 lora 进行微调得到。目前该项目已公开训练的语料和模型权重文件（两个型号），供开发者可使用自己各种大小的语料，训练自己的语言模型，并适用到对应的垂直领域。

### 中文LLaMA&Alpaca大模型
- https://github.com/ymcui/Chinese-LLaMA-Alpaca

以ChatGPT、GPT-4等为代表的大语言模型（Large Language Model, LLM）掀起了新一轮自然语言处理领域的研究浪潮，展现出了类通用人工智能（AGI）的能力，受到业界广泛关注。然而，由于大语言模型的训练和部署都极为昂贵，为构建透明且开放的学术研究造成了一定的阻碍。

为了促进大模型在中文NLP社区的开放研究，本项目开源了中文LLaMA模型和经过指令精调的Alpaca大模型。这些模型在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，在中文LLaMA的基础上，本项目使用了中文指令数据进行指令精调，显著提升了模型对指令的理解和执行能力。

### 中文对话式大语言模型Firefly
- https://mp.weixin.qq.com/s/tyH9Ifcvw4DKqoIoYjT6Kg
- https://github.com/yangjianxin1/Firefly

Firefly（流萤） 是一个开源的中文对话式大语言模型，使用指令微调（Instruction Tuning）在中文数据集上进行调优。同时使用了词表裁剪、ZeRO、张量并行等技术，有效降低显存消耗和提高训练效率。 在训练中，我们使用了更小的模型参数量，以及更少的计算资源。

我们构造了许多与中华文化相关的数据，以提升模型这方面的表现，如对联、作诗、文言文翻译、散文、金庸小说等。

### 凤凰
- https://mp.weixin.qq.com/s/beAAh_MdqssV8bEKsccElg
- https://github.com/FreedomIntelligence/LLMZoo

LLM Zoo is a project that provides data, models, and evaluation benchmark for large language models.

### 复旦MOSS
- https://github.com/OpenLMLab/MOSS
- https://mp.weixin.qq.com/s/LjToZVWjQ-ot5KJFCFtA3g

MOSS是一个支持中英双语和多种插件的开源对话语言模型，moss-moon系列模型具有160亿参数，在FP16精度下可在单张A100/A800或两张3090显卡运行，在INT4/8精度下可在单张3090显卡运行。MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。

### BELLE: Bloom-Enhanced Large Language model Engine
- https://huggingface.co/BelleGroup
- https://github.com/LianjiaTech/BELLE

本项目基于 Stanford Alpaca ，Stanford Alpaca 的目标是构建和开源一个基于LLaMA的模型。 Stanford Alpaca 的种子任务都是英语，收集的数据也都是英文，因此训练出来的模型未对中文优化。


本项目目标是促进中文对话大模型开源社区的发展。本项目针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。

### Bloom
- https://huggingface.co/blog/bloom
- https://huggingface.co/bigscience/bloom

BLOOM is an autoregressive Large Language Model (LLM), trained to continue text from a prompt on vast amounts of text data using industrial-scale computational resources. As such, it is able to output coherent text in 46 languages and 13 programming languages that is hardly distinguishable from text written by humans. BLOOM can also be instructed to perform text tasks it hasn't been explicitly trained for, by casting them as text generation tasks.

### BiLLa: A Bilingual LLaMA with Enhanced Reasoning Ability
- https://zhuanlan.zhihu.com/p/628688680
- https://github.com/Neutralzz/BiLLa

BiLLa是开源的推理能力增强的中英双语LLaMA模型。模型的主要特性有：
- 较大提升LLaMA的中文理解能力，并尽可能减少对原始LLaMA英文能力的损伤；
- 训练过程增加较多的任务型数据，利用ChatGPT生成解析，强化模型理解任务求解逻辑；
- 全量参数更新，追求更好的生成效果。

### BLOOMChat176B
- https://mp.weixin.qq.com/s/cY6ORD8CUyXRL0l20EjwqQ
- https://sambanova.ai/blog/introducing-bloomchat-176b-the-multilingual-chat-based-llm/
- https://huggingface.co/spaces/sambanovasystems/BLOOMChat
- https://github.com/sambanova/bloomchat

开源对话模型一直跟闭源模型在多语言能力上存在差距。SambaNova 和斯坦福 Together Computer 开源可商用的多语言聊天模型 BLOOMChat 176B，支持中文。BLOOMChat 在SambaNova 自研芯片 RDU 上完成训练，借助 SambaNova 的独特可重构数据流架构，利用 BLOOM 开源模型的核心能力，通过在 OpenChatKit、Dolly 2.0 和 OASST1 的 OIG 上进行微调。在基于六种语言的早期双盲测试中，BLOOMChat 在 66%的测评数据上产生的对话表现优于近期的开源对话模型。同时在与 GPT4 的基于六种语言的人工测评对比中，BLOOMChat 得到 45%对 55%的胜率，大大缩小开源和闭源模型的多语言对话能力差距。当前 BLOOMChat 开源模型文件，支持在 huggingface 在线推理试用。

### ChatYuan
- https://github.com/clue-ai/ChatYuan
- https://modelscope.cn/models/ClueAI/ChatYuan-large

元语功能型对话大模型, 这个模型可以用于问答、结合上下文做对话、做各种生成任务，包括创意性写作，也能回答一些像法律、新冠等领域问题。它基于PromptCLUE-large结合数亿条功能对话多轮对话数据进一步训练得到。

PromptCLUE-large在1000亿token中文语料上预训练，累计学习1.5万亿中文token，并且在数百种任务上进行Prompt任务式训练。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对多种生成任务，可以进行采样自由生成。

### ChatGLM-6B
- https://github.com/THUDM/ChatGLM-6B
- https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning

ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。更多信息请参考我们的博客。

### Chinese-Transformer-XL
- https://github.com/THUDM/Chinese-Transformer-XL

本项目提供了智源研究院"文汇" 预训练模型Chinese-Transformer-XL的预训练和文本生成代码。

### ChatMed-TCM & ChatMed-Consult
- https://github.com/michael-wzhu/ChatMed

🚀 ChatMed-Consult : 基于中文医疗在线问诊数据集ChatMed_Consult_Dataset的50w+在线问诊+ChatGPT回复作为训练集。模型主干为LlaMA-7b,融合了Chinese-LlaMA-Alpaca的LoRA权重与中文扩展词表，然后再进行基于LoRA的参数高效微调。我们将全部代码都进行了公开。我们也将部署一个在线Gradio demo, 敬请关注。

⏳ ChatMed-TCM : 大模型赋能中医药传承。这一模型的训练数据为中医药指令数据集ChatMed_TCM_Dataset。以我们开源的中医药知识图谱为基础，采用以实体为中心的自指令方法(entity-centric self-instruct)，调用ChatGPT得到2.6w+的围绕中医药的指令数据。ChatMed-TCM模型也是以LlaMA为底座，采用LoRA微调得到。

### ChatGLM-Med
- https://github.com/SCIR-HI/Med-ChatGLM

基于中文医学知识的ChatGLM模型微调，本项目开源了经过中文医学指令精调/指令微调(Instruct-tuning) 的ChatGLM-6B模型。我们通过医学知识图谱和GPT3.5 API构建了中文医学指令数据集，并在此基础上对ChatGLM-6B进行了指令微调，提高了ChatGLM在医疗领域的问答效果。

### DoctorGLM
- https://github.com/xionghonglin/DoctorGLM

DoctorGLM，基于 ChatGLM-6B的中文问诊模型。

### EVA: 大规模中文开放域对话系统
- https://github.com/thu-coai/EVA

EVA 是目前最大的开源中文预训练对话模型，拥有28亿参数，主要擅长开放域闲聊，目前有 1.0 和 2.0 两个版本。其中，1.0版本在 WudaoCorpus-Dialog 上训练而成，2.0 版本在从 WudaoCorpus-Dialog 中清洗出的更高质量的对话数据上训练而成，模型性能也明显好于 EVA1.0。

### GPT2 for Multiple Language
- https://github.com/imcaspar/gpt2-ml

- 简化整理 GPT2 训练代码（based on Grover, supporting TPUs）
- 移植 bert tokenizer，添加多语言支持
- 15亿参数 GPT2 中文预训练模型( 15G 语料，训练 10w 步 )
- 开箱即用的模型生成效果 demo #
- 15亿参数 GPT2 中文预训练模型( 30G 语料，训练 22w 步 )

### LawGPT_zh 中文法律大模型（獬豸）
- https://mp.weixin.qq.com/s/Pk4NdFQq5G6iZ3QmcyyFUg
- https://github.com/LiuHC0428/LAW-GPT

我们的愿景是为让所有人在遇到法律问题时能第一时间获得专业可靠的回答。因为专业的律师服务只有真正触手可及，才会让人们习惯运用，一如二十年前的搜索引擎，十年前的快递业务。我们希望让法律走进日常生活，为构建法治社会贡献我们的力量。项目海报由Midjourney生成。

本项目开源的中文法律通用模型由ChatGLM-6B LoRA 16-bit指令微调得到。数据集包括现有的法律问答数据集和基于法条和真实案例指导的self-Instruct构建的高质量法律文本问答，提高了通用语言大模型在法律领域的表现，提高了模型回答的可靠性和专业程度。

### Linly伶荔说
- https://github.com/CVI-SZU/Linly
- https://mp.weixin.qq.com/s/zSxsArP1pxYNubNDZua7iA

“伶荔说”模型具有以下优势：1. 在32*A100 GPU上训练了不同量级和功能的中文模型，对模型充分训练并提供强大的baseline。据我们所知33B的Linly-Chinese-LLAMA是目前最大的中文LLaMA模型。2. 公开所有训练数据、代码、参数细节以及实验结果，确保项目的可复现性，用户可以选择合适的资源直接用于自己的流程中。3. 项目具有高兼容性和易用性，提供可用于CUDA和CPU的量化推理框架，并支持Huggingface格式。

目前公开可用的模型有：

Linly-Chinese-LLaMA：中文基础模型，基于LLaMA在高质量中文语料上增量训练强化中文语言能力，现已开放 7B、13B 和 33B 量级，65B正在训练中。

Linly-ChatFlow：中文对话模型，在400万指令数据集合上对中文基础模型指令精调，现已开放7B、13B对话模型。

Linly-ChatFlow-int4 ：ChatFlow 4-bit量化版本，用于在CPU上部署模型推理。

进行中的项目：
Linly-Chinese-BLOOM：基于BLOOM中文增量训练的中文基础模型，包含7B和175B模型量级，可用于商业场景。

### MedicalGPT-zh
- github.com/MediaBrain-SJTU/MedicalGPT-zh

该开源了基于ChatGLM-6B LoRA 16-bit指令微调的中文医疗通用模型。基于共计28科室的中文医疗共识与临床指南文本，我们生成医疗知识覆盖面更全，回答内容更加精准的高质量指令数据集。

### PromptCLUE
- https://github.com/clue-ai/PromptCLUE

PromptCLUE：大规模多任务Prompt预训练中文开源模型。

中文上的三大统一：统一模型框架，统一任务形式，统一应用方式。

支持几十个不同类型的任务，具有较好的零样本学习能力和少样本学习能力。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对生成任务，可以进行采样自由生成。

千亿中文token上大规模预训练，累计学习1.5万亿中文token，亿级中文任务数据上完成训练，训练任务超过150+。比base版平均任务提升7个点+；具有更好的理解、生成和抽取能力，并且支持文本改写、纠错、知识图谱问答。

### SkyText-Chinese-GPT3
- https://github.com/SkyWorkAIGC/SkyText-Chinese-GPT3

SkyText是由奇点智源发布的中文GPT3预训练大模型，可以进行聊天、问答、中英互译等不同的任务。 应用这个模型，除了可以实现基本的聊天、对话、你问我答外，还能支持中英文互译、内容续写、对对联、写古诗、生成菜谱、第三人称转述、创建采访问题等多种功能。

## 3 其他小伙伴的资料
### 总结开源可用的Instruct/Prompt Tuning数据
- https://zhuanlan.zhihu.com/p/615277009

### 总结当下可用的大模型LLMs
- https://zhuanlan.zhihu.com/p/611403556

### 针对聊天对话数据摘要生成任务微调 FLAN-T5
- https://www.philschmid.de/fine-tune-flan-t5

### 使用 DeepSpeed 和 Hugging Face 🤗 Transformer 微调 FLAN-T5 XL/XXL
- https://zhuanlan.zhihu.com/p/615528315

### ChatGPT等大模型高效调参大法——PEFT库的算法简介
- https://zhuanlan.zhihu.com/p/613863520

### Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU
- https://huggingface.co/blog/trl-peft

### 可以微调类ChatGPT模型啦！开源Alpaca-LoRA+RTX 4090就能搞定
- https://mp.weixin.qq.com/s/vzIm-fOxxPEU69ArAowoIg

### 0门槛克隆ChatGPT！30分钟训完，60亿参数性能堪比GPT-3.5
- https://mp.weixin.qq.com/s/RMrXIHGOy3cPu8ybQNWonA

### 训练个中文版ChatGPT没那么难：不用A100，开源Alpaca-LoRA+RTX 4090就能搞定
- https://mp.weixin.qq.com/s/k7T-vfoH3xvxl6uqImP7DQ

### GPT fine-tune实战： 训练我自己的 ChatGPT
- https://zhuanlan.zhihu.com/p/616504594

### 笔记本就能运行的ChatGPT平替来了，附完整版技术报告
- https://mp.weixin.qq.com/s/crpG4dtfQFe3Q7hR3oeyxQ

### 【官方教程】ChatGLM-6B微调，最低只需7GB显存
- https://mp.weixin.qq.com/s/miML4PXioK5iM8UI0cTSCQ

### 特制自己的ChatGPT：多接口统一的轻量级LLM-IFT平台
- https://mp.weixin.qq.com/s/Q5Q3RpQ80XmpbfhSxq2R1Q

### ChatDoctor：基于LLaMA在医学领域知识上微调的医学对话模型
- https://mp.weixin.qq.com/s/-IqECOgCs4cS6Ya-EccXOA
- https://github.com/Kent0n-Li/ChatDoctor

### 也谈ChatGPT的低成本“平替”当下实现路线：语言模型+指令微调数据+微调加速架构下的代表项目和开放数据
- https://mp.weixin.qq.com/s/CJ4cCjti5jHOpDZqd42stw

### StackLLaMA: A hands-on guide to train LLaMA with RLHF
- https://huggingface.co/blog/stackllama

### 成本不到100美元！UC伯克利再开源类ChatGPT模型「考拉」：数据量大没有用，高质量才是王道
- https://zhuanlan.zhihu.com/p/621078208

### NLP大模型必备-FudanNLP开源中文图书集合CBook-150K
- https://mp.weixin.qq.com/s/X2SmjkALVVOE5hOrizcqqw
- https://github.com/FudanNLPLAB/CBook-150K
- http://www.doc-ai.cn/

### COIG：首个大规模、可商用的中文开源指令数据！
- https://mp.weixin.qq.com/s/1hSU5AROH0ZGuDo9oD0bFw
- https://huggingface.co/datasets/BAAI/COIG

### 以竞赛为例--GPT/BART/CPT的预训练和微调全流程
- https://mp.weixin.qq.com/s/fNb9tmEXLUtDoWKibNFLEQ

### 生成式专利语言模型(PatentGPT)评估
- https://mp.weixin.qq.com/s/hnmH8AzQupIZH1lWX2ZSNw

### 极低资源微调大模型方法LoRA以及BLOOM-LORA实现代码
- https://zhuanlan.zhihu.com/p/625488835

### “超越”(MMCU)中文通用大语言模型测试集--国内首个多领域多任务数据集
- https://mp.weixin.qq.com/s/sZqqK51PamKHOz3DFcA_4A

数据集的测试内容涵盖四大领域：医疗、法律、心理学和教育。通过综合评估模型在多个学科上的知识广度和深度，能够帮助研究者更精准地找出模型的缺陷，并对模型的能力进行打分。

### CCKS2023-PromptCBLUE中文医疗大模型评测比赛
- https://mp.weixin.qq.com/s/LjOiZ_S7oLJBvqdKotA9zA

为推动LLM在医疗领域的发展和落地，华东师范大学计算机学院王晓玲教授团队联合阿里巴巴天池平台、复旦大学、复旦大学附属华山医院、东北大学、哈尔滨工业大学（深圳）、鹏城实验室与同济大学推出PromptCBLUE评测基准(https://github.com/michael-wzhu/PromptCBLUE)，对CBLUE基准(https://tianchi.aliyun.com/dataset/95414)进行二次开发，将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务，形成首个中文医疗场景的LLM评测基准。PromptCBLUE将作为CCKS-2023的评测任务之一，已在阿里巴巴天池大赛平台上线进行开放评测，欢迎各位师生报名参赛(刷榜)。

### 也看垂直领域大模型微调落地-以医疗领域为例：从PMC-LLaMA增量预训到MedicalGPT-zh指令微调项目概述
- https://mp.weixin.qq.com/s/Pk4NdFQq5G6iZ3QmcyyFUg

### HuggingFace宣布在transformers库中引入首个RNN模型：RWKV，一个结合了RNN与Transformer双重优点的模型
- https://zhuanlan.zhihu.com/p/629637598

### LLM评价模型PandaLM技术前瞻
- https://zhuanlan.zhihu.com/p/630173415
- https://github.com/WeOpenML/PandaLM

### 小数据也能助力大发现！CancerGPT成功预测药物组合，惊人数字证明其准确性！
- https://mp.weixin.qq.com/s/xswnXhnLOkVOQwfKNFdPQA

### 逐步蒸馏！用更少的数据，训练更小的模型：性能却堪比大2000倍的模型
- https://mp.weixin.qq.com/s/dtKaeSO4hZPGOuPcHRmBQw

### 国内首个可复现的 RLHF 基准，北大团队开源PKU-Beaver | 料见闭门交流
- https://github.com/PKU-Alignment/safe-rlhf
- https://mp.weixin.qq.com/s/ZpkgszXbisl5xf63EfTNjQ

### Meta AI 重磅推出LIMA！媲美GPT-4、无需RLHF就能对齐！
- https://zhuanlan.zhihu.com/p/631508237

### 逼近GPT-4！BLOOMChat: 开源可商用支持多语言的大语言模型
- https://zhuanlan.zhihu.com/p/631036519

### 手把手复现一个ChatGPT
- https://zhuanlan.zhihu.com/p/631690198

### 关于hippocratic.ai和glass.health的产品讨论
- https://mp.weixin.qq.com/s/yl_aPKg74yHKNdfPhGss5g

### 越小越好: Q8-Chat，在英特尔至强 CPU 上体验高效的生成式 AI
- https://mp.weixin.qq.com/s/O55qgGeD5lDKl9tGVmBN3g

### * 开源原驼（Guanaco）及背后的QLoRA技术，将微调65B模型的显存需求从780GB以上降低到48GB以下，效果直逼GPT-4，技术详解
- https://zhuanlan.zhihu.com/p/632236718

### * 使用LoRA对BELLE发布的BELLE-7B-2M进行微调
- https://zhuanlan.zhihu.com/p/632317500

### * 【LLM系列之Tokenizer】如何科学地训练一个LLM分词器
- https://mp.weixin.qq.com/s/z6wUY1p8_AVv8YEQ6FRYIA

### * 金融领域大模型效果，低成本，Just-in-Time，场景落地
- https://mp.weixin.qq.com/s/5Nm1I10eLi0xhNIxqyEOMA

### * 首个大规模使用工具的大模型来了：伯克利发布Gorilla
- https://mp.weixin.qq.com/s/p9tx3q3Lpr4fNqdyxWhzyA

### * NBCE：使用朴素贝叶斯扩展LLM的Context处理长度
- https://kexue.fm/archives/9617

> 持续更新中 (Continuously Updated)... 

