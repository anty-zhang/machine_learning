

# TEXT生成模型

## Text to iamge

Dalle-2: open AI，基于transformer建模，采用对比学习训练。

Imagen: Google，基于 Transformer 模型搭建，其中语言模型在纯文本数据集上进行了预训练。Imagen 增加了语言模型参数量，发现效果比提升扩散模型参数量更好。

Stable Diffusion: 由慕尼黑大学的 CompVis 小组开发，基于潜在扩散模型打造。

Muse: 由谷歌开发，基于 Transformer 模型取得了比扩散模型更好的结果，只有 900M 参数，但在推理时间上比 Stable Diffusion1.4 版本快 3 倍，比 Imagen-3B 和 Parti-3B 快 10 倍。


## Text to 3D

DreamFusion: 由谷歌和 UC 伯克利开发，基于预训练文本-2D 图像扩散模型实现文本生成 3D 模型。采用类似 NeRF 的三维场景参数化定义映射，无需任何 3D 数据或修改扩散模型，就能实现文本生成 3D 图像的效果。

Magic3D: 由英伟达开发，旨在缩短 DreamFusion 图像生成时间、同时提升生成质量。

## Text to video

Phenaki: 由谷歌打造，基于新的编解码器架构 C-ViViT 将视频压缩为离散嵌入，能够在时空两个维度上压缩视频，在时间上保持自回归的同时，还能自回归生成任意长度的视频。

Soundify 是 Runway 开发的一个系统，目的是将声音效果与视频进行匹配，即制作音效

## text to audio

AudioLM 由谷歌开发，将输入音频映射到一系列离散标记中，并将音频生成转换成语言建模任务，学会基于提示词产生自然连贯的音色。

Jukebox 由 OpenAI 开发的音乐模型，可生成带有唱词的音乐

Whisper 由 OpenAI 开发，实现了多语言语音识别、翻译和语言识别，目前模型已经开源并可以用 pip 安装

## text to text

ChatGPT 由 OpenAI 生成，是一个对话生成 AI

LaMDA Google，基于 Transformer 打造，利用了其在文本中呈现的长程依赖关系能力。

PEER 由 Meta AI 打造，基于维基百科编辑历史进行训练，直到模型掌握完整的写作流程。

Speech from Brain 由 Meta AI 打造，用于帮助无法通过语音、打字或手势进行交流的人，通过对比学习训练 wave2vec 2.0 自监督模型，基于非侵入式脑机接口发出的脑电波进行解读，并解码大脑生成的内容，从而合成对应语音。


## text to code

Codex 是 OpenAI 打造的编程模型，基于 GPT-3 微调，可以基于文本需求生成代码。

AlphaCode 由 DeepMind 打造，基于 Transformer 模型打造，通过采用 GitHub 中 715.1GB 的代码进行预训练，并从 Codeforces 中引入一个数据集进行微调，随后基于 Codecontests 数据集进行模型验证，并进一步改善了模型输出性能。

## text to science

Galatica 是 Meta AI 推出的 1200 亿参数论文写作辅助模型，又被称之为“写论文的 Copilot 模型”

Minerva 由谷歌开发，目的是通过逐步推理解决数学定量问题，可以主动生成相关公式。

## image to text

Flamingo: 是 DeepMind 推出的小样本学习模型，基于可以分析视觉场景的视觉模型和执行基本推理的大语言模型打造，其中大语言模型基于文本数据集训练。

VisualGPT: 是 OpenAI 制作的图像-文本模型，基于预训练 GPT-2 提出了一种新的注意力机制，来衔接不同模态之间的语义差异，无需大量图像-文本数据训练，就能提升文本生成效率。


[ChatGPT is not all you need. A State of the Art Review of large Generative AI models](https://arxiv.org/abs/2301.04655)

[2022 生成模型进展有多快，新论文盘点 9 类生成模型代表作](https://www.ithome.com/0/669/955.htm)
