![](image_p1_0.png)

EXPERT INSIGHT In color

4178 44

LLM

29483

Handbook

Master the art of engineering large

language models from concept to production

Forewords by Julien Chaumond

Co-founder and CTO, Hugging Face

Hamza Tahir

Co-founder and CTO, ZenML pa ckb

Paul lusztin | Maxime Labonne

-----

# LLM Engineer's Handbook

##### Master the art of engineering large language models from concept to production

Paul Iusztin Maxime Labonne

-----

##### LLM Engineer's Handbook

Copyright © 2024 Packt Publishing

*All rights reserved. No part of this book may be reproduced, stored in a retrieval system, or transmitted in* any form or by any means, without the prior written permission of the publisher, except in the case of brief quotations embedded in critical articles or reviews. Every effort has been made in the preparation of this book to ensure the accuracy of the information presented. However, the information contained in this book is sold without warranty, either express or implied. Neither the authors, nor Packt Publishing or its dealers and distributors, will be held liable for any damages caused or alleged to have been caused directly or indirectly by this book. Packt Publishing has endeavored to provide trademark information about all of the companies and products mentioned in this book by the appropriate use of capitals. However, Packt Publishing cannot guarantee the accuracy of this information.

###### Senior Publishing Product Manager: Gebin George Acquisition Editor - Peer Reviews: Swaroop Singh Project Editor: Amisha Vathare

###### Content Development Editor: Tanya D'cruz Copy Editor: Safis Editing Technical Editor: Karan Sonawane Proofreader: Safis Editing

###### Indexer: Manju Arasan Presentation Designer: Rajesh Shirsath Developer Relations Marketing Executive: Anamika Singh

First published: October 2024

Production reference: 2171024

Published by Packt Publishing Ltd. Grosvenor House 11 St Paul's Square Birmingham B3 1RB, UK.

ISBN 978-1-83620-007-9

```
www.packt.com
```

-----

# Forewords

As my co-founder at Hugging Face, Clement Delangue, and I often say, AI is becoming the default way of building technology. Over the past 3 years, LLMs have already had a profound impact on technology, and they are bound to have an even greater impact in the coming 5 years. They will be embedded in more and more products and, I believe, at the center of any human activity based on knowledge or creativity. For instance, coders are already leveraging LLMs and changing the way they work, focusing on higher-order thinking and tasks while collaborating with machines. Studio musicians rely on AI-powered tools to explore the musical creativity space faster. Lawyers are increasing their impact through retrieval-augmented generation (RAG) and large databases of case law. At Hugging Face, we've always advocated for a future where not just one company or a small number of scientists control the AI models used by the rest of the population, but instead for a future where as many people as possible-from as many different backgrounds as possible-are capable of diving into how cutting-edge machine learning models actually work. Maxime Labonne and Paul Iusztin have been instrumental in this movement to democratize LLMs by writing this book and making sure that as many people as possible can not only use them but also adapt them, fine-tune them, quantize them, and make them efficient enough to actually deploy in the real world. Their work is essential, and I'm glad they are making this resource available to the community. This expands the convex hull of human knowledge. *Julien Chaumond* *Co-founder and CTO, Hugging Face*

-----

As someone deeply immersed in the world of machine learning operations, I'm thrilled to endorse The LLM Engineer's Handbook. This comprehensive guide arrives at a crucial time when the demand for LLM expertise is skyrocketing across industries. What sets this book apart is its practical, end-to-end approach. By walking readers through the creation of an LLM Twin, it bridges the often daunting gap between theory and real-world application. From data engineering and model fine-tuning to advanced topics like RAG pipelines and inference optimization, the authors leave no stone unturned. I'm particularly impressed by the emphasis on MLOps and LLMOps principles. As organizations increasingly rely on LLMs, understanding how to build scalable, reproducible, and robust systems is paramount. The inclusion of orchestration strategies and cloud integration showcases the authors' commitment to equipping readers with truly production-ready skills. Whether you're a seasoned ML practitioner looking to specialize in LLMs or a software engineer aiming to break into this exciting field, this handbook provides the perfect blend of foundational knowledge and cutting-edge techniques. The clear explanations, practical examples, and focus on best practices make it an invaluable resource for anyone serious about mastering LLM engineering. In an era where AI is reshaping industries at breakneck speed, The LLM Engineer's Handbook stands out as an essential guide for navigating the complexities of large language models. It's not just a book; it's a roadmap to becoming a proficient LLM engineer in today's AI-driven landscape. *Hamza Tahir* *Co-founder and CTO, ZenML*

-----

# Contributors

##### About the authors

**Paul Iusztin is a senior ML and MLOps engineer with over seven years of experience building**

GenAI, Computer Vision and MLOps solutions. His latest contribution was at Metaphysic, where he served as one of their core engineers in taking large neural networks to production. He previously worked at CoreAI, Everseen, and Continental. He is the Founder of Decoding ML, an educational channel on production-grade ML that provides posts, articles, and open-source courses to help others build real-world ML systems.

**Maxime Labonne is the Head of Post-Training at Liquid AI. He holds a PhD. in ML from the**

Polytechnic Institute of Paris and is recognized as a Google Developer Expert in AI/ML. As an active blogger, he has made significant contributions to the open-source community, including the LLM Course on GitHub, tools such as LLM AutoEval, and several state-of-the-art models like NeuralDaredevil. He is the author of the best-selling book Hands-On Graph Neural Networks Using *Python, published by Packt.*

*I want to thank my family and partner. Your unwavering support and patience made this book possible.*

-----

##### About the reviewer

**Rany ElHousieny is an AI solutions architect and AI engineering manager with over two decades**

of experience in AI, NLP, and ML. Throughout his career, he has focused on the development and deployment of AI models, authoring multiple articles on AI systems architecture and ethical AI deployment. He has led groundbreaking projects at companies like Microsoft, where he spearheaded advancements in NLP and the Language Understanding Intelligent Service (LUIS). Currently, he plays a pivotal role at Clearwater Analytics, driving innovation in GenAI and AI-driven financial and investment management solutions.

*I would like to thank Clearwater Analytics for providing a supportive and learning environment that fosters growth and innovation. The vision of our leaders, always staying ahead with the latest technologies, has been a constant source of inspiration. Their commitment to AI advancements made my experience of reviewing this book insightful and enriching. Special thanks to my family for their ongoing encouragement throughout this journey.*

-----

##### Join our book's Discord space

Join our community's Discord space for discussions with the authors and other readers:

```
https://packt.link/llmeng
```

-----

```text

```

-----

# Table of Contents

Preface xxi

Chapter 1: Understanding the LLM Twin Concept and Architecture 1

**Understanding the LLM Twin concept  2**

What is an LLM Twin? • 2 Why building an LLM Twin matters • 3 Why not use ChatGPT (or another similar chatbot)? • 5

**Planning the MVP of the LLM Twin product  6**

What is an MVP? • 6 Defining the LLM Twin MVP • 7

**Building ML systems with feature/training/inference pipelines  8**

The problem with building ML systems • 8 The issue with previous solutions • 10 The solution - ML pipelines for ML systems • 13

*The feature pipeline • 14* *The training pipeline • 14* *The inference pipeline • 14* Benefits of the FTI architecture • 15

**Designing the system architecture of the LLM Twin  16**

Listing the technical details of the LLM Twin architecture • 16 How to design the LLM Twin architecture using the FTI pipeline design • 17

*Data collection pipeline • 19*

-----

x *Table of Contents*

---

*Feature pipeline • 19* *Training pipeline • 21* *Inference pipeline • 22* Final thoughts on the FTI design and the LLM Twin architecture • 22

**Summary  23 References  23**

Chapter 2: Tooling and Installation 25

**Python ecosystem and project installation  26**

Poetry: dependency and virtual environment management • 27 Poe the Poet: task execution tool • 29

**MLOps and LLMOps tooling  30**

Hugging Face: model registry • 31 ZenML: orchestrator, artifacts, and metadata • 32

*Orchestrator • 33* *Artifacts and metadata • 39* *How to run and configure a ZenML pipeline • 43* Comet ML: experiment tracker • 45 Opik: prompt monitoring • 46

**Databases for storing unstructured and vector data  47**

MongoDB: NoSQL database • 47 Qdrant: vector database • 47

**Preparing for AWS  48**

Setting up an AWS account, an access key, and the CLI • 48 SageMaker: training and inference compute • 50

*Why AWS SageMaker? • 51*

**Summary  52 References  53**

Chapter 3: Data Engineering 55

**Designing the LLM Twin's data collection pipeline  56**

-----

*Table of Contents* xi

---

Implementing the LLM Twin's data collection pipeline • 61

ZenML pipeline and steps • 61 The dispatcher: How do you instantiate the right crawler? • 66 The crawlers • 69

*Base classes • 69* *GitHubCrawler class • 73* *CustomArticleCrawler class • 75* *MediumCrawler class • 77* The NoSQL data warehouse documents • 79

*The ORM and ODM software patterns • 80 Implementing the ODM class • 82 Data categories and user document classes • 87*

**Gathering raw data into the data warehouse  89**

Troubleshooting • 94

*Selenium issues • 95 Import our backed-up data • 95*

**Summary  96 References  96**

Chapter 4: RAG Feature Pipeline 99

**Understanding RAG  100**

Why use RAG? • 100

*Hallucinations • 101* *Old information • 101* The vanilla RAG framework • 101

*Ingestion pipeline • 104* *Retrieval pipeline • 105* *Generation pipeline • 105* What are embeddings? • 107

*Why embeddings are so powerful • 109*

-----

xii *Table of Contents*

---

*How are embeddings created? • 111* *Applications of embeddings • 114* More on vector DBs • 115

*How does a vector DB work? • 115 Algorithms for creating the vector index • 116 DB operations • 116*

**An overview of advanced RAG  117**

Pre-retrieval • 119 Retrieval • 122 Post-retrieval • 124

**Exploring the LLM Twin's RAG feature pipeline architecture  127**

The problem we are solving • 127 The feature store • 128 Where does the raw data come from? • 128 Designing the architecture of the RAG feature pipeline • 129

*Batch pipelines • 130 Batch versus streaming pipelines • 130 Core steps • 134 Change data capture: syncing the data warehouse and feature store • 136 Why is the data stored in two snapshots? • 138 Orchestration • 138*

**Implementing the LLM Twin's RAG feature pipeline  139**

Settings • 139 ZenML pipeline and steps • 140

*Querying the data warehouse • 143* *Cleaning the documents • 146* *Chunk and embed the cleaned documents • 147* *Loading the documents to the vector DB • 150* Pydantic domain entities • 150

*OVM • 154* The dispatcher layer • 160

-----

*Table of Contents* xiii

---

The handlers • 162

*The cleaning handlers • 163 The chunking handlers • 165 The embedding handlers • 169*

**Summary  173 References  174**

Chapter 5: Supervised Fine-Tuning 177

**Creating an instruction dataset  178**

General framework • 178

*Data quantity • 180* Data curation • 182 Rule-based filtering • 182 Data deduplication • 184 Data decontamination • 185 Data quality evaluation • 186 Data exploration • 189 Data generation • 191 Data augmentation • 193

**Creating our own instruction dataset  196 Exploring SFT and its techniques  206**

When to fine-tune • 206 Instruction dataset formats • 208 Chat templates • 208 Parameter-efficient fine-tuning techniques • 211

*Full fine-tuning • 211* *LoRA • 213* *QLoRA • 215* Training parameters • 216

*Learning rate and scheduler • 216 Batch size • 216*

-----

xiv *Table of Contents*

---

*Maximum length and packing • 217 Number of epochs • 218 Optimizers • 218 Weight decay • 219 Gradient checkpointing • 219*

**Fine-tuning in practice  219 Summary  226 References  227**

Chapter 6: Fine-Tuning with Preference Alignment 229

**Understanding preference datasets  230**

Preference data • 230

*Data quantity • 232* Data generation and evaluation • 233

*Generating preferences • 233 Tips for data generation • 234 Evaluating preferences • 235*

**Creating our own preference dataset  237 Preference alignment  245**

Reinforcement Learning from Human Feedback • 246 Direct Preference Optimization • 248

**Implementing DPO  250 Summary  257 References  258**

Chapter 7: Evaluating LLMs 261

**Model evaluation  261**

Comparing ML and LLM evaluation • 262 General-purpose LLM evaluations • 263 Domain-specific LLM evaluations • 265 Task-specific LLM evaluations • 267

-----

*Table of Contents* xv

---

**RAG evaluation  271**

Ragas • 272 ARES • 274

**Evaluating TwinLlama-31-8B  275**

Generating answers • 276 Evaluating answers • 278 Analyzing results • 283

**Summary  286 References  287**

Chapter 8: Inference Optimization 289

**Model optimization strategies  290**

KV cache • 291 Continuous batching • 294 Speculative decoding • 295 Optimized attention mechanisms • 297

**Model parallelism  298**

Data parallelism • 299 Pipeline parallelism • 300 Tensor parallelism • 301 Combining approaches • 303

**Model quantization  303**

Introduction to quantization • 304 Quantization with GGUF and llama.cpp • 309 Quantization with GPTQ and EXL2 • 311 Other quantization techniques • 313

**Summary  314 References  315**

Chapter 9: RAG Inference Pipeline 317

**Understanding the LLM Twin's RAG inference pipeline  318**

-----

xvi *Table of Contents*

---

**Exploring the LLM Twin's advanced RAG techniques  321**

Advanced RAG pre-retrieval optimizations: query expansion and self-querying • 324

*Query expansion • 324* *Self-querying • 328* Advanced RAG retrieval optimization: filtered vector search • 332 Advanced RAG post-retrieval optimization: reranking • 334

**Implementing the LLM Twin's RAG inference pipeline  338**

Implementing the retrieval module • 339 Bringing everything together into the RAG inference pipeline • 346

**Summary  351 References  351**

Chapter 10: Inference Pipeline Deployment 355

**Criteria for choosing deployment types  356**

Throughput and latency • 356 Data • 357

**Understanding inference deployment types  359**

Online real-time inference • 360 Asynchronous inference • 361 Offline batch transform • 362

**Monolithic versus microservices architecture in model serving  363**

Monolithic architecture • 365 Microservices architecture • 365 Choosing between monolithic and microservices architectures • 367

**Exploring the LLM Twin's inference pipeline deployment strategy  368**

The training versus the inference pipeline • 371

**Deploying the LLM Twin service  372**

Implementing the LLM microservice using AWS SageMaker • 373

*What are Hugging Face's DLCs? • 373 Configuring SageMaker roles • 374*

-----

*Table of Contents* xvii

---

*Deploying the LLM Twin model to AWS SageMaker • 375* *Calling the AWS SageMaker Inference endpoint • 386* Building the business microservice using FastAPI • 390

**Autoscaling capabilities to handle spikes in usage  393**

Registering a scalable target • 396 Creating a scalable policy • 397 Minimum and maximum scaling limits • 398

*Cooldown period • 398*

**Summary  399 References  400**

Chapter 11: MLOps and LLMOps 401

**The path to LLMOps: Understanding its roots in DevOps and MLOps  402**

DevOps • 403

*The DevOps lifecycle • 403* *The core DevOps concepts • 404* MLOps • 405

*MLOps core components • 407* *MLOps principles • 407* *ML vs. MLOps engineering • 409* LLMOps • 410

*Human feedback • 411 Guardrails • 411 Prompt monitoring • 413*

**Deploying the LLM Twin's pipelines to the cloud  415**

Understanding the infrastructure • 416 Setting up MongoDB • 418 Setting up Qdrant • 419 Setting up the ZenML cloud • 421

*Containerize the code using Docker • 424*

-----

xviii *Table of Contents*

---

*Run the pipelines on AWS • 428 Troubleshooting the ResourceLimitExceeded error after running a ZenML pipeline on SageMaker • 432*

**Adding LLMOps to the LLM Twin  434**

LLM Twin's CI/CD pipeline flow • 434

*More on formatting errors • 436* *More on linting errors • 436* Quick overview of GitHub Actions • 437 The CI pipeline • 438

*GitHub Actions CI YAML file • 438* The CD pipeline • 442 Test out the CI/CD pipeline • 445 The CT pipeline • 446

*Initial triggers • 448* *Trigger downstream pipelines • 449* Prompt monitoring • 451 Alerting • 457

**Summary  458 References  459**

Appendix: MLOps Principles 461

**1 Automation or operationalization  461 2 Versioning  463 3 Experiment tracking  464 4 Testing  464**

Test types • 464 What do we test? • 465 Test examples • 465

**5 Monitoring  468**

Logs • 468 Metrics • 468

-----

*Table of Contents* xix

---

*System metrics • 469 Model metrics • 469 Drifts • 469 Monitoring vs. observability • 472 Alerts • 473*

**6 Reproducibility  473**

Other Books You May Enjoy 477

Index 481

-----

```text

```

-----

# Preface

The field of LLM engineering has rapidly emerged as a critical area in artificial intelligence and machine learning. As LLMs continue to revolutionize natural language processing and generation, the demand for professionals who can effectively implement, optimize, and deploy these models in real-world scenarios has grown exponentially. LLM engineering encompasses a wide range of disciplines, from data preparation and model fine-tuning to inference optimization and production deployment, requiring a unique blend of software engineering, machine learning expertise, and domain knowledge.

**Machine Learning Operations (MLOps) plays a crucial role in the successful implementation of**

LLMs in production environments. MLOps extends the principles of DevOps to machine learning projects, focusing on automating and streamlining the entire ML lifecycle. For LLMs, MLOps is particularly important due to the complexity and scale of these models. It addresses challenges such as managing large datasets, handling model versioning, ensuring reproducibility, and maintaining model performance over time. By incorporating MLOps practices, LLM projects can achieve greater efficiency, reliability, and scalability, ultimately leading to more successful and impactful deployments.

**The LLM Engineer's Handbook is a comprehensive guide to applying best practices to the new**

field of LLM engineering. Throughout the chapters, readers will find simplified key concepts, practical techniques, and experts tips for every stage of the LLM lifecycle. The book covers topics such as data engineering, supervised fine-tuning, model evaluation, inference optimization, and

**Retrieval-Augmented Generation (RAG) pipeline development.**

To illustrate these concepts in action, an end-to-end project called the LLM Twin will be developed throughout the book., with the goal of imitating someone's writing style and personality. This use case will demonstrate how to build a minimum viable product to solve a specific problem, using various aspects of LLM engineering and MLOps.

-----

xxii *Preface*

---

Readers can expect to gain a deeper understanding of how to collect and prepare data for LLMs, fine-tune models for specific tasks, optimize inference performance, and implement RAG pipelines. They will learn how to evaluate LLM performance, align models with human preferences, and deploy LLM-based applications. The book also covers essential MLOps principles and practices, enabling readers to build scalable, reproducible, and robust LLM applications.

##### Who this book is for

This book is intended for a wide range of technology professionals and enthusiasts interested in the practical applications of LLMs. It's ideal for software engineers aiming to transition into AI projects. While some familiarity with software development is beneficial, the book explains many concepts from the ground up, making it accessible even to those who are new to AI and machine learning. For those already working with machine learning , this book will enhance your skills in implementing and deploying LLM-based systems. We provide a deep dive into the fundamentals of MLOps, guiding you through the process of creating a minimum viable product using an opensource LLM to solve real-world problems.

##### What this book covers

*Chapter 1, Understanding the LLM Twin Concept and Architecture, introduces the LLM Twin project,* which is used throughout the book as an end-to-end example of a production-level LLM application, and defines the FTI architecture for building scalable ML systems and applies it to the LLM Twin use case. *Chapter 2, Tooling and Installation, presents Python, MLOps, and cloud tools used to build re-* al-world LLM applications, such as an orchestrator, experiment tracker, prompt monitoring and LLM evaluation tool. It shows how to use and install them locally for testing and development. *Chapter 3, Data Engineering, shows the implementation of a data collection pipeline that scrapes* multiple sites, such as Medium, GitHub and Substack and stores the raw data in a data warehouse. It emphasizes collecting raw data from dynamic sources over static datasets for real-world ML applications. *Chapter 4, RAG Feature Pipeline, introduces RAG fundamental concepts, such as embeddings, the* vanilla RAG framework, vector databases, and how to optimize RAG applications. It applies the RAG theory by architecting and implementing LLM Twin's RAG feature pipeline using software best practices.

-----

*Preface* xxiii

---

*Chapter 5, Supervised Fine-Tuning, explores the process of refining pre-trained language models* for specific tasks using instruction-answer pairs. It covers creating high-quality datasets, implementing fine-tuning techniques like full fine-tuning, LoRA, and QLoRA, and provides a practical demonstration of fine-tuning a Llama 3.1 8B model on a custom dataset. *Chapter 6, Fine-Tuning with Preference Alignment, introduces techniques for aligning language* models with human preferences, focusing on Direct Preference Optimization (DPO). It covers creating custom preference datasets, implementing DPO, and provides a practical demonstration of aligning the TwinLlama-3.1-8B model using the Unsloth library. *Chapter 7, Evaluating LLMs, details various methods for assessing the performance of language* models and LLM systems. It introduces general-purpose and domain-specific evaluations and discusses popular benchmarks. The chapter includes a practical evaluation of the TwinLlama-3.1-8B model using multiple criteria. *Chapter 8, Inference Optimization, covers key optimization strategies such as speculative decoding,* model parallelism, and weight quantization. It discusses how to improve inference speed, reduce latency, and minimize memory usage, introducing popular inference engines and comparing their features. *Chapter 9, RAG Inference Pipeline, explores advanced RAG techniques by implementing methods* such as self-query, reranking, and filtered vector search from scratch. It covers designing and implementing the LLM Twin's RAG inference pipeline and a custom retrieval module similar to what you see in popular frameworks such as LangChain. *Chapter 10, Inference Pipeline Deployment, introduces ML deployment strategies, such as online,* asynchronous and batch inference, which will help in architecting and deploying the LLM Twin fine-tuned model to AWS SageMaker and building a FastAPI microservice to expose the RAG inference pipeline as a RESTful API. *Chapter 11, MLOps and LLMOps, presents what LLMOps is, starting with its roots in DevOps and* MLOps. This chapter explains how to deploy the LLM Twin project to the cloud, such as the ML pipelines to AWS and shows how to containerize the code using Docker and build a CI/CD/CT pipeline. It also adds a prompt monitoring layer on top of LLM Twin's inference pipeline. *Appendix, MLOps Principles, covers the six MLOps principles used to build scalable, reproducible,* and robust ML applications.

-----

xxiv *Preface*

---

##### To get the most out of this book

To maximize your learning experience, you are expected to have, at the very least, a foundational understanding of software development principles and practices. Familiarity with Python programming is particularly beneficial, as the book's examples and code snippets are predominantly in Python. While prior experience with machine learning concepts is advantageous, it is not strictly necessary, as the book provides explanations for many fundamental AI and ML concepts. However, you should be comfortable with basic data structures, algorithms, and have some experience working with APIs and cloud services. Familiarity with version control systems like Git is assumed, as this book has a GitHub repository for code examples. While this book is designed to be accessible to those who are new to AI and LLMs, if you have some background in these areas, you will find it easier to grasp the more advanced concepts and techniques we present.

###### Download the example code files

The code bundle for the book is hosted on GitHub at `https://github.com/PacktPublishing/`

```
LLM-Engineers-Handbook. We also have other code bundles from our rich catalog of books and
```

videos available at `https://github.com/PacktPublishing/` . Check them out!

###### Download the color images

We also provide a PDF file that has color images of the screenshots/diagrams used in this book.

```
You can download it here: https://packt .link/gbp/9781836200079.
```

###### Conventions used

There are a number of text conventions used throughout this book.

```
CodeInText: Indicates code words in text, database table names, folder names, filenames, file
```

extensions, pathnames, dummy URLs, user input, and Twitter handles. For example: "In the `format\_samples` function, we apply the Alpaca chat template to each individual message." A block of code is set as follows:

```python
def format_samples(example):
example["prompt"] = alpaca_template.format(example["prompt"])
example["chosen"] = example['chosen'] + EOS_TOKEN
example["rejected"] = example['rejected'] + EOS_TOKEN
return {"prompt": example["prompt"], "chosen": example["chosen"],
"rejected": example["rejected"]}
```

-----

*Preface* xxv

---

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

| Any command-line input or output is written as follows: | def "rejected": Any command-line input or output is written as follows: | def format_samples(example): example["prompt"] example["chosen"] example["rejected"] return {"prompt": "rejected": Any command-line input or output is written as follows: | format_samples(example): example["prompt"] = alpaca_template.format(example["prompt"]) example["chosen"] = example['chosen'] + EOS_TOKEN example["rejected"] = example['rejected'] + EOS_TOKEN return {"prompt": example["prompt"], "chosen": example["chosen"], "rejected": example["rejected"]} Any command-line input or output is written as follows: |
|---|---|---|---|
|  | poetry | install | --without aws |
| Bold: words tings click | Indicates in tab on | a menus at the | new term, an important word, or words that you see on the screen. For instance, or dialog boxes appear in the text like this. For example: "To do so, go to the Set- the top of the forked repository in GitHub. In the left panel, in the Security section, Secrets and Variables toggle and, finally, click on Actions." |
|  |  | & | Warnings or important notes appear like this. |
|  |  |  | Tips and tricks appear like this. |

##### Get in touch

Feedback from our readers is always welcome. **General feedback: Email** `feedback@packtpub.com` and mention the book's title in the subject of your message. If you have questions about any aspect of this book, please email us at `questions@`

```
packtpub .com.
```

**Errata: Although we have taken every care to ensure the accuracy of our content, mistakes do**

happen. If you have found a mistake in this book, we would be grateful if you reported this to us.

```
Please visit http://www.packtpub .com/submit-errata, click Submit Errata, and fill in the form.
```

-----

xxvi *Preface*

---

**Piracy: If you come across any illegal copies of our works in any form on the internet, we would**

be grateful if you would provide us with the location address or website name. Please contact us at `copyright@packtpub.com` with a link to the material.

**If you are interested in becoming an author: If there is a topic that you have expertise in and you**

are interested in either writing or contributing to a book, please visit `http://authors.packtpub.`

```
com.
```

##### Share your thoughts

Once you've read LLM Engineer's Handbook, First Edition, we'd love to hear your thoughts! Please

```
click here to go straight to the Amazon review page for this book and share your feedback.
```

Your review is important to us and the tech community and will help us make sure we're delivering excellent quality content.

-----

##### Download a free PDF copy of this book

Thanks for purchasing this book! Do you like to read on the go but are unable to carry your print books everywhere? Is your eBook purchase not compatible with the device of your choice? Don't worry, now with every Packt book you get a DRM-free PDF version of that book at no cost. Read anywhere, any place, on any device. Search, copy, and paste code from your favorite technical books directly into your application. The perks don't stop there, you can get exclusive access to discounts, newsletters, and great free content in your inbox daily. Follow these simple steps to get the benefits:

1. Scan the QR code or visit the link below:

![](image_p28_0.png)

```
https://packt.link/free-ebook/9781836200079
```

2. Submit your proof of purchase.
3. That's it! We'll send your free PDF and other benefits to your email directly.

-----

```text

```

-----

1

# Understanding the LLM Twin Concept and Architecture

By the end of this book, we will have walked you through the journey of building an end-to-end

**large language model (LLM) product. We firmly believe that the best way to learn about LLMs**

and production machine learning (ML) is to get your hands dirty and build systems. This book will show you how to build an LLM Twin, an AI character that learns to write like a particular person by incorporating its style, voice, and personality into an LLM. Using this example, we will walk you through the complete ML life cycle, from data gathering to deployment and monitoring. Most of the concepts learned while implementing your LLM Twin can be applied in other LLMbased or ML applications. When starting to implement a new product, from an engineering point of view, there are three planning steps we must go through before we start building. First, it is critical to understand the problem we are trying to solve and what we want to build. In our case, what exactly is an LLM Twin, and why build it? This step is where we must dream and focus on the "Why." Secondly, to reflect a real-world scenario, we will design the first iteration of a product with minimum functionality. Here, we must clearly define the core features required to create a working and valuable product. The choices are made based on the timeline, resources, and team's knowledge. This is where we bridge the gap between dreaming and focusing on what is realistic and eventually answer the following question: "What are we going to build?". Finally, we will go through a system design step, laying out the core architecture and design choices used to build the LLM system. Note that the first two components are primarily product-related, while the last one is technical and focuses on the "How."

-----

2 *Understanding the LLM Twin Concept and Architecture*

---

These three steps are natural in building a real-world product. Even if the first two do not require much ML knowledge, it is critical to go through them to understand "how" to build the product with a clear vision. In a nutshell, this chapter covers the following topics:

- Understanding the LLM Twin concept
- Planning the MVP of the LLM Twin product
- Building ML systems with feature/training/inference pipelines
- Designing the system architecture of the LLM Twin By the end of this chapter, you will have a clear picture of what you will learn to build throughout the book.

##### Understanding the LLM Twin concept

The first step is to have a clear vision of what we want to create and why it's valuable to build it. The concept of an LLM Twin is new. Thus, before diving into the technical details, it is essential to understand what it is, what we should expect from it, and how it should work. Having a solid intuition of your end goal makes it much easier to digest the theory, code, and infrastructure presented in this book.

###### What is an LLM Twin?

In a few words, an LLM Twin is an AI character that incorporates your writing style, voice, and personality into an LLM, which is a complex AI model. It is a digital version of yourself projected into an LLM. Instead of a generic LLM trained on the whole internet, an LLM Twin is fine-tuned on yourself. Naturally, as an ML model reflects the data it is trained on, this LLM will incorporate your writing style, voice, and personality. We intentionally used the word "projected." As with any other projection, you lose a lot of information along the way. Thus, this LLM will not be you; it will copy the side of you reflected in the data it was trained on. It is essential to understand that an LLM reflects the data it was trained on. If you feed it Shakespeare, it will start writing like him. If you train it on Billie Eilish, it will start writing songs in her style. This is also known as style transfer. This concept is prevalent in generating images, too. For example, let's say you want to create a cat image using Van Gogh's style. We will leverage the style transfer strategy, but instead of choosing a personality, we will do it on our own persona. To adjust the LLM to a given style and voice along with fine-tuning, we will also leverage various advanced retrieval-augmented generation (RAG) techniques to condition the autoregressive process with previous embeddings of ourselves.

-----

*Chapter 1* 3

---

We will explore the details in Chapter 5 on fine-tuning and Chapters 4 and 9 on RAG, but for now, let's look at a few examples to intuitively understand what we stated previously. Here are some scenarios of what you can fine-tune an LLM on to become your twin:

- **LinkedIn posts and X threads: Specialize the LLM in writing social media content.**
- **Messages with your friends and family: Adapt the LLM to an unfiltered version of yourself.**
- **Academic papers and articles: Calibrate the LLM in writing formal and educative content.**
- **Code: Specialize the LLM in implementing code as you would.** All the preceding scenarios can be reduced to one core strategy: collecting your digital data (or some parts of it) and feeding it to an LLM using different algorithms. Ultimately, the LLM reflects the voice and style of the collected data. Easy, right? Unfortunately, this raises many technical and moral issues. First, on the technical side, how can we access this data? Do we have enough digital data to project ourselves into an LLM? What kind of data would be valuable? Secondly, on the moral side, is it OK to do this in the first place? Do we want to create a copycat of ourselves? Will it write using our voice and personality, or just try to replicate it? Remember that the role of this section is not to bother with the "What" and "How" but with the

"Why." Let's understand why it makes sense to have your LLM Twin, why it can be valuable, and

why it is morally correct if we frame the problem correctly.

###### Why building an LLM Twin matters

As an engineer (or any other professional career), building a personal brand is more valuable than a standard CV. The biggest issue with creating a personal brand is that writing content on platforms such as LinkedIn, X, or Medium takes a lot of time. Even if you enjoy writing and creating content, you will eventually run out of inspiration or time and feel like you need assistance. We don't want to transform this section into a pitch, but we have to understand the scope of this product/project clearly. We want to build an LLM Twin to write personalized content on LinkedIn, X, Instagram, Substack, and Medium (or other blogs) using our style and voice. It will not be used in any immoral scenarios, but it will act as your writing co-pilot. Based on what we will teach you in this book, you can get creative and adapt it to various use cases, but we will focus on the niche of generating social media content and articles. Thus, instead of writing the content from scratch, we can feed the skeleton of our main idea to the LLM Twin and let it do the grunt work.

-----

4 *Understanding the LLM Twin Concept and Architecture*

---

Ultimately, we will have to check whether everything is correct and format it to our liking (more on the concrete features in the Planning the MVP ofthe LLM Twin product section). Hence, we project ourselves into a content-writing LLM Twin that will help us automate our writing process. It will likely fail if we try to use this particular LLM in a different scenario, as this is where we will specialize the LLM through fine-tuning, prompt engineering, and RAG. So, why does building an LLM Twin matter? It helps you do the following:

- Create your brand
- Automate the writing process
- Brainstorm new creative ideas

| What's the difference between a co-pilot and an LLM Twin? A co-pilot and digital twin are two different concepts that work together and can be combined into a powerful solution: • The co-pilot is an AI assistant or tool that augments human users in various |
|---|
| programming, writing, or content creation tasks. |
| • The twin serves as a 1:1 digital representation of a real-world entity, often |
| using AI to bridge the gap between the physical and digital worlds. For in- stance, an LLM Twin is an LLM that learns to mimic your voice, personality, and writing style. With these definitions in mind, a writing and content creation AI assistant who writes like you is your LLM Twin co-pilot. |

Also, it is critical to understand that building an LLM Twin is entirely moral. The LLM will be fine-tuned only on our personal digital data. We won't collect and use other people's data to try to impersonate anyone's identity. We have a clear goal in mind: creating our personalized writing copycat. Everyone will have their own LLM Twin with restricted access.

Of course, many security concerns are involved, but we won't go into that here as it could be a book in itself.

-----

*Chapter 1* 5

---

###### Why not use ChatGPT (or another similar chatbot)?

This subsection will refer to using ChatGPT (or another similar chatbot) just in the context of generating personalized content.

We have already provided the answer. ChatGPT is not personalized to your writing style and voice. Instead, it is very generic, unarticulated, and wordy. Maintaining an original voice is critical for long-term success when building your brand. Thus, directly using ChatGPT or Gemini will not yield the most optimal results. Even if you are OK with sharing impersonalized content, mindlessly using ChatGPT can result in the following:

- **Misinformation due to hallucination: Manually checking the results for hallucinations or**

using third-party tools to evaluate your results is a tedious and unproductive experience.

- **Tedious manual prompting: You must manually craft your prompts and inject external**

information, which is a tiresome experience. Also, the generated answers will be hard to replicate between multiple sessions as you don't have complete control over your prompts and injected data. You can solve part of this problem using an API and a tool such as LangChain, but you need programming experience to do so. From our experience, if you want high-quality content that provides real value, you will spend more time debugging the generated text than writing it yourself. The key of the LLM Twin stands in the following:

- What data we collect
- How we preprocess the data
- How we feed the data into the LLM
- How we chain multiple prompts for the desired results
- How we evaluate the generated content The LLM itself is important, but we want to highlight that using ChatGPT's web interface is exceptionally tedious in managing and injecting various data sources or evaluating the outputs. The solution is to build an LLM system that encapsulates and automates all the following steps (manually replicating them each time is not a long-term and feasible solution):
- Data collection
- Data preprocessing

-----

6 *Understanding the LLM Twin Concept and Architecture*

---

- Data storage, versioning, and retrieval
- LLM fine-tuning
- RAG
- Content generation evaluation Note that we never said not to use OpenAI's GPT API, just that the LLM framework we will present is LLM-agnostic. Thus, if it can be manipulated programmatically and exposes a fine-tuning interface, it can be integrated into the LLM Twin system we will learn to build. The key to most successful ML products is to be data-centric and make your architecture model-agnostic. Thus, you can quickly experiment with multiple models on your specific data.

##### Planning the MVP of the LLM Twin product

Now that we understand what an LLM Twin is and why we want to build it, we must clearly define the product's features. In this book, we will focus on the first iteration, often labeled the minimum

**viable product (MVP), to follow the natural cycle of most products. Here, the main objective is**

to align our ideas with realistic and doable business objectives using the available resources to produce the product. Even as an engineer, as you grow up in responsibilities, you must go through these steps to bridge the gap between the business needs and what can be implemented.

###### What is an MVP?

An MVP is a version of a product that includes just enough features to draw in early users and test the viability of the product concept in the initial stages of development. Usually, the purpose of the MVP is to gather insights from the market with minimal effort. An MVP is a powerful strategy because of the following reasons:

- **Accelerated time-to-market: Launch a product quickly to gain early traction**
- **Idea validation: Test it with real users before investing in the full development of the**

product

- **Market research: Gain insights into what resonates with the target audience**
- **Risk minimization: Reduces the time and resources needed for a product that might not**

achieve market success Sticking to the V in MVP is essential, meaning the product must be viable. The product must provide an end-to-end user journey without half-implemented features, even if the product is minimal. It must be a working product with a good user experience that people will love and want to keep using to see how it evolves to its full potential.

-----

*Chapter 1* 7

---

###### Defining the LLM Twin MVP

As a thought experiment, let's assume that instead of building this project for this book, we want to make a real product. In that case, what are our resources? Well, unfortunately, not many:

- We are a team of three people with two ML engineers and one ML researcher
- Our laptops
- Personal funding for computing, such as training LLMs
- Our enthusiasm As you can see, we don't have many resources. Even if this is just a thought experiment, it reflects the reality for most start-ups at the beginning of their journey. Thus, we must be very strategic in defining our LLM Twin MVP and what features we want to pick. Our goal is simple: we want to maximize the product's value relative to the effort and resources poured into it. To keep it simple, we will build the features that can do the following for the LLM Twin:
- Collect data from your LinkedIn, Medium, Substack, and GitHub profiles
- Fine-tune an open-source LLM using the collected data
- Populate a vector database (DB) using our digital data for RAG
- Create LinkedIn posts leveraging the following:
      - User prompts
      - RAG to reuse and reference old content
      - New posts, articles, or papers as additional knowledge to the LLM
- Have a simple web interface to interact with the LLM Twin and be able to do the following:
      - Configure your social media links and trigger the collection step
      - Send prompts or links to external resources That will be the LLM Twin MVP. Even if it doesn't sound like much, remember that we must make this system cost effective, scalable, and modular.

Even if we focus only on the core features of the LLM Twin defined in this section, we

& will build the product with the latest LLM research and best software engineering

and MLOps practices in mind. We aim to show you how to engineer a cost-effective and scalable LLM application.

-----

8 *Understanding the LLM Twin Concept and Architecture*

---

Until now, we have examined the LLM Twin from the users' and businesses' perspectives. The last step is to examine it from an engineering perspective and define a development plan to understand how to solve it technically. From now on, the book's focus will be on the implementation of the LLM Twin.

##### Building ML systems with feature/training/inference pipelines

Before diving into the specifics of the LLM Twin architecture, we must understand an ML system pattern at the core of the architecture, known as the feature/training/inference (FTI) architecture. This section will present a general overview of the FTI pipeline design and how it can structure an ML application. Let's see how we can apply the FTI pipelines to the LLM Twin architecture.

###### The problem with building ML systems

Building production-ready ML systems is much more than just training a model. From an engineering point of view, training the model is the most straightforward step in most use cases. However, training a model becomes complex when deciding on the correct architecture and hyperparameters. That's not an engineering problem but a research problem. At this point, we want to focus on how to design a production-ready architecture. Training a model with high accuracy is extremely valuable, but just by training it on a static dataset, you are far from deploying it robustly. We have to consider how to do the following:

- Ingest, clean, and validate fresh data
- Training versus inference setups
- Compute and serve features in the right environment
- Serve the model in a cost-effective way
- Version, track, and share the datasets and models
- Monitor your infrastructure and models
- Deploy the model on a scalable infrastructure
- Automate the deployments and training These are the types of problems an ML or MLOps engineer must consider, while the research or data science team is often responsible for training the model.

-----

*Chapter 1* 9

---

![](image_p38_0.png)

Data

collection

Testing and debugging

Resource

management

Model

analysis

Serving

infrastructure

Metadata Process

##### management management

*Figure 1.1: Common elements from an ML system*

The preceding figure shows all the components the Google Cloud team suggests that a mature ML and MLOps system requires. Along with the ML code, there are many moving pieces. The rest of the system comprises configuration, automation, data collection, data verification, testing and debugging, resource management, model analysis, process and metadata management, serving infrastructure, and monitoring. The point is that there are many components we must consider when productionizing an ML model. Thus, the critical question is this: How do we connect all these components into a single homogenous system? We must create a boilerplate for clearly designing ML systems to answer that question.

Similar solutions exist for classic software. For example, if you zoom out, most software applications can be split between a DB, business logic, and UI layer. Every layer can be as complex as needed, but at a high-level overview, the architecture of standard software can be boiled down to the previous three components. Do we have something similar for ML applications? The first step is to examine previous solutions and why they are unsuitable for building scalable ML systems.

-----

10 *Understanding the LLM Twin Concept and Architecture*

---

###### The issue with previous solutions

In Figure 1.2, you can observe the typical architecture present in most ML applications. It is based on a monolithic batch architecture that couples the feature creation, model training, and inference into the same component. By taking this approach, you quickly solve one critical problem in the ML world: the training-serving skew. The training-serving skew happens when the features passed to the model are computed differently at training and inference time. In this architecture, the features are created using the same code. Hence, the training-serving skew issue is solved by default. This pattern works fine when working with small data. The pipeline runs on a schedule in batch mode, and the predictions are consumed by a third-party application such as a dashboard.

![](image_p39_0.png)

Training

Data

Model

Registry

###### Upload:

Download: Model Weights Model Weights

Create Train

Features Model

Predictions

Inference

Data

*Figure 1.2: Monolithic batch pipeline architecture*

Unfortunately, building a monolithic batch system raises many other issues, such as the following:

- Features are not reusable (by your system or others)
- If the data increases, you have to refactor the whole code to support PySpark or Ray
- It's hard to rewrite the prediction module in a more efficient language such as C++, Java,

or Rust

-----

*Chapter 1* 11

---

- It's hard to share the work between multiple teams between the features, training, and

prediction modules

- It's impossible to switch to streaming technology for real-time training In Figure 1.3, we can see a similar scenario for a real-time system. This use case introduces another issue in addition to what we listed before. To make the predictions, we have to transfer the whole state through the client request so the features can be computed and passed to the model. Consider the scenario of computing movie recommendations for a user. Instead of simply passing the user ID, we must transmit the entire user state, including their name, age, gender, movie history, and more. This approach is fraught with potential errors, as the client must understand how to access this state, and it's tightly coupled with the model service. Another example would be when implementing an LLM with RAG support. The documents we add as context along the query represent our external state. If we didn't store the records in a vector DB, we would have to pass them with the user query. To do so, the client must know how to query and retrieve the documents, which is not feasible. It is an antipattern for the client application to know how to access or compute the features. If you don't understand how RAG works, we will explain it in detail in Chapters 8 and 9.

![](image_p40_0.png)

Training

Data

Model

Registry

Upload:

Download: Model Weights Model Weights

Create Features

Train Make Model Predictions

Request Result

Predictions

*Figure 1.3: Stateless real-time architecture*

-----

12 *Understanding the LLM Twin Concept and Architecture*

---

In conclusion, our problem is accessing the features to make predictions without passing them at the client's request. For example, based on our first user movie recommendation example, how can we predict the recommendations solely based on the user's ID? Remember these questions, as we will answer them shortly. Ultimately, on the other spectrum, Google Cloud provides a production-ready architecture, as shown in Figure 1.4. Unfortunately, even if it's a feasible solution, it's very complex and not intuitive. You will have difficulty understanding this if you are not highly experienced in deploying and keeping ML models in production. Also, it is not straightforward to understand how to start small and grow the system in time.

The following image is reproduced from work created and shared by Google and used according to terms described in the Creative Commons 4.0 Attribution License:

![](image_p41_0.png)

ata toe

analysis

oftine

pti

Model

preparation ©

validation prepara veining taining

Model

evaluation Joa anaiysis

Model

validation ca

Source

code 0

Pipeline deployment

i

‘experimentation/development/test

Feature

Batch

(etching

Trained

registry Model

model

Dz 0: Model oc Model Model

ata ata —

a todel todel

7

validation preparation training evaluation validation

‘Trigger

+ metadata store

ML

performance

‘monitoring

Predcton service.

*Figure 1.4: ML pipeline automation for CT (source: https://cloud.google.com/architecture/ mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)*

-----

*Chapter 1* 13

---

But here is where the FTI pipeline architectures kick in. The following section will show you how to solve these fundamental issues using an intuitive ML design.

###### The solution - ML pipelines for ML systems

The solution is based on creating a clear and straightforward mind map that any team or person can follow to compute the features, train the model, and make predictions. Based on these three critical steps that any ML system requires, the pattern is known as the FTI pipeline. So, how does this differ from what we presented before?

The pattern suggests that any ML system can be boiled down to these three pipelines: feature, training, and inference (similar to the DB, business logic, and UI layers from classic software). This is powerful, as we can clearly define the scope and interface of each pipeline. Also, it's easier to understand how the three components interact. Ultimately, we have just three instead of 20 moving pieces, as suggested in Figure 1.4, which is much easier to work with and define.

As shown in Figure 1.5, we have the feature, training, and inference pipelines. We will zoom in on each of them and understand their scope and interface.

![](image_p42_0.png)

#1. Feature Pipeline.

Transform data info features & labels Tools: Pandas, Polars,

Spark, DBT, Flink, Bytewax

#3. Inference Pipe

Make predictions with models & features

new

Tools: PyTorch, TensorFlow,

Scikitleor, Jax

##### Terns

& Labels

‘models with

Trai

feature &

Tools: PyTorch, TensorFlow, Scikitlearn, Jax

Features om

= 8 g3

Store

Model

Registry

*Figure 1.5: FTI pipelines architecture*

-----

14 *Understanding the LLM Twin Concept and Architecture*

---

Before going into the details, it is essential to understand that each pipeline is a different component that can run on a different process or hardware. Thus, each pipeline can be written using a different technology, by a different team, or scaled differently. The key idea is that the design is very flexible to the needs of your team. It acts as a mind map for structuring your architecture.

###### The feature pipeline

The feature pipeline takes raw data as input, processes it, and outputs the features and labels required by the model for training or inference. Instead of directly passing them to the model, the features and labels are stored inside a feature store. Its responsibility is to store, version, track, and share the features. By saving the features in a feature store, we always have a state of our features. Thus, we can easily send the features to the training and inference pipelines. As the data is versioned, we can always ensure that the training and inference time features match. Thus, we avoid the training-serving skew problem.

###### The training pipeline

The training pipeline takes the features and labels from the features stored as input and outputs a train model or models. The models are stored in a model registry. Its role is similar to that of feature stores, but this time, the model is the first-class citizen. Thus, the model registry will store, version, track, and share the model with the inference pipeline. Also, most modern model registries support a metadata store that allows you to specify essential aspects of how the model was trained. The most important are the features, labels, and their version used to train the model. Thus, we will always know what data the model was trained on.

###### The inference pipeline

The inference pipeline takes as input the features and labels from the feature store and the trained model from the model registry. With these two, predictions can be easily made in either batch or real-time mode. As this is a versatile pattern, it is up to you to decide what you do with your predictions. If it's a batch system, they will probably be stored in a DB. If it's a real-time system, the predictions will be served to the client who requested them. Additionally, the features, labels, and models are versioned. We can easily upgrade or roll back the deployment of the model. For example, we will always know that model v1 uses features F1, F2, and F3, and model v2 uses F2, F3, and F4. Thus, we can quickly change the connections between the model and features.

-----

*Chapter 1* 15

---

###### Benefits of the FTI architecture

To conclude, the most important thing you must remember about the FTI pipelines is their interface:

- The feature pipeline takes in data and outputs the features and labels saved to the feature

store.

- The training pipeline queries the features store for features and labels and outputs a

model to the model registry.

- The inference pipeline uses the features from the feature store and the model from the

model registry to make predictions. It doesn't matter how complex your ML system gets, these interfaces will remain the same. Now that we understand better how the pattern works, we want to highlight the main benefits of using this pattern:

- As you have just three components, it is intuitive to use and easy to understand.
- Each component can be written into its tech stack, so we can quickly adapt them to specific

needs, such as big or streaming data. Also, it allows us to pick the best tools for the job.

- As there is a transparent interface between the three components, each one can be de-

veloped by a different team (if necessary), making the development more manageable and scalable.

- Every component can be deployed, scaled, and monitored independently. The final thing you must understand about the FTI pattern is that the system doesn't have to contain only three pipelines. In most cases, it will include more. For example, the feature pipeline can be composed of a service that computes the features and one that validates the data. Also, the training pipeline can be composed of the training and evaluation components. The FTI pipelines act as logical layers. Thus, it is perfectly fine for each to be complex and contain multiple services. However, what is essential is to stick to the same interface on how the FTI pipelines interact with each other through the feature store and model registries. By doing so, each FTI component can evolve differently, without knowing the details of each other and without breaking the system on new changes.

-----

16 *Understanding the LLM Twin Concept and Architecture*

---

To learn more about the FTI pipeline pattern, consider reading From MLOps to ML

Z Systems with Feature/Training/Inference Pipelines by Jim Dowling, CEO and co-founder

```
of Hopsworks: https://www.hopsworks.ai/post/mlops-to-ml-systems-with-
```

`fti-pipelines` . His article inspired this section. Now that we understand the FTI pipeline architecture, the final step of this chapter is to see how it can be applied to the LLM Twin use case.

##### Designing the system architecture of the LLM Twin

In this section, we will list the concrete technical details of the LLM Twin application and understand how we can solve them by designing our LLM system using the FTI architecture. However, before diving into the pipelines, we want to highlight that we won't focus on the tooling or the tech stack at this step. We only want to define a high-level architecture of the system, which is language-, framework-, platform-, and infrastructure-agnostic at this point. We will focus on each component's scope, interface, and interconnectivity. In future chapters, we will cover the implementation details and tech stack.

###### Listing the technical details of the LLM Twin architecture

Until now, we defined what the LLM Twin should support from the user's point of view. Now, let's clarify the requirements of the ML system from a purely technical perspective:

- On the data side, we have to do the following:
      - Collect data from LinkedIn, Medium, Substack, and GitHub completely autono-

mously and on a schedule

- Standardize the crawled data and store it in a data warehouse
- Clean the raw data
- Create instruct datasets for fine-tuning an LLM
- Chunk and embed the cleaned data. Store the vectorized data into a vector DB

for RAG.

- For training, we have to do the following:
      - Fine-tune LLMs of various sizes (7B, 14B, 30B, or 70B parameters)
      - Fine-tune on instruction datasets of multiple sizes
      - Switch between LLM types (for example, between Mistral, Llama, and GPT)
      - Track and compare experiments

-----

*Chapter 1* 17

---

- Test potential production LLM candidates before deploying them
- Automatically start the training when new instruction datasets are available.
- The inference code will have the following properties:
- A REST API interface for clients to interact with the LLM Twin
- Access to the vector DB in real time for RAG
- Inference with LLMs of various sizes
- Autoscaling based on user requests
- Automatically deploy the LLMs that pass the evaluation step.
- The system will support the following LLMOps features:
- Instruction dataset versioning, lineage, and reusability
- Model versioning, lineage, and reusability
- Experiment tracking
- **Continuous training, continuous integration, and continuous delivery (CT/**

###### CI/CD)

- Prompt and system monitoring

& If any technical requirement doesn't make sense now, bear with us. To avoid repetition, we will examine the details in their specific chapter.

The preceding list is quite comprehensive. We could have detailed it even more, but at this point, we want to focus on the core functionality. When implementing each component, we will look into all the little details. But for now, the fundamental question we must ask ourselves is this: How can we apply the FTI pipeline design to implement the preceding list of requirements?

###### How to design the LLM Twin architecture using the FTI pipeline design

We will split the system into four core components. You will ask yourself this: "Four? Why not three, as the FTI pipeline design clearly states?" That is a great question. Fortunately, the answer is simple. We must also implement the data pipeline along the three feature/training/inference pipelines. According to best practices:

- The data engineering team owns the data pipeline
- The ML engineering team owns the FTI pipelines.

-----

18 *Understanding the LLM Twin Concept and Architecture*

---

Given our goal of building an MVP with a small team, we must implement the entire application. This includes defining the data collection and FTI pipelines. Tackling a problem end to end is often encountered in start-ups that can't afford dedicated teams. Thus, engineers have to wear many hats, depending on the state of the product. Nevertheless, in any scenario, knowing how an end-to-end ML system works is valuable for better understanding other people's work. *Figure 1.6 shows the LLM system architecture. The best way to understand it is to review the four* components individually and explain how they work.

![](image_p47_0.png)

Data Collection Pipeline Feature Pipeline

Articles

Posts

Code

Data for

Fine-tuning

& RAG

Training Pipeline

Experiment

Tracker

MR)

um

Fine-tuning

|

J

Fine-tuning

Data

LM Production

Candidate

Retrieval

cl ient

Instruct

Vector DB Dataset

Logical Feature Store loy

Accepted LLM Model

Registry

“Write a post

Prompt &

LM —

System

Twin Generated Post

Monitoring

Inference Pipeline *Figure 1.6: LLM Twin high-level architecture*

-----

*Chapter 1* 19

---

###### Data collection pipeline

The data collection pipeline involves crawling your personal data from Medium, Substack, LinkedIn, and GitHub. As a data pipeline, we will use the extract, load, transform (ETL) pattern to extract data from social media platforms, standardize it, and load it into a data warehouse.

It is critical to highlight that the data collection pipeline is designed to crawl data ed only from your social media platform. It will not have access to other people. As an

example for this book, we agreed to make our collected data available for learning purposes. Otherwise, using other people's data without their consent is not moral. The output of this component will be a NoSQL DB, which will act as our data warehouse. As we work with text data, which is naturally unstructured, a NoSQL DB fits like a glove. Even though a NoSQL DB, such as MongoDB, is not labeled as a data warehouse, from our point of view, it will act as one. Why? Because it stores standardized raw data gathered by various ETL pipelines that are ready to be ingested into an ML system. The collected digital data is binned into three categories:

- Articles (Medium, Substack)
- Posts (LinkedIn)
- Code (GitHub) We want to abstract away the platform where the data was crawled. For example, when feeding an article to the LLM, knowing it came from Medium or Substack is not essential. We can keep the source URL as metadata to give references. However, from the processing, fine-tuning, and RAG points of view, it is vital to know what type of data we ingested, as each category must be processed differently. For example, the chunking strategy between a post, article, and piece of code will look different. Also, by grouping the data by category, not the source, we can quickly plug data from other platforms, such as X into the posts or GitLab into the code collection. As a modular system, we must attach an additional ETL in the data collection pipeline, and everything else will work without further code modifications.

###### Feature pipeline

The feature pipeline's role is to take raw articles, posts, and code data points from the data warehouse, process them, and load them into the feature store.

-----

20 *Understanding the LLM Twin Concept and Architecture*

---

The characteristics of the FTI pattern are already present. Here are some custom properties of the LLM Twin's feature pipeline:

- It processes three types of data differently: articles, posts, and code
- It contains three main processing steps necessary for fine-tuning and RAG: cleaning,

chunking, and embedding

- It creates two snapshots of the digital data, one after cleaning (used for fine-tuning) and

one after embedding (used for RAG)

- It uses a logical feature store instead of a specialized feature store Let's zoom in on the logical feature store part a bit. As with any RAG-based system, one of the central pieces of the infrastructure is a vector DB. Instead of integrating another DB, more concretely, a specialized feature store, we used the vector DB, plus some additional logic to check all the properties of a feature store our system needs. The vector DB doesn't offer the concept of a training dataset, but it can be used as a NoSQL DB. This means we can access data points using their ID and collection name. Thus, we can easily query the vector DB for new data points without any vector search logic. Ultimately, we will wrap the retrieved data into a versioned, tracked, and shareable artifact-more on artifacts in *Chapter 2. For now, you must know it is an MLOps concept used to wrap data and enrich it with* the properties listed before. How will the rest of the system access the logical feature store? The training pipeline will use the instruct datasets as artifacts, and the inference pipeline will query the vector DB for additional context using vector search techniques. For our use case, this is more than enough because of the following reasons:
- The artifacts work great for offline use cases such as training
- The vector DB is built for online access, which we require for inference. In future chapters, however, we will explain how the three data categories (articles, posts, and code) are cleaned, chunked, and embedded. To conclude, we take in raw article, post, or code data points, process them, and store them in a feature store to make them accessible to the training and inference pipelines. Note that trimming all the complexity away and focusing only on the interface is a perfect match with the FTI pattern. Beautiful, right?

-----

*Chapter 1* 21

---

###### Training pipeline

The training pipeline consumes instruct datasets from the feature store, fine-tunes an LLM with it, and stores the tuned LLM weights in a model registry. More concretely, when a new instruct dataset is available in the logical feature store, we will trigger the training pipeline, consume the artifact, and fine-tune the LLM. In the initial stages, the data science team owns this step. They run multiple experiments to find the best model and hyperparameters for the job, either through automatic hyperparameter tuning or manually. To compare and pick the best set of hyperparameters, we will use an experiment tracker to log everything of value and compare it between experiments. Ultimately, they will pick the best hyperparameters and fine-tuned LLM and propose it as the LLM production candidate. The proposed LLM is then stored in the model registry. After the experimentation phase is over, we store and reuse the best hyperparameters found to eliminate the manual restrictions of the process. Now, we can completely automate the training process, known as continuous training. The testing pipeline is triggered for a more detailed analysis than during fine-tuning. Before pushing the new model to production, assessing it against a stricter set of tests is critical to see that the latest candidate is better than what is currently in production. If this step passes, the model is ultimately tagged as accepted and deployed to the production inference pipeline. Even in a fully automated ML system, it is recommended to have a manual step before accepting a new production model. It is like pushing the red button before a significant action with high consequences. Thus, at this stage, an expert looks at a report generated by the testing component. If everything looks good, it approves the model, and the automation can continue. The particularities of this component will be on LLM aspects, such as the following:

- How do you implement an LLM agnostic pipeline?
- What fine-tuning techniques should you use?
- How do you scale the fine-tuning algorithm on LLMs and datasets of various sizes?
- How do you pick an LLM production candidate from multiple experiments?
- How do you test the LLM to decide whether to push it to production or not? By the end of this book, you will know how to answer all these questions. One last aspect we want to clarify is CT. Our modular design allows us to quickly leverage an ML orchestrator to schedule and trigger different system parts. For example, we can schedule the data collection pipeline to crawl data every week.

-----

22 *Understanding the LLM Twin Concept and Architecture*

---

Then, we can trigger the feature pipeline when new data is available in the data warehouse and the training pipeline when new instruction datasets are available.

###### Inference pipeline

The inference pipeline is the last piece of the puzzle. It is connected to the model registry and logical feature store. It loads a fine-tuned LLM from the model registry, and from the logical feature store, it accesses the vector DB for RAG. It takes in client requests through a REST API as queries. It uses the fine-tuned LLM and access to the vector DB to carry out RAG and answer the queries. All the client queries, enriched prompts using RAG, and generated answers are sent to a prompt monitoring system to analyze, debug, and better understand the system. Based on specific requirements, the monitoring system can trigger alarms to take action either manually or automatically. At the interface level, this component follows exactly the FTI architecture, but when zooming in, we can observe unique characteristics of an LLM and RAG system, such as the following:

- A retrieval client used to do vector searches for RAG
- Prompt templates used to map user queries and external information to LLM inputs
- Special tools for prompt monitoring

###### Final thoughts on the FTI design and the LLM Twin architecture

We don't have to be highly rigid about the FTI pattern. It is a tool used to clarify how to design ML systems. For example, instead of using a dedicated features store just because that is how it is done, in our system, it is easier and cheaper to use a logical feature store based on a vector DB and artifacts. What was important to focus on were the required properties a feature store provides, such as a versioned and reusable training dataset. Ultimately, we will explain the computing requirements of each component briefly. The data collection and feature pipeline are mostly CPU-based and do not require powerful machines. The training pipeline requires powerful GPU-based machines that could load an LLM and fine-tune it. The inference pipeline is somewhere in the middle. It still needs a powerful machine but is less compute-intensive than the training step. However, it must be tested carefully, as the inference pipeline directly interfaces with the user. Thus, we want the latency to be within the required parameters for a good user experience. However, using the FTI design is not an issue. We can pick the proper computing requirements for each component.

-----

*Chapter 1* 23

---

Also, each pipeline will be scaled differently. The data and feature pipelines will be scaled horizontally based on the CPU and RAM load. The training pipeline will be scaled vertically by adding more GPUs. The inference pipeline will be scaled horizontally based on the number of client requests. To conclude, the presented LLM architecture checks all the technical requirements listed at the beginning of the section. It processes the data as requested, and the training is modular and can be quickly adapted to different LLMs, datasets, or fine-tuning techniques. The inference pipeline supports RAG and is exposed as a REST API. On the LLMOps side, the system supports dataset and model versioning, lineage, and reusability. The system has a monitoring service, and the whole ML architecture is designed with CT/CI/CD in mind. This concludes the high-level overview of the LLM Twin architecture.

##### Summary

This first chapter was critical to understanding the book's goal. As a product-oriented book that will walk you through building an end-to-end ML system, it was essential to understand the concept of an LLM Twin initially. Afterward, we walked you through what an MVP is and how to plan our LLM Twin MVP based on our available resources. Following this, we translated our concept into a practical technical solution with specific requirements. In this context, we introduced the FTI design pattern and showcased its real-world application in designing systems that are both modular and scalable. Ultimately, we successfully applied the FTI pattern to design the architecture of the LLM Twin to fit all our technical requirements. Having a clear vision of the big picture is essential when building systems. Understanding how a single component will be integrated into the rest of the application can be very valuable when working on it. We started with a more abstract presentation of the LLM Twin architecture, focusing on each component's scope, interface, and interconnectivity. The following chapters will explore how to implement and deploy each component. On the MLOps side, we will walk you through using a computing platform, orchestrator, model registry, artifacts, and other tools and concepts to support all MLOps best practices.

##### References

- Dowling, J. (2024a, July 11). From MLOps to ML Systems with Feature/Training/Inference

```
Pipelines. Hopsworks. https://www.hopsworks.ai/post/mlops-to-ml-systems-with-
fti-pipelines
```

-----

24 *Understanding the LLM Twin Concept and Architecture*

---

- Dowling, J. (2024b, August 5). Modularity and Composability for AI Systems with AI Pipe-

```
lines and Shared Storage. Hopsworks. https://www.hopsworks.ai/post/modularity-and-
composability-for-ai-systems-with-ai-pipelines-and-shared-storage
```

- Joseph, M. (2024, August 23). The Taxonomy for Data Transformations in AI Systems. Hop-

```
sworks. https://www.hopsworks.ai/post/a-taxonomy-for-data-transformations-
in-ai-systems
```

- *MLOps: Continuous delivery and automation pipelines in machine learning. (2024, August*

```
28). Google Cloud. https://cloud.google.com/architecture/mlops-continuous-
delivery-and-automation-pipelines-in-machine-learning
```

- Qwak. (2024a, June 2). CI/CD for Machine Learning in 2024: Best Practices to build, test,

```
and Deploy | Infer. Medium. https://medium.com/infer-qwak/ci-cd-for-machine-
learning-in-2024-best-practices-to-build-test-and-deploy-c4ad869824d2
```

- Qwak. (2024b,July 23). 5 Best Open Source Tools to build End-to-End MLOPs Pipeline in 2024.

```
Medium. https://medium.com/infer-qwak/building-an-end-to-end-mlops-pipeline-
with-open-source-tools-d8bacbf4184f
```

- Salama, K., Kazmierczak,J., & Schut, D. (2021). Practitioners guide to MLOPs: A framework

*for continuous delivery and automation of machine learning (1st ed.) [PDF]. Google Cloud.*

```
https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_
whitepaper.pdf
```

##### Join our book's Discord space

Join our community's Discord space for discussions with the authors and other readers:

```
https://packt.link/llmeng
```

-----

2

# Tooling and Installation

This chapter presents all the essential tools that will be used throughout the book, especially in implementing and deploying the LLM Twin project. At this point in the book, we don't plan to present in-depth LLM, RAG, MLOps, or LLMOps concepts. We will quickly walk you through our tech stack and prerequisites to avoid repeating ourselves throughout the book on how to set up a particular tool and why we chose it. Starting with Chapter 3, we will begin exploring our LLM Twin use case by implementing a data collection ETL that crawls data from the internet. In the first part of the chapter, we will present the tools within the Python ecosystem to manage multiple Python versions, create a virtual environment, and install the pinned dependencies required for our project to run. Alongside presenting these tools, we will also show how to install the `LLM-Engineers-Handbook` repository on your local machine (in case you want to try out the

```
code yourself): https://github .com/PacktPublishing/LLM-Engineers-Handbook.
```

Next, we will explore all the MLOps and LLMOps tools we will use, starting with more generic tools, such as a model registry, and moving on to more LLM-oriented tools, such as LLM evaluation and prompt monitoring tools. We will also understand how to manage a project with multiple ML pipelines using ZenML, an orchestrator bridging the gap between ML and MLOps. Also, we will quickly explore what databases we will use for NoSQL and vector storage. We will show you how to run all these components on your local machine using Docker. Lastly, we will quickly review AWS and show you how to create an AWS user and access keys and install and configure the AWS CLI to manipulate your cloud resources programmatically. We will also explore SageMaker and why we use it to train and deploy our open-source LLMs.

-----

26 *Tooling and Installation*

---

If you are familiar with these tools, you can safely skip this chapter. We also explain how to install the project and set up all the necessary components in the repository's `README` . Thus, you also have the option to use that as more concise documentation if you plan to run the code while reading the book. To sum all that up, in this chapter, we will explore the following topics:

- Python ecosystem and project installation
- MLOps and LLMOps tooling
- Databases for storing unstructured and vector data
- Preparing for AWS By the end of this chapter, you will be aware of all the tools we will use across the book. Also, you will have learned how to install the `LLM-Engineers-Handbook` repository, set up the rest of the tools, and use them if you run the code while reading the book.

##### Python ecosystem and project installation

Any Python project needs three fundamental tools: the Python interpreter, dependency management, and a task execution tool. The Python interpreter executes your Python project as expected. All the code within the book is tested with Python 3.11.8. You can download the Python interpreter from here: `https://www.python.org/downloads/` . We recommend installing the exact Python version (Python 3.11.8) to run the LLM Twin project using pyenv, making the installation process straightforward. Instead of installing multiple global Python versions, we recommend managing them using pyenv, a Python version management tool that lets you manage multiple Python versions between projects. You can install it using this link: `https://github.com/pyenv/pyenv?tab=readme-ov-`

```
file#installation .
```

After you have installed pyenv, you can install the latest version of Python 3.11, using pyenv, as follows:

| pyenv install 3.11.8 |
|---|
| Now list all installed Python versions to see that it was installed correctly: |
| pyenv versions |
| You should see something like this: |
| # * system |

-----

*Chapter 2* 27

| # 3.11.8 |
|---|
| To make Python 3.11.8 the default version across your entire system (whenever you open a new terminal), use the following command: |
| pyenv global 3.11.8 |
| However, we aim to use Python 3.11.8 locally only in our repository. To achieve that, first, we have to clone the repository and navigate to it: |
| git clone https://github.com/PacktPublishing/LLM-Engineers-Handbook.git cd LLM-Engineers-Handbook |
| Because we defined a .python-version file within the repository, pyenv will know to pick up the version from that file and use it locally whenever you are working within that folder. To double-check that, run the following command while you are in the repository: |
| python --version |
| It should output: |
| # Python 3.11.8 |
| To create the .python-version file, you must run pyenv local 3.11.8 once. Then, pyenv will always know to use that Python version while working within a specific directory. Now that we have installed the correct Python version using pyenv, let's move on to Poetry, which we will use as our dependency and virtual environment manager. Poetry: dependency and virtual environment management Poetry is one of the most popular dependency and virtual environment managers within the Python ecosystem. But let's start by clarifying what a dependency manager is. In Python, a depen- dency manager allows you to specify, install, update, and manage external libraries or packages (dependencies) that a project relies on. For example, this is a simple Poetry requirements file that uses Python 3.11 and the requests and numpy Python packages. |
| [tool.poetry.dependencies] python = "^3.11" requests = "^2.25.1" numpy = "^1.19.5" [build-system] |

-----

28 *Tooling and Installation*

| requires = ["poetry-core"] build-backend = "poetry.core.masonry.api" |
|---|
| By using Poetry to pin your dependencies, you always ensure that you install the correct version of the dependencies that your projects work with. Poetry, by default, saves all its requirements in pyproject.toml files, which are stored at the root of your repository, as you can see in the cloned LLM-Engineers-Handbook repository. Another massive advantage of using Poetry is that it creates a new Python virtual environment in which it installs the specified Python version and requirements. A virtual environment allows you to isolate your project's dependencies from your global Python dependencies and other projects. By doing so, you ensure there are no version clashes between projects. For example, let's assume that Project A needs numpy == 1.19 .5, and Project B needs numpy == 1.26 .0. If you keep both projects in the global Python environment, that will not work, as Project B will override Project A's numpy installation, which will corrupt Project A and stop it from working. Using Poetry, you can isolate each project in its own Python environment with its own Python dependencies, avoiding any dependency clashes. You can install Poetry from here: https://python-poetry.org/docs/ . We use Poetry 1.8.3 throughout the book. Once Poetry is installed, navigate to your cloned LLM-Engineers-Hand- book repository and run the following command to install all the necessary Python dependencies: |
| poetry install --without aws |

This command knows to pick up all the dependencies from your repository that are listed in the `pyproject.toml` and `poetry.lock` files. After the installation, you can activate your Poetry environment by running `poetry shell` in your terminal or by prefixing all your CLI commands

```
as follows: poetry run <your command> .
```

One final note on Poetry is that it locks down the exact versions of the dependency tree in the `poetry.lock` file based on the definitions added to the `project.toml` file. While the `pyproject.` `toml` file may specify version ranges (e.g., `requests = "^2.25 .1"), the poetry.lock` file records

```
the exact version (e.g., requests = "2.25 .1") that was installed. It also locks the versions of
```

sub-dependencies (dependencies of your dependencies), which may not be explicitly listed in your `pyproject.toml` file. By locking all the dependencies and sub-dependencies to specific versions, the `poetry.lock` file ensures that all project installations use the same versions of each package. This consistency leads to predictable behavior, reducing the likelihood of encountering "works on my machine" issues.

-----

*Chapter 2* 29

---

Other tools similar to Poetry are Venv and Conda for creating virtual environments. Still, they lack the dependency management option. Thus,you must do it through Python's default `requirements.` `txt` files, which are less powerful than Poetry's `lock` files. Another option is Pipenv, which feature-wise is more like Poetry but slower, and uv, which is a replacement for Poetry built in Rust, making it blazing fast. `uv` has lots of potential to replace Poetry, making it worthwhile to test out:

```
https://github .com/astral-sh/uv.
```

The final piece of the puzzle is to look at the task execution tool we used to manage all our CLI commands.

###### Poe the Poet: task execution tool

Poe the Poet is a plugin on top of Poetry that is used to manage and execute all the CLI commands required to interact with the project. It helps you define and run tasks within your Python project, simplifying automation and script execution. Other popular options are Makefile, Invoke, or shell scripts, but Poe the Poet eliminates the need to write separate shell scripts or Makefiles for managing project tasks, making it an elegant way to manage tasks using the same configuration file that Poetry already uses for dependencies. When working with Poe the Poet, instead of having all your commands documented in a README file or other document, you can add them directly to your `pyproject.toml` file and execute them in the command line with an alias. For example, using Poe the Poet, we can define the following tasks in a file:

```
pyproject.toml
```

| [tool.poe.tasks] test = "pytest" format = "black ." start = "python main.py" |
|---|
| You can then run these tasks using the poe command: |
| poetry poe test poetry poe format poetry poe start |
| You can install Poe the Poet as a Poetry plugin, as follows: |
| poetry self add 'poethepoet[poetry_plugin]' |

-----

30 *Tooling and Installation*

---

To conclude, using a tool as a façade over all your CLI commands is necessary to run your application. It significantly simplifies the application's complexity and enhances collaboration as it acts as out-of-the-box documentation. Assuming you have `pyenv` and Poetry installed, here are all the commands you need to run to clone the repository and install the dependencies and Poe the Poet as a Poetry plugin:

```
git clone https://github.com/PacktPublishing/LLM-Engineers-Handbook.gitcd
LLM-Engineers-Handbook
poetry install --without aws
poetry self add 'poethepoet[poetry_plugin]'
```

To make the project fully operational, there are still a few steps to follow, such as filling out a `.env` file with your credentials and getting tokens from OpenAI and Hugging Face. But this book isn't an installation guide, so we've moved all these details into the repository's README as they are useful only if you plan to run the repository: `https://github.com/PacktPublishing/`

```
LLM-Engineers-Handbook.
```

Now that we have installed our Python project, let's present the MLOps tools we will use in the book. If you are already familiar with these tools, you can safely skip the following tooling section and move on to the Databases for storing unstructured and vector data section.

##### MLOps and LLMOps tooling

This section will quickly present all the MLOps and LLMOps tools we will use throughout the book and their role in building ML systems using MLOps best practices. At this point in the book, we don't aim to detail all the MLOps components we will use to implement the LLM Twin use case, such as model registries and orchestrators, but only provide a quick idea of what they are and how to use them. As we develop the LLM Twin project throughout the book, you will see hands-on examples of how we use all these tools. In Chapter 11, we will dive deeply into the theory of MLOps and LLMOps and connect all the dots. As the MLOps and LLMOps fields are highly practical, we will leave the theory of these aspects to the end, as it will be much easier to understand it after you go through the LLM Twin use case implementation. Also, this section is not dedicated to showing you how to set up each tool. It focuses primarily on what each tool is used for and highlights the core features used throughout this book. Still, using Docker, you can quickly run the whole infrastructure locally. If you want to run the steps within the book yourself, you can host the application locally with these three simple steps:

1. Have Docker 27.1.1 (or higher) installed.

-----

*Chapter 2* 31

---

2. Fill your `.env` file with all the necessary credentials as explained in the repository README.

```
3. Run poetrypoelocal-infrastructure-up to locally spin up ZenML(http://127 .0.0 .1:8237/)
```

and the MongoDB and Qdrant databases. You can read more details on how to run everything locally in the LLM-Engineers-Handbook re-

```
pository README: https://github .com/PacktPublishing/LLM-Engineers-Handbook. Within
```

the book, we will also show you how to deploy each component to the cloud.

###### Hugging Face: model registry

A model registry is a centralized repository that manages ML models throughout their lifecycle. It stores models along with their metadata, version history, and performance metrics, serving as a single source of truth. In MLOps, a model registry is crucial for tracking, sharing, and documenting model versions, facilitating team collaboration. Also, it is a fundamental element in the deployment process as it integrates with continuous integration and continuous deployment (CI/CD) pipelines. We used Hugging Face as our model registry, as we can leverage its ecosystem to easily share our fine-tuned LLM Twin models with anyone who reads the book. Also, by following the Hugging Face model registry interface, we can easily integrate the model with all the frameworks around the LLMs ecosystem, such as Unsloth for fine-tuning and SageMaker for inference. Our fine-tuned LLMs are available on Hugging Face at:

```
• TwinLlama 31 8B (after fine-tuning): https://huggingface.co/mlabonne/TwinLlama-
3.1-8B
```

- **TwinLlama 31 8B DPO (after preference alignment):** `https://huggingface.co/`

```
mlabonne/TwinLlama-3.1-8B-DPO
```

![](image_p60_0.png)

@ mlabonne TwinLlama-3.1-8B 0 ©

like 7 Text Generation Transformers Safetensors mlabon

&

& apache-2.0

License:

Model card Files and versions ~~ Community

#

*Figure 2.1: Hugging Face model registry example*

-----

32 *Tooling and Installation*

---

For a quick demo, we have them available on Hugging Face Spaces:

```
• TwinLlama 31 8B: https://huggingface.co/spaces/mlabonne/TwinLlama-3.1-8B
• TwinLlama 31 8B DPO: https://huggingface.co/spaces/mlabonne/TwinLlama-3.1-
8B-DPO
```

Most ML tools provide model registry features. For example, ZenML, Comet,and SageMaker, which we will present in future sections, also offer their own model registries. They are good options, but we picked Hugging Face solely because of its ecosystem, which provides easy shareability and integration throughout the open-source environment. Thus, you will usually select the model registry that integrates the most with your project's tooling and requirements.

###### ZenML: orchestrator, artifacts, and metadata

ZenML acts as the bridge between ML and MLOps. Thus, it offers multiple MLOps features that make your ML pipeline traceability, reproducibility, deployment, and maintainability easier. At its core, it is designed to create reproducible workflows in machine learning. It addresses the challenge of transitioning from exploratory research in Jupyter notebooks to a production-ready ML environment. It tackles production-based replication issues, such as versioning difficulties, reproducing experiments, organizing complex ML workflows, bridging the gap between training and deployment, and tracking metadata. Thus, ZenML's main features are orchestrating ML pipelines, storing and versioning ML pipelines as outputs, and attaching metadata to artifacts for better observability. Instead of being another ML platform, ZenML introduced the concept of a stack, which allows you to run ZenML on multiple infrastructure options. A stack will enable you to connect ZenML to different cloud services, such as:

- An orchestrator and compute engine (for example, AWS SageMaker or Vertex AI)
- Remote storage (for instance, AWS S3 or Google Cloud Storage buckets)
- A container registry (for example, Docker Registry or AWS ECR) Thus, ZenML acts as a glue that brings all your infrastructure and tools together in one place through its stack feature, allowing you to quickly iterate through your development processes and easily monitor your entire ML system. The beauty of this is that ZenML doesn't vendor-lock you into any cloud platform. It completely abstracts away the implementation of your Python code from the infrastructure it runs on. For example, in our LLM Twin use case, we used the AWS stack:
- SageMaker as our orchestrator and compute

-----

*Chapter 2* 33

---

- S3 as our remote storage used to store and track artifacts
- ECR as our container registry However, the Python code contains no S3 or ECR particularities, as ZenML takes care of them. Thus, we can easily switch to other providers, such as Google Cloud Storage or Azure. For more details on ZenML stacks, you can start here: `https://docs.zenml.io/user-guide/production-`

```
guide/understand-stacks .
```

We will focus only on the ZenML features used throughout the book, such as orches- 4 trating, artifacts, and metadata. For more details on ZenML, check out their starter

```
guide: https://docs.zenml .io/user-guide/starter-guide.
```

The local version of the ZenML server comes installed as a Python package. Thus, when running

```
poetry install, it installs a ZenML debugging server that you can use locally. In Chapter 11, we
```

will show you how to use their cloud serverless option to deploy the ML pipelines to AWS.

###### Orchestrator

An orchestrator is a system that automates, schedules, and coordinates all your ML pipelines. It ensures that each pipeline-such as data ingestion, preprocessing, model training, and deployment-executes in the correct order and handles dependencies efficiently. By managing these processes, an orchestrator optimizes resource utilization, handles failures gracefully, and enhances scalability, making complex ML pipelines more reliable and easier to manage.

**How does ZenML work as an orchestrator? It works with pipelines and steps. A pipeline is a**

high-level object that contains multiple steps. A function becomes a ZenML pipeline by being decorated with @pipeline, and a step when decorated with @step. This is a standard pattern when using orchestrators: you have a high-level function, often called a pipeline, that calls multiple units/steps/tasks. Let's explore how we can implement a ZenML pipeline with one of the ML pipelines implemented for the LLM Twin project. In the code snippet below, we defined a ZenML pipeline that queries the database for a user based on its full name and crawls all the provided links under that user:

```python
from zenml import pipeline
from steps.etl import crawl_links, get_or_create_user
@pipeline
```

-----

34 *Tooling and Installation*

---

```python
def digital_data_etl(user_full_name: str, links: list[str]) -> None:
user = get_or_create_user(user_full_name)
crawl_links(user=user, links=links)
```

You can run the pipeline with the following CLI command: `poetry poe run-digital-data-etl` . To visualize the pipeline run, you can go to your ZenML dashboard (at `http://127.0.0 .1:8237/)` and, on the left panel, click on the Pipelines tab and then on the digital\_data\_etl pipeline, as illustrated in Figure 2.2:

![](image_p63_0.png)

|¢ Pipelines

Production Setup >

0/2

Overview

Search.

Pipeline

Chi

J Models

Artifacts

=

Stacks

©

“E generate\_datasets ©)

(©

°E feature\_engineering ©

*Figure 2.2: ZenML Pipelines dashboard*

After clicking on the digital\_data\_etl pipeline, you can visualize all the previous and current pipeline runs, as seen in Figure 2.3. You can see which one succeeded, failed, or is still running. Also, you can see the stack used to run the pipeline, where the default stack is the one used to run your ML pipelines locally.

-----

*Chapter 2* 35

---

![](image_p64_0.png)

©

etn saa.

### ©

tn

5

### ©

5 pm

[=] i a 0 default

24/08/2024, 12:28:41 © detautt

[©] default

26/08/2024, 1114:06 © defaut

*Figure 2.3: ZenML digital\_data\_etl pipeline dashboard. Example of a specific pipeline*

Now, after clicking on the latest digital\_data\_etl pipeline run (or any other run that succeeded or is still running), we can visualize the pipeline's steps, outputs, and insights, as illustrated in Figure *2.4. This structure is often called a directed acyclic graph (DAG). More on DAGs in Chapter 11.*

![](image_p64_1.png)

© digital\_data\_etl\_run\_2024\_09\_26.15\_38\_51 ©

S| c

(© get\_or\_create\_user

g user

© overview 32 Configuration

Tce

Stas © completed

Pioine

| Author | o aout |
|---|---|
| © start Time | 20091202, 153851 |
| ena Time | 2610912024, 15:38:51 |

crawled.links

seek LE builtins list v

Artfact store *Figure 2.4: ZenML digital\_data\_etl pipeline run dashboard (example of a specific pipeline run)*

-----

36 *Tooling and Installation*

---

By clicking on a specific step, you can get more insights into its code and configuration. It even aggregates the logs output by that specific step to avoid switching between tools, as shown in *Figure 2.5.*

![](image_p65_0.png)

© get\_or\_create\_user completes

0 Overview 5 Code Logs #2 Configuration

Logs

Step get\_or\_create\_user has started.

2024-09-26 15:38:51,597 | INFO | steps.etl.get\_or\_create\_user:get\_or\_create\_user:11

Getting or creating user: Maxime Labonne

~

Step get\_or\_create\_user has finished in 0.073s.

Step get\_or\_create\_user completed successfully.

*Figure 2.5: Example of insights from a specific step of the digital\_data\_etl pipeline run*

Now that we understand how to define a ZenML pipeline and how to look it up in the dashboard, let's quickly look at how to define a ZenML step. In the code snippet below, we defined the `get\_` `or\_create\_user()` step, which works just like a normal Python function but is decorated with

```
@step. We won't go into the details of the logic, as we will cover the ETL logic in Chapter 3. For
```

now, we will focus only on the ZenML functionality.

```python
from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step
from llm_engineering.application import utils
```

-----

*Chapter 2* 37

---

```python
from llm_engineering.domain.documents import UserDocument
@step
def get_or_create_user(user_full_name: str) -> Annotated[UserDocument,
"user"]:
logger.info(f"Getting or creating user: {user_full_name}")
first_name, last_name = utils.split_user_full_name(user_full_name)
user = UserDocument.get_or_create(first_name=first_name, last_
name=last_name)
return user
```

Within a ZenML step, you can define any Python logic your use case needs. In this simple example, we are just creating or retrieving a user, but we could replace that code with anything, starting from data collection to feature engineering and training. What is essential to notice is that to integrate ZenML with your code, you have to write modular code, where each function does just one thing. The modularity of your code makes it easy to decorate your functions with `@step` and then glue multiple steps together within a main function decorated with @pipeline. One design choice that will impact your application is deciding the granularity of each step, as each will run as a different unit on a different machine when deployed in the cloud.

To decouple our code from ZenML, we encapsulated all the application and domain logic into the `llm\_engineering` Python module. We also defined the `pipelines` and `steps` folders, where we defined our ZenML logic. Within the `steps` module, we only used what we needed from the `llm\_engineering` Python module (similar to how you use a Python package). In the `pipelines` module, we only aggregated ZenML steps to glue them into the final pipeline. Using this design, we can easily swap ZenML with another orchestrator or use our application logic in other use cases, such as a REST API. We only have to replace the ZenML code without touching the `llm\_engineering` module where all our logic resides.

-----

38 *Tooling and Installation*

---

This folder structure is reflected at the root of the LLM-Engineers-Handbook repository, as illustrated in Figure 2.6:

![](image_p67_0.png)

© LLM-Engineers-Handbook (Pubic) 2 (© @)

wateh

| ~

#### | co - |

# 7Branches © 0Tags

main -

@ {D

iusztinpaul docs: Improve README 4312314 hours ago. 116 Commits githubjworkfiows fix: Loading Settings from ZenML secrets 2 months ago vscode feat: Add DE pipeline logic 4 months ago

8 code\_snippets feat: Add custom ODM example last week configs docs: Extend README yesterday dummy\_dataset added finetuning script v1 2 months ago images docs: Update README with env details ago docs: Extend README yesterday

pipelines feat: Add dataset generation logic with prefernce support 2 weeks ago

| steps | feat: Add dataset generation logic with prefernce support | 2 weeks ago |
|---|---|---|
| tools | feat: Add dataset generation logic with prefernce support | 2 weeks ago |

*Figure 2.6: LLM-Engineers-Handbook repository folder structure*

One last thing to consider when writing ZenML steps is that if you return a value, it should be serializable. ZenML can serialize most objects that can be reduced to primitive data types, but there are a few exceptions. For example, we used UUID types as IDs throughout the code, which aren't natively supported by ZenML. Thus, we had to extend ZenML's materializer to support UUIDs. We raised this issue to ZenML. Hence, in future ZenML versions, UUIDs will be supported, but it was an excellent example of the serialization aspect of transforming function outputs in artifacts.

-----

*Chapter 2* 39

---

###### Artifacts and metadata

As mentioned in the previous section, ZenML transforms any step output into an artifact. First, let's quickly understand what an artifact is. In MLOps, an artifact is any file(s) produced during the machine learning lifecycle, such as datasets, trained models, checkpoints, or logs. Artifacts are crucial for reproducing experiments and deploying models. We can transform anything into an artifact. For example, the model registry is a particular use case for an artifact. Thus, artifacts have these unique properties: they are versioned, sharable, and have metadata attached to them to understand what's inside quickly. For example, when wrapping your dataset with an artifact, you can add to its metadata the size of the dataset, the train-test split ratio, the size, types of labels, and anything else useful to understand what's inside the dataset without actually downloading it. Let's circle back to our digital\_data\_etl pipeline example, where we had as a step output an artifact, the crawled links, which are an artifact, as seen in Figure 2.7

![](image_p68_0.png)

© get\_or\_create\_user

8 =

lim\_engineering.domain.documents.Use.

© crawl\_links

crawled\_links

*Figure 2.7: ZenML artifact example using the digital\_data\_etl pipeline as an example*

-----

40 *Tooling and Installation*

---

By clicking on the `crawled\_links` artifact and navigating to the Metadata tab, we can quickly see all the domains we crawled for a particular author, the number of links we crawled for each domain, and how many were successful, as illustrated in Figure 2.8:

![](image_p69_0.png)

crawled\_links ©

0 overview [) Metadata [I] Visualization

> Uncategorized

miabonne.github.io

successful

2

total

2

maximelabonne.substack.com

successful

2

total

2

*Figure 2.8: ZenML metadata example using the digital\_data\_etl pipeline as an example*

A more interesting example of an artifact and its metadata is the generated dataset artifact. In *Figure 2.9, we can visualize the metadata of the* `instruct\_datasets` artifact, which was automatically generated and will be used to fine-tune the LLM Twin model. More details on the `instruction datasets` are in Chapter 5. For now, we want to highlight that within the dataset's metadata, we have precomputed a lot of helpful information about it, such as how many data categories it contains, its storage size, and the number of samples per training and testing split.

-----

*Chapter 2* 41

---

![](image_p70_0.png)

E instruct\_datasets 1

0 overview Metadata Visualization

Uncategorized

data\_categories

articles

storage\_size 493.23 KB

test\_split\_size

01

> schema train\_num\_samples\_per\_category

articles 738 test\_num\_samples\_per\_category

articles 82

*Figure 2.9: ZenML metadata example for the instruct\_datasets artifact*

The metadata is manually added to the artifact, as shown in the code snippet below. Thus, you can precompute and attach to the artifact's metadata anything you consider helpful for dataset discovery across your business and projects:

```python
… # More imports
from zenml import ArtifactConfig, get_step_context, step
@step
def generate_intruction_dataset(
prompts: Annotated[dict[DataCategory,
list[GenerateDatasetSamplesPrompt]], "prompts"]) -> Annotated[
```

-----

42 *Tooling and Installation*

---

| InstructTrainTestSplit, ArtifactConfig( name="instruct_datasets", tags=["dataset", "instruct", "cleaned"], ), ]: datasets = … # Generate datasets step_context = get_step_context() step_context.add_output_metadata(output_name="instruct_datasets", metadata=_get_metadata_instruct_dataset(datasets)) return datasets def _get_metadata_instruct_dataset(datasets: InstructTrainTestSplit) -> dict[str, Any]: instruct_dataset_categories = list(datasets.train.keys()) train_num_samples = { category: instruct_dataset.num_samples for category, instruct_ dataset in datasets.train.items() } test_num_samples = {category: instruct_dataset.num_samples for category, instruct_dataset in datasets.test.items()} return { "data_categories": instruct_dataset_categories, "test_split_size": datasets.test_split_size, "train_num_samples_per_category": train_num_samples, "test_num_samples_per_category": test_num_samples, } |
|---|
| Also, you can easily download and access a specific version of the dataset using its Universally Unique Identifier (UUID), which you can find using the ZenML dashboard or CLI: |
| from zenml.client import Client artifact = Client().get_artifact_version('8bba35c4-8ff9-4d8f-a039- 08046efc9fdc') loaded_artifact = artifact.load() |

-----

*Chapter 2* 43

---

The last step in exploring ZenML is understanding how to run and configure a ZenML pipeline.

###### How to run and configure a ZenML pipeline

All the ZenML pipelines can be called from the `run.py` file, accessed at `tools/run.py` in our GitHub repository. Within the `run.py` file, we implemented a simple CLI that allows you to specify what pipeline to run. For example, to call the `digital\_data\_etl` pipeline to crawl Maxime's content, you have to run:

| python -m tools.run --run-etl --no-cache --etl-config-filename digital_ data_etl_maxime_labonne.yaml |
|---|
| Or, to crawl Paul's content, you can run: |
| python -m tools.run --run-etl --no-cache --etl-config-filename digital_ data_etl_paul_iusztin.yaml |
| As explained when introducing Poe the Poet, all our CLI commands used to interact with the proj- ect will be executed through Poe to simplify and standardize the project. Thus, we encapsulated these Python calls under the following poe CLI commands: |
| poetry poe run-digital-data-etl-maxime poetry poe run-digital-data-etl-paul |
| We only change the ETL config file name when scraping content for different people. ZenML allows us to inject specific configuration files at runtime as follows: |
| config_path = root_dir / "configs" / etl_config_filename assert config_path.exists(), f"Config file not found: { config_path }" run_args_etl = { "config_path": config_path, "run_name": f"digital_data_etl_run_{dt.now(). strftime('%Y_%m_%d_%H_%M_%S')}" } digital_data_etl.with_options()(**run_args_etl) |
| In the config file, we specify all the parameters that will input the pipeline as parameters. For ex- ample, the configs/digital_data_etl_maxime_labonne.yaml configuration file looks as follows: |
| parameters: user_full_name: Maxime Labonne # [First Name(s)] [Last Name] links: # Personal Blog |

-----

44 *Tooling and Installation*

---

| - https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31. html - https://mlabonne.github.io/blog/posts/2024-07-15_The_Rise_of_ Agentic_Data_Generation.html # Substack - https://maximelabonne.substack.com/p/uncensor-any-llm-with- abliteration-d30148b7d43e … # More links |
|---|
| Where the digital_data_etl function signature looks like this: |
| @pipeline def digital_data_etl(user_full_name: str, links: list[str]) -> str: |

This approach allows us to configure each pipeline at runtime without modifying the code. We can also clearly track the inputs for all our pipelines, ensuring reproducibility. As seen in Figure *2.10, we have one or more configs for each pipeline.*

![](image_p73_0.png)

LLM-Engineering / configs / 2

@ iusztinpaul feat: Add dataset generation

Name

digital\_data\_etl\_maxime\_labonne.yam|

end\_to\_end\_data.yam!

3 export\_artifact\_to\_json.yam!

3

3 generate\_instruct\_datasets.yam!

generate\_preference\_datasets.yam|

training.yam!

*Figure 2.10: ZenML pipeline configs*

-----

*Chapter 2* 45

---

Other popular orchestrators similar to ZenML that we've personally tested and consider powerful are Airflow, Prefect, Metaflow, and Dagster. Also, if you are a heavy user of Kubernetes, you can opt for Agro Workflows or Kubeflow, the latter of which works only on top of Kubernetes. We still consider ZenML the best trade-off between ease of use, features, and costs. Also, none of these tools offer the stack feature that is offered by ZenML, which allows it to avoid vendor-locking you in to any cloud ecosystem. In Chapter 11, we will explore in more depth how to leverage an orchestrator to implement MLOps best practices. But now that we understand ZenML, what it is helpful for, and how to use it, let's move on to the experiment tracker.

###### Comet ML: experiment tracker

Training ML models is an entirely iterative and experimental process. Unlike traditional software development, it involves running multiple parallel experiments, comparing them based on predefined metrics, and deciding which one should advance to production. An experiment tracking tool allows you to log all the necessary information, such as metrics and visual representations of your model predictions, to compare all your experiments and quickly select the best model. Our LLM project is no exception. As illustrated in Figure 2.11, we used Comet to track metrics such as training and evaluation loss or the value of the gradient norm across all our experiments.

![](image_p74_0.png)

*Figure 2.11: Comet ML training metrics example*

-----

46 *Tooling and Installation*

---

Using an experiment tracker, you can go beyond training and evaluation metrics and log your training hyperparameters to track different configurations between experiments. It also logs out-of-the-box system metrics such as GPU, CPU, or memory utilization to give you a clear picture of what resources you need during training and where potential bottlenecks slow down your training, as seen in Figure 2.12.

![](image_p75_0.png)

*Figure 2.12: Comet ML system metrics example*

You don't have to set up Comet locally. We will use their online version for free without any constraints throughout this book. Also, if you want to look more in-depth into the Comet ML experiment tracker, we made the training experiments tracked with Comet ML public while fine-tuning our LLM Twin models. You can access them here: `https://www.comet.com/mlabonne/`

```
llm-twin-training/view/new/panels .
```

Other popular experiment trackers are W&B, MLflow, and Neptune. We've worked with all of them and can state that they all have mostly the same features, but Comet ML differentiates itself through its ease of use and intuitive interface. Let's move on to the final piece of the MLOps puzzle: Opik for prompt monitoring.

###### Opik: prompt monitoring

You cannot use standard tools and techniques when logging and monitoring prompts. The reason for this is complicated. We will dig into it in Chapter 11. However, to quickly give you some understanding, you cannot use standard logging tools as prompts are complex and unstructured chains.

-----

*Chapter 2* 47

---

When interacting with an LLM application, you chain multiple input prompts and the generated output into a trace, where one prompt depends on previous prompts. Thus, instead of plain text logs, you need an intuitive way to group these traces into a specialized dashboard that makes debugging and monitoring traces of prompts easier. We used Opik, an open-source tool made by Comet, as our prompt monitoring tool because it follows Comet's philosophy of simplicity and ease of use, which is currently relatively rare in the LLM landscape. Other options offering similar features are Langfuse (open source, `https://langfuse.`

```
com), Galileo (not open source, rungalileo .io), and LangSmith (not open source, https://www.
langchain .com/langsmith), but we found their solutions more cumbersome to use and imple-
```

ment. Opik, along with its serverless option, also provides a free open-source version that you have complete control over. You can read more on Opik at `https://github .com/comet-ml/opik.`

##### Databases for storing unstructured and vector data

We also want to present the NoSQL and vector databases we will use within our examples. When working locally, they are already integrated through Docker. Thus, when running `poetry poe`

```
local-infrastructure-up, as instructed a few sections above, local images of Docker for both
```

databases will be pulled and run on your machine. Also, when deploying the project, we will show you how to use their serverless option and integrate it with the rest of the LLM Twin project.

###### MongoDB: NoSQL database

MongoDB is one of today's most popular, robust, fast, and feature-rich NoSQL databases. It integrates well with most cloud ecosystems, such as AWS, Google Cloud, Azure, and Databricks. Thus, using MongoDB as our NoSQL database was a no-brainer. When we wrote this book, MongoDB was used by big players such as Novo Nordisk, Delivery Hero, Okta, and Volvo. This widespread adoption suggests that MongoDB will remain a leading NoSQL database for a long time. We use MongoDB as a NoSQL database to store the raw data we collect from the internet before processing it and pushing it into the vector database. As we work with unstructured text data, the flexibility of the NoSQL database fits like a charm.

###### Qdrant: vector database

Qdrant (https://qdrant.tech/) is one of the most popular, robust, and feature-rich vector databases. We could have used almost any vector database for our small MVP, but we wanted to pick something light and likely to be used in the industry for many years to come.

-----

48 *Tooling and Installation*

---

We will use Qdrant to store the data from MongoDB after it's processed and transformed for GenAI usability. Qdrant is used by big players such as X (formerly Twitter), Disney, Microsoft, Discord, and Johnson & Johnson. Thus, it is highly probable that Qdrant will remain in the vector database game for a long time. While writing the book, other popular options were Milvus, Redis, Weaviate, Pinecone, Chroma, and pgvector (a PostgreSQL plugin for vector indexes). We found that Qdrant offers the best trade-off between RPS, latency, and index time, making it a solid choice for many generative AI applications. Comparing all the vector databases in detail could be a chapter in itself. We don't want to do that here. Still, if curious, you can check the Vector DB Comparison resource from Superlinked at

```
https://superlinked .com/vector-db-comparison, which compares all the top vector databases
```

in terms of everything you can think about, from the license and release year to database features, embedding models, and frameworks supported.

##### Preparing for AWS

This last part of the chapter will focus on setting up an AWS account (if you don't already have one), an AWS access key, and the CLI. Also, we will look into what SageMaker is and why we use it. We picked AWS as our cloud provider because it's the most popular out there and the cloud in which we (the writers) have the most experience. The reality is that other big cloud providers, such as GCP or Azure, offer similar services. Thus, depending on your specific application, there is always a trade-off between development time (in which you have the most experience), features, and costs. But for our MVP, AWS, it's the perfect option as it provides robust features for everything we need, such as S3 (object storage), ECR (container registry), and SageMaker (compute for training and inference).

###### Setting up an AWS account, an access key, and the CLI

As AWS could change its UI/UX, the best way to instruct you on how to create an AWS account is by redirecting you to their official tutorial: `https://docs.aws.amazon.com/accounts/latest/`

```
reference/manage-acct-creating.html .
```

After successfully creating an AWS account, you can access the AWS console at `http://console.`

```
aws.amazon .com. Select Sign in using root user email (found under the Sign in button), then
```

enter your account's email address and password.

-----

*Chapter 2* 49

---

Next, we must generate access keys to access AWS programmatically. The best option to do so is first to create an IAM user with administrative access as described in this AWS official tutorial:

```
https://docs.aws.amazon.com/streams/latest/dev/setting-up.html
```

For production accounts, it is best practice to grant permissions with a policy of least privilege, giving each user only the permissions they require to perform their role. However, to simplify the setup of our test account, we will use the `AdministratorAccess` managed policy, which gives our user full access, as explained in the tutorial above and illustrated in Figure 2.13.

| | | Security credentials | Last Accessed

![](image_p78_0.png)

Permissions Groups

Permissions policies 2 Remove [Add

permissions v

@ Administratoraccess

C

ERC

Fiterby Type.

NES

managed function

AWS - job v

|

v

Directly

*Figure 2.13: IAM user permission policies example*

Next, you have to create an access key for the IAM user you just created using the following tutorial:

```
https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html .
```

The access keys will look as follows:

| aws_access_key_id = <your_access_key_id> aws_secret_access_key = <your_secret_access_key> |
|---|
| Just be careful to store them somewhere safe, as you won't be able to access them after you cre- ate them. Also, be cautious with who you share them, as they could be used to access your AWS account and manipulate various AWS resources. The last step is to install the AWS CLI and configure it with your newly created access keys. You can install the AWS CLI using the following link: https://docs.aws.amazon.com/cli/latest/ userguide/getting-started-install.html . After installing the AWS CLI, you can configure it by running aws configure . Here is an example of our AWS configuration: |
| [default] aws_access_key_id = ************* aws_secret_access_key = ************ |

-----

50 *Tooling and Installation*

| For more details on how to configure the AWS CLI, check out the following tutorial: docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html Also, to configure the project with your AWS credentials, you must fill in the following variables within your | region = output = For more details on how to configure the AWS CLI, check out the following tutorial: docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html Also, to configure the project with your AWS credentials, you must fill in the following variables within your | region = output = For more details on how to configure the AWS CLI, check out the following tutorial: docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html Also, to configure the project with your AWS credentials, you must fill in the following variables within your | region = output = For more details on how to configure the AWS CLI, check out the following tutorial: docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html Also, to configure the project with your AWS credentials, you must fill in the following variables within your | region = output = For more details on how to configure the AWS CLI, check out the following tutorial: docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html Also, to configure the project with your AWS credentials, you must fill in the following variables within your | region = eu-central-1 output = json For more details on how to configure the AWS CLI, check out the following tutorial: https:// docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html . Also, to configure the project with your AWS credentials, you must fill in the following variables within your .env file: |
|---|---|---|---|---|---|
|  |  | use |  |  | AWS_REGION="eu-central-1" # Change it with your AWS region. By default, we "eu-central-1". AWS_ACCESS_KEY="<your_aws_access_key>" AWS_SECRET_KEY="<your_aws_secret_key>" |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

###### An important note about costs associated with hands-on tasks in this book

All the cloud services used across the book stick to their freemium option, except AWS. Thus, if you use a personal AWS account, you will be responsible for AWS costs as you follow along in this book. While some services may fall under AWS Free Tier usage, others will not. Thus, you are responsible for checking your billing console regularly.

W

Most of the costs will come when testing SageMaker for training and inference. Based on our tests, the AWS costs can vary between $50 and $100 using the specifications provided in this book and repository. See the AWS documentation on setting up billing alarms to monitor your costs

```
at https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/
monitor_estimated_charges_with_cloudwatch.html .
```

###### SageMaker: training and inference compute

The last topic of this chapter is understanding SageMaker and why we decided to use it. SageMaker is an ML platform used to train and deploy ML models. An official definition is as follows: AWS SageMaker is a fully managed machine learning service by AWS that enables developers and data scientists to build, train, and deploy machine learning models at scale. It simplifies the process by handling the underlying infrastructure, allowing users to focus on developing high-quality models efficiently.

-----

*Chapter 2* 51

---

We will use SageMaker to fine-tune and operationalize our training pipeline on clusters of GPUs and to deploy our custom LLM Twin model as a REST API that can be accessed in real time from anywhere in the world.

###### Why AWS SageMaker?

We must also discuss why we chose AWS SageMaker over simpler and more cost-effective options, such as AWS Bedrock. First, let's explain Bedrock and its benefits. Amazon Bedrock is a serverless solution for deploying LLMs. Serverless means that there are no servers or infrastructure to manage. It provides pre-trained models, which you can access directly through API calls. When we wrote this book, they provided support only for Mistral, Flan, Llama 2, and Llama 3 (quite a limited list of options). You can send input data and receive predictions from the models without managing the underlying infrastructure or software. This approach significantly reduces the complexity and time required to integrate AI capabilities into applications, making it more accessible to developers with limited machine learning expertise. However, this ease of integration comes at the cost of limited customization options, as you're restricted to the pre-trained models and APIs provided by Amazon Bedrock. In terms of pricing, Bedrock uses a simple pricing model based on the number of API calls. This straightforward pricing structure makes it more efficient to estimate and control costs. Meanwhile, SageMaker provides a comprehensive platform for building, training, and deploying machine learning models. It allows you to customize your ML processes entirely or even use the platform for research. That's why SageMaker is mainly used by data scientists and machine learning experts who know how to program, understand machine learning concepts, and are comfortable working with cloud platforms such as AWS. SageMaker is a double-edged sword regarding costs, following a pay-as-you-go pricing model similar to most AWS services. This means you have to pay for the usage of computing resources, storage, and any other services required to build your applications. In contrast to Bedrock, even if the SageMaker endpoint is not used, you will still pay for the deployed resources on AWS, such as online EC2 instances. Thus, you have to design autoscaling systems that delete unused resources. To conclude, Bedrock offers an out-of-the-box solution that allows you to quickly deploy an API endpoint powered by one of the available foundation models. Meanwhile, SageMaker is a multi-functional platform enabling you to customize your ML logic fully.

-----

52 *Tooling and Installation*

---

So why did we choose SageMaker over Bedrock? Bedrock would have been an excellent solution for quickly prototyping something, but this is a book on LLM engineering, and our goal is to dig into all the engineering aspects that Bedrock tries to mask away. Thus, we chose SageMaker because of its high level of customizability, allowing us to show you all the engineering required to deploy a model. In reality, even SageMaker isn't fully customizable. If you want complete control over your deployment, use EKS, AWS's Kubernetes self-managed service. In this case, you have direct access to the virtual machines, allowing you to fully customize how you build your ML pipelines, how they interact, and how you manage your resources. You could do the same thing with AWS ECS, AWS's version of Kubernetes. Using EKS or ECS, you could also reduce the costs, as these services cost considerably less. To conclude, SageMaker strikes a balance between complete control and customization and a fully managed service that hides all the engineering complexity behind the scenes. This balance ensures that you have the control you need while also benefiting from the managed service's convenience.

##### Summary

In this chapter, we reviewed the core tools used across the book. First, we understood how to install the correct version of Python that supports our repository. Then, we looked over how to create a virtual environment and install all the dependencies using Poetry. Finally, we understood how to use a task execution tool like Poe the Poet to aggregate all the commands required to run the application. The next step was to review all the tools used to ensure MLOps best practices, such as a model registry to share our models, an experiment tracker to manage our training experiments, an orchestrator to manage all our ML pipelines and artifacts, and metadata to manage all our files and datasets. We also understood what type of databases we need to implement the LLM Twin use case. Finally, we explored the process of setting up an AWS account, generating an access key, and configuring the AWS CLI for programmatic access to the AWS cloud. We also gained a deep understanding of AWS SageMaker and the reasons behind choosing it to build our LLM Twin application. In the next chapter, we will explore the implementation of the LLM Twin project by starting with the data collection ETL that scrapes posts, articles, and repositories from the internet and stores them in a data warehouse.

-----

*Chapter 2* 53

---

##### References

- Acsany, P. (2024, February 19). Dependency Management With Python Poetry. `https://`

```
realpython.com/dependency-management-python-poetry/
```

- Comet.ml. (n.d.). comet-ml/opik: Open-source end-to-end LLM Development Platform. GitHub.

```
https://github.com/comet-ml/opik
```

- Czakon, J. (2024, September 25). ML Experiment Tracking: What It Is, Why It Matters, and

```
How to Implement It. neptune.ai. https://neptune.ai/blog/ml-experiment-tracking
```

- Hopsworks. (n.d.). ML Artifacts (ML Assets)? Hopsworks. `https://www.hopsworks.ai/`

```
dictionary/ml-artifacts
```

- *Introduction | Documentation | Poetry - Python dependency management and packaging made*

```
easy. (n.d.). https://python-poetry.org/docs
```

- Jones, L. (2024, March 21). Managing Multiple Python Versions With pyenv. `https://`

```
realpython.com/intro-to-pyenv/
```

- Kaewsanmua, K. (2024, January 3). Best Machine Learning Workflow and Pipeline Orches-

```
tration Tools. neptune.ai. https://neptune.ai/blog/best-workflow-and-pipeline-
orchestration-tools
```

- MongoDB. (n.d.). What is NoSQL? NoSQL databases explained. `https://www.mongodb.`

```
com/resources/basics/databases/nosql-explained
```

- Nat-N. (n.d.). nat-n/poethepoet: A task runner that works well with poetry. GitHub. `https://`

```
github.com/nat-n/poethepoet
```

- Oladele, S. (2024, August 29). ML Model Registry: The Ultimate Guide. neptune.ai. `https://`

```
neptune.ai/blog/ml-model-registry
```

- Schwaber-Cohen, R. (n.d.). What is a Vector Database & How Does it Work? Use Cases + Ex-

```
amples. Pinecone. https://www.pinecone.io/learn/vector-database/
```

- *Starter guide | ZenML Documentation. (n.d.).* `https://docs.zenml.io/user-guide/`

```
starter-guide
• Vector DB Comparison. (n.d.). https://superlinked.com/vector-db-comparison
```

-----

54 *Tooling and Installation*

---

##### Join our book's Discord space

Join our community's Discord space for discussions with the authors and other readers:

```
https://packt.link/llmeng
```

-----

3

# Data Engineering

This chapter will begin exploring the LLM Twin project in more depth. We will learn how to design and implement the data collection pipeline to gather the raw data we will use in all our LLM use cases, such as fine-tuning or inference. As this is not a book on data engineering, we will keep this chapter short and focus only on what is strictly necessary to collect the required raw data. Starting with Chapter 4, we will concentrate on LLMs and GenAI, exploring its theory and concrete implementation details. When working on toy projects or doing research, you usually have a static dataset with which you work. But in our LLM Twin use case, we want to mimic a real-world scenario where we must gather and curate the data ourselves. Thus, implementing our data pipeline will connect the dots regarding how an end-to-end ML project works. This chapter will explore how to design and implement an Extract, Transform, Load (ETL) pipeline that crawls multiple social platforms, such as Medium, Substack, or GitHub, and aggregates the gathered data into a MongoDB data warehouse. We will show you how to implement various crawling methods, standardize the data, and load it into a data warehouse. We will begin by designing the LLM Twin's data collection pipeline and explaining the architecture of the ETL pipeline. Afterward, we will move directly to implementing the pipeline, starting with ZenML, which will orchestrate the entire process. We will investigate the crawler implementation and understand how to implement a dispatcher layer that instantiates the right crawler class based on the domain of the provided link while following software best practices. Next, we will learn how to implement each crawler individually. Also, we will show you how to implement a data layer on top of MongoDB to structure all our documents and interact with the database.

-----

56 *Data Engineering*

---

Finally, we will explore how to run the data collection pipeline using ZenML and query the collected data from MongoDB. Thus, in this chapter, we will study the following topics:

- Designing the LLM Twin's data collection pipeline
- Implementing the LLM Twin's data collection pipeline
- Gathering raw data into the data warehouse By the end of this chapter, you will know how to design and implement an ETL pipeline to extract, transform, and load raw data ready to be ingested into the ML application.

##### Designing the LLM Twin's data collection pipeline

Before digging into the implementation, we must understand the LLM Twin's data collection ETL architecture, illustrated in Figure 3.1. We must explore what platforms we will crawl to extract data from and how we will design our data structures and processes. However, the first step is understanding how our data collection pipeline maps to an ETL process. An ETL pipeline involves three fundamental steps:

1. We extract data from various sources. We will crawl data from platforms like Medium,

Substack, and GitHub to gather raw data.

2. We transform this data by cleaning and standardizing it into a consistent format suitable

for storage and analysis.

3. We load the transformed data into a data warehouse or database. For our project, we use MongoDB as our NoSQL data warehouse. Although this is not a standard approach, we will explain the reasoning behind this choice shortly.

-----

*Chapter 3* 57

---

![](image_p86_0.png)

Digital Data

Sources

links to crawl:

medium fink link 1

«+

github 2

+

substock link N

+

Data Collection Pipeline

Medium Custom Article ETL (Arficles)

&

NoSQL Data

GitHub ETL (Repository)

Linkedin ETL (Posts)

RAG Feature Pipeline if

Fine-Tuning &

[TE

Vector DB

*Figure 3.1: LLM Twin's data collection ETL pipeline architecture*

We want to design an ETL pipeline that inputs a user and a list of links as input. Afterward, it crawls each link individually, standardizes the collected content, and saves it under that specific author in a MongoDB data warehouse.

-----

58 *Data Engineering*

---

Hence, the signature of the data collection pipeline will look as follows:

- **Input: A list of links and their associated user (the author)**
- **Output: A list of raw documents stored in the NoSQL data warehouse** We will use `user` and `author` interchangeably, as in most scenarios across the ETL pipeline, a user is the author of the extracted content. However, within the data warehouse, we have only a user collection. The ETL pipeline will detect the domain of each link, based on which it will call a specialized crawler. We implemented four different crawlers for three different data categories, as seen in *Figure 3.2. First, we will explore the three fundamental data categories we will work with across* the book. All our collected documents can be boiled down to an article, repository (or code), and post. It doesn't matter where the data comes from. We are primarily interested in the document's format. In most scenarios, we will have to process these data categories differently. Thus, we created a different domain entity for each, where each entity will have its class and collection in MongoDB. As we save the source URL within the document's metadata, we will still know its source and can reference it in our GenAI use cases.

![](image_p87_0.png)

Data Medium Blogs

Linkedin

Sources

Substack

Crawlers | Medium

Linkedin

Data Article

#### Categories Document

Repository

Document Post

Document

NoSQL Data Warehouse

*Figure 3.2: The relationship between the crawlers and the data categories*

-----

*Chapter 3* 59

---

Our codebase supports four different crawlers:

- **Medium crawler: Used to collect data from Medium. It outputs an article document. It** logs in to Medium and crawls the HTML of the article's link. Then, it extracts, cleans, and normalizes the text from the HTML and loads the standardized text of the article into the NoSQL data warehouse.
- **Custom article crawler: It performs similar steps to the Medium crawler but is a more** generic implementation for collecting articles from various sites. Thus, as it doesn't implement any particularities of any platform, it doesn't perform the login step and blindly gathers all the HTML from a particular link. This is enough for articles freely available online, which you can find on Substack and people's blogs. We will use this crawler as a safety net when the link's domain isn't associated with the other supported crawlers. For example, when providing a Substack link, it will default to the custom article crawler, but when providing a Medium URL, it will use the Medium crawler.
- **GitHub crawler: This collects data from GitHub. It outputs a repository document. It** clones the repository, parses the repository file tree, cleans and normalizes the files, and loads them to the database.
- **LinkedIn crawler: This is used to collect data from LinkedIn. It outputs multiple post** documents. It logs in to LinkedIn, navigates to the user's feed, and crawls all the user's latest posts. For each post, it extracts its HTML, cleans and normalizes it, and loads it to MongoDB.

In the next section, we will examine each crawler's implementation in detail. For now, note that each crawler accesses a specific platform or site in a particular way and extracts HTML from it. Afterward, all the crawlers parse the HTML, extract the text from it, and clean and normalize it so it can be stored in the data warehouse under the same interface. By reducing all the collected data to three data categories and not creating a new data category for every new data source, we can easily extend this architecture to multiple data sources with minimal effort. For example, if we want to start collecting data from X, we only have to implement a new crawler that outputs a post document, and that's it. The rest of the code will remain untouched. Otherwise, if we introduced the source dimension in the class and document structure, we would have to add code to all downstream layers to support any new data source. For example, we would have to implement a new document class for each new source and adapt the feature pipeline to support it.

-----

60 *Data Engineering*

---

For our proof of concept, crawling a few hundred documents is enough, but if we want to scale it to a real-world product, we would probably need more data sources to crawl from. LLMs are data-hungry. Thus, you need thousands of documents for ideal results instead of just a few hundred. But in many projects, it's an excellent strategy to implement an end-to-end project version that isn't the most accurate and iterate through it later. Thus, by using this architecture, you can easily add more data sources in future iterations to gather a larger dataset. More on LLM fine-tuning and dataset size will be covered in the next chapter.

**How is the ETL process connected to the feature pipeline? The feature pipeline ingests the raw**

data from the MongoDB data warehouse, cleans it further, processes it into features, and stores it in the Qdrant vector DB to make it accessible for the LLM training and inference pipelines. Chap*ter 4 provides more information on the feature pipeline. The ETL process is independent of the* feature pipeline. The two pipelines communicate with each other strictly through the MongoDB data warehouse. Thus, the data collection pipeline can write data for MongoDB, and the feature pipeline can read from it independently and on different schedules.

**Why did we use MongoDB as a data warehouse? Using a transactional database, such as Mon-**

goDB, as a data warehouse is uncommon. However, in our use case, we are working with small amounts of data, which MongoDB can handle. Even if we plan to compute statistics on top of our MongoDB collections, it will work fine at the scale of our LLM Twin's data (hundreds of documents). We picked MongoDB to store our raw data primarily because of the nature of our unstructured data: text crawled from the internet. By mainly working with unstructured text, selecting a NoSQL database that doesn't enforce a schema made our development easier and faster. Also, MongoDB is stable and easy to use. Their Python SDK is intuitive. They provide a Docker image that works out of the box locally and a cloud freemium tier that is perfect for proofs of concept, such as the LLM Twin. Thus, we can freely work with it locally and in the cloud. However, when working with big data (millions of documents or more), using a dedicated data warehouse such as Snowflake or BigQuery will be ideal. Now that we've understood the architecture of the LLM Twin's data collection pipeline, let's move on to its implementation.

-----

*Chapter 3* 61

---

###### Implementing the LLM Twin's data collection pipeline

As we presented in Chapter 2, the entry point to each pipeline from our LLM Twin project is a ZenML pipeline, which can be configured at runtime through YAML files and run through the ZenML ecosystem. Thus, let's start by looking into the ZenML `digital\_data\_etl` pipeline. You'll notice that this is the same pipeline we used as an example in Chapter 2 to illustrate ZenML. But this time, we will dig deeper into the implementation, explaining how the data collection works behind the scenes. After understanding how the pipeline works, we will explore the implementation of each crawler used to collect data from various sites and the MongoDB documents used to store and query data from the data warehouse.

###### ZenML pipeline and steps

In the code snippet below, we can see the implementation of the ZenML `digital\_data\_etl` pipeline, which inputs the user's full name and a list of links that will be crawled under that user (considered the author of the content extracted from those links). Within the function, we call two steps. In the first one, we look up the user in the database based on its full name. Then, we loop through all the links and crawl each independently. The pipeline's implementation is available

```python
in our repository at pipelines/digital_data_etl .py.
from zenml import pipeline
from steps.etl import crawl_links, get_or_create_user
```

```python
@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
user = get_or_create_user(user_full_name)
last_step = crawl_links(user=user, links=links)
return last_step.invocation_id
```

-----

62 *Data Engineering*

---

*Figure 3.3 shows a run of the* `digital\_data\_etl` pipeline on the ZenML dashboard. The next phase is to explore the `get\_or\_create\_user` and `crawl\_links` ZenML steps individually. The step implementation is available in our repository at `steps/etl` .

![](image_p91_0.png)

© digital\_data\_etl\_run\_2024\_09\_26.15\_38\_51 ©

+ 51 Run Insights o © Overview 3%

Configuration

[<}

© get\_or\_create\_user

Details

cc

stats © competed

g user

im\_engineering.domain.documents.Use. ToL

Author ©

Time

| © crawl_inks | Start | 2610972028, |
|---|---|---|
|  | End Time | 2610972028, 153851 |

crawled.links Soon

stack builtins list

*Figure 3.3: Example of a digital\_data\_etl pipeline run from ZenML's dashboard* We will start with the `get\_or\_create\_user` ZenML step. We begin by importing the necessary modules and functions used throughout the script.

| from loguru import logger from typing_extensions import Annotated from zenml import get_step_context, step from llm_engineering.application import utils from llm_engineering.domain.documents import UserDocument |
|---|
| Next, we define the function's signature, which takes a user's full name as input and retrieves an existing user or creates a new one in the MongoDB database if it doesn't exist: |
| @step def get_or_create_user(user_full_name: str) -> Annotated[UserDocument, "user"]: |

-----

*Chapter 3* 63

---

Using a utility function, we split the full name into first and last names. Then, we attempt to retrieve the user from the database or create a new one if it doesn't exist. We also retrieve the current step context and add metadata about the user to the output, which will be reflected in the metadata of the `user` ZenML output artifact:

```
logger.info(f"Getting or creating user: {user_full_name}")
```

| first_name, last_name = utils.split_user_full_name(user_full_name) user = UserDocument.get_or_create(first_name=first_name, last_ name=last_name) step_context = get_step_context() step_context.add_output_metadata(output_name="user", metadata=_get_ metadata(user_full_name, user)) return user |
|---|
| Additionally, we define a helper function called _get_metadata(), which builds a dictionary containing the query parameters and the retrieved user information, which will be added as metadata to the user artifact: |
| def _get_metadata(user_full_name: str, user: UserDocument) -> dict: return { "query": { "user_full_name": user_full_name, }, "retrieved": { "user_id": str(user.id), "first_name": user.first_name, "last_name": user.last_name, }, } |
| We will move on to the crawl_links ZenML step, which collects the data from the provided links. The code begins by importing essential modules and libraries for web crawling: |
| from urllib.parse import urlparse from loguru import logger |

-----

64 *Data Engineering*

---

| from tqdm import tqdm from typing_extensions import Annotated from zenml import get_step_context, step from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher from llm_engineering.domain.documents import UserDocument |
|---|
| Following the imports, the main function inputs a list of links written by a specific author. Within this function, a crawler dispatcher is initialized and configured to handle specific domains such as LinkedIn, Medium, and GitHub: |
| @step def crawl_links(user: UserDocument, links: list[str]) -> Annotated[list[str], "crawled_links"]: dispatcher = CrawlerDispatcher.build().register_linkedin().register_ medium().register_github() logger.info(f"Starting to crawl {len(links)} link(s).") |
| The function initializes variables to store the output metadata and count successful crawls. It then iterates over each link. It attempts to crawl and extract data for each link, updating the count of successful crawls and accumulating metadata about each URL: |
| metadata = {} successfull_crawls = 0 for link in tqdm(links): successfull_crawl, crawled_domain = _crawl_link(dispatcher, link, user) successfull_crawls += successfull_crawl metadata = _add_to_metadata(metadata, crawled_domain, successfull_ crawl) |
| After processing all links, the function attaches the accumulated metadata to the output artifact: |
| step_context = get_step_context() step_context.add_output_metadata(output_name="crawled_links", metadata=metadata) logger.info(f"Successfully crawled {successfull_crawls} / {len(links)} |

-----

*Chapter 3* 65

---

| links.") return links |
|---|
| The code includes a helper function that attempts to extract information from each link using the appropriate crawler based on the link's domain. It handles any exceptions that may occur during extraction and returns a tuple indicating the crawl's success and the link's domain: |
| def _crawl_link(dispatcher: CrawlerDispatcher, link: str, user: UserDocument) -> tuple[bool, str]: crawler = dispatcher.get_crawler(link) crawler_domain = urlparse(link).netloc try: crawler.extract(link=link, user=user) return (True, crawler_domain) except Exception as e: logger.error(f"An error occurred while crawling: {e!s}") return (False, crawler_domain) |
| Another helper function is provided to update the metadata dictionary with the results of each crawl: |
| def _add_to_metadata(metadata: dict, domain: str, successfull_crawl: bool) -> dict: if domain not in metadata: metadata[domain] = {} metadata[domain]["successful"] = metadata.get(domain, {}). get("successful", 0) + successfull_crawl metadata[domain]["total"] = metadata.get(domain, {}).get("total", 0) + 1 return metadata |

As seen in the abovementioned `\_crawl\_link()` function, the `CrawlerDispatcher` class knows what crawler to initialize based on each link's domain. The logic is then abstracted away under the crawler's `extract()` method. Let's zoom in on the `CrawlerDispatcher` class to understand how this works fully.

-----

66 *Data Engineering*

---

**The dispatcher: How do you instantiate the right crawler?**

The entry point to our crawling logic is the `CrawlerDispatcher` class. As illustrated in Figure *3.4, the dispatcher acts as the intermediate layer between the provided links and the crawlers. It* knows what crawler to associate with each URL. The `CrawlerDispatcher` class knows how to extract the domain of each link and initialize the proper crawler that collects the data from that site. For example, if it detects the `https://medium.` `com` domain when providing a link to an article, it will build an instance of the `MediumCrawler` used to crawl that particular platform. With that in mind, let's explore the implementation of

```
the CrawlerDispatcher class.
```

\\& All the crawling logic is available in the GitHub repository at `llm\_engineering/`

K

```
application/crawlers .
```

![](image_p95_0.png)

Links

pipe /linkedin.com/...

### hitps:/ /github.com/...

https://medium.com/...

Nye!

xN links hitps://linkedin.com/. Hitps://medium.com/.

hitps:/ com’.

/substack

Crawlers

*Figure 3.4: The relationship between the provided links, the CrawlerDispatcher, and the crawlers*

-----

*Chapter 3* 67

---

We begin by importing the necessary Python modules for URL handling and regex, along with importing our crawler classes:

| import re from urllib.parse import urlparse from loguru import logger from .base import BaseCrawler from .custom_article import CustomArticleCrawler from .github import GithubCrawler from .linkedin import LinkedInCrawler from .medium import MediumCrawler |
|---|
| The CrawlerDispatcher class is defined to manage and dispatch appropriate crawler instances based on given URLs and their domains. Its constructor initializes a registry to store the regis- tered crawlers. |
| class CrawlerDispatcher: def __init__(self) -> None: self._crawlers = {} |
| As we are using the builder creational pattern to instantiate and configure the dispatcher, we define a build() class method that returns an instance of the dispatcher: |
| @classmethod def build(cls) -> "CrawlerDispatcher": dispatcher = cls() return dispatcher |
| The dispatcher includes methods to register crawlers for specific platforms like Medium, Linke- dIn, and GitHub. These methods use a generic register() method under the hood to add each crawler to the registry. By returning self, we follow the builder creational pattern (more on the builder pattern: https://refactoring .guru/design-patterns/builder). We can chain mul- tiple register_*() methods when instantiating the dispatcher as follows: CrawlerDispatcher. build().register_linkedin().register_medium() . |
| def register_medium(self) -> "CrawlerDispatcher": self.register("https://medium.com", MediumCrawler) |

-----

68 *Data Engineering*

---

```python
return self
def register_linkedin(self) -> "CrawlerDispatcher":
self.register("https://linkedin.com", LinkedInCrawler)
```

```python
return self
def register_github(self) -> "CrawlerDispatcher":
self.register("https://github.com", GithubCrawler)
```

| return self |
|---|
| The generic register() method normalizes each domain to ensure its format is consistent be- fore it's added as a key to the self._crawlers registry of the dispatcher. This is a critical step, as we will use the key of the dictionary as the domain pattern to match future links with a crawler: |
| def register(self, domain: str, crawler: type[BaseCrawler]) -> None: parsed_domain = urlparse(domain) domain = parsed_domain.netloc self._crawlers[r"https://(www\\.)?{}/*".format(re.escape(domain))] = crawler |
| Finally, the get_crawler() method determines the appropriate crawler for a given URL by match- ing it against the registered domains. If no match is found, it logs a warning and defaults to using the CustomArticleCrawler. |
| def get_crawler(self, url: str) -> BaseCrawler: for pattern, crawler in self._crawlers.items(): if re.match(pattern, url): return crawler() else: logger.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.") return CustomArticleCrawler() |

The next step in understanding how the data collection pipeline works is analyzing each crawler individually.

-----

*Chapter 3* 69

---

###### The crawlers

Before exploring each crawler's implementation, we must present their base class, which defines a unified interface for all the crawlers. As shown in Figure 3.4, we can implement the dispatcher layer because each crawler follows the same signature. Each class implements the `extract()` method, allowing us to leverage OOP techniques such as polymorphism, where we can work with abstract objects without knowing their concrete subclass. For example, in the `\_crawl\_link()` function from the ZenML steps, we had the following code:

| crawler = dispatcher.get_crawler(link) crawler.extract(link=link, user=user) |
|---|
| Note how we called the extract() method without caring about what specific type of crawler we instantiated. To conclude, working with abstract interfaces ensures core reusability and ease of extension. Base classes Now, let's explore the BaseCrawler interface, which can be found in the repository at https:// github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/ application/crawlers/base .py. |
| from abc import ABC, abstractmethod class BaseCrawler(ABC): model: type[NoSQLBaseDocument] @abstractmethod def extract(self, link: str, **kwargs) -> None: ... |

As mentioned above, the interface defines an `extract()` method that takes as input a link. Also, it defines a model attribute at the class level that represents the data category document type used to save the extracted data into the MongoDB data warehouse. Doing so allows us to customize each subclass with different data categories while preserving the same attributes at the class level. We will soon explore the `NoSQLBaseDocument` class when digging into the document entities. We also extend the `BaseCrawler` class with a `BaseSeleniumCrawler` class, which implements reusable functionality that uses Selenium to crawl various sites, such as Medium or LinkedIn.

**Selenium is a tool for automating web browsers. It's used to interact with web pages program-**

matically (like logging into LinkedIn, navigating through profiles, etc.).

-----

70 *Data Engineering*

---

Selenium can programmatically control various browsers such as Chrome, Firefox, or Brave. For these specific platforms, we need Selenium to manipulate the browser programmatically to log in and scroll through the newsfeed or article before being able to extract the entire HTML. For other sites, where we don't have to go through the login step or can directly load the whole page, we can extract the HTML from a particular URL using more straightforward methods than Selenium.

4 For the Selenium-based crawlers to work, you must install Chrome on your machine

(or a Chromium-based browser such as Brave).

The code begins by setting up the necessary imports and configurations for web crawling using Selenium and the ChromeDriver initializer. The `chromedriver\_autoinstaller` ensures that the appropriate version of ChromeDriver is installed and added to the system path, maintaining compatibility with the installed version of your Google Chrome browser (or other Chromium-based browser). Selenium will use the ChromeDriver to communicate with the browser and open a headless session, where we can programmatically manipulate the browser to access various URLs, click on specific elements, such as buttons, or scroll through the newsfeed. Using the chromedriver\_autoinstaller, we ensure we always have the correct ChromeDriver version installed that matches our machine's Chrome browser version.

```python
import time
from tempfile import mkdtemp
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from llm_engineering.domain.documents import NoSQLBaseDocument
```

```
# Check if the current version of chromedriver exists
# and if it doesn't exist, download it automatically,
# then add chromedriver to path
chromedriver_autoinstaller.install()
```

Next, we define the `BaseSeleniumCrawler` class for use cases where we need Selenium to collect the data, such as collecting data from Medium or LinkedIn.

-----

*Chapter 3* 71

---

Its constructor initializes various Chrome options to optimize performance, enhance security, and ensure a headless browsing environment. These options disable unnecessary features like GPU rendering, extensions, and notifications, which can interfere with automated browsing. These are standard configurations when crawling in headless mode:

| class BaseSeleniumCrawler(BaseCrawler, ABC): def __init__(self, scroll_limit: int = 5) -> None: options = webdriver.ChromeOptions() options.add_argument("--no-sandbox") options.add_argument("--headless=new") options.add_argument("--disable-dev-shm-usage") options.add_argument("--log-level=3") options.add_argument("--disable-popup-blocking") options.add_argument("--disable-notifications") options.add_argument("--disable-extensions") options.add_argument("--disable-background-networking") options.add_argument("--ignore-certificate-errors") options.add_argument(f"--user-data-dir={mkdtemp()}") options.add_argument(f"--data-path={mkdtemp()}") options.add_argument(f"--disk-cache-dir={mkdtemp()}") options.add_argument("--remote-debugging-port=9226") |
|---|
| After configuring the Chrome options, the code allows subclasses to set any additional driver options by calling the set_extra_driver_options() method. It then initializes the scroll limit and creates a new instance of the Chrome driver with the specified options: |
| self.set_extra_driver_options(options) self.scroll_limit = scroll_limit self.driver = webdriver.Chrome( options=options, ) |
| The BaseSeleniumCrawler class includes placeholder methods for set_extra_driver_options() and login(), which subclasses can override to provide specific functionality. This ensures mod- ularity, as every platform has a different login page with a different HTML structure: |
| def set_extra_driver_options(self, options: Options) -> None: |

-----

72 *Data Engineering*

---

| Finally, the such as LinkedIn, up to a specified scroll limit. It scrolls to the bottom of the page, waits for new content to load, and repeats the process until it reaches the end of the page or the scroll limit is exceeded. This method is essential for feeds where the content appears as the user scrolls: | Finally, the such as LinkedIn, up to a specified scroll limit. It scrolls to the bottom of the page, waits for new content to load, and repeats the process until it reaches the end of the page or the scroll limit is exceeded. This method is essential for feeds where the content appears as the user scrolls: | Finally, the such as LinkedIn, up to a specified scroll limit. It scrolls to the bottom of the page, waits for new content to load, and repeats the process until it reaches the end of the page or the scroll limit is exceeded. This method is essential for feeds where the content appears as the user scrolls: | def Finally, the such as LinkedIn, up to a specified scroll limit. It scrolls to the bottom of the page, waits for new content to load, and repeats the process until it reaches the end of the page or the scroll limit is exceeded. This method is essential for feeds where the content appears as the user scrolls: | def Finally, the such as LinkedIn, up to a specified scroll limit. It scrolls to the bottom of the page, waits for new content to load, and repeats the process until it reaches the end of the page or the scroll limit is exceeded. This method is essential for feeds where the content appears as the user scrolls: | pass def login(self) -> None: pass scroll_page() method implements a scrolling mechanism to navigate through pages, such as LinkedIn, up to a specified scroll limit. It scrolls to the bottom of the page, waits for new content to load, and repeats the process until it reaches the end of the page or the scroll limit is exceeded. This method is essential for feeds where the content appears as the user scrolls: |
|---|---|---|---|---|---|
|  |  |  |  | def | scroll_page(self) -> None: """Scroll through the LinkedIn page based on the scroll limit.""" current_scroll = 0 last_height = self.driver.execute_script("return document.body. scrollHeight") while True: self.driver.execute_script("window.scrollTo(0, document.body. scrollHeight);") time.sleep(5) new_height = self.driver.execute_script("return document.body. scrollHeight") if new_height == last_height or (self.scroll_limit and current_scroll >= self.scroll_limit): break last_height = new_height current_scroll += 1 |
| We've implementation | understood of • • • | what the | the following | base specific | classes of our crawlers look like. Next, we will look into the crawlers: GitHubCrawler(BaseCrawler) CustomArticleCrawler(BaseCrawler) MediumCrawler(BaseSeleniumCrawler) |

You can find the implementation of the above crawlers in the GitHub repository at

```
of https://github.com/PacktPublishing/LLM-Engineers-Handbook/tree/main
/llm_engineering/application/crawlers .
```

-----

*Chapter 3* 73

---

###### GitHubCrawler class

The `GithubCrawler` class is designed to scrape GitHub repositories, extending the functionality of the BaseCrawler. We don't have to log in to GitHub through the browser, as we can leverage Git's clone functionality. Thus, we don't have to leverage any Selenium functionality. Upon initialization, it sets up a list of patterns to ignore standard files and directories found in GitHub repositories, such as .git, .toml, .lock, and .png, ensuring that unnecessary files are excluded from the scraping process:

| class GithubCrawler(BaseCrawler): model = RepositoryDocument def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None: super().__init__() self._ignore = ignore |
|---|
| Next, we implement the extract() method, where the crawler first checks if the repository has already been processed and stored in the database. If it exists, it exits the method to prevent storing duplicates: |
| def extract(self, link: str, **kwargs) -> None: old_model = self.model.find(link=link) if old_model is not None: logger.info(f"Repository already exists in the database: {link}") return |
| If the repository is new, the crawler extracts the repository name from the link. Then, it creates a temporary directory to clone the repository to ensure that the cloned repository is cleaned up from the local disk after it's processed: |
| logger.info(f"Starting scrapping GitHub repository: {link}") repo_name = link.rstrip("/").split("/")[-1] local_temp = tempfile.mkdtemp() |
| Within a try block, the crawler changes the current working directory to the temporary directory and executes the git clone command in a different process: |
| try: |

-----

74 *Data Engineering*

---

| os.chdir(local_temp) subprocess.run(["git", "clone", link]) |
|---|
| After successfully cloning the repository, the crawler constructs the path to the cloned repository. It initializes an empty dictionary used to aggregate the content of the files in a standardized way. It walks through the directory tree, skipping over any directories or files that match the ignore patterns. For each relevant file, it reads the content, removes any spaces, and stores it in the dic- tionary with the file path as the key: |
| repo_path = os.path.join(local_temp, os.listdir(local_temp)[0]) # tree = {} for root, _, files in os.walk(repo_path): dir = root.replace(repo_path, "").lstrip("/") if dir.startswith(self._ignore): continue for file in files: if file.endswith(self._ignore): continue file_path = os.path.join(dir, file) with open(os.path.join(root, file), "r", errors="ignore") as f: tree[file_path] = f.read().replace(" ", "") |
| It then creates a new instance of the RepositoryDocument model, populating it with the repos- itory content, name, link, platform information, and author details. The instance is then saved to MongoDB: |
| user = kwargs["user"] instance = self.model( content=tree, name=repo_name, link=link, platform="github", author_id=user.id, author_full_name=user.full_name, ) instance.save() |

-----

*Chapter 3* 75

---

Finally, whether the scraping succeeds or an exception occurs, the crawler ensures that the temporary directory is removed to clean up any resources used during the process:

| except Exception: raise finally: shutil.rmtree(local_temp) logger.info(f"Finished scrapping GitHub repository: {link}") |
|---|
| CustomArticleCrawler class The CustomArticleCrawler class takes a different approach to collecting data from the in- ternet. It leverages the AsyncHtmlLoader class to read the entire HTML from a link and the Html2TextTransformer class to extract the text from that HTML. Both classes are made available by the langchain_community Python package, as seen below, where we import all the necessary Python modules: |
| from urllib.parse import urlparse from langchain_community.document_loaders import AsyncHtmlLoader from langchain_community.document_transformers.html2text import Html2TextTransformer from loguru import logger from llm_engineering.domain.documents import ArticleDocument |
|  |

```python
from .base import BaseCrawler
```

Next, we define the `CustomArticleCrawler` class, which inherits from BaseCrawler. As before, we don't need to log in or use the scrolling functionality provided by Selenium. In the `extract` method, we first check if the article exists in the database to avoid duplicating content:

```python
class CustomArticleCrawler(BaseCrawler):
model = ArticleDocument
def extract(self, link: str, **kwargs) -> None:
old_model = self.model.find(link=link)
if old_model is not None:
```

-----

76 *Data Engineering*

---

| logger.info(f"Article already exists in the database: {link}") return |
|---|
| If the article doesn't exist, we proceed to scrape it. We use the AsyncHtmlLoader class to load the HTML from the provided link. After, we transform it into plain text using the Html2TextTransformer class, which returns a list of documents. We are only interested in the first document. As we dele- gate the whole logic to these two classes, we don't control how the content is extracted and parsed. That's why we used this class as a fallback system for domains where we don't have anything cus- tom implemented. These two classes follow the LangChain paradigm, which provides high-level functionality that works decently in most scenarios. It is fast to implement but hard to customize. That is one of the reasons why many developers avoid using LangChain in production use cases: |
| logger.info(f"Starting scrapping article: {link}") loader = AsyncHtmlLoader([link]) docs = loader.load() html2text = Html2TextTransformer() docs_transformed = html2text.transform_documents(docs) doc_transformed = docs_transformed[0] |
| We get the page content from the extracted document, plus relevant metadata such as the title, subtitle, content, and language: |
| content = { "Title": doc_transformed.metadata.get("title"), "Subtitle": doc_transformed.metadata.get("description"), "Content": doc_transformed.page_content, "language": doc_transformed.metadata.get("language"), } |
| Next, we parse the URL to determine the platform (or domain) from which the article was scraped: |
| parsed_url = urlparse(link) platform = parsed_url.netloc |
| We then create a new instance of the article model, populating it with the extracted content. Finally, we save this instance to the MongoDB data warehouse: |
| user = kwargs["user"] |

-----

*Chapter 3* 77

---

| instance = self.model( content=content, link=link, platform=platform, author_id=user.id, author_full_name=user.full_name, ) instance.save() logger.info(f"Finished scrapping custom article: {link}") |
|---|
| So far, we have seen how to crawl GitHub repositories and random sites using LangChain utility functions. Lastly, we must explore a crawler using Selenium to manipulate the browser program- matically. Thus, we will continue with the MediumCrawler implementation. MediumCrawler class The code begins by importing essential libraries and defining the MediumCrawler class, which inherits from BaseSeleniumCrawler: |
| from bs4 import BeautifulSoup from loguru import logger from llm_engineering.domain.documents import ArticleDocument |

```python
from .base import BaseSeleniumCrawler
class MediumCrawler(BaseSeleniumCrawler):
model = ArticleDocument
```

Within the `MediumCrawler` class, we leverage the `set\_extra\_driver\_options()` method to extend the default driver options used by Selenium:

```python
def set_extra_driver_options(self, options) -> None:
options.add_argument(r"--profile-directory=Profile 2")
```

The `extract()` method implements the core functionality, first checking whether the article exists in the database to prevent duplicate entries.

-----

78 *Data Engineering*

---

If the article is new, the method proceeds to navigate to the article's link and scroll through the page to ensure all content is loaded:

```python
def extract(self, link: str, **kwargs) -> None:
old_model = self.model.find(link=link)
if old_model is not None:
logger.info(f"Article already exists in the database: {link}")
return
logger.info(f"Starting scrapping Medium article: {link}")
```

| self.driver.get(link) self.scroll_page() |
|---|
| After fully loading the page, the method uses BeautifulSoup to parse the HTML content and extract the article's title, subtitle, and full text. BeautifulSoup is a popular Python library for web scraping and parsing HTML or XML documents. Thus, we used it to extract all the HTML elements we needed from the HTML accessed with Selenium. Finally, we aggregate everything into a dictionary: |
| soup = BeautifulSoup(self.driver.page_source, "html.parser") title = soup.find_all("h1", class_="pw-post-title") subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph") data = { "Title": title[0].string if title else None, "Subtitle": subtitle[0].string if subtitle else None, "Content": soup.get_text(), } |
| Finally,the method closes the WebDriver to free up resources. It then creates a new ArticleDocument instance, populates it with the extracted content and user information provided via kwargs, and saves it to the database: |
| self.driver.close() user = kwargs["user"] instance = self.model( |

-----

*Chapter 3* 79

---

```
platform="medium",
content=data,
link=link,
author_id=user.id,
author_full_name=user.full_name,
)
instance.save()
```

```
logger.info(f"Successfully scraped and saved article: {link}")
```

With that, we conclude the `MediumCrawler` implementation. The LinkedIn crawler follows a similar pattern to the Medium one, where it uses Selenium to log in and access the feed of a user's latest posts. Then, it extracts the posts and scrolls through the feed to load the next page until a limit is hit. You can check the full implementation in our repository at `https://github.`

```
com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/application/
crawlers/linkedin .py.
```

With the rise of LLMs, collecting data from the internet has become a critical step in many real-world AI applications. Hence, more high-level tools have appeared in the Python ecosystem, such as Scrapy (https://github.com/scrapy/scrapy), which crawls websites and extracts structured data from their pages, and Crawl4AI (https://github.com/unclecode/crawl4ai), which is highly specialized in crawling data for LLMs and AI applications. In this section, we've looked at implementing three types of crawlers: one that leverages the `git` executable in a subprocess to clone GitHub repositories, one that uses LangChain utilities to extract the HTML of a single web page, and one that leverages Selenium for more complex scenarios where we have to navigate through the login page, scroll the article to load the entire HTML, and extract it into text format. The last step is understanding how the document classes we've used across the chapter, such as the ArticleDocument, work.

###### The NoSQL data warehouse documents

We had to implement three document classes to structure our data categories. These classes define the specific attributes we require for a document, such as the content, author, and source link. It is best practice to structure your data in classes instead of dictionaries, as the attributes we expect for each item are more verbose, reducing run errors. For example, when accessing a value from a Python dictionary, we can never be sure it is present or its type is current. By wrapping our data items with classes, we can ensure each attribute is as expected.

-----

80 *Data Engineering*

---

By leveraging Python packages such as Pydantic, we have out-of-the-box type validation, which ensures consistency in our datasets. Thus, we modeled the data categories as the following document classes, which we already used in the code up until point:

```
• ArticleDocument class
• PostDocument class
• RepositoryDocument class
```

These are not simple Python data classes or Pydantic models. They support read and write operations on top of the MongoDB data warehouse. To inject the read-and-write functionality into all the document classes without repeating any code, we used the Object-Document Mapping (ODM) software pattern, which is based on the object-relational mapping (ORM) pattern. Thus, let's first explore ORM, then move to ODM, and, finally, dig into our custom ODM implementation and document classes.

###### The ORM and ODM software patterns

Before we talk about software patterns, let's see what ORM is. It's a technique that lets you query and manipulate data from a database using an object-oriented paradigm. Instead of writing SQL or API-specific queries, you encapsulate all the complexity under an ORM class that knows how to handle all the database operations, most commonly CRUD operations. Thus, working with ORM removes the need to handle the database operations manually and reduces the need to write boilerplate code manually. An ORM interacts with a SQL database, such as PostgreSQL or MySQL. Most modern Python applications use ORMs when interacting with the database. Even though SQL is still a popular choice in the data world, you rarely see raw SQL queries in Python backend components. The most popular Python ORM is SQLAlchemy (https://www `.sqlalchemy .org/).` Also, with the rise of FastAPI, SQLModel is (https://github.com/fastapi/sqlmodel) a common choice, which is a wrapper over SQLAlchemy that makes the integration easier with FastAPI. For example, using SQLAlchemy, we defined a `User` ORM with the ID and name fields. The `User` ORM is mapped to the `users` table within the SQL database. Thus, when we create a new user and commit it to the database, it is automatically saved to the `users` table. The same applies to all the CRUD operations on top of the `User` class.

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
Base = declarative_base()
```

-----

*Chapter 3* 81

---

| Using the line of SQL. Note that an ORM usually supports all CRUD operations. Here is a code snippet that shows how to save an instance of the User ORM to a SQLite database: | # Using the line of SQL. Note that an ORM usually supports all CRUD operations. Here is a code snippet that shows how to save an instance of the User ORM to a SQLite database: | # Define a Using the line of SQL. Note that an ORM usually supports all CRUD operations. Here is a code snippet that shows how to save an instance of the User ORM to a SQLite database: | Define a class User(Base): __tablename__ = id name Using the line of SQL. Note that an ORM usually supports all CRUD operations. Here is a code snippet that shows how to save an instance of the User ORM to a SQLite database: | Define a class User(Base): __tablename__ = id name Using the User line of SQL. Note that an ORM usually supports all CRUD operations. Here is a code snippet that shows how to save an instance of the User ORM to a SQLite database: | Define a class that maps to the users table. class User(Base): __tablename__ = "users" = Column(Integer, primary_key=True) name = Column(String) User ORM, we can quickly insert or query users directly from Python without writing a line of SQL. Note that an ORM usually supports all CRUD operations. Here is a code snippet that shows how to save an instance of the User ORM to a SQLite database: |
|---|---|---|---|---|---|
|  |  | # | engine = Session = session = # Add a new_user = | Create a | create_engine("sqlite:///:memory:") Base.metadata.create_all(engine) session used to interact with the database. sessionmaker(bind=engine) Session() new user. User(name="Alice") session.add(new_user) session.commit() |
| Also, | this | is | how | we | can query a user from the users SQLite table: |
|  |  | user if |  | = user: print(f"User print(f"User | session.query(User).first() ID: {user.id}") name: {user.name}") |

& Find the entire script and how to run it in the GitHub repository at code\_snippets/03\_

```
orm .py.
```

The ODM pattern is extremely similar to ORM, but instead of working with SQL databases and tables, it works with NoSQL databases (such as MongoDB) and unstructured collections. As we work with NoSQL databases, the data structure is centered on collections, which store JSON-like documents rather than rows in tables.

-----

82 *Data Engineering*

---

To conclude, ODM simplifies working with document-based NoSQL databases and maps object-oriented code to JSON-like documents. We will implement a light ODM module on top of MongoDB to fully understand how ODM works.

###### Implementing the ODM class

This section will explore how to implement an ODM class from scratch. This is an excellent exercise to learn how ODM works and sharpen our skills in writing modular and reusable Python classes. Hence, we will implement a base ODM class called NoSQLBaseDocument, from which all the other documents will inherit to interact with the MongoDB data warehouse.

& The class can be found in our repository at llm\_engineering/domain/base/nosql.

```
py.
```

The code starts by importing essential modules and setting up the database connection. Through the `\_database` variable, we establish a connection to the database specified in the settings, which is by default called twin:

```python
import uuid
from abc import ABC
from typing import Generic, Type, TypeVar
from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import errors
from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings
```

```
_database = connection.get_database(settings.DATABASE_NAME)
```

Next, we define a type variable `T` bound to the `NoSQLBaseDocument` class. The variable leverages Python's generic module, allowing us to generalize the class's types. For example, when we implement the `ArticleDocument` class, which will inherit from the `NoSQLBaseDocument` class, all the instances where `T` was used will be replaced with the `ArticleDocument` type when analyzing the signature of functions (more on Python generics: `https://realpython .com/python312-typing).`

-----

*Chapter 3* 83

---

The `NoSQLBaseDocument` class is then declared as an abstract base class inheriting from Pydantic's BaseModel, Python's Generic (which provides the functionality described earlier), and `ABC` (making the class abstract) classes. This class serves as the foundational ODM class:

| T = TypeVar("T", bound="NoSQLBaseDocument") class NoSQLBaseDocument(BaseModel, Generic[T], ABC): |
|---|
| Within the NoSQLBaseDocument class, an id field is defined as a UUID4, with a default factory generating a unique UUID. The class also implements the __eq__ and __hash__ methods to allow instances to be compared and used in hashed collections like sets or as dictionary keys based on their unique id attribute: |
| id: UUID4 = Field(default_factory=uuid.uuid4) def __eq__(self, value: object) -> bool: if not isinstance(value, self.__class__): return False return self.id == value.id def __hash__(self) -> int: return hash(self.id) |
| The class provides methods for converting between MongoDB documents and class instances. The from_mongo() class method transforms a dictionary retrieved from MongoDB into an instance of the class. The to_mongo() instance method converts the model instance into a dictionary suitable for MongoDB insertion: |
| @classmethod def from_mongo(cls: Type[T], data: dict) -> T: if not data: raise ValueError("Data is empty.") id = data.pop("_id") return cls(**dict(data, id=id)) def to_mongo(self: T, **kwargs) -> dict: |

-----

84 *Data Engineering*

---

| exclude_unset = kwargs.pop("exclude_unset", False) by_alias = kwargs.pop("by_alias", True) parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_ alias, **kwargs) if "_id" not in parsed and "id" in parsed: parsed["_id"] = str(parsed.pop("id")) for key, value in parsed.items(): if isinstance(value, uuid.UUID): parsed[key] = str(value) return parsed |
|---|
| The save() method allows an instance of the model to be inserted into a MongoDB collection. It retrieves the appropriate collection, converts the instance into a MongoDB-compatible document leveraging the to_mongo() method described above, and attempts to insert it into the database, handling any write errors that may occur: |
| def save(self: T, **kwargs) -> T \| None: collection = _database[self.get_collection_name()] try: collection.insert_one(self.to_mongo(**kwargs)) return self except errors.WriteError: logger.exception("Failed to insert document.") return None |
| The get_or_create() class method attempts to find a document in the database matching the provided filter options. If a matching document is found, it is converted into an instance of the class. If not, a new instance is created with the filter options as its initial data and saved to the database: |
| @classmethod def get_or_create(cls: Type[T], **filter_options) -> T: collection = _database[cls.get_collection_name()] try: |

-----

*Chapter 3* 85

---

| instance = collection.find_one(filter_options) if instance: return cls.from_mongo(instance) new_instance = cls(**filter_options) new_instance = new_instance.save() return new_instance except errors.OperationFailure: logger.exception(f"Failed to retrieve document with filter options: {filter_options}") raise |
|---|
| The bulk_insert() class method allows multiple documents to be inserted into the database at once: |
| @classmethod def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool: collection = _database[cls.get_collection_name()] try: collection.insert_many([doc.to_mongo(**kwargs) for doc in documents]) return True except (errors.WriteError, errors.BulkWriteError): logger.error(f"Failed to insert documents of type {cls.__name__}") return False |
| The find() class method searches for a single document in the database that matches the given filter options: |
| @classmethod def find(cls: Type[T], **filter_options) -> T \| None: collection = _database[cls.get_collection_name()] try: instance = collection.find_one(filter_options) |

-----

86 *Data Engineering*

---

| if instance: return cls.from_mongo(instance) return None except errors.OperationFailure: logger.error("Failed to retrieve document.") return None |
|---|
| Similarly, the bulk_find() class method retrieves multiple documents matching the filter options. It converts each retrieved MongoDB document into a model instance, collecting them into a list: |
| @classmethod def bulk_find(cls: Type[T], **filter_options) -> list[T]: collection = _database[cls.get_collection_name()] try: instances = collection.find(filter_options) return [document for instance in instances if (document := cls. from_mongo(instance)) is not None] except errors.OperationFailure: logger.error("Failed to retrieve document.") return [] |
| Finally, the get_collection_name() class method determines the name of the MongoDB collec- tion associated with the class. It expects the class to have a nested Settings class with a name at- tribute specifying the collection name. If this configuration is missing, an ImproperlyConfigured exception will be raised specifying that the subclass should define a nested Settings class: |
| @classmethod def get_collection_name(cls: Type[T]) -> str: if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"): raise ImproperlyConfigured( "Document should define an Settings configuration class with the name of the collection." ) return cls.Settings.name |

-----

*Chapter 3* 87

---

We can configure each subclass using the nested `Settings` class, such as defining the collection name, or anything else specific to that subclass. Within the Python ecosystem, there is an ODM implementation on top of MongoDB, called mongoengine, which you can find on GitHub. It follows a pattern similar to ours but more comprehensive. We implemented it by ourselves, as it was an excellent exercise to practice writing modular and generic code following best OOP principles, which are essential for implementing production-level code.

###### Data categories and user document classes

The last piece of the puzzle is to see the implementation of the subclasses that inherit from the `NoSQLBaseDocument` base class. These are the concrete classes that define our data categories. You've seen these classes used across the chapter when working with articles, repositories, and posts within the crawler classes. We begin by importing the essential Python modules and the ODM base class:

```python
from abc import ABC
from typing import Optional
from pydantic import UUID4, Field
from .base import NoSQLBaseDocument
from .types import DataCategory
```

We define an `enum` class, where we centralize all our data category types. These variables will act as constants in configuring all our ODM classes throughout the book.

©

The class can be found in the repository at `llm\_engineering/domain/types .py.`

```python
from enum import StrEnum
class DataCategory(StrEnum):
PROMPT = "prompt"
QUERIES = "queries"
INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
```

-----

88 *Data Engineering*

---

| INSTRUCT_DATASET = "instruct_dataset" PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples" PREFERENCE_DATASET = "preference_dataset" POSTS = "posts" ARTICLES = "articles" REPOSITORIES = "repositories" |
|---|
| The Document class is introduced as an abstract base model for other documents on top of the NoSQLBaseDocument ODM class. It includes common attributes like content, platform, and author details, providing a standardized structure for documents that will inherit from it: |
| class Document(NoSQLBaseDocument, ABC): content: dict platform: str author_id: UUID4 = Field(alias="author_id") author_full_name: str = Field(alias="author_full_name") |
| Finally, specific document types are defined by extending the Document class. The RepositoryDocument, PostDocument, and ArticleDocument classes represent different catego- ries of data, each with unique fields and settings that specify their respective collection names in the database: |
| class RepositoryDocument(Document): name: str link: str class Settings: name = DataCategory.REPOSITORIES class PostDocument(Document): image: Optional[str] = None link: str \| None = None class Settings: name = DataCategory.POSTS |

-----

*Chapter 3* 89

---

| class ArticleDocument(Document): link: str class Settings: name = DataCategory.ARTICLES |
|---|
| Finally, we define the UserDocument class, which is used to store and query all the users from the LLM Twin project: |
| class UserDocument(NoSQLBaseDocument): first_name: str last_name: str class Settings: name = "users" @property def full_name(self): return f"{self.first_name} {self.last_name}" |

By implementing the `NoSQLBaseDocument` ODM class, we had to focus solely on the fields and specific functionality of each document or domain entity. All the CRUD functionality is delegated to the parent class. Also, by leveraging Pydantic to define the fields, we have out-of-the-box type validation. For example, when creating an instance of the `ArticleDocument` class, if the provided link is `None` or not a string, it will throw an error signaling that the data is invalid. With that, we've finished implementing our data collection pipeline, starting with the ZenML components. Then, we looked into the implementation of the crawlers and, finally, wrapped it up with the ODM class and data category documents. The last step is to run the data collection pipeline and ingest raw data into the MongoDB data warehouse.

##### Gathering raw data into the data warehouse

ZenML orchestrates the data collection pipeline. Thus, leveraging ZenML, the data collection pipeline can be run manually, scheduled, or triggered by specific events. Here, we will show you how to run it manually, while we will discuss the other scenarios in Chapter 11 when digging deeper into MLOps.

-----

90 *Data Engineering*

---

We configured a different pipeline run for each author. We provided a ZenML configuration file for Paul Iusztin's or Maxime Labonne's data. To call the data collection pipeline to collect Maxime's data, for example, you can run the following CLI command:

| poetry poe run-digital-data-etl-maxime |
|---|
| That will call the pipeline with the following ZenML YAML configuration file: |
| parameters: user_full_name: Maxime Labonne # [First Name(s)] [Last Name] links: # Personal Blog - https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31. html - https://mlabonne.github.io/blog/posts/2024-07-15_The_Rise_of_ Agentic_Data_Generation.html # Substack - https://maximelabonne.substack.com/p/uncensor-any-llm-with- abliteration-d30148b7d43e - https://maximelabonne.substack.com/p/create-mixtures-of-experts- with-mergekit-11b318c99562 - https://maximelabonne.substack.com/p/merge-large-language-models- with-mergekit-2118fb392b54 … # More Substack links |

In Figure 3.3 earlier, we saw the pipeline's run DAG and details in ZenML's dashboard. Meanwhile, *Figure 3.5 shows the* `user` output artifact generated by this data collection pipeline. You can inspect the query `user\_full\_name` and the retrieved `user` from the MongoDB database, for which we collected the links in this specific run.

-----

*Chapter 3* 91

---

![](image_p120_0.png)

data\_etl\_.. 0

dataetl digital Overview

J digital\_data\_etl\_run\_2024\_10\_02\_12 overview Metadata [i] Visualization

### c © get

103 bytes

8 user

lim\_engineering.domain.d| query

|

user Maxine Labonne

©

oe

|

## |

lastname ene

i

0

userid

*Figure 3.5: Example of the user output artifact after running the data collection pipeline using Maxime's configuration file*

Also, in Figure 3.6, you can observe the `crawled\_links` output artifact, which lists all the domains from which we collected data, the total number of links crawled for each domain, and the number of successfully collected links.

-----

92 *Data Engineering*

---

We want to highlight again the power of these artifacts, as they trace each pipeline's results and metadata, making it extremely easy to monitor and debug each pipeline run individually.

![](image_p121_0.png)

= (i)

dataetl Overview

crawled\_links © (J digital\_data\_etl\_run\_2024\_10\_02\_12

() = [i]

overview Metadata Visualization

+ Uncategorized

c ©)

nan 2

storage\_size 2.43KB user

lim\_engineering.domain.d

successful 2

© crawl\_lin

0 crawled\_li total 2 maximelabonne.substack.com

successful

24

2

*Figure 3.6: Example of the crawled\_links output artifact after running the data collection pipeline using Maxime's configuration file*

Now, we can download the `crawled\_links` artifact anywhere in our code by running the following code, where the `ID` of the artifact can be found in ZenML and is unique for every artifact version:

```python
from zenml.client import Client
artifact = Client().get_artifact_version('8349ce09-0693-4e28-8fa2-
20f82c76ddec')
loaded_artifact = artifact.load()
```

-----

*Chapter 3* 93

---

For example, we can easily run the same data collection pipeline but with Paul Iusztin's YAML configuration, listed below:

| parameters: user_full_name: Paul Iusztin # [First Name(s)] [Last Name] links: # Medium - https://medium.com/decodingml/an-end-to-end-framework-for- production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f - https://medium.com/decodingml/a-real-time-retrieval-system-for-rag- on-social-media-data-9cc01d50a2a0 - https://medium.com/decodingml/sota-python-streaming-pipelines-for- fine-tuning-llms-and-rag-in-real-time-82eb07795b87 … # More Medium links # Substack - https://decodingml.substack.com/p/real-time-feature-pipelines- with?r=1ttoeh - https://decodingml.substack.com/p/building-ml-systems-the-right- way?r=1ttoeh - https://decodingml.substack.com/p/reduce-your-pytorchs-code- latency?r=1ttoeh … # More Substack links |
|---|
| To run the pipeline using Paul's configuration, we call the following poe command: |
| poetry poe run-digital-data-etl-paul |
| That, under the hood, calls the following CLI command that references Paul's config file: |
| poetry run python -m tools.run --run-etl --no-cache --etl-config-filename digital_data_etl_paul_iusztin.yaml |
| You can find all the configs in the repository in the configs/ directory. Also, using poe, we con- figured a command that calls the data collection pipeline for all the supported authors: |
| poetry poe run-digital-data-etl |
| We can easily query the MongoDB data warehouse using our ODM classes. For example, let's query all the articles collected for Paul Iusztin: |
| from llm_engineering.domain.documents import ArticleDocument, UserDocument |

-----

94 *Data Engineering*

---

```
user = UserDocument.get_or_create(first_name="Paul", last_name="Iusztin")
articles = ArticleDocument.bulk_find(author_id=str(user.id))
```

| print(f"User ID: {user.id}") print(f"User name: {user.first_name} {user.last_name}") print(f"Number of articles: {len(articles)}") print("First article link:", articles[0].link) |
|---|
| The output of the code from above is: |
| User ID: 900fec95-d621-4315-84c6-52e5229e0b96 User name: Paul Iusztin Number of articles: 50 First article link: https://medium.com/decodingml/an-end-to-end-framework- for-production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f |

With only two lines of code, we can query and filter our MongoDB data warehouse using any ODM defined within our project. Also, to ensure that your data collection pipeline works as expected, you can search your MongoDB collections using your IDE's MongoDB plugin, which you must install separately. For example,

```
you can use this plugin for VSCode: https://www.mongodb .com/products/tools/vs-code. For
```

other IDEs, you can use similar plugins or external NoSQL visualization tools. After connecting to the MongoDB visualization tool, you can connect to our local database using the following

```
URI: mongodb://llm_engineering:llm_engineering@127.0.0.1:27017 . For a cloud MongoDB
```

cluster, you must change the URI, which we will explore in Chapter 11. And just like that, you've learned how to run the data collection pipeline with different ZenML configs and how to visualize the output artifacts of each run. We also looked at how to query the data warehouse for a particular data category and author. Thus, we've finalized our data engineering chapter and can move to the conclusion.

###### Troubleshooting

The raw data stored in the MongoDB database is central to all future steps. Thus, if you haven't successfully run the code from this chapter due to any issues with the crawlers, this section provides solutions for fixing potential issues to allow you to move forward.

-----

*Chapter 3* 95

---

###### Selenium issues

It is a well-known issue that running Selenium can cause problems due to issues with the browser driver, such as the ChromeDriver. Thus, if the crawlers that use Selenium, such as the

```
MediumCrawler, fail due to problems with your ChromeDriver, you can easily bypass this by
```

commenting out the Medium links added to the data collection YAML configs. To do so, go to the `configs/` directory and find all the YAML files that start with digital\_data\_etl\_\*, such as `digital\_data\_etl\_maxime\_labonne.yaml` . Open them and comment on all the Medium-related URLs, as illustrated in Figure 3.7. You can leave out the Substack or personal blog URLs as these use the CustomArticleCrawler, which is not dependent on Selenium.

/ configs

![](image_p124_0.png)

LLM-Engineers-Handbook / ©

@ iusatinpaul docs: Comment out Medium urls

| | +

Code Blame 62 ines 61 loc) 4.62 K8

| 1 2 | settings: docker: |  |
|---|---|---|
| 3 | parent_inage: True | 992382797823.dk.ecr.eu-central-1.anazonaws. con/zennl-rlwlcs: latest |
| 5 | orchestrator. sagenaker: |  |
| 6 | synchronous: | false |

7

8 parameters: 9 user\_full\_name: Paul Iusztin # [First Nane(s)] [Last Name]

Links!

n # Mediu (only articles that under the wall work)

- are not paid

2 # con/d

n /a-real-tine-retrieval-systen-for-rag-on-social-nedia-data-Scco1d50a2a0

1

= i

15 # /the-4-advanced-rag-algor 1119942

| 1 | # = | 82299 |
|---|---|---|
| I | # | Substack |

on/p

| 1 | subs tack, | real-t ine-feature-pipelines-with?ralttoeh |
|---|---|---|
| 1 | subs tack. |  |
| 2 | subs tack. | reduce-your-pytorchs—code-latency if ied?r=1ttoeh |

subs tack.

*Figure 3.7: Fix Selenium issues when crawling raw data*

###### Import our backed-up data

If nothing works, there is the possibility of populating the MongoDB database with your backed-

```
up data saved under the data/data_warehouse_raw_data directory. This will allow you to
```

proceed to the fine-tuning and inference sections without running the data collection ETL code. To import all the data within this directory, run:

```
poetry poe run-import-data-warehouse-from-json
```

-----

96 *Data Engineering*

---

After running the CLI command from above, you will have a one-to-one replica of the dataset we used while developing the code. To ensure the import is completed successfully, you should have 88 articles and 3 users in your MongoDB database.

##### Summary

In this chapter, we've learned how to design and build the data collection pipeline for the LLM Twin use case. Instead of relying on static datasets, we collected our custom data to mimic real-world situations, preparing us for real-world challenges in building AI systems. First, we examined the architecture of LLM Twin's data collection pipeline, which functions as an ETL process. Next, we started digging into the pipeline implementation. We began by understanding how we can orchestrate the pipeline using ZenML. Then, we looked into the crawler implementation. We learned how to crawl data in three ways: using CLI commands in subprocesses or using utility functions from LangChain or Selenium to build custom logic that programmatically manipulates the browser. Finally, we looked into how to build our own ODM class, which we used to define our document class hierarchy, which contains entities such as articles, posts, and repositories. At the end of the chapter, we learned how to run ZenML pipelines with different YAML configuration files and explore the results in the dashboard. We also saw how to interact with the MongoDB data warehouse through the ODM classes. In the next chapter, we will cover the key steps of the RAG feature pipeline, including chunking and embedding documents, ingesting these documents into a vector DB, and applying pre-retrieval optimizations to improve performance. We will also set up the necessary infrastructure programmatically using Pulumi and conclude by deploying the RAG ingestion pipeline to AWS.

##### References

- Breuss, M. (2023, July 26). Beautiful Soup: Build a Web Scraper With Python. `https://`

```
realpython.com/beautiful-soup-web-scraper-python/
```

- David, D. (2024,July 8). Guide to Web Scraping with Selenium in 2024. Bright Data. `https://`

```
brightdata.com/blog/how-tos/using-selenium-for-web-scraping
```

- Hjelle, G. A. (2023, October 21). Python 3.12 Preview: Static Typing Improvements. `https://`

```
realpython.com/python312-typing/
```

- *ORM Quick Start - SQLAlchemy 2.0 documentation. (n.d.).* `https://docs.sqlalchemy.`

```
org/en/20/orm/quickstart.html
```

-----

*Chapter 3* 97

---

- Ramos, L. P. (2023, August 4). Python and MongoDB: Connecting to NoSQL Databases.

```
https://realpython.com/introduction-to-mongodb-and-python/
```

- Refactoring.Guru. (2024, January 1). Builder. `https://refactoring.guru/design-`

```
patterns/builder
```

- *What is ETL? A complete guide. (n.d.). Qlik.* `https://www.qlik.com/us/etl`

##### Join our book's Discord space

Join our community's Discord space for discussions with the authors and other readers:

```
https://packt.link/llmeng
```

-----

```text

```

-----

4

# RAG Feature Pipeline

**Retrieval-augmented generation (RAG) is fundamental in most generative AI applications. RAG's**

core responsibility is to inject custom data into the large language model (LLM) to perform a given action (e.g., summarize, reformulate, and extract the injected data). You often want to use the LLM on data it wasn't trained on (e.g., private or new data). As fine-tuning an LLM is a highly costly operation, RAG is a compelling strategy that bypasses the need for constant fine-tuning to access that new data. We will start this chapter with a theoretical part that focuses on the fundamentals of RAG and how it works. We will then walk you through all the components of a naïve RAG system: chunking, embedding, and vector DBs. Ultimately, we will present various optimizations used for an advanced RAG system. Then, we will continue exploring LLM Twin's RAG feature pipeline architecture. At this step, we will apply all the theoretical aspects we discussed at the beginning of the chapter. Finally, we will go through a practical example by implementing the LLM Twin's RAG feature pipeline based on the system design described throughout the book. The main sections of this chapter are:

- Understanding RAG
- An overview of advanced RAG
- Exploring the LLM Twin's RAG feature pipeline architecture
- Implementing the LLM Twin's RAG feature pipeline By the end of this chapter, you will have a clear and comprehensive understanding of what RAG is and how it is applied to our LLM Twin use case.

-----

100 *RAG Feature Pipeline*

---

##### Understanding RAG

RAG enhances the accuracy and reliability of generative AI models with information fetched from external sources. It is a technique complementary to the internal knowledge of the LLMs. Before going into the details, let's understand what RAG stands for:

- **Retrieval: Search for relevant data**
- **Augmented: Add the data as context to the prompt**
- **Generation: Use the augmented prompt with an LLM for generation** Any LLM is bound to understand the data it was trained on, sometimes called parameterized knowledge. Thus, even if the LLM can perfectly answer what happened in the past, it won't have access to the newest data or any other external sources on which it wasn't trained. Let's take the most powerful model from OpenAI as an example, which, in the summer of 2024, is GPT-4o. The model is trained on data up to October 2023. Thus, if we ask what happened during the 2020 pandemic, it can be answered perfectly due to its parametrized knowledge. However, it will not know the answer if we ask about the 2024 European Football Championship results due to its bounded parametrized knowledge. Another scenario is that it will start confidently hallucinating and provide a faulty answer. RAG overcomes these two limitations of LLMs. It provides access to external or latest data and prevents hallucinations, enhancing generative AI models' accuracy and reliability.

###### Why use RAG?

We briefly explained the importance of using RAG in generative AI applications earlier. Now, we will dig deeper into the "why," following which we will focus on what a naïve RAG framework looks like. For now, to get an intuition about RAG, you have to know that when using RAG, we inject the necessary information into the prompt to answer the initial user question. After that, we pass the augmented prompt to the LLM for the final answer. Now, the LLM will use the additional context to answer the user question. There are two fundamental problems that RAG solves:

- Hallucinations
- Old or private information

-----

*Chapter 4* 101

---

###### Hallucinations

If a chatbot without RAG is asked a question about something it wasn't trained on, there is a high chance that it will give you a confident answer about something that isn't true. Let's take the 2024 European Football Championship as an example. If the model is trained up to October 2023 and we ask it something about the tournament, it will most likely come up with a random answer that is hard to differentiate between reality and truth. Even if the LLM doesn't hallucinate all the time, it raises concerns about the trustworthiness of its answers. Thus, we must ask ourselves: "When can we trust the LLM's answers?" and "How can we evaluate if the answers are correct?". By introducing RAG, we enforce the LLM to always answer solely based on the introduced context. The LLM will act as the reasoning engine, while the additional information added through RAG will act as the single source of truth for the generated answer. By doing so, we can quickly evaluate if the LLM's answer is based on the external data or not.

###### Old information

Any LLM is trained or fine-tuned on a subset of the total world knowledge dataset. This is due to three main issues:

- **Private data: You cannot train your model on data you don't own or have the right to use.**
- **New data: New data is generated every second. Thus, you would have to constantly train**

your LLM to keep up.

- **Costs: Training or fine-tuning an LLM is an extremely costly operation. Hence, it is not**

feasible to do it on an hourly or daily basis. RAG solves these issues, as you no longer have to constantly fine-tune your LLM on new data (or even private data). Directly injecting the necessary data to respond to user questions into the prompts that are fed to the LLM is enough to generate correct and valuable answers. To conclude, RAG is key for a robust and flexible generative AI system. But how do we inject the right data into the prompt based on the user's questions? We will dig into the technical aspects of RAG in the next sections.

###### The vanilla RAG framework

Every RAG system is similar at its roots. We will first focus on understanding RAG in its simplest form. Later, we will gradually introduce more advanced RAG techniques to improve the system's accuracy. Note that we will use vanilla and naive RAG interchangeably to avoid repetition.

-----

102 *RAG Feature Pipeline*

---

A RAG system is composed of three main modules independent of each other:

- **Ingestion pipeline: A batch or streaming pipeline used to populate the vector DB**
- **Retrieval pipeline: A module that queries the vector DB and retrieves relevant entries to**

the user's input

- **Generation pipeline: The layer that uses the retrieved data to augment the prompt and**

an LLM to generate answers As these three components are classes or services of their own, we will dig into each separately. But for now, let's try to answer the question "How are these three modules connected?". Here is a very simplistic overview:

1. On the backend side, the ingestion pipeline runs either on a schedule or constantly to

populate the vector DB with external data.

2. On the client side, the user asks a question.
3. The question is passed to the retrieval module, which preprocesses the user's input and

queries the vector DB.

4. The generation pipelines use a prompt template, user input, and retrieved context to

create the prompt.

5. The prompt is passed to an LLM to generate the answer.
6. The answer is shown to the user.

-----

*Chapter 4* 103

---

![](image_p132_0.png)

xN Embedded

Chunks & Metadata

Vector DB

User Input

Embedding

Prompt Elements dding

del

##### [

User

*Figure 4.1: Vanilla RAG architecture*

You must implement RAG in your generative AI application when you need access to any type of external information. For example, when implementing a financial assistant, you most likely need access to the latest news, reports, and prices before providing valuable answers. Or, if you build a traveling recommender, you must retrieve and parse a list of potential attractions, restaurants, and activities. At training time, LLMs don't have access to your specific data, so you will often have to implement a RAG strategy in your generative AI project. Now, let's dig into the ingestion, retrieval, and generation pipelines.

-----

104 *RAG Feature Pipeline*

---

###### Ingestion pipeline

The RAG ingestion pipeline extracts raw documents from various data sources (e.g., data warehouse, data lake, web pages, etc.). Then, it cleans, chunks (splits into smaller sections), and embeds the documents. Ultimately, it loads the embedded chunks into a vector DB (or other similar vector storage). Thus, the RAG ingestion pipeline is split into the following:

- The data extraction module gathers all the necessary data from various sources such as

DBs, APIs, or web pages. This module is highly dependent on your data. It can be as easy as querying your data warehouse or something more complex such as crawling Wikipedia.

- A cleaning layer standardizes and removes unwanted characters from the extracted data.

For example, you must remove all invalid characters from your input text, such as non-AS- CII and bold and italic characters. Another popular cleaning strategy is to replace URLs with placeholders. However, your cleaning strategy will vary depending on your data source and embedding model.

- The chunking module splits the cleaned documents into smaller ones. As we want to

pass the document's content to an embedding model, this is necessary to ensure it doesn't exceed the model's input maximum size. Also, chunking is required to separate specific regions that are semantically related. For example, when chunking a book's chapter, the most optimal way is to group similar paragraphs into the same section or chunk. By doing so, at the retrieval time, you will add only the essential data to the prompt.

- The embedding component uses an embedding model to take the chunk's content (text,

images, audio, etc.) and project it into a dense vector packed with semantic value-more on embeddings in the What are embeddings? section below.

- The loading module takes the embedded chunks along with a metadata document. The

metadata will contain essential information such as the embedded content, the URL to the source of the chunk, and when the content was published on the web. The embedding is used as an index to query similar chunks, while the metadata is used to access the information added to augment the prompt. At this point, we have a RAG ingestion pipeline that takes raw documents as input, processes them, and populates a vector DB. The next step is to retrieve relevant data from the vector store correctly.

-----

*Chapter 4* 105

---

###### Retrieval pipeline

The retrieval components take the user's input (text, image, audio, etc.), embed it, and query the vector DB for similar vectors to the user's input. The primary function of the retrieval step is to project the user's input into the same vector space as the embeddings used as an index in the vector DB. This allows us to find the top K's most similar entries by comparing the embeddings from the vector storage with the user's input vector. These entries then serve as content to augment the prompt that is passed to the LLM to generate the answer. You must use a distance metric to compare two vectors, such as the Euclidean or Manhattan distance. But the most popular one is the cosine distance, which is equal to 1 minus the cosine of the angle between two vectors, as follows:

|| || ||||

It ranges from `-1` to 1, with a value of `-1` when vectors A and B are in opposite directions, `0` if they are orthogonal, and `1` if they point in the same direction. Most of the time, the cosine distance works well in non-linear complex vector spaces. However, it is essential to notice that choosing the proper distance between two vectors depends on your data and the embedding model you use. One critical factor to highlight is that the user's input and embeddings must be in the same vector space. Otherwise, you cannot compute the distance between them. To do so, it is essential to preprocess the user input in the same way you processed the raw documents in the RAG ingestion pipeline. This means you must clean, chunk (if necessary), and embed the user's input using the same functions, models, and hyperparameters. This is similar to how you have to preprocess the data into features in the same way between training and inference; otherwise, the inference will yield inaccurate results-a phenomenon also known as the training-serving skew.

###### Generation pipeline

The last step of the RAG system is to take the user's input, retrieve data, pass it to an LLM, and generate a valuable answer.

-----

106 *RAG Feature Pipeline*

---

The final prompt results from a system and prompt template populated with the user's query and retrieved context. You might have a single prompt template or multiple prompt templates, depending on your application. Usually, all the prompt engineering is done at the prompt template level. Below, you can see a dummy example of what a generic system and prompt template look like and how they are used together with the retrieval logic and the LLM to generate the final answer:

```
system_template = """
You are a helpful assistant who answers all the user's questions politely.
"""
prompt_template = """
Answer the user's question using only the provided context. If you cannot
answer using the context, respond with "I don't know."
Context: {context}
User question: {user_question}
"""
user_question = "<your_question>"
retrieved_context = retrieve(user_question)
prompt = f"{system_template}\n"
prompt += prompt_template.format(context=retrieved_context, user_
question=user_question)
answer = llm(prompt)
```

As the prompt templates evolve, each change should be tracked and versioned using machine

**learning operations (MLOps) best practices. Thus, during training or inference time, you always**

know that a given answer was generated by a specific version of the LLM and prompt template(s). You can do this through Git, store the prompt templates in a DB, or use specific prompt management tools such as LangFuse.

As we've seen in the retrieval pipeline, some critical aspects that directly impact the accuracy of your RAG system are the embeddings of the external data, usually stored in vector DBs, the embedding of the user's query, and how we can find similarities between the two using functions such as the cosine distance. To better understand this part of the RAG algorithm, let's zoom in on what embeddings are and how they are computed.

-----

*Chapter 4* 107

---

###### What are embeddings?

Imagine you're trying to teach a computer to understand the world. Embeddings are like a particular translator that turns these things into a numerical code. This code isn't random, though, because similar words or items end up with codes that are close to each other. It's like a map where words with similar meanings are clustered together. With that in mind, a more theoretical definition is that embeddings are dense numerical representations of objects encoded as vectors in a continuous vector space, such as words, images, or items in a recommendation system. This transformation helps capture the semantic meaning and relationships between the objects. For instance, in natural language processing (NLP), embeddings translate words into vectors where semantically similar words are positioned closely together in the vector space.

![](image_p136_0.png)

Embedding

Model

House

*Figure 4.2: What are embeddings?*

A popular method is visualizing the embeddings to understand and evaluate their geometrical relationship. As the embeddings often have more than 2 or 3 dimensions, usually between 64 and 2048, you must project them again to 2D or 3D.

-----

108 *RAG Feature Pipeline*

---

For example, you can use UMAP (https://umap-learn `.readthedocs.io/en/latest/index.`

```
html), a dimensionality reduction method well known for keeping the geometrical properties
```

between the points when projecting the embeddings to 2D or 3D. Another popular algorithm for dimensionality reduction when visualizing vectors is t-SNE (https://scikit-learn `.org/stable/`

```
modules/generated/sklearn.manifold.TSNE .html). However, compared to UMAP, it is more
```

stochastic and doesn't preserve the topological relationships between the points.

A dimensionality reduction algorithm, such as PCA, UMAP, and t-SNE, is a mathematical technique used to reduce the number of input variables or features in a dataset while preserving the data's essential patterns, structure, and relationships. The

&

goal is to transform high-dimensional data into a lower-dimensional form, making it easier to visualize, interpret, and process while minimizing the loss of important information. These methods help to address the "curse of dimensionality," improve computational efficiency, and often enhance the performance of ML algorithms.

![](image_p137_0.png)

nneighbors=15, min dlst=0.1

*Figure 4.3: Visualize embeddings using UMAP (Source: UMAP's documentation)*

-----

*Chapter 4* 109

---

###### Why embeddings are so powerful

Firstly, ML models work only with numerical values. This is not a problem when working with tabular data, as the data is often in numerical form or can easily be processed into numbers. Embeddings come in handy when we want to feed words, images, or audio data into models. For instance, when working with transformer models, you tokenize all your text input, where each token has an embedding associated with it. The beauty of this process lies in its simplicity; the input to the transformer is a sequence of embeddings, which can be easily and confidently interpreted by the dense layers of the neural network. Based on this example, you can use embeddings to encode any categorical variable and feed it to an ML model. But why not use other simple methods, such as one-hot encoding? When working with categorical variables with high cardinality, such as language vocabularies, you will suffer from the curse of dimensionality when using other classical methods. For example, if your vocabulary has 10,000 tokens, then only one token will have a length of 10,000 after applying one-hot encoding. If the input sequence has N tokens, that will become N \* 10,000 input parameters. If N >= 100, often, when inputting text, the input is too large to be usable. Another issue with other classical methods that don't suffer from the curse of dimensionality, such as hashing, is that you lose the semantic relationships between the vectors.

-----

110 *RAG Feature Pipeline*

---

**One-hot encoding is a technique that converts categorical variables into a binary**

matrix representation. Each category is represented as a unique binary vector. For each categorical variable, a binary vector is created with a length equal to the number of unique categories, where all values are zero except for the index corresponding to the specific category, which is set to one. The method preserves all information about the categories. It is simple and interpretable. However, a significant disadvantage is that it can lead to a high-dimensional feature space if the categorical variable has many unique values, making the method impractical.

**Feature hashing, also known as hashing encoding or the "hash trick," is a technique**

used to convert categorical variables into numerical features by applying a hash

W function to the category values. Compared to one-hot encoding, the method is not

bound to the number of unique categories, but it reduces the dimensionality of the feature space by mapping categories into a fixed number of bins or buckets. Thus, it reduces the dimensionality of the feature space, which is particularly useful when dealing with high-cardinality categorical variables. This makes it efficient in terms of memory usage and computational time. However, there is a risk of collisions, where different categories might map to the same bin, leading to a loss of information. The mapping makes the method uninterpretable. Also, it is difficult to understand the relationship between the original categories and the hashed features. Embeddings help us encode categorical variables while controlling the output vector's dimension. They also use ingenious ways to condense information into a lower dimension space than naive hashing tricks. Secondly, embedding your input reduces the size of its dimension and condenses all of its semantic meaning into a dense vector. This is an extremely popular technique when working with images, where a CNN encoder module maps the high-dimensional meaning into an embedding, which is later processed by a CNN decoder that performs the classification or regression steps. The following image shows a typical CNN layout. Imagine tiny squares within each layer. Those are the "receptive fields." Each square feeds information to a single neuron in the previous layer. As you move through the network, two key things are happening:

- **Shrinking the picture: Special "subsampling" operations make the layers smaller, fo-**

cusing on essential details.

- **Learning features: "Convolution" operations, on the other hand, actually increase the**

layer size as the network learns more complex features from the image.

-----

*Chapter 4* 111

---

Finally, a fully connected layer at the end takes all this processed information and transforms it into the final vector embedding, a numerical image representation.

![](image_p140_0.png)

Feature maps

Input

rs

Subsampling ~~ Fully connected

Convolutions Subsampling

*Figure 4.4: Creating embeddings from an image using a CNN (Image source)*

| The preceding image is sourced from Wikimedia Commons (https://commons . |
|---|
| © wikimedia .org/wiki/File:Typical_cnn .png) and licensed under the Creative |
| Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0: https:// |
| creativecommons.org/licenses/by-sa/4.0/deed .en). |

###### How are embeddings created?

Embeddings are created by deep learning models that understand the context and semantics of your input and project it into a continuous vector space. Various deep learning models can be used to create embeddings, varying by the data input type. Thus, it is fundamental to understand your data and what you need from it before picking an embedding model. For example, when working with text data, one of the early methods used to create embeddings for your vocabulary is Word2Vec and GloVe. These are still popular methods used today for simpler applications. Another popular method is to use encoder-only transformers, such as BERT, or other methods from its family, such as RoBERTa. These models leverage the encoder of the transformer architecture to smartly project your input into a dense vector space that can later be used as embeddings. To quickly compute the embeddings in Python, you can conveniently leverage the Sentence Transformers Python package (also available in Hugging Face's transformer package). This tool provides a user-friendly interface, making the embedding process straightforward and efficient.

-----

112 *RAG Feature Pipeline*

---

In the code snippet below, you can see how we loaded a model from SentenceTransformer, computed the embeddings for three sentences, and, ultimately, computed the cosine similarity between them. The similarity between one sentence and itself is always 1. Also, the similarity between the first and second sentences is approximately 0, as the sentences have nothing in common. In contrast, the value between the first and third one is higher as there is some overlapping context:

```python
from sentence_transformers import SentenceTransformer
```

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
"The dog sits outside waiting for a treat.",
"I am going swimming.",
"The dog is swimming."
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# Output: [3, 384]
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# Output:
# tensor([[ 1.0000, -0.0389, 0.2692],
# [-0.0389, 1.0000, 0.3837],
# [ 0.2692, 0.3837, 1.0000]])
#
# similarities[0, 0] = The similarity between the first sentence and
itself.
# similarities[0, 1] = The similarity between the first and second
sentence.
# similarities[2, 1] = The similarity between the third and second
sentence.
```

The source code for the preceding snippet can be found at `https://github.com/PacktPublishing/`

```
LLM-Engineering/blob/main/code_snippets/08_text_embeddings .py.
```

-----

*Chapter 4* 113

---

& The examples in the embeddings section can be run within the virtual environment

used across the book, as it contains all the required dependencies.

The best-performing embedding model can change with time and your specific use case. You can find particular models on the Massive Text Embedding Benchmark (MTEB) on Hugging Face. Depending on your needs, you can consider the best-performing model, the one with the best accuracy, or the one with the smallest memory footprint. This decision is solely based on your requirements (e.g., accuracy and hardware). However, Hugging Face and SentenceTransformer make switching between different models straightforward. Thus, you can always experiment with various options. When working with images, you can embed them using convolutional neural networks (CNNs). Popular CNN networks are based on the ResNet architecture. However, we can't directly use image embedding techniques for audio recordings. Instead, we can create a visual representation of the audio, such as a spectrogram, and then apply image embedding models to those visuals. This allows us to capture the essence of images and sounds in a way computers can understand. By leveraging models like CLIP, you can practically embed a piece of text and an image in the same vector space. This allows you to find similar images using a sentence as input, or the other way around, demonstrating the practicality of CLIP. In the following code snippet, we use CLIP to encode a crazy cat image and three sentences. Ultimately, we use cosine similarity to compute the resemblance between the picture and the sentences:

```python
from io import BytesIO
```

```python
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer
```

```
response = requests.get(
"https://github.com/PacktPublishing/LLM-Engineering/blob/main/images/
crazy_cat.jpg?raw=true"
)
```

-----

114 *RAG Feature Pipeline*

---

```
image = Image.open(BytesIO(response.content))
model = SentenceTransformer("clip-ViT-B-32")
img_emb = model.encode(image)
text_emb = model.encode(
["A crazy cat smiling.",
"A white and brown cat with a yellow bandana.",
"A man eating in the garden."]
)
print(text_emb.shape) # noqa
# Output: (3, 512)
similarity_scores = model.similarity(img_emb, text_emb)
print(similarity_scores) # noqa
# Output: tensor([[0.3068, 0.3300, 0.1719]])
The source code can be found at https://github.com/PacktPublishing/LLM-Engineering/
blob/main/code_snippets/08_text_image_embeddings .py.
```

Here, we provided a small introduction to how embeddings can be computed. The realm of specific implementations is vast, but what is important to know is that embeddings can be computed for most digital data categories, such as words, sentences, documents, images, videos, and graphs. It's crucial to grasp that you must use specialized models when you need to compute the distance between two different data categories, such as the distance between the vector of a sentence and of an image. These models are designed to project both data types into the same vector space, such as CLIP, ensuring accurate distance computation.

###### Applications of embeddings

Due to the generative AI revolution, which uses RAG, embeddings have become extremely popular in information retrieval tasks, such as semantic search for text, code, images, and audio, and long-term memory of agents. But before generative AI, embeddings were already heavily used in:

- Representing categorical variables (e.g., vocabulary tokens) that are fed to an ML model
- Recommender systems by encoding the users and items and finding their relationship
- Clustering and outlier detection
- Data visualization by using algorithms such as UMAP

-----

*Chapter 4* 115

---

- Classification by using the embeddings as features
- Zero-shot classification by comparing the embedding of each class and picking the most

similar one The last step to fully understanding how RAG works is to examine vector DBs and how they leverage embeddings to retrieve data.

###### More on vector DBs

Vector DBs are specialized DBs designed to efficiently store, index, and retrieve vector embeddings. Traditional scalar-based DBs struggle with the complexity of vector data, making vector DBs crucial for tasks like real-time semantic search. While standalone vector indices like FAISS are effective for similarity search, they lack vector DBs' comprehensive data management capabilities. Vector DBs support CRUD operations, metadata filtering, scalability, real-time updates, backups, ecosystem integration, and robust data security, making them more suited for production environments than standalone indices.

###### How does a vector DB work?

Think of how you usually search a DB. You type in something specific, and the system spits out the exact match. That's how traditional DBs work. Vector DBs are different. Instead of perfect matches, we look for the closest neighbors of the query vector. Under the hood, a vector DB uses

**approximate nearest neighbor (ANN) algorithms to find these close neighbors.**

While ANN algorithms don't return the top matches for a given search, standard nearest neighbor algorithms are too slow to work in practice. Also, it is shown empirically that using only approximations of the top matches for a given input query works well enough. Thus, the trade-off between accuracy and latency ultimately favors ANN algorithms. This is a typical workflow of a vector DB:

1. **Indexing vectors: Vectors are indexed using data structures optimized for high-dimen-** sional data. Common indexing techniques include hierarchical navigable small world (HNSW), random projection, product quantization (PQ), and locality-sensitive hashing (LSH).
2. **Querying for similarity: During a search, the DB queries the indexed vectors to find those**

most similar to the input vector. This process involves comparing vectors based on similarity measures such as cosine similarity, Euclidean distance, or dot product. Each has unique advantages and is suitable for different use cases.

-----

116 *RAG Feature Pipeline*

---

3. **Post-processing results: After identifying potential matches, the results undergo post-pro-**

cessing to refine accuracy. This step ensures that the most relevant vectors are returned to the user. Vector DBs can filter results based on metadata before or after the vector search. Both approaches have trade-offs in terms of performance and accuracy. The query also depends on the metadata (along with the vector index), so it contains a metadata index user for filtering operations.

###### Algorithms for creating the vector index

Vector DBs use various algorithms to create the vector index and manage searching data efficiently:

- **Random projection: Random projection reduces the dimensionality of vectors by project-**

ing them into a lower-dimensional space using a random matrix. This technique preserves the relative distances between vectors, facilitating faster searches.

- **PQ: PQ compresses vectors by dividing them into smaller sub-vectors and then quantizing**

these sub-vectors into representative codes. This reduces memory usage and speeds up similarity searches.

- **LSH: LSH maps similar vectors into buckets. This method enables fast approximate near-**

est neighbor searches by focusing on a subset of the data, reducing the computational complexity.

- **HNSW: HNSW constructs a multi-layer graph where each node represents a set of vectors.**

Similar nodes are connected, allowing the algorithm to navigate the graph and find the nearest neighbors efficiently. These algorithms enable vector DBs to efficiently handle complex and large-scale data, making them a perfect fit for a variety of AI and ML applications.

###### DB operations

Vector DBs also share common characteristics with standard DBs to ensure high performance, fault tolerance, and ease of management in production environments. Key operations include:

- **Sharding and replication: Data is partitioned (sharded) across multiple nodes to ensure**

scalability and high availability. Data replication across nodes helps maintain data integrity and availability in case of node failures.

-----

*Chapter 4* 117

---

- **Monitoring: Continuous monitoring of DB performance, including query latency and re-**

source usage (RAM, CPU, disk), helps maintain optimal operations and identify potential issues before they impact the system.

- **Access control: Implementing robust access control mechanisms ensures that only au-**

thorized users can access and modify data. This includes role-based access controls and other security protocols to protect sensitive information.

- **Backups: Regular DB backups are critical for disaster recovery. Automated backup pro-**

cesses ensure that data can be restored to a previous state in case of corruption or loss.

##### An overview of advanced RAG

The vanilla RAG framework we just presented doesn't address many fundamental aspects that impact the quality of the retrieval and answer generation, such as:

- Are the retrieved documents relevant to the user's question?
- Is the retrieved context enough to answer the user's question?
- Is there any redundant information that only adds noise to the augmented prompt?
- Does the latency of the retrieval step match our requirements?
- What do we do if we can't generate a valid answer using the retrieved information? From the questions above, we can draw two conclusions. The first one is that we need a robust evaluation module for our RAG system that can quantify and measure the quality of the retrieved data and generate answers relative to the user's question. We will discuss this topic in more detail in Chapter 9. The second conclusion is that we must improve our RAG framework to address the retrieval limitations directly in the algorithm. These improvements are known as advanced RAG. The vanilla RAG design can be optimized at three different stages:
- **Pre-retrieval: This stage focuses on how to structure and preprocess your data for data**

indexing optimizations as well as query optimizations.

- **Retrieval: This stage revolves around improving the embedding models and metadata**

filtering to improve the vector search step.

-----

118 *RAG Feature Pipeline*

---

- **Post-retrieval: This stage mainly targets different ways to filter out noise from the retrieved** documents and compress the prompt before feeding it to an LLM for answer generation.

![](image_p147_0.png)

RAG Ingestion Pipeline Embedding

Mod

Retrieval xN Embedded

Chunks & Metadata

Vector

xK Retrieved Chunks

DB

Embedding

Prompt Elements Model

*Figure 4.5: The three stages of advanced RAG*

This section is not meant to be an exhaustive list of all the advanced RAG methods available. The goal is to build an intuition about what can be optimized. We will use only examples based on text data, but the principles of advanced RAG remain the same regardless of the data category. Now, let's zoom in on all three components.

-----

*Chapter 4* 119

---

###### Pre-retrieval

The pre-retrieval steps are performed in two different ways:

- **Data indexing: It is part of the RAG ingestion pipeline. It is mainly implemented within**

the cleaning or chunking modules to preprocess the data for better indexing.

- **Query optimization: The algorithm is performed directly on the user's query before em-**

bedding it and retrieving the chunks from the vector DB. As we index our data using embeddings that semantically represent the content of a chunked document, most of the data indexing techniques focus on better preprocessing and structuring the data to improve retrieval efficiency, such as:

- **Sliding window: The sliding window technique introduces overlap between text chunks,**

ensuring that important context near chunk boundaries is retained, which enhances retrieval accuracy. This is particularly beneficial in domains like legal documents, scientific papers, customer support logs, and medical records, where critical information often spans multiple sections. The embedding is computed on the chunk along with the overlapping portion. Hence, the sliding window improves the system's ability to retrieve relevant and coherent information by maintaining context across boundaries.

- **Enhancing data granularity: This involves data cleaning techniques like removing irrel-**

evant details, verifying factual accuracy, and updating outdated information. A clean and accurate dataset allows for sharper retrieval.

- **Metadata: Adding metadata tags like dates, URLs, external IDs, or chapter markers helps**

filter results efficiently during retrieval.

- **Optimizing index structures: It is based on different data index methods, such as various**

chunk sizes and multi-indexing strategies.

- **Small-to-big: The algorithm decouples the chunks used for retrieval and the context used**

in the prompt for the final answer generation. The algorithm uses a small sequence of text to compute the embedding while preserving the sequence itself and a wider window around it in the metadata. Thus, using smaller chunks enhances the retrieval's accuracy, while the larger context adds more contextual information to the LLM.

-----

120 *RAG Feature Pipeline*

---

The intuition behind this is that if we use the whole text for computing the embedding, we might introduce too much noise, or the text could contain multiple topics, which results in a poor overall semantic representation of the embedding.

![](image_p149_0.png)

Vector Search

oot |

Text to SQL

RestAPI Calls

*Figure 4.6: Query routing*

On the query optimization side, we can leverage techniques such as query routing, query rewriting, and query expansion to refine the retrieved information for the LLM further:

- **Query routing: Based on the user's input, we might have to interact with different cate-**

gories of data and query each category differently. Query rooting is used to decide what action to take based on the user's input, similar to if/else statements. Still, the decisions are made solely using natural language instead of logical statements.

-----

*Chapter 4* 121

---

As illustrated in Figure 4.6, let's assume that, based on the user's input, to do RAG, we can retrieve additional context from a vector DB using vector search queries, a standard SQL DB by translating the user query to an SQL command, or the internet by leveraging REST API calls. The query router can also detect whether a context is required, helping us avoid making redundant calls to external data storage. Also, a query router can be used to pick the best prompt template for a given input. For example, in the LLM Twin use case, depending on whether the user wants an article paragraph, a post, or a code snippet, you need different prompt templates to optimize the creation process. The routing usually uses an LLM to decide what route to take or embeddings by picking the path with the most similar vectors. To summarize, query routing is identical to an if/else statement but much more versatile as it works directly with natural language.

- **Query rewriting: Sometimes, the user's initial query might not perfectly align with the**

way your data is structured. Query rewriting tackles this by reformulating the question to match the indexed information better. This can involve techniques like:

- **Paraphrasing: Rephrasing the user's query while preserving its meaning (e.g.,**

"What are the causes of climate change?" could be rewritten as "Factors contributing to global warming").

- **Synonym substitution: Replacing less common words with synonyms to broaden**

the search scope (e.g., " joyful" could be rewritten as "happy").

- **Sub-queries: For longer queries, we can break them down into multiple shorter** and more focused sub-queries. This can help the retrieval stage identify relevant documents more precisely.
- **Hypothetical document embeddings (HyDE): This technique involves having an LLM**

create a hypothetical response to the query. Then, both the original query and the LLM's response are fed into the retrieval stage.

- **Query expansion: This approach aims to enrich the user's question by adding additional**

terms or concepts, resulting in different perspectives of the same initial question. For example, when searching for "disease," you can leverage synonyms and related terms associated with the original query words and also include "illnesses" or "ailments."

- **Self-query: The core idea is to map unstructured queries into structured ones. An LLM**

identifies key entities, events, and relationships within the input text. These identities are used as filtering parameters to reduce the vector search space (e.g., identify cities within the query, for example, "Paris," and add it to your filter to reduce your vector search space).

-----

122 *RAG Feature Pipeline*

---

Both data indexing and query optimization pre-retrieval optimization techniques depend highly on your data type, structure, and source. Thus, as with any data processing pipeline, no method always works, as every use case has its own particularities and gotchas. Optimizing your pre-retrieval RAG layer is experimental. Thus, what is essential is to try multiple methods (such as the ones enumerated in this section), reiterate, and observe what works best.

###### Retrieval

The retrieval step can be optimized in two fundamental ways:

- **Improving the embedding models used in the RAG ingestion pipeline to encode the**

chunked documents and, at inference time, transform the user's input.

- **Leveraging the DB's filter and search features. This step will be used solely at inference**

time when you have to retrieve the most similar chunks based on user input. Both strategies are aligned with our ultimate goal: to enhance the vector search step by leveraging the semantic similarity between the query and the indexed data. When improving the embedding models, you usually have to fine-tune the pre-trained embedding models to tailor them to specific jargon and nuances of your domain, especially for areas with evolving terminology or rare terms. Instead of fine-tuning the embedding model, you can leverage instructor models (https://

```
huggingface .co/hkunlp/instructor-xl) to guide the embedding generation process with an
```

instruction/prompt aimed at your domain. Tailoring your embedding network to your data using such a model can be a good option, as fine-tuning a model consumes more computing and human resources. In the code snippet below, you can see an example of an Instructor model that embeds article titles about AI:

```python
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR("hkunlp/instructor-base")
sentence = "RAG Fundamentals First"
```

```
instruction = "Represent the title of an article about AI:"
```

-----

*Chapter 4* 123

---

|  |  | # | embeddings |  | = model.encode([[instruction, sentence]]) print(embeddings.shape) # noqa Output: (1, 768) |
|---|---|---|---|---|---|
| The | source | code | can | be | found at https://github.com/PacktPublishing/LLM-Engineering/ blob/main/code_snippets/08_instructor_embeddings .py. |

To run the instructor code, you have to create a different virtual environment and

a

activate it:

```
python3 -m venv instructor_venv && source instructor_venv/bin/activate
```

&

And install the required Python dependencies:

```
pip install sentence-transformers==2.2.2 InstructorEmbedding==1.0.1
```

On the other side of the spectrum, here is how you can improve your retrieval by leveraging classic filter and search DB features:

- **Hybrid search: This is a vector and keyword-based search blend. Keyword-based search**

excels at identifying documents containing specific keywords. When your task demands pinpoint accuracy and the retrieved information must include exact keyword matches, hybrid search shines. Vector search, while powerful, can sometimes struggle with finding exact matches, but it excels at finding more general semantic similarities. You leverage both keyword matching and semantic similarities by combining the two methods. You have a parameter, usually called alpha, that controls the weight between the two methods. The algorithm has two independent searches, which are later normalized and unified.

- **Filtered vector search: This type of search leverages the metadata index to filter for specific**

keywords within the metadata. It differs from a hybrid search in that you retrieve the data once using only the vector index and perform the filtering step before or after the vector search to reduce your search space.

-----

124 *RAG Feature Pipeline*

---

In practice, on the retrieval side, you usually start with filtered vector search or hybrid search, as they are fairly quick to implement. This approach gives you the flexibility to adjust your strategy based on performance. If the results are not as expected, you can always fine-tune your embedding model.

###### Post-retrieval

The post-retrieval optimizations are solely performed on the retrieved data to ensure that the LLM's performance is not compromised by issues such as limited context windows or noisy data. This is because the retrieved context can sometimes be too large or contain irrelevant information, both of which can distract the LLM. Two popular methods performed at the post-retrieval step are:

- **Prompt compression: Eliminate unnecessary details while keeping the essence of the data.**
- **Re-ranking: Use a cross-encoder ML model to give a matching score between the user's** input and every retrieved chunk. The retrieved items are sorted based on this score. Only the top N results are kept as the most relevant. As you can see in Figure 4.7, this works because the re-ranking model can find more complex relationships between the user input and some content than a simple similarity search. However, we can't apply this model at the initial retrieval step because it is costly. That is why a popular strategy is to retrieve the data using a similarity distance between the embeddings and refine the retrieved information using a re-raking model, as illustrated in Figure 4.8.

-----

*Chapter 4* 125

---

![](image_p154_0.png)

Bi-Encoder Cross-Encoder

®

Cosine-Similarity post

*Figure 4.7: Bi-encoder (the standard embedding model) versus cross-encoder*

The abovementioned techniques are far from an exhaustive list of all potential solutions. We used them as examples to get an intuition on what you can (and should) optimize at each step in your RAG workflow. The truth is that these techniques can vary tremendously by the type of data you work with.

-----

126 *RAG Feature Pipeline*

---

For example, if you work with multi-modal data such as text and images, most of the techniques from earlier won't work as they are designed for text only.

![](image_p155_0.png)

SN

Query Vector DB

Pre-process

Retrieved chunks

Top N

*Figure 4.8: The re-ranking algorithm* To summarize, the primary goal of these optimizations is to enhance the RAG algorithm at three key stages: pre-retrieval, retrieval, and post-retrieval. This involves preprocessing data for improved vector indexing, adjusting user queries for more accurate searches, enhancing the embedding model, utilizing classic filtering DB operations, and removing noisy data. By keeping these goals in mind, you can effectively optimize your RAG workflow for data processing and retrieval

-----

*Chapter 4* 127

---

##### Exploring the LLM Twin's RAG feature pipeline architecture

Now that you have a strong intuition and understanding of RAG and its workings, we will continue exploring our particular LLM Twin use case. The goal is to provide a hands-on end-to-end example to solidify the theory presented in this chapter. Any RAG system is split into two independent components:

- The ingestion pipeline takes in raw data, cleans, chunks, embeds, and loads it into a

vector DB.

- The inference pipeline queries the vector DB for relevant context and ultimately generates

an answer by levering an LLM. In this chapter, we will focus on implementing the RAG ingestion pipeline, and in Chapter 9, we will continue developing the inference pipeline. With that in mind, let's have a quick refresher on the problem we are trying to solve and where we get our raw data. Remember that we are building an end-to-end ML system. Thus, all the components talk to each other through an interface (or a contract), and each pipeline has a single responsibility. In our case, we ingest raw documents, preprocess them, and load them into a vector DB.

###### The problem we are solving

As presented in the previous chapter, this book aims to show you how to build a production-ready LLM Twin backed by an end-to-end ML system. In this chapter specifically, we want to design a RAG feature pipeline that takes raw social media data (e.g., articles, code repositories, and posts) from our MongoDB data warehouse. The text of the raw documents will be cleaned, chunked, embedded, and ultimately loaded to a feature store. As discussed in Chapter 1, we will implement a logical feature store using ZenML artifacts and a Qdrant vector DB. As we want to build a fully automated feature pipeline, we want to sync the data warehouse and logical feature store. Remember that, at inference time, the context used to generate the answer is retrieved from the vector DB. Thus, the speed of synchronization between the data warehouse and the feature store will directly impact the accuracy of our RAG algorithm. Another key consideration is how to automate the feature pipeline and integrate it with the rest of our ML system. Our goal is to minimize any desynchronization between the two data storages, as this could potentially compromise the integrity of our system.

-----

128 *RAG Feature Pipeline*

---

To conclude, we must design a feature pipeline that constantly syncs the data warehouse and logical feature store while processing the data accordingly. Having the data in a feature store is critical for a production-ready ML system. The LLM Twin inference pipeline will query it for RAG, while the training pipeline will consume tracked and versioned fine-tuning datasets from it.

###### The feature store

The feature store will be the central access point for all the features used within the training and inference pipelines. The training pipeline will use the cleaned data from the feature store (stored as artifacts) to fine-tune LLMs. The inference pipeline will query the vector DB for chunked documents for RAG. That is why we are designing a feature pipeline and not only a RAG ingestion pipeline. In practice, the feature pipeline contains multiple subcomponents, one of which is the RAG logic. Remember that the feature pipeline is mainly used as a mind map to navigate the complexity of ML systems. It clearly states that it takes raw data as input and then outputs features and optional labels, which are stored in the feature store. Thus, a good intuition is to consider that all the logic between the data warehouse and the feature store goes into the feature pipeline namespace, consisting of one or more sub-pipelines. For example, we will implement another pipeline that takes in cleaned data, processes it into instruct datasets, and stores it in artifacts; this also sits under the feature pipeline umbrella as the artifacts are part of the logical feature store. Another example would be implementing a data validation pipeline on top of the raw data or computed features. Another important observation to make is that text data stored as strings are not considered features if you follow the standard conventions. A feature is something that is fed directly into the model. For example, we would have to tokenize the instruct datasets or chunked documents to be considered features. Why? Because the tokens are fed directly to the model and not the sentences as strings. Unfortunately, this makes the system more complex and unflexible. Thus, we will do the tokenization at runtime. But this observation is important to understand as it's a clear example that you don't have to be too rigid about the feature/training/inference (FTI) architecture. You have to take it and adapt it to your own use case.

###### Where does the raw data come from?

As a quick reminder, all the raw documents are stored in a MongoDB data warehouse. The data warehouse is populated by the data collection ETL pipeline presented in Chapter 3. The ETL pipeline crawls various platforms such as Medium and Substack, standardizes the data, and loads it into MongoDB. Check out Chapter 3 for more details on this topic.

-----

*Chapter 4* 129

---

###### Designing the architecture of the RAG feature pipeline

The last step is to architect and go through the design of the RAG feature pipeline of the LLM Twin application. We will use a batch design scheduled to poll data from the MongoDB data warehouse, process it, and load it to a Qdrant vector DB. The first question to ask ourselves is, "Why a batch pipeline?" But before answering that, let's quickly understand how a batch architecture works and behaves relative to a streaming design.

![](image_p158_0.png)

A

Raw

NoSQL DB

#### Cleaned

Docs

Chunked

Docs

Embedded

Cleaned Docs

for Fine-tuning

Cleaned Docs

for Fine-tuning

Vector DB

Model ous

Registry

Accepted

LM

Chunked Docs

for RAG

User Request

Generated Answer

*Figure 4.9: The architecture of the LLM Twin's RAG feature pipeline*

-----

130 *RAG Feature Pipeline*

---

###### Batch pipelines

A batch pipeline in data systems refers to a data processing method where data is collected, processed, and stored in predefined intervals and larger volumes, also known as "batches". This approach differs from real-time or streaming data processing, where data is processed continuously as it arrives. This is what happens in a batch pipeline:

1. **Data collection: Data is collected from various sources and stored until sufficient amounts**

are accumulated for processing. This can include data from DBs, logs, files, and other sources.

2. **Scheduled processing: Data processing is scheduled at regular intervals, for example,**

hourly or daily. During this time, the collected data is processed in bulk. This can involve data cleansing, transformation, aggregation, and other operations.

3. **Data loading: After processing, the data is loaded into the target system, such as a DB, data**

warehouse, data lake, or feature store. This processed data is then available for analysis, querying, or further processing. Batch pipelines are particularly useful when dealing with large volumes of data that do not require immediate processing. They offer several advantages, including:

- **Efficiency: Batch processing can handle large volumes of data more efficiently than re-**

al-time processing, allowing for optimized resource allocation and parallel processing.

- **Complex processing: Batch pipelines can perform complex data transformations and**

aggregations that might be too resource-intensive for real-time processing.

- **Simplicity: Batch processing systems' architectures are often simpler than those of re-**

al-time systems, making them easier to implement and maintain.

###### Batch versus streaming pipelines

When implementing feature pipelines, you have two main design choices: batch and streaming. Thus, it is worthwhile to see the difference between the two and understand why we chose a batch architecture over a streaming one for our LLM Twin use case. You can effortlessly write a dedicated chapter on streaming pipelines, which suggests its complexity over a batch design. However, as streaming architectures become increasingly popular, one must have an intuition of how they work to choose the best option for your application.

-----

*Chapter 4* 131

---

The core elements of streaming applications are a distributed event streaming platform such as Apache Kafka or Redpanda to store events from multiple clients and a streaming engine such as Apache Flink or Bytewax to process the events. To simplify your architecture, you can swap your event streaming platform with queues, such as RabbitMQ, to store the events until processed. *Table 4.1 compares batch and streaming pipelines based on multiple criteria such as processing* schedule and complexity:

| Aspect | Batch pipeline |  |
|---|---|---|
| Processing schedule | Processes data at regular intervals (e.g., every minute, hourly, daily). |  |
| Efficiency | Handles large volumes of data more efficiently, optimizing resource allocation and parallel processing. |  |
| Processing complexity | Capable of performing complex data transformations and aggregations. |  |
| Use cases | Suitable for scenarios where \| immediate data processing is not critical. Commonly used in data warehousing, reporting, ETL processes, and feature pipelines. |  |
| System complexity | Compared to streaming pipelines, systems are generally simpler to implement and maintain. |  |

*Table 4.1: Batch versus streaming pipelines*

###### Streaming pipeline

```
Processes data
continuously, with
minimal latency.
Handles single data
points, providing
immediate insights
and updates, allowing
for rapid response to
changes.
Designed to handle
high-velocity data
streams with low
latency.
Ideal for applications
requiring real-time
analytics, features,
monitoring, and event-
driven architectures.
More complex to
implement and maintain
due to the need for
low-latency processing,
fault tolerance, and
scalability. The
tooling is also
more advanced and
complicated.
```

-----

132 *RAG Feature Pipeline*

---

For example, streaming pipelines are extremely powerful in social media recommender systems like TikTok. When using social media, user behavior changes frequently. A typical scenario is that you want to relax at a certain point in time and mostly look at videos of puppies. Still, after 15 minutes, you get bored and want something more serious, such as educative content or news. This means the recommender system has to capture these behavior changes without delay to keep you engaged. As the transition between interests is cyclical and not predictable, you can't use a batch pipeline that runs every 30 minutes or every hour to generate more content. You can run it every minute to create new content, but, at the same time, it will result in unnecessary costs, as most predictions will not be consumed. By implementing a streaming pipeline, you update the features of specific users in real time, which are then passed to a chain of models that predict the new recommendations. Streaming architectures are also the backbone of real-time fraud detection algorithms, such as those used at Stripe or PayPal. In this context, it's critical to identify potentially fraudulent transactions as they occur, not after a few minutes or hours as a batch pipeline would process them. The same urgency applies to high-frequency trading platforms that make stock predictions based on the constant influx of market data, enabling traders to make decisions within milliseconds. On the other hand, you can use a batch architecture for an offline recommender system. For example, when implementing one for an e-commerce or streaming platform, you don't need the system to be so reactive, as the user's behavior rarely changes. Thus, updating the recommendations periodically, such as every night, based on historical user behavior data using a batch pipeline is easier to implement and cheaper. Another popular example of batch pipelines is the ETL design used to extract, transform, and load data for different use cases. The ETL design is widespread in data pipelines used to move data from one DB to another. Some practical use cases include aggregating data for analytics, where you have to extract data from multiple sources, aggregate it, and load it to a data warehouse connected to a dashboard. The analytics domains can be widespread, from e-commerce and marketing to finance and research.

-----

*Chapter 4* 133

---

The data collection pipeline used in the LLM Twin use case is another example of an ETL pipeline that extracts data from the internet, structures it, and loads it into a data warehouse for future processing. Along with prediction or feature freshness, another disadvantage of batch pipelines over streaming ones is that you usually make redundant predictions. Let's take the example of a recommender system for a streaming platform like Netflix. Every night, you make the predictions for all users. There is a significant chance that a large chunk of users won't log in that day. Also, users usually don't browse all the recommendations but stick to the first ones. Thus, only a portion of predictions are used, wasting computing power on all the others. That's why a popular strategy is to start with a batch architecture, as it's faster and easier to implement. After the product is in place, you gradually move to a streaming design to reduce costs and improve the user experience. To conclude, we have used a batch architecture (and not a streaming one) to implement the LLM Twin's feature pipeline for the following reasons:

- **Does not require immediate data processing: Even if syncing the data warehouse and** feature store is critical for an accurate RAG system, a delay of a few minutes is acceptable. Thus, we can schedule the batch pipeline to run every minute, constantly syncing the two data storages. This technique works because the data volume is small. The whole data warehouse will have only thousands of records, not millions or billions. Hence, we can quickly iterate through them and sync the two DBs.
- **Simplicity: As stated earlier, implementing a streaming pipeline is two times more com-** plex. In the real world, you want to keep your system as simple as possible, making it easier to understand, debug, and maintain. Also, simplicity usually translates to lower infrastructure and development costs.

-----

134 *RAG Feature Pipeline*

---

In Figure 8.10, we compare what tools you can use based on your architecture (streaming versus batch) and the quantity of data you have to process (small versus big data). In our use case, we are in the smaller data and batch quadrant, where we picked a combination of vanilla Python and generative AI tools such as LangChain, Sentence Transformers, and Unstructured.

![](image_p163_0.png)

Streaming

Beam

Bytewax Spark Streaming

Flink Kafka Streams

Bigger

Data

« >

Smaller

Data Dask

Pandas Spark

Ray

Vanilla Pyth

anilla

##### BigQuery

Polars

Snowflake Batch

¥

*Figure 4.10: Tools on the streaming versus batch and smaller versus bigger data spectrum*

In the Change data capture: syncing the data warehouse and feature store section later in this chapter, we will discuss when switching from a batch architecture to a streaming one makes sense.

###### Core steps

Most of the RAG feature pipelines are composed of five core steps. The one implemented in the LLM Twin architecture makes no exception. Thus, you can quickly adapt this pattern for other RAG applications, but here is what the LLM Twin's RAG feature pipeline looks like:

1. **Data extraction: Extract the latest articles, code repositories, and posts from the Mon-** goDB data warehouse. At the extraction step, you usually aggregate all the data you need for processing.

-----

*Chapter 4* 135

---

2. **Cleaning: The data from the data warehouse is standardized and partially clean, but we**

have to ensure that the text contains only useful information, is not duplicated, and can be interpreted by the embedding model. For example, we must clean and normalize all non-ASCII characters before passing the text to the embedding model. Also, to keep the information semantically dense, we decided to replace all the URLs with placeholders and remove all emojis. The cleaning step is more art than science. Hence, after you have the first iteration with an evaluation mechanism in place, you will probably reiterate and improve it.

3. **Chunking: You must adopt various chunking strategies based on each data category**

and embedding model. For example, when working with code repositories, you want the chunks broader, whereas when working with articles, you want them narrower or scoped at the paragraph level. Depending on your data, you must decide if you split your document based on the chapter, section, paragraph, sentence, or just a fixed window size. Also, you have to ensure that the chunk size doesn't exceed the maximum input size of the embedding model. That is why you usually chunk a document based on your data structure and the maximum input size of the model.

4. **Embedding: You pass each chunk individually to an embedding model of your choice.**

Implementation-wise, this step is usually the simplest, as tools such as SentenceTransformer and Hugging Face provide high-level interfaces for most embedding models. As explained in the What are embeddings? section of this chapter, at this step, the most critical decisions are to decide what model to use and whether to fine-tune it or not. For example, we used an `"all-mpnet-base-v2"` embedding model from SentenceTransformer, which is relatively tiny and runs on most machines. However, we provide a configuration file where you can quickly configure the embedding model with something more powerful based on the state of the art when reading this book. You can quickly find other options on the MTEB on Hugging Face (https://huggingface.co/spaces/mteb/leaderboard).

5. **Data loading: The final step combines the embedding of a chunked document and its**

metadata, such as the author and the document ID, content, URL, platform, and creation date. Ultimately, we wrap the vector and the metadata into a structure compatible with Qdrant and push it to the vector DB. As we want to use Qdrant as the single source of truth for the features, we also push the cleaned documents (before chunking) to Qdrant. We can push data without vectors, as the metadata index of Qdrant behaves like a NoSQL DB. Thus, pushing metadata without a vector attached to it is like using a standard NoSQL engine.

-----

136 *RAG Feature Pipeline*

---

###### Change data capture: syncing the data warehouse and feature store

As highlighted a few times in this chapter, data is constantly changing, which can result in DBs, data lakes, data warehouses, and feature stores getting out of sync. Change data capture (CDC) is a strategy that allows you to optimally keep two or more data storage types in sync without computing and I/O overhead. It captures any CRUD operation done on the source DB and replicates it on a target DB. Optionally, you can add preprocessing steps in between the replication. The syncing issues also apply when building a feature pipeline. One key design choice concerns how to sync the data warehouse with the feature store to have data fresh enough for your particular use case. In our LLM Twin use case, we chose a naïve approach out of simplicity. We implemented a batch pipeline that is triggered periodically or manually. It reads all the raw data from the data warehouse, processes it in batches, and inserts new records or updates old ones from the Qdrant vector DB. This works fine when you are working with a small number of records, at the order of thousands or tens of thousands. But our naïve approach raises the following questions:

- What happens if the data suddenly grows to millions of records (or higher)?
- What happens if a record is deleted from the data warehouse? How is this reflected in

the feature store?

- What if we want to process only the new or updated items from the data warehouse and

not all of them? Fortunately, the CDC pattern can solve all of these issues. When implementing CDC, you can take multiple approaches, but all of them use either a push or pull strategy:

- **Push: The source DB is the primary driver in the push approach. It actively identifies**

and transmits data modifications to target systems for processing. This method ensures near-instantaneous updates at the target, but data loss can occur if target systems are inaccessible. To mitigate this, a messaging system is typically employed as a buffer.

- **Pull: The pull method assigns a more passive role to the source DB, which only records**

data changes. Target systems periodically request these changes and handle updates accordingly. While this approach lightens the load on the source, it introduces a delay in data propagation. A messaging system is again essential to prevent data loss during periods of target system unavailability.

-----

*Chapter 4* 137

---

In summary, the push method is ideal for applications demanding immediate data access, whereas the pull method is better suited for large-scale data transfers where real-time updates aren't critical. With that in mind, there are different methods to detect changes in data. Thus, let's list the main CDC patterns that are used in the industry:

- **Timestamp-based: The approach involves adding a modification time column to DB** tables, usually called `LAST\_MODIFIED` or LAST\_UPDATED. Downstream systems can query this column to identify records that have been updated since their last check. While simple to implement, this method is limited to tracking changes, not deletions, and imposes performance overhead due to the need to scan entire tables.
- **Trigger-based: The trigger-based approach utilizes DB triggers to automatically record**

data modifications in a separate table upon INSERT, UPDATE, or DELETE operations, often known as the event table. This method provides comprehensive change tracking but can impact the DB performance due to the additional write operations involved for each event.

- **Log-based: DBs maintain transaction logs to record all data modifications, including**

timestamps. Primarily used for recovery, these logs can also be leveraged to propagate changes to target systems in real time. This approach minimizes the performance impact on the source DB. As a huge advantage, it avoids additional processing overhead on the source DB, captures all data changes, and requires no schema modification. But on the opposite side, it lacks standardized log formats, leading to vendor-specific implementations.

Z For more details on CDC, I recommend What is Change Data Capture? from Conflu-

```
ent's blog: https://www.confluent.io/en-gb/learn/change-data-capture/ .
```

With these CDC techniques in mind, we could quickly implement a pull timestamp-based strategy in our RAG feature pipeline to sync the data warehouse and feature store more optimally when the data grows. Our implementation is still pull-based but doesn't check any last updated field in the source DB; it just pulls everything from the data warehouse. However, the most popular and optimal technique in the industry is the log-based one. It doesn't add any I/O overhead to the source DB, has low latency, and supports all CRUD operations. The biggest downside is its development complexity, which requires a queue to capture all the CRUD events and a streaming pipeline to process them.

-----

138 *RAG Feature Pipeline*

---

As this is an LLM book and not a data engineering one, we wanted to keep things simple, but it's important to know that these techniques exist, and you can always upgrade your current implementation when it doesn't fit your application requirements anymore.

###### Why is the data stored in two snapshots?

We store two snapshots of our data in the logical feature store:

- **After the data is cleaned: For fine-tuning LLMs**
- **After the documents are chunked and embedded: For RAG** *Why did we design it this way? Remember that the features should be accessed solely from the feature* store for training and inference. Thus, this adds consistency to our design and makes it cleaner. Also, storing the data cleaned specifically for our fine-tuning and embedding use case in the MongoDB data warehouse would have been an antipattern. The data from the warehouse is shared all across the company. Thus, processing it for a specific use case is not good practice. Imagine another summarization use case where we must clean and preprocess the data differently. We must create a new "Cleaned Data" table prefixed with the use case name. We have to repeat that for every new use case. Therefore, to avoid having a spaghetti data warehouse, the data from the data warehouse is generic and is modeled to specific applications only in downstream components, which, in our case, is the feature store. Ultimately, as we mentioned in the Core steps section, you can leverage the metadata index of a vector DB as a NoSQL DB. Based on these factors, we decided to keep the cleaned data in Qdrant, along with the chunked and embedded versions of the documents. As a quick reminder, when operationalizing our LLM Twin system, the create instruct dataset pipeline, explained in Chapter 5, will read the cleaned documents from Qdrant, process them, and save them under a versioned ZenML artifact. The training pipeline requires a dataset and not plain documents. This is a reminder that our logical feature store comprises the Qdrant vector DB for online serving and ZenML artifacts for offline training.

###### Orchestration

ZenML will orchestrate the batch RAG feature pipeline. Using ZenML, we can schedule it to run on a schedule, for example, every hour, or quickly manually trigger it. Another option is to trigger it after the ETL data collection pipeline finishes. By orchestrating the feature pipeline and integrating it into ZenML (or any other orchestration tool), we can operationalize the feature pipeline with the end goal of continuous training (CT).

-----

*Chapter 4* 139

---

We will go into all the details of orchestration, scheduling, and CT in Chapter 11.

##### Implementing the LLM Twin's RAG feature pipeline

The last step is to review the LLM Twin's RAG feature pipeline code to see how we applied everything we discussed in this chapter. We will walk you through the following:

- ZenML code
- Pydantic domain objects
- A custom object-vector mapping (OVM) implementation
- The cleaning, chunking, and embedding logic for all our data categories We will take a top-down approach. Thus, let's start with the Settings class and ZenML pipeline.

###### Settings

```
We use Pydantic Settings (https://docs .pydantic .dev/latest/concepts/pydantic_settings/)
```

to define a global Settings class that loads sensitive or non-sensitive variables from a `.env` file. This approach also gives us all the benefits of Pydantic, such as type validation. For example, if we provide a string for the `QDRANT\_DATABASE\_PORT` variable instead of an integer, the program will crash. This behavior makes the whole application more deterministic and reliable. Here is what the `Settings` class looks like with all the variables necessary to build the RAG feature pipeline:

```python
from pydantic import BaseSettings
class Settings(BaseSettings):
class Config:
env_file = ".env"
env_file_encoding = "utf-8"
… # Some other settings…
# RAG
TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-
L6-v2"
RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-
MiniLM-L-4-v2"
RAG_MODEL_DEVICE: str = "cpu"
```

-----

140 *RAG Feature Pipeline*

---

| # QdrantDB Vector DB USE_QDRANT_CLOUD: bool = False QDRANT_DATABASE_HOST: str = "localhost" QDRANT_DATABASE_PORT: int = 6333 QDRANT_CLOUD_URL: str = "str" QDRANT_APIKEY: str \| None = None … # More settings… settings = Settings() |
|---|
| As stated in the internal Config class, all the variables have default values or can be overridden by providing a .env file. ZenML pipeline and steps The ZenML pipeline is the entry point for the RAG feature engineering pipeline. It reflects the five core phases of RAG ingestion code: extracting raw documents, cleaning, chunking, embed- ding, and loading them to the logical feature store. The calls within the feature_engineering() function are ZenML steps, representing a single execution unit performing the five phases of RAG. The code is available in the GitHub repository at https://github.com/PacktPublishing/LLM- Engineers-Handbook/blob/main/pipelines/feature_engineering .py: |
| from zenml import pipeline from llm_engineering.interfaces.orchestrator.steps import feature_ engineering as fe_steps @pipeline def feature_engineering(author_full_names: list[str]) -> None: raw_documents = fe_steps.query_data_warehouse(author_full_names) |

```
cleaned_documents = fe_steps.clean_documents(raw_documents)
last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)
```

```
embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
last_step_2 = fe_steps.load_to_vector_db(embedded_documents)
return [last_step_1.invocation_id, last_step_2.invocation_id]
```

-----

*Chapter 4* 141

---

*Figure 4.11 shows how multiple feature engineering pipeline runs look in ZenML's dashboard.*

![](image_p170_0.png)

“i feature\_engineering

Search.

rot |

C on run

5 foture\_engineeing.

feature un

5 engineering. ©)

##### fears rn

5 ©

B ato © cea awa 005031 cesar

©

*Figure 4.11: Feature pipeline runs in the ZenML dashboard*

*Figure 8.12 shows the DAG of the RAG feature pipeline, where you can follow all the pipeline steps* and their output artifacts. Remember that whatever is returned from a ZenML step is automatically saved as an artifact, stored in ZenML's artifact registry, versioned, and shareable across the application.

![](image_p170_1.png)

© query\_data\_warehouse raw\_documents

builtins list

© clean\_documents

cleaned\_documents

builtins. list

© chunk\_and\_embed © load\_to\_vector\_db builtins list

© load\_to\_vector\_db\_2

*Figure 4.12: Feature pipeline DAG in the ZenML dashboard*

-----

142 *RAG Feature Pipeline*

---

The final puzzle piece is understanding how to configure the RAG feature pipeline dynamically. All its available settings are exposed as function parameters. Here, we need only a list of author's names, as seen in the function's signature: `feature\_engineering(author\_full\_names:` `list[str])` . We inject a YAML configuration file at runtime that contains all the necessary values based on different use cases. For example, the following configuration includes a list of all the authors of this book as we want to populate the feature store with data from all of us (available

```
in the GitHub repository at configs/feature_engineering .yaml):
```

| parameters: author_full_names: - Alex Vesa - Maxime Labonne - Paul Iusztin |
|---|
| The beauty of this approach is that you don't have to modify the code to configure the feature pipeline with different input values. You have to provide a different configuration file when run- ning it, as follows: |
| feature_engineering.with_options(config_path="…/feature_engineering.yaml") () |
| You can either hardcode the path to the config file or provide the config_path from the CLI, which allows you to modify the pipeline's configuration between different runs. Out of simplicity, we hard-coded the configuration file. Thus, we can call the feature engineering pipeline calling the run.py script as follows: |
| python -m tools.run --no-cache --run-feature-engineering |
| However, you can easily add another CLI argument to pass the config_path variable. Also, you can run the feature pipeline using the following poe command: |
| poetry poe run-feature-engineering-pipeline |

Let's move forward to the ZenML steps and sequentially zoom in on all of them. The source code for all the feature engineering pipeline steps is available on GitHub at `"steps/feature\_engineering"` . We will begin with the first step, which involves querying the data warehouse for new content to process into features.

-----

*Chapter 4* 143

---

###### Querying the data warehouse

The first thing to notice is that a step is a Python function decorated with @step, similar to how a ZenML pipeline works. The function below takes as input a list of authors' full names and performs the following core steps:

- It attempts to get or create a `UserDocument` instance using the first and last names, ap-

pending this instance to the `authors` list. If the user doesn't exist, it throws an error.

- It fetches all the raw data for the user from the data warehouse and extends the `documents`

list to include these user documents.

- Ultimately, it computes a descriptive metadata dictionary logged and tracked in ZenML.

```python
… # other imports
from zenml import get_step_context, step
@step
def query_data_warehouse(
author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
documents = []
authors = []
for author_full_name in author_full_names:
logger.info(f"Querying data warehouse for user: {author_full_
name}")
first_name, last_name = utils.split_user_full_name(author_full_
name)
logger.info(f"First name: {first_name}, Last name: {last_name}")
user = UserDocument.get_or_create(first_name=first_name, last_
name=last_name)
authors.append(user)
results = fetch_all_data(user)
user_documents = [doc for query_result in results.values() for doc
in query_result]
documents.extend(user_documents)
step_context = get_step_context()
```

-----

144 *RAG Feature Pipeline*

---

| step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents)) return documents |
|---|
| The fetch function leverages a thread pool that runs each query on a different thread. As we have multiple data categories, we have to make a different query for the articles, posts, and reposi- tories, as they are stored in different collections. Each query calls the data warehouse, which is bounded by the network I/O and data warehouse latency, not by the machine's CPU. Thus, by moving each query to a different thread, we can parallelize them. Ultimately, instead of adding the latency of each query as the total timing, the time to run this fetch function will be the max between all the calls. Using threads to parallelize I/O-bounded calls is good practice in Python, as they are not locked by the Python Global Interpreter Lock (GIL). In contrast, adding each call to a different process would add too much overhead, as a process takes longer to spin off than a thread. In Python, you want to parallelize things with processes only when the operations are CPU or memory-bound because the GIL affects them. Each process has a different GIL. Thus, paralleliz- ing your computing logic, such as processing a batch of documents or images already loaded in memory, isn't affected by Python's GIL limitations. |
| def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]: user_id = str(user.id) with ThreadPoolExecutor() as executor: future_to_query = { executor.submit(__fetch_articles, user_id): "articles", executor.submit(__fetch_posts, user_id): "posts", executor.submit(__fetch_repositories, user_id): "repositories", } results = {} for future in as_completed(future_to_query): query_name = future_to_query[future] try: results[query_name] = future.result() except Exception: |

-----

*Chapter 4* 145

---

| logger.exception(f"'{query_name}' request failed.") results[query_name] = [] return results |
|---|
| The _get_metadata() function takes the list of queried documents and authors and counts the number of them relative to each data category: |
| def _get_metadata(documents: list[Document]) -> dict: metadata = { "num_documents": len(documents), } for document in documents: collection = document.get_collection_name() if collection not in metadata: metadata[collection] = {} if "authors" not in metadata[collection]: metadata[collection]["authors"] = list() metadata[collection]["num_documents"] = metadata[collection]. get("num_documents", 0) + 1 metadata[collection]["authors"].append(document.author_full_name) for value in metadata.values(): if isinstance(value, dict) and "authors" in value: value["authors"] = list(set(value["authors"])) return metadata |

We will expose this metadata in the ZenML dashboard to quickly see some statistics on the loaded data. For example, in Figure 4.13, we accessed the metadata tab of the `query\_data\_warehouse()` step, where you can see that, within that particular run of the feature pipeline, we loaded 76 documents from three authors. This is extremely powerful for monitoring and debugging batch pipelines.

-----

146 *RAG Feature Pipeline*

---

You can always extend it with anything that makes sense for your use case.

![](image_p175_0.png)

f80ace2c-af55-42d7-964b-012¢81f17511

raw\_documents 6:

(©) Overview Metadata [i] Visualization

> Uncategorized

Vv articles

num\_documents 76

“authors

0 Paul lusztin

1 Maxime Labonne

2 Alex Vesa

*Figure 4.13: Metadata of the "query the data warehouse" ZenML step*

###### Cleaning the documents

In the cleaning step, we iterate through all the documents and delegate all the logic to a `CleaningDispatcher` who knows what cleaning logic to apply based on the data category. Remember that we want to apply, or have the ability to apply in the future, different cleaning techniques on articles, posts, and code repositories.

```python
@step
def clean_documents(
documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
cleaned_documents = []
for document in documents:
```

-----

*Chapter 4* 147

---

| cleaned_document = CleaningDispatcher.dispatch(document) cleaned_documents.append(cleaned_document) step_context = get_step_context() step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents)) return cleaned_documents |
|---|
| The computed metadata is similar to what we logged in the query_data_warehouse() step. Thus, let's move on to chunking and embedding. Chunk and embed the cleaned documents Similar to how we cleaned the documents, we delegate the chunking and embedding logic to a dispatcher who knows how to handle each data category. Note that the chunking dispatcher returns a list instead of a single object, which makes sense as the document is split into multiple chunks. We will dig into the dispatcher in the "The dispatcher layer" section of this chapter. |
| @step def chunk_and_embed( cleaned_documents: Annotated[list, "cleaned_documents"], ) -> Annotated[list, "embedded_documents"]: metadata = {"chunking": {}, "embedding": {}, "num_documents": len(cleaned_documents)} embedded_chunks = [] for document in cleaned_documents: chunks = ChunkingDispatcher.dispatch(document) metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"]) for batched_chunks in utils.misc.batch(chunks, 10): batched_embedded_chunks = EmbeddingDispatcher. dispatch(batched_chunks) embedded_chunks.extend(batched_embedded_chunks) metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"]) metadata["num_chunks"] = len(embedded_chunks) |

-----

148 *RAG Feature Pipeline*

---

```
metadata["num_embedded_chunks"] = len(embedded_chunks)
step_context = get_step_context()
step_context.add_output_metadata(output_name="embedded_documents",
metadata=metadata)
return embedded_chunks
```

In Figure 4.14, you can see the metadata of the chunking and embedding ZenML step. For example, you can quickly understand that we transformed 76 documents into 2,373 chunks, or the properties we used for chunking articles, such as a `chunk\_size` of 500 and a `chunk\_overlap` of 50.

![](image_p177_0.png)

56260613-5a4d-45bb-b17a-2747905151e2

embedded\_documents 2s

0 overview Metadata [I] Visualization

Uncategorized

length 2373

num\_chunks 2373

num\_documents

76

num\_embedded\_chunks

2373

storage\_size

~ chunking

articles chunk\_size

num\_chunks

> authors

50

500

2373

*Figure 4.14: Metadata of the embedding and chunking ZenML step, detailing the uncategorized and chunking dropdowns*

-----

*Chapter 4* 149

---

In Figure 4.15, the rest of the ZenML metadata from the embedding and chunking step details the embedding model and its properties used to compute the vectors.

![](image_p178_0.png)

56260613-5a4d-45bb-b17a-27479d5151e2

embedded\_documents

0 Overview Metadata [I] Visualization

> Uncategorized

> chunking

Vv embedding

Vv articles embedding\_model\_id sentence-transformers/all-MiniLM-L6-v2

##### embedding_size

max\_input\_length

Vv authors

384

256

Paul lusztin

1 Maxime Labonne

2 Alex Vesa

*Figure 4.15: Metadata of the embedding and chunking ZenML step, detailing the embedding dropdown*

-----

150 *RAG Feature Pipeline*

---

As ML systems can break at any time while in production due to drifts or untreated use cases, leveraging the metadata section to monitor the ingested data can be a powerful tool that will save debugging days, translating to tens of thousands of dollars or more for your business.

###### Loading the documents to the vector DB

As each article, post, or code repository sits in a different collection inside the vector DB, we have to group all the documents based on their data category. Then, we load each group in bulk in the Qdrant vector DB:

| Pydantic domain entities Before investigating the dispatchers, we must understand the domain objects we work with. To some extent, in implementing the LLM Twin, we are following the domain-driven design (DDD) principles, which state that domain entities are the core of your application. Thus, before pro- ceeding, it's important to understand the hierarchy of the domain classes we are working with. | @step def ) database.") collection_name()}") Pydantic domain entities Before investigating the dispatchers, we must understand the domain objects we work with. To some extent, in implementing the LLM Twin, we are following the domain-driven design (DDD) principles, which state that domain entities are the core of your application. Thus, before pro- ceeding, it's important to understand the hierarchy of the domain classes we are working with. | @step def ) -> database.") collection_name()}") Pydantic domain entities Before investigating the dispatchers, we must understand the domain objects we work with. To some extent, in implementing the LLM Twin, we are following the domain-driven design (DDD) principles, which state that domain entities are the core of your application. Thus, before pro- ceeding, it's important to understand the hierarchy of the domain classes we are working with. | @step def load_to_vector_db( documents: -> None: logger.info(f"Loading {len(documents)} documents database.") grouped_documents for collection_name()}") return True Pydantic domain entities Before investigating the dispatchers, we must understand the domain objects we work with. To some extent, in implementing the LLM Twin, we are following the domain-driven design (DDD) principles, which state that domain entities are the core of your application. Thus, before pro- ceeding, it's important to understand the hierarchy of the domain classes we are working with. | @step load_to_vector_db( documents: None: logger.info(f"Loading {len(documents)} documents database.") grouped_documents for collection_name()}") return True Pydantic domain entities Before investigating the dispatchers, we must understand the domain objects we work with. To some extent, in implementing the LLM Twin, we are following the domain-driven design (DDD) principles, which state that domain entities are the core of your application. Thus, before pro- ceeding, it's important to understand the hierarchy of the domain classes we are working with. | load_to_vector_db( documents: Annotated[list, "documents"], None: logger.info(f"Loading {len(documents)} documents into the vector database.") grouped_documents = VectorBaseDocument.group_by_class(documents) for document_class, documents in grouped_documents.items(): logger.info(f"Loading documents into {document_class.get_ collection_name()}") for documents_batch in utils.misc.batch(documents, size=4): try: document_class.bulk_insert(documents_batch) except Exception: return False return True Pydantic domain entities Before investigating the dispatchers, we must understand the domain objects we work with. To some extent, in implementing the LLM Twin, we are following the domain-driven design (DDD) principles, which state that domain entities are the core of your application. Thus, before pro- ceeding, it's important to understand the hierarchy of the domain classes we are working with. |
|---|---|---|---|---|---|
|  | & | The | code | for | the domain entities is available on GitHub at https://github.com/ PacktPublishing/LLM-Engineering/tree/main/llm_engineering/domain . |

-----

*Chapter 4* 151

---

We used Pydantic to model all our domain entities. When we wrote the book, choosing Pydantic was a no-brainer, as it is the go-to Python package for writing data structures with out-of-the-box type validation. As Python is a dynamically typed language, using Pydantic for type validation at runtime makes your system order of times more robust, as you can be sure that you are always working with the right type of data. The domain of our LLM Twin application is split into two dimensions:

- **The data category: Post, article, and repository**
- **The state of the data: Cleaned, chunked, and embedded** We decided to create a base class for each state of the document, resulting in having the following base abstract classes:

```
• class CleanedDocument(VectorBaseDocument, ABC)
• class Chunk(VectorBaseDocument, ABC)
• class EmbeddedChunk(VectorBaseDocument, ABC)
```

Note that all of them inherit the `VectorBaseDocument` class, which is our custom OVM implementation, which we will explain in the next section of this chapter. Also, it inherits from ABC, which makes the class abstract. Thus, you cannot initialize an object out of these classes; you may only inherit from them. That is why base classes are always marked as abstract. Each base abstract class from above (which models the state) will have a subclass that adds the data category dimension. For example, the `CleanedDocument` class will have the following subclasses:

```
• class CleanedPostDocument(CleanedDocument)
• class CleanedArticleDocument(CleanedDocument)
• class CleanedRepositoryDocument(CleanedDocument)
```

As we can see in Figure 8.16, we will repeat the same logic for the `Chunk` and `EmbeddedChunk` base abstract classes. We will implement a specific document class for each data category and state combination, resulting in nine types of domain entities. For example, when ingesting a raw document, the cleaning step will yield a `CleanedArticleDocument` instance, the chunking step will return a list of `ArticleChunk` objects, and the embedding operation will return `EmbeddedArticleChunk` instances that encapsulate the embedding and all the necessary metadata to ingest in the vector DB.

-----

152 *RAG Feature Pipeline*

---

The same will happen for the posts and repositories.

![](image_p181_0.png)

*Figure 4.16: Domain entities class hierarchy and their interaction*

We chose this design because the list of states will rarely change, and we want to extend the list of data categories. Thus, structuring the classes after the state allows us to plug another data category by inheriting these base abstract classes. Let's see the complete code for the hierarchy of the cleaned document. All the attributes of a cleaned document will be saved within the metadata of the vector DB. For example, the metadata of a cleaned article document will always contain the content, platform, author ID, author full name, and link of the article. Another fundamental aspect is the `Config` internal class, which defines the name of the collection within the vector DB, the data category of the entity, and whether to leverage the vector index when creating the collection:

```python
class CleanedDocument(VectorBaseDocument, ABC):
content: str
platform: str
author_id: UUID4
author_full_name: str
```

-----

*Chapter 4* 153

---

| class CleanedPostDocument(CleanedDocument): image: Optional[str] = None class Config: name = "cleaned_posts" category = DataCategory.POSTS use_vector_index = False class CleanedArticleDocument(CleanedDocument): link: str class Config: name = "cleaned_articles" category = DataCategory.ARTICLES use_vector_index = False class CleanedRepositoryDocument(CleanedDocument): name: str link: str class Config: name = "cleaned_repositories" category = DataCategory.REPOSITORIES use_vector_index = False |
|---|
| To conclude this section, let's also take a look at the base abstract class of the chunk and embed- ded chunk: |
| class Chunk(VectorBaseDocument, ABC): content: str platform: str document_id: UUID4 author_id: UUID4 author_full_name: str metadata: dict = Field(default_factory=dict) … # PostChunk, ArticleChunk, RepositoryChunk |

-----

154 *RAG Feature Pipeline*

---

| We also defined an enum that aggregates all our data categories in a single structure of constants: | class EmbeddedChunk(VectorBaseDocument, … We also defined an enum that aggregates all our data categories in a single structure of constants: | class EmbeddedChunk(VectorBaseDocument, … # We also defined an enum that aggregates all our data categories in a single structure of constants: | class EmbeddedChunk(VectorBaseDocument, content: str embedding: platform: document_id: author_id: author_full_name: metadata: # EmbeddedPostChunk, We also defined an enum that aggregates all our data categories in a single structure of constants: | class EmbeddedChunk(VectorBaseDocument, content: str embedding: platform: document_id: author_id: author_full_name: metadata: EmbeddedPostChunk, We also defined an enum that aggregates all our data categories in a single structure of constants: | class EmbeddedChunk(VectorBaseDocument, ABC): content: str embedding: list[float] \| None platform: str document_id: UUID4 author_id: UUID4 author_full_name: str metadata: dict = Field(default_factory=dict) EmbeddedPostChunk, EmbeddedArticleChunk, EmbeddedRepositoryChunk We also defined an enum that aggregates all our data categories in a single structure of constants: |
|---|---|---|---|---|---|
|  |  |  |  |  | class DataCategory(StrEnum): POSTS = "posts" ARTICLES = "articles" REPOSITORIES = "repositories" |
| The ter data Similar example leveraging | term 3. and to is OOP | The last OVM OVM We SQL what simple, best | is called tables. we it's practices | VectorBaseDocument inspired it Otherwise, did a and | step to fully understand how the domain objects work is to zoom into the OVM class. by the object-relational mapping (ORM) pattern we discussed in Chap- OVM because we work with embedding and vector DBs instead of structured it follows the same principles as an ORM pattern. in Chapter 3, we will implement our own OVM version. Even if our custom powerful example of how to write modular and extendable classes by principles. |
|  |  |  |  |  | The full implementation of the VectorBaseDocument class is available on GitHub at https://github.com/PacktPublishing/LLM-Engineering/blob/main/llm_ engineering/domain/base/vector .py. |

Our OVM base class is called `VectorBaseDocument` . It will support CRUD operations on top of Qdrant. Based on our application's demands, we limited it only to create and read operations, but it can easily be extended to update and delete functions.

-----

*Chapter 4* 155

---

Let's take a look at the definition of the `VectorBaseDocument` class:

```python
from pydantic import UUID4, BaseModel
from typing import Generic
from llm_engineering.infrastructure.db.qdrant import connection
```

```python
T = TypeVar("T", bound="VectorBaseDocument")
class VectorBaseDocument(BaseModel, Generic[T], ABC):
id: UUID4 = Field(default_factory=uuid.uuid4)
@classmethod
def from_record(cls: Type[T], point: Record) -> T:
_id = UUID(point.id, version=4)
payload = point.payload or {}
attributes = {
"id": _id,
**payload,
}
if cls._has_class_attribute("embedding"):
payload["embedding"] = point.vector or None
return cls(**attributes)
def to_point(self: T, **kwargs) -> PointStruct:
exclude_unset = kwargs.pop("exclude_unset", False)
by_alias = kwargs.pop("by_alias", True)
payload = self.dict(exclude_unset=exclude_unset, by_alias=by_
alias, **kwargs)
_id = str(payload.pop("id"))
vector = payload.pop("embedding", {})
if vector and isinstance(vector, np.ndarray):
```

-----

156 *RAG Feature Pipeline*

---

| vector = vector.tolist() return PointStruct(id=_id, vector=vector, payload=payload) |
|---|
| • The VectorBaseDocument class inherits from Pydantic's BaseModel and helps us structure a single record's attributes from the vector DB. Every OVM will be initialized by default with UUID4 as its unique identifier. Using generics-more precisely, by inheriting from Generic[T]-the signatures of all the subclasses of the VectorBaseDocument class will adapt to that given class. For example, the from_record() method of the Chunk() class, which inherits VectorBaseDocument, will return the Chunk type, which drastically helps the static analyzer and type checkers such as mypy (https://mypy .readthedocs.io/en/ stable/). The from_record() method adapts a data point from Qdrant's format to our internal structure based on Pydantic. On the other hand, the to_point() method takes the attributes of the current instance and adapts them to Qdrant's PointStruct() format. We will leverage these two methods for our create and read operations. Ultimately, all operations made to Qdrant will be done through the connection instance, which is instantiated in the application's infrastructure layer. The bulk_insert() method maps each document to a point. Then, it uses the Qdrant connection instance to load all the points to a given collection in Qdrant. If the insertion fails once, it tries to create the collection and do the insertion again. Often, it is good practice to split your logic into two functions. One private function contains the logic, in our case _bulk_insert(), and one public function handles all the errors and failure scenarios. |
| class VectorBaseDocument(BaseModel, Generic[T], ABC): … # Rest of the class @classmethod def bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> bool: try: cls._bulk_insert(documents) except exceptions.UnexpectedResponse: logger.info( f"Collection '{cls.get_collection_name()}' does not exist. Trying to create the collection and reinsert the documents." |

-----

*Chapter 4* 157

---

| ) cls.create_collection() try: cls._bulk_insert(documents) except exceptions.UnexpectedResponse: logger.error(f"Failed to insert documents in '{cls.get_ collection_name()}'.") return False return True @classmethod def _bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> None: points = [doc.to_point() for doc in documents] connection.upsert(collection_name=cls.get_collection_name(), points=points) |
|---|
| The collection name is inferred from the Config class defined in the subclasses inheriting the OVM: |
| class VectorBaseDocument(BaseModel, Generic[T], ABC): … # Rest of the class @classmethod def get_collection_name(cls: Type[T]) -> str: if not hasattr(cls, "Config") or not hasattr(cls.Config, "name"): raise ImproperlyConfigured( "The class should define a Config class with" "the 'name' property that reflects the collection's name." ) return cls.Config.name |

-----

158 *RAG Feature Pipeline*

---

Now, we must define a method that lets us read all the records from the vector DB (without using vector similarity search logic). The `bulk\_find()` method enables us to scroll (or list) all the records from a collection. The function below scrolls the Qdrant vector DB, which returns a list of data points, which are ultimately mapped to our internal structure using the `from\_record()` method. The limit parameters control how many items we return at once, and the offset signals the ID of the point from which Qdrant starts returning records.

```python
class VectorBaseDocument(BaseModel, Generic[T], ABC):
… # Rest of the class
@classmethod
def bulk_find(cls: Type[T], limit: int = 10, **kwargs) ->
tuple[list[T], UUID | None]:
try:
documents, next_offset = cls._bulk_find(limit=limit, **kwargs)
except exceptions.UnexpectedResponse:
logger.error(f"Failed to search documents in '{cls.get_
collection_name()}'.")
documents, next_offset = [], None
return documents, next_offset
@classmethod
def _bulk_find(cls: Type[T], limit: int = 10, **kwargs) ->
tuple[list[T], UUID | None]:
collection_name = cls.get_collection_name()
offset = kwargs.pop("offset", None)
offset = str(offset) if offset else None
records, next_offset = connection.scroll(
collection_name=collection_name,
limit=limit,
with_payload=kwargs.pop("with_payload", True),
with_vectors=kwargs.pop("with_vectors", False),
offset=offset,
```

-----

*Chapter 4* 159

---

| **kwargs, ) documents = [cls.from_record(record) for record in records] if next_offset is not None: next_offset = UUID(next_offset, version=4) return documents, next_offset |
|---|
| The last piece of the puzzle is to define a method that performs a vector similarity search on a provided query embedding. Like before, we defined a public search() and private _search() method. The search is performed by Qdrant when calling the connection.search() function. |
| class VectorBaseDocument(BaseModel, Generic[T], ABC): … # Rest of the class @classmethod def search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]: try: documents = cls._search(query_vector=query_vector, limit=limit, **kwargs) except exceptions.UnexpectedResponse: logger.error(f"Failed to search documents in '{cls.get_ collection_name()}'.") documents = [] return documents @classmethod def _search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]: collection_name = cls.get_collection_name() records = connection.search( collection_name=collection_name, query_vector=query_vector, limit=limit, with_payload=kwargs.pop("with_payload", True), with_vectors=kwargs.pop("with_vectors", False), |

-----

160 *RAG Feature Pipeline*

---

| **kwargs, ) documents = [cls.from_record(record) for record in records] return documents |
|---|
| Now that we understand what our domain entities look like and how the OVM works, let's move on to the dispatchers who clean, chunk, and embed the documents. The dispatcher layer A dispatcher inputs a document and applies dedicated handlers based on its data category (article, post, or repository). A handler can either clean, chunk, or embed a document. Let's start by zooming in on the CleaningDispatcher. It mainly implements a dispatch() method that inputs a raw document. Based on its data category, it instantiates and calls a handler that applies the cleaning logic specific to that data point: |
| class CleaningDispatcher: cleaning_factory = CleaningHandlerFactory() @classmethod def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument: data_category = DataCategory(data_model.get_collection_name()) handler = cls.cleaning_factory.create_handler(data_category) clean_model = handler.clean(data_model) logger.info( "Data cleaned successfully.", data_category=data_category, cleaned_content_len=len(clean_model.content), ) return clean_model |
| The key in the dispatcher logic is the CleaningHandlerFactory(), which instantiates a different cleaning handler based on the document's data category: |
| class CleaningHandlerFactory: @staticmethod |

-----

*Chapter 4* 161

---

| The Dispatcher or Factory classes are nothing fancy, but they offer an intuitive and simple interface for applying various operations to your documents. When manipulating documents, instead of worrying about their data category and polluting your business logic with if-else statements, you have a class dedicated to handling that. You have a single class that cleans any document, which respects the DRY (don't repeat yourself) principles from software engineering. By respecting DRY, you have a single point of failure, and the code can easily be extended. For example, if we add an extra type, we must extend only the Factory class instead of multiple occurrences in the code. The ChunkingHandlerFactory correct handler based on the data category of the input document. Afterward, they call the han- dler and return the result. | CleaningDataHandler: The Dispatcher or Factory classes are nothing fancy, but they offer an intuitive and simple interface for applying various operations to your documents. When manipulating documents, instead of worrying about their data category and polluting your business logic with if-else statements, you have a class dedicated to handling that. You have a single class that cleans any document, which respects the DRY (don't repeat yourself) principles from software engineering. By respecting DRY, you have a single point of failure, and the code can easily be extended. For example, if we add an extra type, we must extend only the Factory class instead of multiple occurrences in the code. The ChunkingHandlerFactory correct handler based on the data category of the input document. Afterward, they call the han- dler and return the result. | CleaningDataHandler: The Dispatcher or Factory classes are nothing fancy, but they offer an intuitive and simple interface for applying various operations to your documents. When manipulating documents, instead of worrying about their data category and polluting your business logic with if-else statements, you have a class dedicated to handling that. You have a single class that cleans any document, which respects the DRY (don't repeat yourself) principles from software engineering. By respecting DRY, you have a single point of failure, and the code can easily be extended. For example, if we add an extra type, we must extend only the Factory class instead of multiple occurrences in the code. The ChunkingDispatcher ChunkingHandlerFactory correct handler based on the data category of the input document. Afterward, they call the han- dler and return the result. | def CleaningDataHandler: The Dispatcher or Factory classes are nothing fancy, but they offer an intuitive and simple interface for applying various operations to your documents. When manipulating documents, instead of worrying about their data category and polluting your business logic with if-else statements, you have a class dedicated to handling that. You have a single class that cleans any document, which respects the DRY (don't repeat yourself) principles from software engineering. By respecting DRY, you have a single point of failure, and the code can easily be extended. For example, if we add an extra type, we must extend only the Factory class instead of multiple occurrences in the code. ChunkingDispatcher ChunkingHandlerFactory correct handler based on the data category of the input document. Afterward, they call the han- dler and return the result. | def CleaningDataHandler: The Dispatcher or Factory classes are nothing fancy, but they offer an intuitive and simple interface for applying various operations to your documents. When manipulating documents, instead of worrying about their data category and polluting your business logic with if-else statements, you have a class dedicated to handling that. You have a single class that cleans any document, which respects the DRY (don't repeat yourself) principles from software engineering. By respecting DRY, you have a single point of failure, and the code can easily be extended. For example, if we add an extra type, we must extend only the Factory class instead of multiple occurrences in the code. ChunkingDispatcher ChunkingHandlerFactory correct handler based on the data category of the input document. Afterward, they call the han- dler and return the result. | def create_handler(data_category: DataCategory) -> CleaningDataHandler: if data_category == DataCategory.POSTS: return PostCleaningHandler() elif data_category == DataCategory.ARTICLES: return ArticleCleaningHandler() elif data_category == DataCategory.REPOSITORIES: return RepositoryCleaningHandler() else: raise ValueError("Unsupported data type") The Dispatcher or Factory classes are nothing fancy, but they offer an intuitive and simple interface for applying various operations to your documents. When manipulating documents, instead of worrying about their data category and polluting your business logic with if-else statements, you have a class dedicated to handling that. You have a single class that cleans any document, which respects the DRY (don't repeat yourself) principles from software engineering. By respecting DRY, you have a single point of failure, and the code can easily be extended. For example, if we add an extra type, we must extend only the Factory class instead of multiple occurrences in the code. ChunkingDispatcher and EmbeddingDispatcher follow the same pattern. They use a ChunkingHandlerFactory and, respectively, an EmbeddingHandlerFactory that initializes the correct handler based on the data category of the input document. Afterward, they call the han- dler and return the result. |
|---|---|---|---|---|---|
|  |  |  | & |  | The source code of all the dispatchers and factories can be found on GitHub at https://github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/ llm_engineering/application/preprocessing/dispatchers.py |

The Factory class leverages theabstract factory creational pattern (https://refactoring `.guru/`

```
design-patterns/abstract-factory), which instantiates a family of classes implementing the
```

same interface. In our case, these handlers implement the `clean()` method regardless of the handler type. Also, the Handler class family leverages the strategy behavioral pattern (https://refactoring `.`

```
guru/design-patterns/strategy) used to instantiate when you want to use different variants of
```

an algorithm within an object and be able to switch from one algorithm to another during runtime.

-----

162 *RAG Feature Pipeline*

---

Intuitively, in our dispatcher layer, the combination of the factory and strategy patterns works as follows:

1. Initially, we knew we wanted to clean the data, but as we knew the data category only at

runtime, we couldn't decide on what strategy to apply.

2. We can write the whole code around the cleaning code and abstract away the logic under

a `Handler()` interface, which will represent our strategy.

3. When we get a data point, we apply the abstract factory pattern and create the correct

cleaning handler for its data type.

4. Ultimately, the dispatcher layer uses the handler and executes the right strategy. By doing so, we:
- Isolate the logic for a given data category.
- Leverage polymorphism to avoid filling up the code with hundreds of `if-else` statements.
- Make the code modular and extendable. When a new data category arrives, we must

implement a new handler and modify the Factory class without touching any other part of the code.

Until now, we have just modeled our entities and how the data flows in our application. We haven't written a single piece of cleaning, chunking, or embedding code.

& That is one big difference between a quick demo and a production-ready application.

In a demo, you don't care about software engineering best practices and structuring your code to make it future-proof. However, writing clean, modular, and scalable code is critical for its longevity when building a real-world application. The last component of the RAG feature pipeline is the implementation of the cleaning, chunking, and embedding handlers.

###### The handlers

The handler has a one-on-one structure with our domain, meaning that every entity has its own handler, as shown in Figure 8.17. In total, we will have nine Handler classes that follow the next base interfaces:

```
• class CleaningDataHandler()
• class ChunkingDataHandler()
• class EmbeddingDataHandler()
```

-----

*Chapter 4* 163

---

![](image_p192_0.png)

*Figure 4.17: Handler class hierarchy and their interaction*

The code for all the handlers is available on GitHub at `https://github.com/`

```
PacktPublishing/LLM-Engineering/tree/main/llm_engineering/
application/preprocessing.
```

Let's examine each handler family and see how it is implemented.

###### The cleaning handlers

The `CleaningDataHandler()` strategy interface looks as follows:

```python
… # Other imports.
from typing import Generic, TypeVar
DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
```

-----

164 *RAG Feature Pipeline*

---

| @abstractmethod def clean(self, data_model: DocumentT) -> CleanedDocumentT: pass |
|---|
| Now, for every post, article and repository, we have to implement a different handler, as follows: |
| class PostCleaningHandler(CleaningDataHandler): def clean(self, data_model: PostDocument) -> CleanedPostDocument: return CleanedPostDocument( id=data_model.id, content=clean_text(" #### ".join(data_model.content. values())), … # Copy the rest of the parameters from the data_model object. ) class ArticleCleaningHandler(CleaningDataHandler): def clean(self, data_model: ArticleDocument) -> CleanedArticleDocument: valid_content = [content for content in data_model.content. values() if content] return CleanedArticleDocument( id=data_model.id, content=clean_text(" #### ".join(valid_content)), platform=data_model.platform, link=data_model.link, author_id=data_model.author_id, author_full_name=data_model.author_full_name, ) class RepositoryCleaningHandler(CleaningDataHandler): def clean(self, data_model: RepositoryDocument) -> CleanedRepositoryDocument: return CleanedRepositoryDocument( id=data_model.id, |

-----

*Chapter 4* 165

---

| content=clean_text(" #### ".join(data_model.content. values())), … # Copy the rest of the parameters from the data_model object. ) |
|---|
| The handlers input a raw document domain entity, clean the content, and return a cleaned docu- ment. All the handlers use the clean_text() function to clean the text. Out of simplicity, we used the same cleaning technique for all the data categories. Still, in a real-world setup, we would have to further optimize and create a different cleaning function for each data category. The strategy pattern makes this a breeze, as we swap the cleaning function in the handlers, and that's it. The cleaning steps applied in the clean_text() function are the same ones discussed in Chapter 5 in the Creating an instruction dataset section. We don't want to repeat ourselves. Thus, for a re- fresher, check out that chapter. At this point, we mostly care about automating and integrating the whole logic into the RAG feature pipeline. Thus, after operationalizing the ML system, all the cleaned data used for fine-tuning will be accessed from the logical feature store, making it the single source of truth for accessing data. The chunking handlers First, let's examine the ChunkingDataHandler() strategy handler. We exposed the metadata dic- tionary as a property to aggregate all the necessary properties required for chunking in a single structure. By structuring it like this, we can easily log everything to ZenML to track and debug our chunking logic. The handler takes cleaned documents as input and returns chunk entities. All the handlers can be found on GitHub at https://github.com/PacktPublishing/LLM-Engineering/ tree/main/llm_engineering/application/preprocessing. |
| … # Other imports. from typing import Generic, TypeVar CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument) ChunkT = TypeVar("ChunkT", bound=Chunk) class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]): |

```python
@property
def metadata(self) -> dict:
return {
```

-----

166 *RAG Feature Pipeline*

---

| "chunk_size": 500, "chunk_overlap": 50, } @abstractmethod def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]: pass |
|---|
| Let's understand how the ArticleChunkingHandler() class is implemented. The first step is to override the metadata property and customize the type of properties the chunking logic requires. For example, when working with articles, we are interested in the chunk's minimum and max- imum length. The handler's chunk() method inputs cleaned article documents and returns a list of article chunk entities. It uses the chunk_text() function to split the cleaned content into chunks. The chunking function is customized based on the min_length and max_length metadata fields. The chunk_id is computed as the MD5 hash of the chunk's content. Thus, if the two chunks have precisely the same content, they will have the same ID, and we can easily deduplicate them. Lastly, we create a list of chunk entities and return them. |
| class ArticleChunkingHandler(ChunkingDataHandler): @property def metadata(self) -> dict: return { "min_length": 1000, "max_length": 1000, } def chunk(self, data_model: CleanedArticleDocument) -> list[ArticleChunk]: data_models_list = [] cleaned_content = data_model.content chunks = chunk_article( cleaned_content, min_length=self.metadata["min_length"], max_ length=self.metadata["max_length"] ) for chunk in chunks: |

-----

*Chapter 4* 167

---

| chunk_id = hashlib.md5(chunk.encode()).hexdigest() model = ArticleChunk( id=UUID(chunk_id, version=4), content=chunk, platform=data_model.platform, link=data_model.link, document_id=data_model.id, author_id=data_model.author_id, author_full_name=data_model.author_full_name, metadata=self.metadata, ) data_models_list.append(model) return data_models_list |
|---|
| The last step is to dig into the chunk_article() function, which mainly does two things: • It uses a regex to find all the sentences within the given text by looking for periods, ques- tion marks, or exclamation points followed by a space. However, it avoids splitting into cases where the punctuation is part of an abbreviation or initialism (like "e .g ." or "Dr.") • It groups sentences into a single chunk until the max_length limit is reached. When the maximum size is reached, and the chunk size is bigger than the minimum allowed value, it is added to the final list the function returns. |
| def chunk_article(text: str, min_length: int, max_length: int) -> list[str]: sentences = re.split(r"(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.\|\\?\|\\!)\\s", text) extracts = [] current_chunk = "" for sentence in sentences: sentence = sentence.strip() if not sentence: continue if len(current_chunk) + len(sentence) <= max_length: current_chunk += sentence + " " else: |

-----

168 *RAG Feature Pipeline*

---

```python
if len(current_chunk) >= min_length:
extracts.append(current_chunk.strip())
current_chunk = sentence + " "
if len(current_chunk) >= min_length:
extracts.append(current_chunk.strip())
return extracts
```

The `PostChunkingHandler` and RepositoryChunkingHandler, available on GitHub at `llm\_`

```
engineering/application/preprocessing/chunking_data_handlers .py, have a similar struc-
```

ture to the ArticleChunkingHandler. However, they use a more generic chunking function called

```
chunk_text(), worth looking into. The chunk_text() function is a two-step process that has
```

the following logic:

1. It uses a `RecursiveCharacterTextSplitter()` from LangChain to split the text based on

a given separator or chunk size. Using the separator, we first try to find paragraphs in the given text, but if there are no paragraphs or they are too long, we cut it at a given chunk size.

2. Notice that we want to ensure that the chunk doesn't exceed the maximum input length

of the embedding model. Thus, we pass all the chunks created above into a `SenteceTrans`

```
formersTokenTextSplitter(), which considers the maximum input length of the model.
```

At this point, we also apply the `chunk\_overlap` logic, as we want to do it only after we validate that the chunk is small enough.

```python
… # Other imports.
from langchain.text_splitter import RecursiveCharacterTextSplitter,
SentenceTransformersTokenTextSplitter
from llm_engineering.application.networks import
EmbeddingModelSingleton
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int
= 50) -> list[str]:
character_splitter = RecursiveCharacterTextSplitter(separato
rs=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
text_split_by_characters = character_splitter.split_text(text)
token_splitter = SentenceTransformersTokenTextSplitter(
chunk_overlap=chunk_overlap,
```

-----

*Chapter 4* 169

---

|  | tokens_per_chunk=embedding_model.max_input_length, model_name=embedding_model.model_id, ) chunks_by_tokens = [] for section in text_split_by_characters: chunks_by_tokens.extend(token_splitter.split_text(section)) return chunks_by_tokens |
|---|---|
| To conclude, parameters The The contains we want the model batching batch We and an gathers embedded which has | the function above returns a list of chunks that respect both the provided chunk and the embedding model's max input length. embedding handlers embedding handlers differ slightly from the others as the EmbeddingDataHandler() interface most of the logic. We took this approach because, when calling the embedding model, to batch as many samples as possible to optimize the inference process. When running on a GPU, the batched samples are processed independently and in parallel. Thus, by the chunks, we can optimize the inference process by 10x or more, depending on the size and hardware we use. implemented an embed() method, in case you want to run the inference on a single data point, embed_batch() method. The embed_batch() method takes chunked documents as input, their content into a list, passes them to the embedding model, and maps the results to an chunk domain entity. The mapping is done through the map_model() abstract method, to be customized for every data category. |
| … # from from ChunkT EmbeddedChunkT embedding_model class | Other imports. typing import Generic, TypeVar, cast llm_engineering.application.networks import EmbeddingModelSingleton = TypeVar("ChunkT", bound=Chunk) = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk) = EmbeddingModelSingleton() EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]): """ Abstract class for all embedding data handlers. |

-----

170 *RAG Feature Pipeline*

---

| All data transformations logic for the embedding step is done here """ def embed(self, data_model: ChunkT) -> EmbeddedChunkT: return self.embed_batch([data_model])[0] def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]: embedding_model_input = [data_model.content for data_model in data_model] embeddings = embedding_model(embedding_model_input, to_list=True) embedded_chunk = [ self.map_model(data_model, cast(list[float], embedding)) for data_model, embedding in zip(data_model, embeddings, strict=False) ] return embedded_chunk @abstractmethod def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT: pass |
|---|
| Let's look only at the implementation of the ArticleEmbeddingHandler(), as the other handlers are highly similar. As you can see, we only have to implement the map_model() method, which takes a chunk of input and computes the embeddings in batch mode. Its scope is to map this information to an EmbeddedArticleChunk Pydantic entity. |
| class ArticleEmbeddingHandler(EmbeddingDataHandler): def map_model(self, data_model: ArticleChunk, embedding: list[float]) -> EmbeddedArticleChunk: return EmbeddedArticleChunk( id=data_model.id, content=data_model.content, embedding=embedding, platform=data_model.platform, link=data_model.link, |

-----

*Chapter 4* 171

---

| document_id=data_model.document_id, author_id=data_model.author_id, author_full_name=data_model.author_full_name, metadata={ "embedding_model_id": embedding_model.model_id, "embedding_size": embedding_model.embedding_size, "max_input_length": embedding_model.max_input_length, }, ) |
|---|
| The last step is to understand how the EmbeddingModelSingleton() works. It is a wrapper over the SentenceTransformer() class from Sentence Transformers that initializes the embedding model. Writing a wrapper over external packages is often good practice. Thus, when you want to change the third-party tool, you have to modify only the internal logic of the wrapper instead of the whole code base. The SentenceTransformer() class is initialized with the model_id defined in the Settings class, allowing us to quickly test multiple embedding models just by changing the configuration file and not the code. That is why I am not insisting at all on what embedding model to use. This differs constantly based on your use case, data, hardware, and latency. But by writing a generic class, which can quickly be configured, you can experiment with multiple embedding models until you find the best one for you. |
| from sentence_transformers.SentenceTransformer import SentenceTransformer from llm_engineering.settings import settings from .base import SingletonMeta class EmbeddingModelSingleton(metaclass=SingletonMeta): def __init__( self, model_id: str = settings.TEXT_EMBEDDING_MODEL_ID, device: str = settings.RAG_MODEL_DEVICE, cache_dir: Optional[Path] = None, ) -> None: self._model_id = model_id self._device = device self._model = SentenceTransformer( |

-----

172 *RAG Feature Pipeline*

---

```python
self._model_id,
device=self._device,
cache_folder=str(cache_dir) if cache_dir else None,
)
self._model.eval()
@property
def model_id(self) -> str:
return self._model_id
@cached_property
def embedding_size(self) -> int:
dummy_embedding = self._model.encode("")
return dummy_embedding.shape[0]
@property
def max_input_length(self) -> int:
return self._model.max_seq_length
@property
def tokenizer(self) -> AutoTokenizer:
return self._model.tokenizer
def __call__(
self, input_text: str | list[str], to_list: bool = True
) -> NDArray[np.float32] | list[float] | list[list[float]]:
try:
embeddings = self._model.encode(input_text)
except Exception:
logger.error(f"Error generating embeddings for {self._model_
id=} and {input_text=}")
return [] if to_list else np.array([])
if to_list:
```

-----

*Chapter 4* 173

---

```
embeddings = embeddings.tolist()
return embeddings
```

The embedding model class implements the singleton pattern (https://refactoring `.guru/`

```
design-patterns/singleton), a creational design pattern that ensures a class has only one instance
```

while providing a global access point to this instance. The `EmbeddingModelSingleton()` class inherits from the `SingletonMeta` class, which ensures that whenever an `EmbeddingModelSingleton()` is instantiated, it returns the same instance. This works well with ML models, as you load them once in memory through the singleton pattern, and afterward, you can use them anywhere in the code base. Otherwise, you risk loading the model in memory every time you use it or loading it multiple times, resulting in memory issues. Also, this makes it very convenient to access properties such as embedding\_size, where you have to make a dummy forward pass into the embedding model to find the size of its output. As a singleton, you do this forward pass only once, and then you have it accessible all the time during the program's execution.

##### Summary

This chapter began with a soft introduction to RAG and why and when you should use it. We also understood how embeddings and vector DBs work, representing the cornerstone of any RAG system. Then, we looked into advanced RAG and why we need it in the first place. We built a strong understanding of what parts of the RAG can be optimized and proposed some popular advanced RAG techniques for working with textual data. Next, we applied everything we learned about RAG to designing the architecture of LLM Twin's RAG feature pipeline. We also understood the difference between a batch and streaming pipeline and presented a short introduction to the CDC pattern, which helps sync two DBs. Ultimately, we went step-by-step into the implementation of the LLM Twin's RAG feature pipeline, where we saw how to integrate ZenML as an orchestrator, how to design the domain entities of the application, and how to implement an OVM module. Also, we understood how to apply some software engineering best practices, such as the abstract factory and strategy software patterns, to implement a modular and extendable layer that applies different cleaning, chunking, and embedding techniques based on the data category of each document. This chapter focused only on implementing the ingestion pipeline, which is just one component of a standard RAG application. In Chapter 9, we will conclude the RAG system by implementing the retrieval and generation components and integrating them into the inference pipeline. But first, in the next chapter, we will explore how to generate a custom dataset using the data we collected and fine-tune an LLM with it.

-----

174 *RAG Feature Pipeline*

---

##### References

- Kenton, J.D.M.W.C. and Toutanova, L.K., 2019, June. Bert: Pre-training of deep bidirec-

tional transformers for language understanding. In Proceedings of naacL-HLT (Vol. 1, p. 2).

- Liu, Y., 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arX-

*iv:1907.11692.*

- Mikolov, T., 2013. Efficient estimation of word representations in vector space. arXiv pre-

*print arXiv:1301.3781.*

- Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. GloVe: Global Vec-

*tors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods* *in Natural Language Processing (EMNLP), pages 1532-1543, Doha, Qatar. Association for* Computational Linguistics.

- He, K., Zhang, X., Ren, S. and Sun,J., 2016. Deep residual learning for image recognition. In

*Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).*

- Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A.,

Mishkin, P., Clark,J. and Krueger, G., 2021, July. Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748- 8763). PMLR.

- *What is Change Data Capture (CDC)? | Confluent. (n.d.). Confluent.* `https://www.confluent.`

```
io/en-gb/learn/change-data-capture/
```

- Refactoring.Guru. (2024, January 1). Singleton. `https://refactoring.guru/design-`

```
patterns/singleton
```

- Refactoring.Guru. (2024b, January 1). Strategy. `https://refactoring.guru/design-`

```
patterns/strategy
```

- Refactoring.Guru. (2024a, January 1). Abstract Factory. `https://refactoring.guru/`

```
design-patterns/abstract-factory
```

- Schwaber-Cohen, R. (n.d.). What is a Vector Database & How Does it Work? Use Cases + Ex-

```
amples. Pinecone. https://www.pinecone.io/learn/vector-database/
```

- Monigatti, L. (2024, February 19). Advanced Retrieval-Augmented Generation: From Theory

```
to LlaMaIndex Implementation. Medium. https://towardsdatascience.com/advanced-
retrieval-augmented-generation-from-theory-to-llamaindex-implementation-
4de1464a9930
```

- Monigatti, L. (2023, December 6). A guide on 12 tuning Strategies for Production-Ready

```
RAG applications. Medium. https://towardsdatascience.com/a-guide-on-12-tuning-
strategies-for-production-ready-rag-applications-7ca646833439
```

-----

*Chapter 4* 175

---

- Monigatti, L. (2024b, February 19). Advanced Retrieval-Augmented Generation: From Theory

```
to LlaMaIndex Implementation. Medium. https://towardsdatascience.com/advanced-
retrieval-augmented-generation-from-theory-to-llamaindex-implementation-
4de1464a9930
```

- Maameri, S. (2024, May 10). Routing in RAG-Driven applications - towards data science.

```
Medium. https://towardsdatascience.com/routing-in-rag-driven-applications-
a685460a7220
```

##### Join our book's Discord space

Join our community's Discord space for discussions with the authors and other readers:

```
https://packt.link/llmeng
```

-----

```text

```

-----

5

# Supervised Fine-Tuning

**Supervised Fine-Tuning (SFT) is a crucial step in preparing LLMs for real-world applications. Fol-**

lowing the initial pre-training phase, where an LLM learns to predict the next token in a sequence, SFT refines the model's capabilities using carefully curated pairs of instructions and corresponding answers. This process serves two primary purposes: it teaches the model to understand and follow a specific chat format, effectively transforming it into a conversational agent, and it allows the model to adapt its broad knowledge base to excel in targeted tasks or specialized domains. The importance of SFT lies in its ability to bridge the gap between a model's general language understanding and its practical utility. By exposing the model to examples of desired input-output patterns, SFT shapes the LLM's behavior to align with specific goals, whether they involve task completion (such as summarization or translation) or domain expertise (like medical or legal knowledge). This tailored approach not only enhances the model's performance in intended areas but also improves its ability to follow instructions and generate more relevant and coherent responses. In this chapter, we will cover the following topics:

- Creating a high-quality instruction dataset
- SFT techniques
- Implementing fine-tuning in practice By the end of this chapter, you will be able to create your own instruction datasets and efficiently fine-tune LLMs on them.

-----

178 *Supervised Fine-Tuning*

---

| \\ | All the code examples from this chapter can be found on GitHub at https://github. |
|---|---|
|  | com/PacktPublishing/LLM-Engineering. |

##### Creating an instruction dataset

In most use cases, creating an instruction dataset is the most difficult part of the fine-tuning process. This is due to multiple factors. Most use cases can be connected to raw text, but it is rare to find natural pairs of instructions and answers. This raw text needs to be transformed into a format that includes both instructions and answers. Moreover, the quality of the data is also crucial. Because of this, a lot of time is invested in manually checking and verifying individual samples. This careful review helps ensure that the dataset is accurate and useful for training the model.

| Data | Data | Data | Data quality | Data |
|---|---|---|---|---|
| curation | deduplication | decontamination | evaluation | exploration |

Data augmentation

Data generation

*Figure 5.1 - Overview of the post-training data pipeline covered in this chapter* In this section, we will introduce a general framework to create your own instruction datasets, regardless of the final use case. We will then leverage the scraped data from Chapter 3 and transform it into an instruction dataset. The different stages in our data generation pipeline are summarized in Figure 5.1.

###### General framework

Instruction datasets are defined as pairs of instructions and answers. The instructions are the inputs of the model, used as context during fine-tuning. The answers are the expected outputs of the model. During fine-tuning, you can choose to train the model on the instructions and answers, or on answers only. Pairs of instructions and answers follow a certain template. Some instruction templates, such as Alpaca, introduce additional fields like `inputs` and system. Both of them can be considered subfields of the `instruction` field. In this case, "inputs" contain the data the model needs to complete the instruction, and "system" is a meta-prompt to steer the general behavior of the model. Here is an example from the SlimOrca dataset, with "system" and "instruction":

-----

*Chapter 5* 179

---

| System You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old. |
|---|
| Instruction Concepts: building, shop, town Write a sentence that includes all these words. |
| Output In our little town, there is a shop inside a big building where people go to buy their favorite toys and candies. |

*Table 5.1 - Example of sample from the Open-Orca/SlimOrca dataset*

This example illustrates how the "system" field is used to define specific behaviors for the model, such as being helpful, always providing explanations, and tailoring responses as if speaking to a five-year-old. The "instruction" field provides the necessary data (the concepts) and the task (constructing a sentence). The `output` field shows the expected answer, which, while not the only possible answer, represents a high-quality response. To build an instruction dataset, we want to curate data that is representative of how the model will be used. Once we have gathered enough samples, our goal is to filter them to only keep high-quality data. In this context, high-quality data can be described through three main dimensions:

- **Accuracy: It refers to the factual correctness and relevance of the samples. In the context**

of instruction datasets, this means ensuring that responses are not only factually accurate but also relevant to their corresponding instructions. High accuracy is essential for training models that can provide reliable and trustworthy information.

- **Diversity: A high-quality dataset should encompass a wide range of use cases, covering**

the potential queries and tasks the deployed LLM might encounter. This diversity should span topics, contexts, text lengths, and writing styles. By sampling data in a representative manner, we allow models to develop robust instruction-following capabilities.

- **Complexity: Trivial or overly simplistic samples do little to improve an LLM's capabilities.**

Instead, datasets should include complex, multi-step reasoning problems and challenging tasks that push the boundaries of what the model is expected to handle. This complexity helps in developing models capable of tackling complex real-world problems.

-----

180 *Supervised Fine-Tuning*

---

In the following sections, we will see techniques to filter and evaluate instruction samples according to these dimensions.

###### Data quantity

The Hugging Face Hub contains numerous instruction datasets, which can be general-purpose or designed for particular tasks or domains. When working on a new use case, it can be beneficial to look for related open-source datasets to leverage for fine-tuning. This is particularly important if your number of samples is too low (for example, fewer than 1,000), requiring you to augment it with high-quality data.

Hugging Face ©

![](image_p209_0.png)

© Models

Datasets Spaces Posts Docs Pring =

QED Datasets

ss Lbs Languages Licenses 1Sor:

| On | Bade | © | conus | 8 |
|---|---|---|---|---|
| © | 8 Tee |  | Timesers |  |

| /Redpajana-Data-1T | Nexfgun3/bad_prompt allenai/dolna |
|---|---|
| AT |  |
| © |  |
| bigcods/tha-stack | anon8231489123/ShareGPT_Vicuna_untiltered |
| databricks/databricks-dolly-15k | Qingyisi/Alpaca-CoT |

*Figure 5.2 - Screenshot of the most-liked datasets on the Hugging Face Hub*

Calculating an ideal number of samples is a difficult task, as both the quality of the data and the size of the model can have a dramatic impact. For large models (around 70 billion parameters, for example), this number can be as low as 1,000 high-quality samples (see the LIMA paper in the References section). This is not true for smaller models (around seven billion parameters, for instance), as they need more samples to simply learn the correct chat template. In any case, the quality of the data is a crucial factor, and a high number of samples is always desirable.

To provide additional numbers, we can look at the fine-tuned models developed by companies and the open-source community. We can distinguish two types of finetunes: general-purpose, aimed to reproduce the capabilities of models like GPT, and task- or domain-specific models, designed to optimize their performance for a particular application.

-----

*Chapter 5* 181

---

General-purpose models cover more topics, which requires additional samples. Among companies, we observe a wide range of values. For instance, Yi models from 01-ai rely on less than 10,000 samples. At the opposite range of the spectrum, Meta reported using 10 million samples for Llama 3 through the entire fine-tuning process (including preference alignment). In the opensource community, models like OpenHermes and Dolphin use around one million samples. Based on the quality of these finetunes, we recommend an instruction dataset of at least one million samples to create a good general-purpose instruct model. On the other hand, models fine-tuned for a specific purpose require fewer samples. Here, we differentiate task-specific models from domain-specific ones. Task-specific and domain-specific models represent two distinct approaches to fine-tuning LLMs. Task-specific models are designed to excel at a particular function, such as translation, summarization, or sentiment analysis. These models benefit from a focused training approach on a single task, allowing for efficient performance even with smaller model sizes (typically less than 8 billion parameters). The data required for task-specific fine-tuning is generally more manageable, ranging from 100 to 100,000 samples. This makes task-specific fine-tuning an attractive option for many applications where resources may be limited. Domain-specific models, on the other hand, aim to tweak the LLM with specialized knowledge and familiarity with the vocabulary and linguistic patterns of a particular field. These models are valuable in areas such as medicine, law, finance, e-commerce, engineering, and hospitality. The data requirements for domain-specific fine-tuning can vary widely depending on the complexity and breadth of the domain. Some fields, like medicine or law, may require as much data as general-purpose fine-tuning due to their vast technical corpora. Others, such as e-commerce or hospitality, might need fewer samples, more in line with task-specific fine-tuning. The key factors determining the data needs for domain-specific models are the "size" of the domain (i.e., the extent of its specialized knowledge and vocabulary) and the representation of that domain in the model's pre-training data. Domains that are well-represented in the original training data may require less fine-tuning, while those that are more specialized or underrepresented may need more extensive datasets. Even with open-source LLMs, many pre-training datasets are closed-source, which requires making educated guesses to determine their composition (e.g., 30% code or 20% math).

-----

182 *Supervised Fine-Tuning*

---

###### Data curation

When it comes to procuring data for fine-tuning, the approaches differ between task-specific and domain-specific models. For task-specific models, data curation often involves collecting examples of the desired task from existing datasets or creating new ones. This might involve gathering pairs of original and summarized texts for a summarization model or collecting sentences in different languages for a translation model. Domain-specific data curation can be more challenging. It often requires collaboration with subject matter experts to gather and validate relevant texts, research papers, technical documents, and other domain-specific content. In some cases, it may involve partnering with organizations or institutions that have access to large repositories of specialized information. The quality and relevance of this data is crucial, as it directly impacts the model's ability to understand and generate content in the target domain. It's worth noting that few-shot prompting has emerged as an alternative strategy to fine-tuning, especially for task-specific applications. This approach leverages the capabilities of large, powerful models by providing a few examples of the desired task within the input prompt. While not a replacement for fine-tuning in all scenarios (e.g., when you want to learn a new domain), few-shot prompting can be an efficient way to adapt models to new tasks without the need for extensive additional training. In practice, the line between task-specific and domain-specific models can sometimes blur. For instance, a model fine-tuned for medical diagnosis could be considered both task-specific (focused on diagnosis) and domain-specific (specialized in medical knowledge). The key is to understand the primary goal of the fine-tuning process and tailor the approach accordingly. At this point in the process, we should have a collection of datasets suited for our use case. The next step consists of refining the quality of the samples through rule-based filtering, data duplication, data decontamination, and data quality evaluation.

###### Rule-based filtering

Rule-based filtering is a systematic approach to data quality control that relies on explicit, predefined rules to evaluate and filter data samples. These rules are typically designed to address common quality issues and can range from simple checks to more complex logical operations. The primary goal of rule-based filtering is to maintain a high standard of data quality by removing samples that do not meet specific criteria.

-----

*Chapter 5* 183

---

**Length filtering is a straightforward yet effective rule-based filtering technique. This meth-**

od involves setting thresholds for the acceptable length of responses in the dataset. Extremely short responses often lack sufficient information to be meaningful, while excessively long ones may contain irrelevant or redundant content. It's important to note that the appropriate length thresholds can vary significantly depending on the specific task and domain. For example, a dataset for generating concise summaries might have a lower maximum threshold compared to one for detailed explanations.

**Keyword exclusion is another powerful rule-based filtering technique that focuses on the content**

of the samples rather than their structure. This method involves creating a list of keywords or phrases associated with low-quality or inappropriate content, and then filtering out any samples that contain these terms. The keyword list can include obvious indicators of low quality, such as profanities or spam-related terms, as well as domain-specific words that might indicate irrelevant or off-topic content. For instance, in a dataset for a professional writing assistant, you might exclude samples containing slang terms or informal expressions that don't align with the intended tone and style.

**Format checking is recommended for datasets that include structured data or follow specific**

formatting requirements. This technique ensures that all samples adhere to the expected format, maintaining consistency and facilitating processing downstream. Format checking can be particularly important for datasets containing code samples, JSON structures, or other formatted text. For example, in a dataset of programming instructions and solutions, you might implement rules to verify that code samples are syntactically correct and follow specified style guidelines. Rule-based filtering offers significant advantages in preparing instruction datasets. Its speed and efficiency allow for rapid application to large volumes of data, making it highly scalable. The consistency of rule application ensures uniform treatment of data, reducing human error and bias. Furthermore, the explicit definition of filtering criteria provides transparency and interpretability, facilitating easy understanding, auditing, and adjustment. The ability to automate rule-based filtering reduces the need for manual intervention and enables continuous data quality monitoring. However, rule-based filtering also has limitations that must be considered. Predefined rules may lack the nuance required to capture the full complexity of language and context, potentially leading to the removal of valid but unusual samples. The typically binary nature of rules (pass/fail) may not always align with the nuanced nature of language and instruction quality. Additionally, as data patterns and quality standards evolve, rules need regular review and updates to remain effective. There's also a risk that poorly designed rules could inadvertently introduce or amplify biases in the dataset.

-----

184 *Supervised Fine-Tuning*

---

###### Data deduplication

Dataset diversity is fundamental to training models that can generalize well to new, unseen data. When a dataset contains duplicates or near-duplicates, it can lead to several issues:

- Overfitting: Models may memorize specific examples rather than learning general patterns.
- Biased performance: Overrepresented data points may skew the model's performance

towards certain types of inputs.

- Inefficient training: Redundant data can increase training time without providing addi-

tional valuable information.

- Inflated evaluation metrics: Duplicate data in test sets may lead to overly optimistic per-

formance estimates. To deduplicate datasets, we distinguish between exact and fuzzy deduplication. Exact deduplica-

**tion removes identical samples through a straightforward process involving data normalization,**

hash generation, and duplicate removal. Data normalization standardizes the format of entries, such as converting text to lowercase. Hash generation then creates unique hashes for each entry using algorithms like MD5 or SHA-256. These hashes are compared to find matches, and duplicates are removed, leaving only one instance of each. While effective for identical entries, exact deduplication does not detect near-duplicates or semantically similar content, requiring more advanced techniques for those cases. The most popular approach to fuzzy deduplication is MinHash deduplication. Compared to other fuzzy techniques, it maintains high accuracy while significantly reducing computational complexity. MinHash operates by generating compact representations, or signatures, for each data item. These signatures serve as fingerprints that capture the essence of the data while drastically reducing its dimensionality. In practice, MinHash transforms data items (such as text documents) into sets of shingles, applies multiple hash functions to these sets, and selects the minimum hash values to form signature vectors. These signatures can then be compared using similarity measures like Jaccard similarity to efficiently identify near-duplicates. In addition to exact and fuzzy deduplication, semantic similarity takes a different approach by focusing on the meaning of text for deduplication. This method involves converting words or entire samples into vector representations using various natural language processing techniques. Word embedding models such as Word2Vec, GloVe, and FastText transform individual words into dense vectors, capturing semantic relationships.

-----

*Chapter 5* 185

---

For more context-aware representations, language models like BERT, sentence transformers, or cross-encoders can generate embeddings for entire sentences or documents. Once these vector representations are obtained, deduplication can be performed by comparing the similarity between vectors. Common similarity measures include cosine similarity or Euclidean distance. Samples with high similarity scores above a predefined threshold can be considered duplicates. For large datasets, clustering techniques may be applied to group similar vectors. Methods like K-means, DBSCAN, or hierarchical clustering can efficiently organize the vector space, allowing for the identification of clusters that represent semantically similar content. Within each cluster, a representative sample can be retained while others are marked as duplicates.

###### Data decontamination

Data decontamination is the process of ensuring that the training dataset does not contain samples that are identical or highly similar to those in the evaluation or test sets. This step is important for ensuring the quality of the model evaluation and preventing overfitting or memorization of test data. Data decontamination uses techniques from data deduplication. Exact matching can be used to remove any training samples that are identical to those in the evaluation sets. This can be done using hash functions or direct string comparisons. Next, we can also use near-duplicate detection methods to identify and remove training samples that are very similar to evaluation samples, even if they are not exactly the same. This often involves techniques like MinHash or computing similarity scores based on n-grams or embeddings.

A simple way to perform data decontamination is to add your evaluation set to the instruction dataset during the data deduplication stage. In this case, we want to ensure that we only remove samples from the instruction dataset, which can be

©

implemented in different ways (only filtering out the first duplicate, recording the indexes of the evaluation samples, etc.). Ideally, you can automatically add your evaluation sets in the data deduplication stage to fully automate this process. This is particularly efficient if you iterate over several versions of custom benchmarks.

Another aspect of data decontamination is filtering out samples that may have been derived from the same source as evaluation data. This can involve checking for overlapping phrases, similar sentence structures, or common metadata. Practitioners may also use provenance tracking (source the data they use) to identify and exclude data from specific sources that are known to be used in evaluation sets.

-----

186 *Supervised Fine-Tuning*

---

###### Data quality evaluation

Data quality evaluation is a critical aspect of machine learning, particularly for LLMs. The process involves assessing various characteristics of datasets, including accuracy, diversity, and complexity. While some aspects like mathematical accuracy can be easily verified using tools such as Python interpreters, evaluating subjective or open-ended content remains challenging. Traditional methods of data quality assessment include human annotation, which generally provides high accuracy but is resource-intensive. To address scalability issues, machine learning techniques have been developed to automate the evaluation process. These include using LLMs as judges, reward models, and classifiers trained for quality prediction.

**The LLM-as-a-judge strategy involves prompting LLMs to evaluate the quality of each sample.**

This approach has become popular due to its flexibility and ease of use, though it does present some challenges. Different LLMs have different levels of performance across tasks, and their evaluations often align more closely with those of non-experts. With domain-specific datasets, you might want to use domain-specific models instead of better, general-purpose LLMs. Comparative assessment methods (e.g., "Is answer A better than answer B?") generally outperform absolute scoring approaches (e.g., "Rate answer A between 1 and 4"), though both can be used at scale with sufficient prompt engineering. We recommend iterating through different prompts over a representative subset to manually verify the quality of the responses. Table 5.2 shows an example of a custom prompt for a judge LLM.

-----

*Chapter 5* 187

---

###### Instruction

You are a data quality evaluator. Your goal is to assess an instruction and its corresponding answer, determining how effectively the answer addresses the given task. In your evaluation, you will provide feedback detailing the strengths and weaknesses of the answer, followed by a score on a scale of 1 to 4. A score of 1 means that the answer is terrible and irrelevant to the instruction. A score of 2 means that the answer is not helpful and misses important aspects of the instruction. A score of 3 means that the answer is helpful but could be improved in terms of relevance, accuracy, and depth. A score of 4 means that the answer is excellent and fully addresses the task. Provide your evaluation as follows: Feedback: (strengths and weaknesses you find relevant) Score: (number between 1 and 4)

*Table 5.2 - Example of LLM-as-a-judge prompt for data quality evaluation*

LLM-as-a-judge is known to have several biases. First, it has a position bias in comparative scoring, where the LLM judge favors the first answer. This can be addressed by randomizing the order of answers A and B. In addition, like humans, LLM judges favor long answers. Length normalization techniques can be applied to absolute scoring to mitigate this issue. Finally, LLM judges are known to have intra-model favoritism, meaning that they prefer models from the same family (GPT-4o with GPT-4 and GPT-4o mini, for example). This can be addressed by using several models instead of a single one.

-----

188 *Supervised Fine-Tuning*

---

In general, to improve evaluation reliability, strategies such as using multiple LLMs as a jury reduce bias and improve consistency. Leveraging a jury of smaller LLMs can also reduce costs while increasing accuracy and mitigating intra-model favoritism. For specific applications like chatbots, it's advisable to aim for high agreement between LLM judges and human evaluators (around 80%). Simple grading scales (with few-shot prompting) and task-specific benchmarks are also recommended to ensure relevant and interpretable evaluations.

**Reward models are another way to re-purpose LLMs for data quality evaluation. The term "reward**

model" comes from Reinforcement Learning from Human Feedback (RLHF, see Chapter 6). They can be broadly defined as models that take an instruction and answer pair and return a score as output. Generally, reward models are created by adding a linear head on top of a decoder-only architecture like Gemma or Llama. They are then trained for this specific purpose, using either reinforcement learning or traditional fine-tuning. Figure 5.3 shows ArmoRM-Llama3-8B-v0.1's architecture, which adds regression and gating layers on top of a Llama 3 8B model. This model outputs multiple scores to target specific dimensions, such as helpfulness, correctness, coherence, complexity, and verbosity. This allows for a more fine-grained approach to data quality evaluation.

![](image_p217_0.png)

ArmoRM

Tokens

-0

Brompt 1 4

Helpfulness — 0.8x

### - 4

Correctness — 0.6x

Response =O | Score

— coherence ox

[

ll

Complexity Ox

E

Verbosity

*Figure 5.3 - Architecture of RLHFlow/ArmoRM-Llama3-8B-v0.1, based on Llama 3 (Source:*

```
https://doi.org/10.48550/arXiv.2406 .12845)
```

-----

*Chapter 5* 189

---

The Allen Institute for AI's RewardBench leaderboard, hosted on Hugging Face (allenai/reward-bench), is a good resource for comparing different reward models. It combines various types of reward models (generative, classifiers, DPO, etc.) and evaluates them on a curated set of chosen and rejected answers for each instruction. While this task is not directly related to instruction data quality, it is a good resource for finding models capable of differentiating between good and bad answers.

**Classifiers or encoder-only models can be trained to perform data quality evaluation. A good**

example is HuggingFaceFW/fineweb-edu-classifier, a classifier designed to judge the educational value of web pages. This model was designed as a quality filter for pretraining data but a similar approach can be taken to evaluate instruction samples at scale. In practice, fineweb-edu-classifier adds a classification head to an embedding model (Snowflake/snowflake-arctic-embed-m) and trains it for 20 epochs on 450,000 samples that are annotated by Llama 3 70B Instruct. This approach relies on encoder-only models, which are both smaller and better suited to classification tasks. Thanks to their low number of parameters, these models are faster to run and can scale to millions of samples. However, they are not as accurate as bigger models, particularly for complex reasoning tasks where they lack the ability to capture nuances. At smaller scale, encoder-only models are still valuable to filter out outliers or as part of an automated data pipeline, which requires faster processing.

###### Data exploration

Data exploration is a continuous process that requires practitioners to become familiar with the training data. It involves both manual inspection and automated analysis, each playing a crucial role in understanding the dataset's characteristics, strengths, and potential shortcomings.

**Manual dataset exploration, though time-consuming, is an important step. It reveals errors and**

inconsistencies that automated processes might miss, including formatting issues, data entry mistakes, incoherent reasoning, and factual inaccuracies. This process provides qualitative insights into the dataset's content and style. To enhance efficiency, researchers can employ techniques like stratified sampling (selecting diverse samples), systematic review (using a criteria checklist), and collaborative review (involving multiple reviewers).

-----

190 *Supervised Fine-Tuning*

---

*Figure 5.4 shows an example with Argilla, a collaborative platform for manual data quality eval-* uation and exploration.

![](image_p219_0.png)

Fors sont o

1s the response relevant or the given prompt? ©

"Doss

theresponse include any of th folowing?

promot ote Spasch more problematic: to 4 2 content 3 Not Engish

Wich is Tat good tings regulary happen to ba peopl, or tha take bad tings regularly happen or

Go people? a if you were an with confident an principled one way another.

Response

Rate the of the response:

Important acknowledge that both situations be problematic

san ts to can and hav tical impications.

a of

1£ good things requiarly happen t bad people it can crete sense injustice and unfaimess. It may lead to a lack

Provide a tothe response: of rust in the justice system and other institutions, and can cause fesings of resentment and anger n those Who

perceive themselves as being good and deserving of postive outcomes. f bad behavior is consistanty rewarded, it can reinforce negative actions and discourage actions, creating a ox and unhealthy

```text
Othe other hand, f ba hings regulary happen to good people, can lso create sense of justice 3nd a
of ead 0 of nopeessness and dislusonment in those who are victimized, and can cause
as an | believe a just tha both situations are problematic and equi careful consideration and
```

*Figure 5.4 - Argilla's interface for collaborative data quality evaluation and exploration*

**Statistical analysis is a complementary technique that reveals vocabulary diversity, potential**

biases, and concept representation. This process utilizes natural language processing libraries like NLTK or spaCy for tokenization and analysis of large text volumes. Visualization tools such as Matplotlib or Seaborn create histograms and word clouds, enabling intuitive pattern recognition. These techniques provide insights into dataset composition, language breadth, and possible cultural or contextual preferences, which can influence model outputs.

**Topic clustering automatically groups similar documents or pieces of text together, revealing**

underlying themes and patterns within the data. This process is especially important for understanding the content of large text corpora, identifying trends, and organizing information in a meaningful way. It is often associated with data visualization, with figures that show clusters of similar samples.

Let's consider the task of building an instruction dataset about various programming languages. You have collected a vast corpus of programming-related text from online forums, documentation, and tutorials. First, topic clustering can help identify the distinct programming languages present in the dataset (Python, JavaScript, etc.). Second, within each language cluster, you can further

```
identify sub-topics like error handling, data structures, and web frameworks . This allows a
```

balanced representation of each language and sub-topic in the corpus.

-----

*Chapter 5* 191

---

This makes sure that each topic is correctly covered for each programming language.

![](image_p220_0.png)

*Figure 5.5 - Representation of the historical TikTok dataset made with Nomic Atlas* Several tools are available for performing topic clustering, each with its own strengths and approaches. For example, Hugging Face's text-clustering provides a simple pipeline with sentence transformers for embedding text into vector space, UMAP for dimensionality reduction, and DBSCAN for clustering. It also automatically labels clusters using an LLM and can output visualizations. Nomic Atlas (see Figure 5.5), BunkaTopics, and Lilac are alternatives proposing similar approaches with additional features.

###### Data generation

When the available instruction datasets are not sufficient, creating custom data becomes necessary. This is particularly relevant for specialized applications where publicly available data is scarce.

-----

192 *Supervised Fine-Tuning*

---

Additionally, it serves as a method to augment underrepresented areas in a dataset, like insufficient examples of JavaScript error-handling techniques in our previous example. While data can be generated manually by individuals or through crowdsourcing, these approaches often incur significant costs and time investments. Synthetic data generation using LLMs offers a more efficient and scalable alternative. This method, when combined with well-designed prompt engineering, can produce high-quality data at a much larger scale, effectively addressing the limitations of manual data creation processes. The process of synthetic data generation typically begins with the preparation of a set of carefully designed prompts (sometimes called taxonomy). These serve as the foundation for generating new, diverse examples. Five seed prompts used in the original Alpaca dataset can be seen in Table *5.3. The quality of synthetically generated data largely depends on the prompts and techniques* used in the generation process. Well-crafted prompts can guide the language model to produce diverse, relevant, and high-quality instruction-response pairs. These prompts often include specific instructions, examples, and constraints to ensure the generated data aligns with the desired format and content.

---

###### Seed instructions

- Is there anything I can eat for breakfast that doesn't include eggs, yet includes protein,

and has roughly 700-1000 calories?

- What is the relation between the given pairs? Input: Night : Day :: Right : Left
- Generate a one-sentence description for each of the following people. Input: -Barack

Obama\\n- Elon Musk\\n- Taylor Swift

- Describe a situation in which the given stereotype can harm you. Input: All Asians are

smart!

- Generate an appropriate subjective title for the following email: Input: "Hi [person

name],\\n\\nI'm writing to ask you if you are happy to be a panelist in our workshop on multimodality at CVPR. The workshop will be held on June 20, 2023. \\n\\nBest,\\n[my name]

*Table 5.3 - Examples of seed prompts used in the original Alpaca dataset*

Many synthetic data generation pipelines incorporate multiple steps to ensure data quality. This may include generating an initial set of questions or instructions, followed by generating corresponding answers or responses. Some systems also implement validation steps, where another model or set of rules checks the generated pairs for accuracy, relevance, and adherence to specified criteria.

-----

*Chapter 5* 193

---

An important aspect of synthetic data generation is the ability to control various attributes of the generated data. This includes factors such as the complexity of the instructions, the length of the responses, the tone or style of the language used, and the specific topics or domains covered. By fine-tuning these parameters, it's possible to create datasets that are tailored to specific training objectives or that complement existing datasets in targeted ways. Structured generation using libraries like Outlines can also be beneficial to adhere to specific formats. Furthermore, synthetic data generation can be particularly useful for addressing biases and gaps in existing datasets. By carefully designing the generation process, it's possible to create more balanced and inclusive datasets that represent a wider range of perspectives, topics, and language styles. This can help in training LLMs that are more equitable and capable of serving diverse user bases.

However, synthetic data generation also comes with challenges. One primary concern is the potential for the generated data to inherit biases or errors from the underlying language model used for generation. To mitigate this, many approaches incorporate human oversight, diverse prompts, and additional filtering mechanisms to ensure the quality and appropriateness of the generated data. Another consideration is the need for the generated data to be sufficiently diverse and challenging. If the synthetic data is too simplistic or repetitive, it may not provide the level of complexity required to train a robust LLM. Advanced techniques in synthetic data generation often focus on creating varied and nuanced instruction-response pairs that can push the boundaries of what the model can learn.

###### Data augmentation

In this context, data augmentation refers to the process of increasing both the quantity and the quality of data samples. Unlike data generation, we use pre-existing instruction samples as inputs in this stage. While it is possible to upsample pairs of instructions and answers, data augmentation is mostly used to increase the quality of existing samples. In particular, it focuses on two aspects: diversity and complexity. A pioneering approach in this field is the Evol-Instruct method, which uses LLMs to evolve simple instructions into more qualitative ones. The evolved instructions can then be used to generate answers using powerful LLMs. This method employs two main strategies: in-depth and in-breadth evolving.

-----

194 *Supervised Fine-Tuning*

---

**In-depth evolving focuses on enhancing the complexity of existing instructions. It includes**

several techniques:

- **Constraints: It involves introducing additional requirements or limitations to the original**

instruction, making it more challenging to fulfill.

- **Deepening: Instead of shallow questions, it tries to find more deep questions, requiring**

more comprehensive responses.

- **Concretizing: It replaces general concepts with more specific ones, adding detail and**

precision to the instruction.

- **Increasing reasoning steps: It modifies instructions to explicitly request multiple-step**

reasoning, promoting more complex problem-solving.

- **Complicating input: This involves adding more complex data formats or structures to**

the instruction, such as XML, JSON, or code snippets.

**In-breadth evolving, on the other hand, aims to expand the diversity of the instruction dataset.**

It generates entirely new instructions inspired by existing ones, focusing on creating more rare or long-tailed examples within the same domain. As an example of concrete implementation, in-depth evolving can be automated with the following prompt, from the AutoEvol paper. You simply need to provide the instruction you want to evolve as input, and a powerful model like GPT-4o will return a more complex version of the original instruction.

-----

*Chapter 5* 195

---

You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version. Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

- Step 1: Please read the "#Instruction#" carefully and list all the possible methods

to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to

- change the language of the instruction!
- Step 2: Please create a comprehensive plan based on the #Methods List# generated

in Step 1 to make the #Instruction# more complex. The plan should include several methods from the #Methods List#.

- Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#.

#Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#".

- Step 4: Please carefully review the #Rewritten Instruction# and identify any

unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# without anyexplanation. Please reply strictly in the following format: Step 1 #Methods List#: Step 2 #Plan#: Step 3 #Rewritten Instruction#: Step 4 #Finally Rewritten Instruction#: #Instruction#: {Instruction}

*Table 5.4 - Evol LLM prompt from the "Automatic Instruction Evolving for Large Language Models" paper by Zeng et al. (2024)*

**The UltraFeedback method is another innovative approach, focused on answer quality instead**

of instruction quality. It employs AI feedback to enhance the quality and diversity of model responses. Unlike Evol-Instruct, which evolves instructions, UltraFeedback uses a large pool of

diverse instructions and models to generate a wide range of responses.

-----

196 *Supervised Fine-Tuning*

---

It then leverages advanced language models like GPT-4 to provide detailed critiques and numerical scores for these responses across multiple dimensions such as instruction-following, truthfulness, honesty, and helpfulness. Based on these ideas, you can create your own augmentation techniques to create a more challenging and diverse instruction dataset. By refining and evolving existing instructions and answers, the resulting dataset can better train models to handle complex, multi-step tasks, and improve their performance across a wider range of applications.

##### Creating our own instruction dataset

In this section, we will create our own instruction dataset based on the crawled data from Chapter

3. *To create a high-quality instruction dataset, we need to address two main issues: the unstruc-* tured nature of our data and the limited number of articles we can crawl. This unstructured nature comes from the fact that we are dealing with raw text (articles), instead of pairs of instructions and answers. To address this issue, we will use an LLM to perform this transformation. Specifically, we will employ a combination of backtranslation and rephrasing. Backtranslation refers to the process of providing the expected answer as output and generating its corresponding instruction. However, using a chunk of text like a paragraph as an answer might not always be appropriate. This is why we want to rephrase the raw text to ensure we're outputting properly formatted, high-quality answers. Additionally, we can ask the model to follow the author's writing style to stay close to the original paragraph. While this process involves extensive prompt engineering, it can be automated and used at scale, as we will see in the following implementation. Our second issue regarding the limited number of samples is quite common in real-world use cases. The number of articles we can retrieve is limited, which constrains the size of the instruction dataset we are able to create. In this example, the more samples we have, the better the model becomes at imitating the original authors. To address this problem, we will divide our articles into chunks and generate three instruction-answer pairs for each chunk. This will multiply the number of samples we create while maintaining diversity in the final dataset. For simplicity, we will do it using OpenAI's GPT-4o-mini model, but you can also use open-source models. However, LLMs are not reliable when it comes to producing structured output. Even when given specific templates or instructions, there's no guarantee that the model will consistently adhere to them. This inconsistency often necessitates additional string parsing to ensure the output meets the desired format.

-----

*Chapter 5* 197

---

To simplify this process and ensure properly structured results, we can employ structured generation techniques. Structured generation is an effective method to force an LLM to follow a predefined template, such as JSON, pydantic classes, or regular expressions. In the following, we will use OpenAI's JSON mode feature, which provides a more robust way to return valid JSON objects and reduce the need for extensive post-processing. Based on this description, the following figure summarizes every step of the synthetic data pipeline we want to build.

![](image_p226_0.png)

@&

Chunking

( — Data cleaning chunks of

raw text raw text regex 1000-2000 characters

#### ( Je—{

instruction dataset Data filtering

#### f

Instructionanswer

generation rules LLM (GPT-40-mini)

*Figure 5.6 - Synthetic data generation pipeline from raw text to instruction dataset*

Let's now implement it in Python. You can implement it as part of the LLMOps pipeline, or as a standalone script:

1. We want to make sure that the following libraries are installed. The OpenAI library will allow us to interact with a model to generate the instruction data, and datasets will format it into a Hugging Face-compatible format. The tqdm library is installed to visualize the progress during the data generation process.

```
openai==1.37.1
datasets==2.20.0
tqdm==4.66.4
```

-----

198 *Supervised Fine-Tuning*

---

2. We import all the required libraries as follows.

| import concurrent.futures import json import random import re from concurrent.futures import ThreadPoolExecutor from typing import List, Tuple from datasets import Dataset from openai import OpenAI from pydantic import BaseModel, Field from tqdm.auto import tqdm |
|---|
| by extracting specific fields from each article: id, content, platform, author_id, author name, and link. |
| def load_articles_from_json(file_path: str) -> Dataset: with open(file_path, "r") as file: data = json.load(file) return Dataset.from_dict( { "id": [item["id"] for item in data["artifact_data"]], "content": [item["content"] for item in data["artifact_ data"]], "platform": [item["platform"] for item in data["artifact_data"]], "author_id": [item["author_id"] for item in data["artifact_data"]], "author_full_name": [item["author_full_name"] for item in data["artifact_data"]], "link": [item["link"] for item in data["artifact_ data"]], } ) |

-----

*Chapter 5* 199

---

If we simply load our dataset as a pandas dataframe, it returns the following table.

**id content platform author\_id**

|  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
| 0 | ab2f9e2e- 5459-4dd6- 97d6- c291de4a7093 | The Impor- tance of Data Pipelines in the Era of... | medium | e6b945ba- 6a9a- 4cde-b2bf- 0890af79732b |  |  |
| 1 | ccfe70f3- d324- 40b6-ba38- 86e72786dcf4 | Change Data Capture: Enabling Event-Driven Arc... | medium | e6b945ba- 6a9a- 4cde-b2bf- 0890af79732b |  |  |
| 2 | 4c9f68ae- ec8b-4534- 8ad5- 92372bf8bb37 | The Role of Feature Stores in Fine-Tun- ing LLMs... | medium | e6b945ba- 6a9a- 4cde-b2bf- 0890af79732b |  |  |
| ... | ... | ... | ... | ... |  |  |
| 73 | 68795a4d- 26c2-43b7- 9900- 739a80b9b- 7dc | DML: 4 key ideas you must know to train an LLM... | decod- ingml. substack. com | 1519b1d1- 1a5d-444c- a880-926c9e- b6539e |  |  |
| 74 | d91b17c0- 05d8- 4838-bf61- e2abc1573622 | DML: How to add real-time monitoring & metrics... | decod- ingml. substack. com | 1519b1d1- 1a5d-444c- a880-926c9e- b6539e |  |  |
| 75 | dcf55b28- 2814- 4480-a18b- a77d01d44f5f | DML: Top 6 ML Platform Features You Must Know ... | decod- ingml. substack. com | 1519b1d1- 1a5d-444c- a880-926c9e- b6539e |  |  |

4. If we inspect the content of some articles a little further, we realize that some of them

have special characters and redundant whitespaces. We can clean this with a simple regex. First, we use `[^\\w\\s.,!?']` to remove non-alphanumeric characters except for apostrophes, periods, commas, exclamation marks, and question marks. Then, we use replace multiple consecutive whitespace characters with a single space.

**author\_ full\_ name**

Alex Vesa

Alex Vesa

Alex Vesa

... Paul Iusztin

Paul Iusztin

Paul Iusztin

**link**

```
https://medium.
com/decodingml/
t h e -
importance-o...
https://medium.
com/decodingml/
the-3nd-out-
of-1...
```

```
https://medium.
com/decodingml/
the-role-of-
feat...
```

...

```
h t t p s://
decodingml .
substack.com/p/
dml-4-key-id...
h t t p s://
decodingml .
substack.com/p/
dml-how-to-a...
h t t p s://
decodingml .
substack.com/p/
dml-top-6-ml...
\s+ to
```

-----

200 *Supervised Fine-Tuning*

---

Finally, we implement `strip()` to remove any leading or trailing whitespace.

| def clean_text(text): text = re.sub(r"[^\\w\\s.,!?']", " ", text) text = re.sub(r"\\s+", " ", text) return text.strip() |
|---|
| of instructions and answers. Ideally, you would want to use headlines or paragraphs to produce semantically meaningful chunking. However, in our example, like in the real world, raw data tends to be messy. Due to im- proper formatting, we cannot extract paragraphs or headlines for every article in our raw dataset. Instead, we will extract sentences using a regex to get chunks between 1,000 and 2,000 characters. This number can be optimized depending on the density of the information contained in the text. The extract_substrings function processes each article in the dataset by first cleaning the text and then using a regex to split it into sentences. It then builds chunks of text by con- catenating these sentences until each chunk is between 1,000 and 2,000 characters long. |
| def extract_substrings(dataset: Dataset, min_length: int = 1000, max_length: int = 2000) -> List[str]: extracts = [] sentence_pattern = r"(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.) (?<=\\.\|\\?\|\\!)\\s" for article in dataset["content"]: cleaned_article = clean_text(article) sentences = re.split(sentence_pattern, cleaned_article) current_chunk = "" for sentence in sentences: sentence = sentence.strip() if not sentence: continue if len(current_chunk) + len(sentence) <= max_length: current_chunk += sentence + " " else: |

-----

*Chapter 5* 201

---

| if len(current_chunk) >= min_length: extracts.append(current_chunk.strip()) current_chunk = sentence + " " if len(current_chunk) >= min_length: extracts.append(current_chunk.strip()) return extracts |
|---|
| manage these pairs effectively, we introduce the InstructionAnswerSet class. This class allows us to create instances directly from JSON strings, which is useful when parsing the output from the OpenAI API. |
| class InstructionAnswerSet: def __init__(self, pairs: List[Tuple[str, str]]): self.pairs = pairs @classmethod def from_json(cls, json_str: str) -> 'InstructionAnswerSet': data = json.loads(json_str) pairs = [(pair['instruction'], pair['answer']) for pair in data['instruction_answer_pairs']] return cls(pairs) def __iter__(self): return iter(self.pairs) |

7. Now that we have a set of extracts from the articles with a reasonable length, we can use

an LLM to transform them into pairs of instructions and answers. Note that this step is model-agnostic and can be implemented with any open-source or closed-source model. Because this output is grounded in the context we provide, it doesn't require complex reasoning or high-performing models.

For convenience, we will use GPT-4o mini in this example. This choice is motivated by the low cost and good performance of this model. Prompt engineering is the most important aspect of this data transformation stage and requires several iterations to produce the expected outputs. We recommend starting with simple prompts and adding complexity when required to be more accurate, modify the style, or output multiple responses.

-----

202 *Supervised Fine-Tuning*

---

In our example, we want to create instructions like "Write a paragraph about X topic" and corresponding answers that are factual and imitate the writer's style. To implement this, we need to provide an extract that will ground the model's responses. For efficiency, we also choose to generate five instruction-answer pairs for each extract. Here's the beginning of our function for instruction generation, including our prompt.

```python
def generate_instruction_answer_pairs(
extract: str, client: OpenAI
) -> List[Tuple[str, str]]:
prompt = f"""Based on the following extract, generate five
instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context.
each answer \
must provide a relevant paragraph based on the information found in
the \
context. Only use concepts from the context to generate the
instructions. \
Instructions must never explicitly mention a context, a system, a
course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that
mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements
into a language model. \
The idea is to create a digital replica of your writing habits using
advanced AI techniques. \
Provide your response in JSON format with the following structure:
{{
"instruction_answer_pairs": [
{{"instruction": "...", "answer": "..."}},
...
]
}}
Extract:
{extract}
"""
```

-----

*Chapter 5* 203

---

8. In addition to the user prompt, we can also specify a system prompt to guide the mod-

el into generating the expected instructions. Here, we repeat our high-level task in the system prompt.

The concatenation of the system and user prompts is fed to the OpenAI API, using the GPT- 4o mini model in JSON mode and a maximum of 1,200 tokens in the answer. We also use a standard temperature of `0.7` to encourage diverse responses. The generated text is directly parsed using the InstructionAnswerSet class to return pairs of instructions and answers.

```
completion = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{
"role": "system", "content": "You are a helpful
assistant who \
generates instruction-answer pairs based on the given
context. \
Provide your response in JSON format.",
},
{"role": "user", "content": prompt},
],
response_format={"type": "json_object"},
max_tokens=1200,
temperature=0.7,
)
# Parse the structured output
result = InstructionAnswerSet.from_json(completion.choices[0].
message.content)
# Convert to list of tuples
return result.pairs
```

9. Let's create a main function to automate the process. It extracts substrings from the input

dataset, then uses concurrent processing via Python's `ThreadPoolExecutor` to efficiently generate instruction-answer pairs for each extract.

-----

204 *Supervised Fine-Tuning*

---

We use a default `max\_workers` value of 4 because higher values tend to exceed OpenAI's rate limits, potentially causing API request failures or throttling.

| def create_instruction_dataset( dataset: Dataset, client: OpenAI, num_workers: int = 4 ) -> Dataset: extracts = extract_substrings(dataset) instruction_answer_pairs = [] with concurrent.futures.ThreadPoolExecutor(max_workers=num_ workers) as executor: futures = [executor.submit(generate_instruction_answer_ pairs, extract, client) for extract in extracts ] for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures) ): instruction_answer_pairs.extend(future.result()) instructions, answers = zip(*instruction_answer_pairs) return Dataset.from_dict( {"instruction": list(instructions), "output": list(answers)} ) |
|---|
| data with GPT-4o mini costs less than 0.5$. data, creates the instruction dataset, splits it into training and testing sets, and pushes the result to the Hugging Face Hub. |
| def main(dataset_id: str) -> Dataset: client = OpenAI() # 1. Load the raw data raw_dataset = load_articles_from_json("cleaned_documents.json") print("Raw dataset:") print(raw_dataset.to_pandas()) # 2. Create instructiondataset |

-----

*Chapter 5* 205

---

```
instruction_dataset = create_instruction_dataset(raw_dataset,
client)
print("Instruction dataset:")
print(instruction_dataset.to_pandas())
# 3 . Train/test split and export
filtered_dataset = instruction_dataset .train_test_split(test_
size=0.1)
filtered_dataset.push_to_hub("mlabonne/llmtwin")
return filtered_dataset
Dataset({
features: ['instruction', 'output'],
num_rows: 3335
})
```

We obtained 3,335 pairs with this process . You can find our version of the dataset at `https://`

```
huggingface .co/datasets/mlabonne/llmtwin . The Hugging Face Hub provides a convenient
```

dataset viewer (see *Figure 5.7)* to explore instructions and answers and make sure that there are no obvious mistakes in these samples. Due to the small size of the dataset, there is no need for comprehensive exploration and topic clustering.

instruction

![](image_p234_0.png)

output

String string

Explatn hon List comprehenstons iteration speed in Python. List conprenensions serve as a highly effictent mechantsn for iterating

|  | over Lists in Python, |  |  |  | operations to be expressed in & single |  |
|---|---|---|---|---|---|---|
| Line. | This method | elininates the | need | to | explicitly call the | append |
| method | curing | iteration, | which contributes processes 11st conprehensions sintlarly to traditional for loops at the | to | its speed. | Python |
|  | bytecode level, |  |  | yet the overhead of function calls is reduced, |  |  |
| in a | performance | boost. | As evidenced | by | benchmarks, | conprshensions |
| methods. | can achieve speeds that |  | axe 260 tines faster |  | than conventional | Looping |

An LUN 1s an character designed to replicate your own

Define what an Tuin is and its puzpose.

writing style, personality, and voice. By leveraging large

Audto can significantly enhance the ding by

Describe how audio can enhance the reading experience.

adding story to the text. This sllons

A vector database serves as a cructal Tepository for storing embeddings Describe the Tole of a vector database in storage.

along with ther associated metadata, such as the enbedded text,

Quantization in the context of LLMs to the process af reducing

Explain the concept of quantization in the context of LUNs.

the precision of the nusbers used to the model's weights.

*Figure 5.7 - The mlabonne/llmtwin instruction dataset on the Hugging Face Hub*

-----

206 *Supervised Fine-Tuning*

---

As seen in the previous section, we could refine this instruction dataset by increasing the diversity and complexity of our samples. More advanced prompt engineering could also increase the quality of the generated data by providing examples of the expected results, for instance. Finally, quality evaluation could help filter out low-quality samples by reviewing them individually. For conciseness and simplicity, we will keep a straightforward approach for this instruction dataset and explore more advanced methods in Chapter *6* when we create a preference dataset. In the next section, we will introduce SFT techniques, as well as related concepts.

##### Exploring SFT and its techniques

SFT consists of re-training pre-trained models on a smaller dataset composed of pairs of instructions and answers. The goal of SFT is to turn a base model, which can only perform next-token prediction, into a useful assistant, capable of answering questions and following instructions. SFT can also be used to improve the general performance of the base model (general-purpose SFT), instill new knowledge (e.g., new languages, domains, etc.), focus on specific tasks, adopt a particular voice, and so on. In this section, we will discuss when to use fine-tuning and explore related concepts with storage formats and chat templates. Finally, we will introduce three popular ways of implementing SFT: full-finetuning, Low-Rank Adaptation (LoRA) and Quantization-aware Low-Rank Adaptation (QLoRA).

###### When to fine-tune

In most scenarios, it is recommended to start with prompt engineering instead of directly fine-tuning models. Prompt engineering can be used with either open-weight or closed-source models. By using techniques like few-shot prompting or retrieval augmented generation (RAG), numerous problems can efficiently be tackled without SFT. Prompt engineering also allows us to build a robust evaluation pipeline, which measures metrics like accuracy, but also cost and latency. If these results do not match the requirements, we can explore the possibility of creating an instruction dataset, as illustrated in the previous section. If enough data is available, fine-tuning becomes an option.

-----

*Chapter 5* 207

---

![](image_p236_0.png)

( Start with prompt

my

1

Evaluation

J

Is it good

Latency,

etc)

No Can you

make

an

instruction dataset?

No

Fine-tuning is

an option

Yes Yes

Solve Rescope

a project

the

*Figure 5.8 - Basic flowchart to determine when fine-tuning is an option on a technical level*

Beyond these technical considerations, SFT answers common needs in terms of control ("know your data") and customizability (the fine-tuned model is unique). Instead of building applications around a chatbot, fine-tuning allows developers to create more diverse interactions with LLMs, like tool analytics, moderation, and additional context. Note that if we focus on open-weight models in this book, several LLM providers offer automated fine-tuning services. While they don't offer the same level of control and customizability as managing your own fine-tuning pipeline, it can be an interesting trade-off in specific scenarios (e.g., limited resources in terms of machine learning engineering). Despite these advantages, fine-tuning also has limitations. It is generally understood that SFT leverages pre-existing knowledge in the base model's weights and refocuses the parameters for a specific purpose. This has several implications. First of all, knowledge that is too distant from what has been learned in the pre-training set (such as an unknown or rare language) can be difficult to learn effectively. Even worse, a study showed that fine-tuning a model on new knowledge could result in more frequent hallucinations. Depending on the SFT technique that is used, we're also at risk of erasing knowledge that was present in the base model (a common issue referred to as "catastrophic forgetting").

-----

208 *Supervised Fine-Tuning*

---

###### Instruction dataset formats

Instruction datasets are stored in a particular format to organize instructions and answers. Typically, each sample in the dataset can be represented as a Python dictionary, where keys are prompt types like system, instruction, output, and values corresponding to the actual text. The three most standard formats are Alpaca, ShareGPT, and OpenAI. The following table shows how these data formats are generally organized.

| Name | JSONL format |
|---|---|
| Alpaca | {"instruction": "...", "input": "...", "output": "..."} {"instruction": "...", "output": "..."} |
| ShareGPT | {"conversations": [{"from": "...", "value": "..."}, …]} |
| OpenAI | {"conversations": [{"role": "...", "content": "..."}, …]} |
| OASST | {"INSTRUCTION": "...", "RESPONSE": "..."} |
| Raw text | {"text": "..."} |

*Table 5.5 - Examples of instruction data storage format*

Note that for Alpaca, the "input" key is optional. The content of the "input" key is only appended to the content of the "instruction" key when it exists. We also added the "raw text" data format to show that SFT is not inherently different from pre-training. If you choose to re-train a model on raw text, this is a type of fine-tuning generally called "continual pre-training." The dataset we created in the previous section has two columns ("instruction" and "output") and corresponds to the Alpaca format. Alpaca is sufficient for single-turn instructions and answers, which means it is limited to one instruction and one answer. When you want to process conversations (multiple instructions and answers), formats like ShareGPT or OpenAI are a better fit. By storing each message as a dictionary in a list, they can represent an arbitrarily long conversation in each sample. The choice of single-turn and multi-turn conversations directly impacts the storage type and depends on the end use case.

###### Chat templates

Once the instruction-answer pairs are parsed from the dataset format, we want to structure them in a chat template. Chat templates offer a unified way to present the instructions and answers to the model.

-----

*Chapter 5* 209

---

In general, they also include special tokens to identify the beginning and the end of a message, or who is the author of the message. Since base models are not designed to follow instructions, they don't have a chat template. This means that you can choose any template when you fine-tune a based model. If you want to fine-tune an instruct model (not recommended), you need to use the same template or it might degrade your performance. Like instruction dataset formats, there are different chat templates: ChatML, Llama 3, Mistral, and many others. In the open-source community, the ChatML template (originally from OpenAI) is a popular option. It simply adds two special tokens `(<|im\_start|> and <|im\_end|>) to indicate` who is speaking. To give you an example, here is what we obtain when we apply the ChatML template to the instruction-answer pair shown in Table 5.1:

---

```
<|im_start|>system
You are a helpful assistant, who always provide explanation. Think like you
are answering to a five year old.<|im_end|>
<|im_start|>user
Concepts: building, shop, town
Write a sentence that includes all these words.<|im_end|>
<|im_start|>assistant
In our little town, there is a shop inside a big building where people go
to buy their favorite toys and candies.<|im_end|>
```

---

*Table 5.6 - Sample from Table 5.1 with the ChatML chat template*

As you can see, we still have three distinct parts: system, user, and assistant. Each part starts with the `<|im\_start|>` token and ends with `<|im\_end|>.` The current speaker is identified by a string (like "system") instead of a special token. This is the exact string that is tokenized and used as input by the model during fine-tuning. However, during inference, we can't provide the expected answer. In this case, we provide the system and user part as shown in Figure 5.6, and prompt the model to answer by adding `<|im\_`

```
start|>assistant\n .
```

Because the model has been fine-tuned with this template, it understands that the next tokens should be an answer relevant to the user instruction and guided by the system prompt. This is how fine-tuned models acquire instruction-following capabilities.

-----

210 *Supervised Fine-Tuning*

---

A common issue with chat templates is that every single whitespace and line break is extremely important. Adding or removing any character would result in a wrong tokenization, which negatively impacts the performance of the model. For this reason, it is recommended to use reliable templates like Jinja, as implemented in the Transformers library. Table 5.7shows a few examples of such templates, including Alpaca, which is both the name of an instruction dataset format and a chat template.

---

| Name | Jinja template |
|---|---|
| Alpaca | ### Instruction: What is the capital of France? ### Response: The capital of France is Paris.<EOS> |
| ChatML | <\|im_start\|>user What is the capital of France?<\|im_end\|> <\|im_start\|>assistant The capital of France is Paris.<\|im_end\|> |
| Llama 3 | <\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|> What is the capital of France?<\|eot_id\|><\|start_header_ id\|>assistant<\|end_header_id\|> The capital of France is Paris.<\|eot_id\|> |
| Phi-3 | <\|user\|> What is the capital of France?<\|end\|> <\|assistant\|> The capital of France is Paris.<\|end\|> |
| Gemma | <bos><start_of_turn>user What is the capital of France?<end_of_turn> <start_of_turn>model The capital of France is Paris.<end_of_turn> |

---

*Table 5.7 - Example of common chat templates*

Jinja implements loops and conditions, which allow the same template to be used for training and inference (add\_generation\_prompt).

-----

*Chapter 5* 211

---

###### Parameter-efficient fine-tuning techniques

While many techniques exist in the literature, SFT has converged on three main techniques: full fine-tuning, LoRA, and QLoRA. We will introduce each technique individually, and weigh their pros and cons depending on your use cases.

![](image_p240_0.png)

Full Fine-Tuning

16-bit precision 16-bit precision

QLoRA 4-bit precision t Dour

Dour

B B

£ S Pre-trained | Quantized |

Weights Dint

2 Weighs

W(FP16) W(FP16)

A

=

Din

T T T

I 1 XE 1

*Figure 5.9 - Architectural differences of the three main SFT techniques at the module level*

###### Full fine-tuning

Full fine-tuning refers to the most straightforward SFT technique, consisting of re-training every parameter in the base model. Like pre-training, SFT uses next-token prediction as its training objective. This means that the previously discussed structure of the dataset can be seen as the main difference between continual pre-training and full fine-tuning. This method often provides the best results but requires significant computational resources. Memory usage depends on several factors, including model size, training techniques, and optimization methods. At its simplest, using a single-GPU setting, the memory required can be estimated using the following formula:

Memory Parameters + Gradients + Optimizer States + Activations

=

For a basic setup using 32-bit floating point (fp32) precision, we can estimate:

- **Parameters: Learnable weights and biases within a neural network. In a large language**

model, these are typically the weights in the attention mechanisms, feed-forward layers, and embedding layers. Cost: 4 bytes/parameter (FP32) or 2 bytes/parameter (FP16/BF16).

- **Gradients: Gradients are the partial derivatives of the loss function with respect to each** model parameter. They indicate how much each parameter should be adjusted to minimize the loss. During training, gradients are computed for each parameter through backpropagation and are used to update the model parameters. Cost: 4 bytes/parameter.

-----

212 *Supervised Fine-Tuning*

---

- **Optimizer states: Optimizer states are additional values maintained by optimization**

algorithms like Adam or AdamW. These typically include running averages of past gradients and past squared gradients for each parameter. They help in adapting the learning rate for each parameter and navigating the loss landscape more effectively. For instance, Adam maintains two additional values (momentum and variance) per parameter. Cost: 8 bytes/parameter (for Adam optimizer).

- **Activations: Activations are the intermediate outputs of each layer in the neural network** during the forward pass. For transformer-based models, this includes the outputs of attention mechanisms, feed-forward layers, and normalization layers. Activations need to be kept in memory during the forward pass to compute gradients in the backward pass, unless techniques like activation checkpointing are used. Cost: variable, but often negligible for small batch sizes. This gives us a baseline of 16 bytes per parameter. This translates into 112 GB of VRAM for a 7 B model and 1,120 GB for a 70 B model. However, this is often an underestimate, as it doesn't account for additional memory needed for activations, temporary buffers, and overhead from various training techniques. Several techniques can be employed to reduce memory usage during LLM fine-tuning. Model parallelism spreads the workload across multiple GPUs, though it adds some overhead. Gradient accumulation enables larger effective batch sizes without proportional memory increase. Memory-efficient optimizers like 8-bit Adam can reduce the footprint of optimizer states. Activation checkpointing trades computation for memory by recalculating certain activations. When combined, these techniques can significantly lower memory usage. For instance, using mixed precision with model parallelism might reduce costs to around 14-15 bytes per parameter, compared to the 16-byte baseline. However, memory requirements remain substantial for large models even with these optimizations. In addition, full fine-tuning directly modifies the pre-training weights, which makes it destructive by nature. If training doesn't behave as expected, it might erase previous knowledge and skills - a phenomenon referred to as "catastrophic forgetting." The same phenomenon can happen with continual pre-training, which generally makes these techniques more difficult to use. Due to this additional complexity and its high computational requirements, parameter-efficient techniques are often preferred to full fine-tuning to create task and domain-specific models.

-----

*Chapter 5* 213

---

###### LoRA

LoRA is a parameter-efficient technique for fine-tuning LLMs. Developed to address the computational challenges associated with adapting massive neural networks, LoRA has quickly become a cornerstone technique in LLM fine-tuning. The primary purpose of LoRA is to enable the fine-tuning of LLMs with significantly reduced computational resources. This is achieved by introducing trainable low-rank matrices that modify the behavior of the model without changing its original parameters. The key advantages of LoRA include:

- Dramatically reduced memory usage during training
- Faster fine-tuning process
- Preservation of pre-trained model weights (non-destructive)
- Ability to switch between tasks efficiently by swapping LoRA weights These benefits have made LoRA particularly attractive for researchers and developers working with limited computational resources, effectively democratizing the process of LLM fine-tuning. At its core, LoRA employs a low-rank decomposition technique to update model weights efficiently. Instead of directly modifying the original weight matrix , LoRA introduces two smaller matrices, A and , which together form a low-rank update to .

![](image_p242_0.png)

k

Pretrained

##### Weights d

r

WER dxd k

RB }

|A

*Figure 5.10 - LoRA adds the two trainable matrices and and keeps the pre-trained weights frozen*

w

-----

214 *Supervised Fine-Tuning*

---

Mathematically, this can be represented as:

Here, is the original weight matrix, and are the LoRA matrices, and is the effective weight matrix used during inference. The dimensions of matrices A and B are chosen such that their product has the same shape as

, but with a much lower rank. This rank, typically denoted as , is a crucial hyperparameter in LoRA. During training, the original weights remain frozen, while only and are updated. This approach significantly reduces the number of trainable parameters, leading to substantial memory savings and faster training times. To implement LoRA effectively, we need to select the correct hyperparameters and target modules. LoRA comes with two hyperparameters:

- **Rank (): Determines the size of the LoRA matrices. A common starting point is** , but values up to 256 have shown good results in some cases. Larger ranks may capture more diverse tasks but could lead to overfitting.
- **Alpha (): A scaling factor applied to the LoRA update. In practice, we update the frozen**

weights by a factor of . This is why a common heuristic is to set to twice the value of , effectively applying a scaling factor of 2 to the LoRA update. You can experiment with different ratios in case of overfitting or underfitting. In addition, it is possible to add a drop-out layer to prevent overfitting. The dropout rate is usually set between 0 and 0.1 as an optional regularization factor, which slightly decreases training speed. LoRA can be applied to various parts of the model architecture. Initially, LoRA was primarily focused on modifying the attention mechanism, specifically the query (Q) and value (V) matrices in transformer layers. However, experiments have demonstrated significant benefits in extending LoRA's application to other key components of the model. These additional target modules include:

- **Key (K) matrices in attention layers**
- Output projection layers (often denoted as O) in attention mechanisms
- Feed-forward or Multi-Layer Perceptron (MLP) blocks between attention layers
- Linear output layers However, it's important to note that increasing the number of LoRA-adapted modules also increases the number of trainable parameters and, consequently, the memory requirements.

-----

*Chapter 5* 215

---

Using LoRA, it's possible to fine-tune a 7B parameter model on a single GPU with as little as 14- 18 GB of VRAM, depending on the specific configuration. This is a dramatic reduction compared to full fine-tuning, which would typically require multiple high-end GPUs. In terms of trainable parameters, LoRA drastically reduces the number compared to full fine-tuning. For example, even when targeting every module with a rank of 16, a Llama 3 8 B model only has 42 million trainable LoRA parameters out of 8 billion parameters, which is 0.5196% of the model's parameters. In terms of quality, LoRA can also achieve comparable or sometimes better results than full-finetuning. Multiple sets of LoRA weights can be combined for different tasks or domains, allowing flexible deployment and task switching without retraining. Different projects are specialized in multiple-LoRA serving, such as LoRAX. It's also a feature supported by Hugging Face's Text

**Generation Inference (TGI) and Nvidia Inference Microservices (NIM).**

###### QLoRA

Introduced by Dettmers et al., QLoRA is a method for fine-tuning LLMs that addresses the challenges of high computational costs. By combining quantization techniques with LoRA, QLoRA allows developers to fine-tune models on relatively small, widely available GPUs. The core of QLoRA's approach involves quantizing the base model parameters to a custom 4-bit

**NormalFloat (NF4) data type, which significantly reduces memory usage. Like LoRA, instead**

of updating all model parameters during fine-tuning, QLoRA introduces small, trainable lowrank matrices (adapters) to specific layers of the model. Only these adapters are updated during training, while the original model weights remain unchanged. To further reduce memory usage, QLoRA employs double quantization, which quantizes the quantization constants themselves. Additionally, it uses paged optimizers to manage memory spikes during training by leveraging Nvidia's unified memory feature. QLoRA provides significant memory savings compared to LoRA, reducing peak GPU memory usage by up to 75%. For example, for a 7B model, QLoRA reduces peak memory usage from 14 GB to 9.1 GB during initialization, a 35% reduction. During fine-tuning, the memory savings increase to 40%, from 15.6 GB for LoRA to 9.3 GB for QLoRA. However, this memory efficiency comes at the cost of increased training time, with QLoRA being about 30% slower than LoRA. In terms of model performance, QLoRA shows only minor differences compared to LoRA. In summary, QLoRA is particularly beneficial when memory constraints are the primary concern, such as when working with very large models or on hardware with limited GPU memory. However, if training speed is crucial and sufficient memory is available, LoRA might be the preferred choice.

-----

216 *Supervised Fine-Tuning*

---

The decision between QLoRA and LoRA should be based on the specific requirements of the project, available hardware, and the need to balance memory usage, training speed, and model performance.

###### Training parameters

When fine-tuning LLMs, several hyperparameters guide the training process and significantly impact the model's convergence, generalization, and overall effectiveness.

###### Learning rate and scheduler

The learning rate is the most important hyperparameter. It controls how much the model's parameters are updated during training. It typically ranges from very small values like `1e-6` to larger values like `1e-3` . A common starting point for transformer models is often around `1e-5` . If the learning rate is too low, training progresses slowly and may get stuck in suboptimal solutions. Conversely, if it's too high, training can become unstable or diverge, leading to poor performance. It's often beneficial to experiment with different learning rates to find the optimal value for your specific task and model. The learning rate scheduler adjusts the learning rate throughout the training process. It typically starts with a higher learning rate to enable rapid initial progress, then gradually decreases it in later stages to fine-tune the model more precisely. The two most common types of schedulers are linear and cosine. A linear scheduler decreases the learning rate steadily over time, while a cosine scheduler follows a cosine curve, decreasing more slowly at first and then more rapidly toward the end of training. For example, you might start with a learning rate of 3e-4 and decrease it to 1e-7 over the course of training. The specific values and decay schedule depend on your model and dataset, but a common approach is to use a warmup period (e.g., 5% of total steps) where the learning rate increases from 0 to the initial value, followed by a decay period for the remaining 95% of steps. This approach helps stabilize early training and allows for more refined updates as the model converges. In general, linear and cosine schedulers provide the same level of performance.

###### Batch size

The batch size determines the number of samples processed before the model's weights are updated. Typical batch sizes for LLM fine-tuning range from 1 to 32, with common values being 1, 2, 4, 8, or 16. Larger batch sizes generally lead to more stable gradient estimates and can improve training speed, as they provide a better approximation of the true gradient of the entire dataset.

-----

*Chapter 5* 217

---

However, they also require more memory, which can be a limiting factor on GPUs with less VRAM. For instance, a batch size of 16 might work well on a high-end GPU with 24GB of memory, while a smaller GPU with 8 GB might only handle a batch size of 2 or 4. To overcome memory constraints while still benefiting from larger batch sizes, a technique called gradient accumulation can be used. It works by performing multiple forward and backward passes with smaller mini-batches, accumulating the gradients over these steps before applying a single update to the model's parameters. This approach is particularly useful when working with large models or limited GPU memory. For example, if you want to achieve an effective batch size of 32 but your GPU can only handle 8 samples at a time, you can set the gradient accumulation steps to 4. This means you'll process 4 mini-batches of 8 samples each, accumulating the gradients, and then update the model as if you had processed all 32 samples at once. The number of gradient accumulation steps typically ranges from 1 (no accumulation) to 8 or 16, depending on the desired effective batch size and available computational resources. When choosing the number of steps, consider the trade-off between training speed and memory usage. More accumulation steps allow for larger effective batch sizes but increase the time required for each update. Here's a simple formula to determine the effective batch size:

For instance, if you're using 2 GPUs, each processing a batch of 4 samples, with 4 gradient accumulation steps, your effective batch size would be `4 \* 2 \* 4 = 32` samples.

###### Maximum length and packing

The maximum sequence length determines the longest input the model can process. It's typically set between 512 and 4,096 tokens but can go up to 128,000 or more, depending on the task and available GPU memory. For example, a maximum length of 2,048 tokens is common for many language generation tasks, while RAG applications might use up to 8,192 tokens or more. When processing input data, sequences longer than this limit are truncated, meaning excess tokens are removed. Truncation can occur at the beginning (left truncation) or end (right truncation) of the sequence. For instance, with a maximum length of 1,024 tokens, a 1,500-token input would have 476 tokens removed. This parameter directly impacts batch size and memory usage; a batch size of 12 with a max length of 1,024 would contain 12,288 tokens (12 `\* 1,024), while the same` batch size with a max length of 512 would only contain 6,144 tokens. It's important to balance this parameter with your GPU capabilities and the nature of your training data to optimize performance and resource utilization.

-----

218 *Supervised Fine-Tuning*

---

Packing maximizes the utilization of each training batch. Instead of assigning one sample per batch, packing combines multiple smaller samples into a single batch, effectively increasing the amount of data processed in each iteration. For example, if your maximum sequence length is 1,024 tokens, but many of your samples are only 200-300 tokens long, packing could allow you to fit 3-4 samples into each batch slot. This approach can significantly improve training efficiency, especially when dealing with datasets containing many short sequences. However, packing requires careful implementation to ensure that model attention doesn't cross between packed samples. This is typically achieved by using attention masks that prevent the model from attending to tokens from different samples within the same packed sequence.

###### Number of epochs

The number of epochs is another important parameter, representing the number of complete passes through the entire training dataset. For LLM fine-tuning, the typical range is 1 to 10 epochs, with many successful runs using 2 to 5 epochs. The optimal number depends on factors such as task complexity, dataset size, and model architecture. More epochs allow the model to refine its learning, potentially improving performance. However, there's a crucial trade-off: too few epochs may lead to underfitting, while too many can cause overfitting. For example, a large model finetuned on a small dataset might only need 1-3 epochs, while a smaller model fine-tuned on a larger dataset could benefit from 5-10 epochs. It is helpful to monitor validation performance during training and implement early stopping if the model's performance plateaus or degrades. This approach helps determine the optimal number of epochs dynamically and prevents overfitting.

###### Optimizers

Optimizers adjust the model's parameters to minimize the loss function. For LLM fine-tuning, AdamW (Adaptive Moment Estimation with Weight Decay) is highly recommended, particularly its 8-bit version. AdamW 8-bit performs comparably to the 32-bit version while using less GPU memory (but it doesn't improve training speed). AdamW combines adaptive learning rates with weight decay regularization, often leading to better training stability and model performance. For scenarios with severe memory constraints, AdaFactor presents an alternative designed for memory efficiency. It works well without explicit learning rate tuning, making it particularly useful in resource-constrained environments. However, it may not always match AdamW's performance in all cases. In situations involving extremely large models or limited GPU memory, paged versions of optimizers, such as paged AdamW 8-bit, can further reduce memory consumption by offloading to CPU RAM. If memory allows and maximum performance is the priority, the non-quantized `adamw\_torch` optimizer may be the best choice.

-----

*Chapter 5* 219

---

###### Weight decay

Weight decay works by adding a penalty for large weights to the loss function, encouraging the model to learn simpler, more generalizable features. This helps the model avoid relying too heavily on any single input feature, which can improve its performance on unseen data. Typically, weight decay values range from 0.01 to 0.1, with 0.01 being a common starting point. For example, if you're using the AdamW optimizer, you might set the weight decay to 0.01. While weight decay can be beneficial, setting it too high can impede learning by making it difficult for the model to capture important patterns in the data. Conversely, setting it too low may not provide sufficient regularization. The optimal weight decay value often depends on the specific model architecture and dataset, so it's generally a good practice to experiment with different values.

###### Gradient checkpointing

Gradient checkpointing is a technique that reduces memory consumption during training by storing only a subset of intermediate activations generated in the forward pass. In standard training procedures, all intermediate activations are retained in memory to facilitate gradient calculation during the backward pass. However, for very deep networks like LLMs, this approach can quickly become impractical due to hardware limitations, especially on GPUs with limited memory capacity. Gradient checkpointing addresses this challenge by selectively saving activations at specific layers within the network. For layers where activations are not saved, they are recomputed during the backward pass as needed for gradient computation. This approach creates a trade-off between computation time and memory usage. While it significantly reduces memory requirements, it may increase overall computation time due to the need to recalculate some activations. Other parameters and techniques exist but play a minor role compared to those previously discussed. In the next section, we will explore how to select and tune these parameters using a concrete example.

##### Fine-tuning in practice

Let's now fine-tune an open-source model on our custom dataset. In this section, we will show an example that implements LoRA and QLoRA for efficiency. Depending on the hardware you have available, you can select the technique that best corresponds to your configuration. There are many efficient open-weight models we can leverage for task or domain-specific use cases. To select the most relevant LLM, we need to consider three main parameters:

-----

220 *Supervised Fine-Tuning*

---

- **License: Some model licenses only allow non-commercial work, which is a problem if**

we want to fine-tune for a company. Custom licenses are common in this field, and can target companies with a certain number of users, for example.

- **Budget: Models with smaller parameter sizes (<10 B) are a lot cheaper to fine-tune and**

deploy for inference than larger models. This is due to the fact that they can be run on cheaper GPUs and process more tokens per second.

- **Performance: Evaluating the base model on general-purpose benchmarks or, even better,**

domain- or task-specific benchmarks relevant to the final use case, is crucial. This helps ensure that the model has the necessary capabilities to perform well on the intended tasks after fine-tuning. In this chapter, we will choose Llama 3.1 8B, an open-weight model released by Meta. It has a permissive custom license ("Llama 3.1 Community License Agreement") that allows commercial use. With 8B parameters, it is small enough to fit on most GPUs while reaching a high level of performance compared to its competitors. We can verify this using the Open LLM Leaderboard, as well as other benchmarks detailed in the model card. There are specialized tools and libraries to fine-tune models. In particular, we recommend the following:

- **TRL: This is a library created and maintained by Hugging Face to train LLMs using SFT**

and preference alignment. It is a popular and reliable library that tends to be the most up-to-date in terms of algorithms. It works in single and multi-GPU settings with FSDP and DeepSpeed.

- **Axolotl: Created by Wing Lian, this tool streamlines the fine-tuning of LLMs with reusable**

YAML configuration files. It is based on TRL but includes many additional features, such as automatically combining datasets stored in various formats. It also supports single- and multi-GPU settings with FSDP and DeepSpeed.

- **Unsloth: Created by Daniel and Michael Han, Unsloth uses custom kernels to speed up**

training (2-5x) and reduce memory use (up to 80% less memory). It is based on TRL and provides many utilities, such as automatically converting models into the GGUF quantization format. At the time of writing, it is only available for single-GPU settings. To maximize efficiency, we will perform fine-tuning using the Unsloth library. The following code is designed as part of our LLMOps pipeline, but can also be used as a stand-alone script. It can also be executed in different environments, like SageMaker, cloud GPUs (like Lambda Labs or RunPod), Google Colab, and many others. We tested it on different GPUs, like A40, A100, and L4.

-----

*Chapter 5* 221

---

To install the Unsloth library and its dependencies, we recommend directly installing from the GitHub repository of the book (https://github.com/PacktPublishing/LLM-Engineering) or Unsloth's repo (https://github.com/unslothai/unsloth). This approach is recommended because the installation steps are regularly updated to address potential conflicts with dependencies:

1. First, we want to access a gated model and (optionally) upload our fine-tuned model to

Hugging Face (https://huggingface.co/). This requires being logged in to an account. If you don't have an account, you can create it and store your API key (Settings | Access

###### Tokens | Create new token) in the .env file:

| HF_TOKEN = YOUR_API_KEY |
|---|
| COMET_API_KEY = YOUR_API_KEY |
| import os import torch from trl import SFTTrainer from datasets import load_dataset, concatenate_datasets from transformers import TrainingArguments, TextStreamerfrom unsloth import FastLanguageModel, is_bfloat16_supported |
| FastLaguageModel class with the .from_pretrained() method. In addition to the mod- el name, we need to specify the max sequence length (2,048 in this example). Finally, the load_in_4bit argument indicates if we want to use QLoRA (quantized pre-trained weights) or LoRA. We'll use LoRA in this example because of faster training and higher quality, but you can easily switch to QLoRA if you don't meet the VRAM requirements. |
| max_seq_length = 2048 model, tokenizer = FastLanguageModel.from_pretrained( model_name="meta-llama/Meta-Llama-3.1-8B", max_seq_length=max_seq_length, load_in_4bit=False, ) |