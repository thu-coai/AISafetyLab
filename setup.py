from setuptools import setup, find_packages

setup(
    name="aisafetylab",
    version="0.0.dev",
    description="AI safety toolkits and resources",
    author="Zhexin Zhang, Leqi Lei, et al.",
    author_email="nonstopfor@gmail.com, leileqi@qq.com",
    url="https://github.com/thu-coai/AISafetyLab",
    packages=find_packages(include=('aisafetylab*',)),
    include_package_data=True,
    install_requires=[
        'transformers>=4.34.0',
        'sentencepiece',
        'datasets',
        'torch>=2.0',
        'openai>=1.0.0',
        'numpy',
        'pandas',
        'fschat',
        'nltk',
        'loguru',
        'jsonlines',
        'spacy',
        'tokenizers >= 0.13.3',
        'tensorboard',
        'wandb',
        'deepspeed',
        'accelerate',
        'optree',
        'scipy',
        'nvitop',
        'rich',
        'typing-extensions',
        'omegaconf',
        'anthropic',
        'tiktoken',
        'torchrl',
        'peft',
        'pytorch_lightning',
        'PyYAML',
        'Requests',
        'safetensors',
        'setproctitle',
        'torchrl',
        'hydra-core'
    ],
    python_requires=">=3.7",
    keywords=['ai safety', 
             ],
    license='GNU General Public License v3.0',
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3"
    ]
)