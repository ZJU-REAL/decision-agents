# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_namespace_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0,<=2.50.0",
    "torchdata",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "transformers<=4.57.3",
    "wandb",
    "packaging>=20.0",
    "qwen-vl-utils[decord]",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy"]
PRIME_REQUIRES = ["pyext"]
GEO_REQUIRES = ["mathruler"]
GPU_REQUIRES = ["liger-kernel", "flash-attn"]
MATH_REQUIRES = ["math-verify"]  # Add math-verify as an optional dependency
VLLM_REQUIRES = ["tensordict>=0.8.0,<=0.10.0,!=0.9.0", "vllm>=0.8.5,<=0.11.0"]
SGLANG_REQUIRES = [
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "sglang[srt,openai]==0.5.5",
    "torch==2.8.0",
]
TRL_REQUIRES = ["trl<=0.9.6"]
MCORE_REQUIRES = ["mbridge"]
TRANSFERQUEUE_REQUIRES = ["TransferQueue==0.1.2.dev0"]

extras_require = {
    "test": TEST_REQUIRES,
    "prime": PRIME_REQUIRES,
    "geo": GEO_REQUIRES,
    "gpu": GPU_REQUIRES,
    "math": MATH_REQUIRES,
    "vllm": VLLM_REQUIRES,
    "sglang": SGLANG_REQUIRES,
    "trl": TRL_REQUIRES,
    "mcore": MCORE_REQUIRES,
    "transferqueue": TRANSFERQUEUE_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Environment adapters and GiGPO use native namespace packages. Limit discovery
# to product roots; never package research directories or test drivers.
product_packages = [name for name in find_namespace_packages(
    where=".", include=("decision_model", "decision_model.*", "verl", "verl.*",
                        "agent_system", "agent_system.*", "gigpo", "gigpo.*",
                        "recipe", "recipe.*"),
    exclude=("*.__pycache__", "*.__pycache__.*", "*.tests", "*.tests.*"),
) if all(part.isidentifier() for part in name.split("."))]

# Keep vendored environment source and runtime assets, including directories
# whose names are not Python identifiers (gym-cards). No downloaded datasets,
# notebook outputs, browser binaries, logs or bytecode enter the wheel.
asset_suffixes = {".py", ".yaml", ".json", ".npy", ".png", ".pddl", ".twl2", ".html",
                  ".css", ".ttf", ".c", ".h", ".l", ".y", ".txt"}
# ALFWorld ships these small domain definitions as runtime resources, not
# downloaded training data. Keep them even though their parent is named data.
builtin_environment_data = {
    "environments/env_package/alfworld/alfworld/data/alfred.pddl",
    "environments/env_package/alfworld/alfworld/data/alfred.twl2",
}
environment_assets = [path.relative_to(this_directory / "agent_system").as_posix()
    for path in sorted((this_directory / "agent_system").rglob("*"))
    if path.is_file() and not {"__pycache__", "logs", ".git"}.intersection(
        path.relative_to(this_directory / "agent_system").parts)
    and ("data" not in path.relative_to(this_directory / "agent_system").parts
         or path.relative_to(this_directory / "agent_system").as_posix() in builtin_environment_data)
    and (path.suffix in asset_suffixes or path.name.startswith("LICENSE") or path.name == "Makefile")]

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=product_packages,
    url="https://github.com/volcengine/verl",
    license="Apache 2.0",
    author="Bytedance - Seed - MLSys",
    author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk",
    description="verl: Volcano Engine Reinforcement Learning for LLM",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
        "agent_system": environment_assets,
        "recipe": ["**/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
