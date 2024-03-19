# Copyright 2024 X.AI Corp.
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

import logging

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model


CKPT_PATH = "./checkpoints/"


def main():
    grok_1_model = LanguageModelConfig( 
        vocab_size=128 * 1024, # 词表大小
        pad_token=0,  # 填充词元
        eos_token=2, # 结束词元
        sequence_len=8192, # 序列大小
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128, # 嵌入大小
            widening_factor=8, # 宽度因子
            key_size=128, # 键大小
            num_q_heads=48, #查询头数量
            num_kv_heads=8, #键值头数量
            num_layers=64, # 层数
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True, # 是否分片激活
            # MoE.
            num_experts=8, # 专家数量
            num_selected_experts=2, # 选择的专家数量
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )
    inference_runner = InferenceRunner(
        pad_sizes=(1024,), # 填充大小
        runner=ModelRunner(
            model=grok_1_model, # grok模型
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model", # 词元分析模型
        local_mesh_config=(1, 8), # 需要8个GPU
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()

    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=100, temperature=0.01))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
