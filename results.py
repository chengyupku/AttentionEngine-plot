deepseek_fwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", "BS1S1\nKV2048", "BS1S1\nKV4096", "BS1S1\nKV8192", "BS8S1\nKV2048", "BS8S1\nKV4096", "BS8S1\nKV8192"]
deepseek_fwd_times_data = [
    ("AttnForge", [1/316, 1/459, 1/539, 1/472, 1/556, 1/592, 0.03, 0.04, 0.05, 0.08, 0.13, 0.23, ]),
    ("FlashAttention-2", [1/120, 1/179, 1/218, 1/161, 1/223, 1/249, 0.05, 0.08, 0.13, 0.23, 0.44, 0.85, ]),
    ("FlashAttention-3", [1/115, 1/205, 1/314, 1/184, 1/254, 1/341, 0.1, 0.18, 0.33, 0.49, 0.93, 1.83, ]),
    ("FlexAttention", [1/64, 1/88, 1/107, 1/80, 1/101, 1/115, 0, 0, 0, 0, 0, 0]),
    ("FlashInfer", [1/80, 1/121, 1/158, 1/155, 1/167, 1/175, 0, 0, 0, 0, 0, 0]),
    ("PyTorch Inductor", [1/74, 1/54, 1/57, 1/49, 1/48, 1/50, 0.1, 0.11, 0.1, 0.2, 0.38, 0.73, ]),
]

deepseek_bwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
deepseek_bwd_times_data = [
    ("AttnForge", [1/140, 1/189, 1/228, 1/178, 1/211, 1/231, ]),
    ("FlashAttention-2", [1/112, 1/157, 1/199, 1/134, 1/174, 1/202, ]),
    ("FlexAttention", [1/49, 1/63, 1/72, 1/55, 1/66, 1/73, ]),
    ("PyTorch Inductor", [1/99, 1/111, 1/113, 1/100, 1/99, 1/97, ]),
]

llama_fwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
llama_fwd_times_data = [
    ("AttnForge", [1/387, 1/493, 1/553, 1/484, 1/550, 1/583, ]),
    ("FlashAttention-2", [1/214, 1/285, 1/330, 1/304, 1/334, 1/356, ]),
    ("FlashAttention-3", [1/387, 1/493, 1/553, 1/484, 1/550, 1/583, ]),
    ("FlexAttention", [1/278, 1/361, 1/441, 1/348, 1/394, 1/428, ]),
    ("FlashInfer", [1/220, 1/293, 1/331, 1/264, 1/289, 1/301, ]),
    ("PyTorch Inductor", [1/65, 1/45, 1/42, 1/42, 1/41, 0]),
]

llama_bwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
llama_bwd_times_data = [
    ("AttnForge", [1/127, 1/312, 1/355, 1/263, 1/321, 1/323, ]),
    ("FlashAttention-2", [1/177, 1/239, 1/284, 1/220, 1/263, 1/297, ]),
    ("FlashAttention-3", [1/249, 1/360, 1/396, 1/304, 1/372, 1/436, ]),
    ("FlexAttention", [1/202, 1/267, 1/310, 1/236, 1/286, 1/316,]),
    ("PyTorch Inductor", [1/91, 1/96, 1/86, 1/87, 1/83, 0, ]),
]


dit_fwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
dit_fwd_times_data = [
    ("AttnForge", [1/219, 1/443, 1/540, 1/470, 1/554, 1/607, ]),
    ("FlashAttention-2", [1/109, 1/154, 1/196, 1/116, 1/199, 1/227, ]),
    ("FlashAttention-3", [1/146, 1/275, 1/377, 1/243, 1/357, 1/449, ]),
    ("FlexAttention", [1/80, 1/109, 1/131, 1/107, 1/128, 1/142, ]),
    ("FlashInfer", [1/103, 1/154, 1/191, 1/184, 1/200, 1/209, ]),
    ("PyTorch Inductor", [1/83, 1/62, 1/66, 1/57, 1/63, 1/57, ]),
]

dit_bwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
dit_bwd_times_data = [
    ("AttnForge", [1/148, 1/203, 1/253, 1/196, 1/233, 1/258, ]),
    ("FlashAttention-2", [1/101, 1/138, 1/173, 1/132, 1/158, 1/175, ]),
    ("FlexAttention", [1/56, 1/71, 1/80, 1/66, 1/76, 1/82, ]),
    ("PyTorch Inductor", [1/99, 1/108, 1/111, 1/102, 1/112, 1/105, ]),
]

sigmoid_fwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
sigmoid_fwd_times_data = [
    ("AttnForge", [1/417, 1/567, 1/587, 1/511, 1/566, 1/602, ]),
    ("FlashSigmoid", [1/247, 1/338, 1/379, 1/355, 1/384, 1/401, ]),
    ("FlashInfer", [1/170, 1/176, 1/179, 0, 0, 0]),
    ("PyTorch Inductor", [1/56, 1/56, 1/56, 1/57, 1/57, 0, ]),
]

sigmoid_bwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
sigmoid_bwd_times_data = [
    ("AttnForge", [1/187, 1/317, 1/351, 1/266, 1/318, 1/353, ]),
    ("FlashSigmoid", [1/178, 1/245, 1/291, 1/222, 1/274, 1/301, ]),
    ("PyTorch Inductor", [1/90, 1/101, 1/104, 1/72, 1/62, 0, ]),
]

relu_fwd_providers = ["BS32\nS512", "BS32\nS1024", "BS32\nS2048", "BS64\nS512", "BS64\nS1024", "BS64\nS2048"]
relu_fwd_times_data = [
    ("AttnForge", [1/133, 1/446, 1/542, 1/264, 1/474, 1/537, ]),
    # ("PyTorch", [1/35, 1/45, 1/51, 1/38, 1/46, 1/51, ]),
    ("PyTorch Inductor", [1/52, 1/63, 1/70, 1/53, 1/61, 1/66, ]),
]

relu_bwd_providers = ["BS32\nS512", "BS32\nS1024", "BS32\nS2048", "BS64\nS512", "BS64\nS1024", "BS64\nS2048"]
relu_bwd_times_data = [
    ("AttnForge", [1/64, 1/255, 1/319, 1/127, 1/267, 1/327, ]),
    # ("PyTorch", [1/60, 1/77, 1/89, 1/64, 1/79, 1/89, ]),
    ("PyTorch Inductor", [1/73, 1/97, 1/115, 1/75, 1/95, 1/110, ]),
]

retnet_fwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS8\nS2048", "BS8\nS4096"]
retnet_fwd_times_data = [
    ("AttnForge", [1/152, 1/165, 1/158, 1/169, ]),
    # ("PyTorch", [1/66, 1/69, 1/55, 1/59, ]),
    ("PyTorch Inductor", [1/123, 1/113, 1/101, 1/105, ]),
]

# retnet_bwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS8\nS2048", "BS8\nS4096"]
# retnet_bwd_times_data = [
#     ("AttnForge", []),
#     ("PyTorch Inductor", []),
# ]

mamba_fwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
mamba_fwd_times_data = [
    ("AttnForge", [0.32, 0.33, 0.3	, 0.48, 0.94, 1.87, ]),
    ("Flash-Linear-Attention", [0.66, 0.7, 0.65, 0.78, 1.78, 3.9, ]),
    ("PyTorch Inductor", [0.84, 2.09, 4, 8, 15.9, 32, ]),
]

mamba_bwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192", ]
mamba_bwd_times_data = [
    ("AttnForge", [1.07, 1.11, 1.67, 2.99, 5.84, 11.5, ]),
    ("Flash-Linear-Attention", [2.85, 2.69, 2.79, 3, 6.33, 13, 	]),
    ("PyTorch Inductor", [1.46, 2.97, 5.62, 11.11, 21.7, 43.7, ]),
]

gla_fwd_providers = ["BS64\nS1024", "BS64\nS2048", "BS64\nS4096"]
gla_fwd_times_data = [
    ("AttnForge", [0.38, 0.74, 1.47, ]),
    ("Flash-Linear-Attention", [0.4, 0.78, 1.53, ]),
    ("PyTorch Inductor", [3.52, 7.01, 15.16, ]),
]


gla_bwd_providers = ["BS64\nS1024", "BS64\nS2048", "BS64\nS4096"]
gla_bwd_times_data = [
    ("AttnForge", [1.59, 3.11, 6.16, ]),
    ("Flash-Linear-Attention", [0, 0, 0]),
    ("PyTorch Inductor", [6.59, 13.13, 27.52, ]),
]


gated_retnet_fwd_providers = ["BS8\nS1024", "BS8\nS2048", "BS8\nS4096"]
gated_retnet_fwd_times_data = [
    ("AttnForge", [0.59, 1.16, 2.31, ]),
    ("Flash-Linear-Attention", [0.96, 1.89, 3.74, ]),
    ("PyTorch Inductor", [1.85, 5.04, 10.36, ]),
]

gated_retnet_bwd_providers = ["BS8\nS1024", "BS8\nS2048", "BS8\nS4096"]
gated_retnet_bwd_times_data = [
    ("AttnForge", [2.3	, 4.53, 9.03, ]),
    ("Flash-Linear-Attention", [4.39, 8.75, 17.6, ]),
    ("PyTorch Inductor", [3.86, 9.41, 19.27, ]),
]

retnet_chunk_fwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS8\nS2048", "BS8\nS4096"]
retnet_chunk_fwd_times_data = [
    ("AttnForge", [0.28, 0.42, 1.59, 3.16, ]),
    ("Flash-Linear-Attention", [0.4, 0.78, 3.02, 6.04, ]),
    ("PyTorch Inductor", [0.65, 1.74, 6.07, 12.77, ]),
]

retnet_chunk_bwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS8\nS2048", "BS8\nS4096"]
retnet_chunk_bwd_times_data = [
    ("AttnForge", [1.05, 1.46, 5.46, 10.88, ]),
    ("Flash-Linear-Attention", [1.11, 2.17, 8.51, 20.34, ]),
    ("PyTorch Inductor", [1.15, 2.83, 10.02, 21.08, ]),
]

# flash_decoding_fwd_providers = ["BS1\nS2048", "BS1\nS4096", "BS1\nS8192", "BS8\nS2048", "BS8\nS4096", "BS8\nS8192"]
# flash_decoding_fwd_times_data = [
#     ("AttnForge", [0.03, 0.04, 0.05, 0.08, 0.13, 0.23, ]),
#     ("FlashAttention-2", [0.05, 0.08, 0.13, 0.23, 0.44, 0.85, ]),
#     ("FlashAttention-3", [0.1, 0.18, 0.33, 0.49, 0.93, 1.83, ]),
# ]