# EN-BG Transformer Translation & Sentence Continuation Model

## Overview
This project is a **Transformer-based model** designed for **English-to-Bulgarian (EN-BG) translation**, with an additional capability to **generate sentence continuations before translation**. Built using **PyTorch**, it leverages a **decoder-only architecture** trained on a **180,000-sentence bilingual corpus**. The model employs **BPA-Dropout** for enhanced tokenization and achieves a **BLEU score of 35+**, ensuring high translation quality.

## Features
- **Sentence Continuation & Translation**: Given an English sentence, the model first **predicts its continuation** before translating it into Bulgarian.
- **Transformer Decoder Architecture**: A decoder-only approach optimized for sequential text generation.
- **High BLEU Score**: Achieves a **BLEU score of 35+**, ensuring reliable translation performance.
- **PyTorch Implementation**: Fully implemented in PyTorch for flexibility and efficiency.
