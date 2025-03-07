import math
import torch
import torch.nn as nn
import parameters as params
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017).
    Injects position information into the embeddings for a decoder-only LM.
    """
    def __init__(self, embed_size, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, embed_size)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_size)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :] # (batch_size, seq_len, embed_size)
        return self.dropout(x)


class LanguageModel(torch.nn.Module):
	def __init__(self,  word2ind, startToken, endToken, unkToken, padToken, transToken):
		super(LanguageModel, self).__init__()


		
		self.word2ind  = word2ind

		self.startTokenIdx = word2ind[startToken]
		self.endTokenIdx = word2ind[endToken]
		self.unkTokenIdx = word2ind[unkToken]
		self.padTokenIdx = word2ind[padToken]
		self.transTokenIdx = word2ind[transToken]

		self.vocab_size = len(word2ind)
		self.embed_size  = params.embed_size
		self.dropout = params.dropout
		self.nhead = params.nhead
		self.dim_feedforward = params.dim_feedforward
		self.decoder_layers = params.decoder_layers

		self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_size,
            padding_idx=self.padTokenIdx
        )

		self.position_encoding = PositionalEncoding(self.embed_size, dropout=self.dropout)

		decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_size,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='relu',
			batch_first=True
        )

		self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.decoder_layers,
        )

		self.out_projection = nn.Linear(self.embed_size, self.vocab_size)

	
	def preparePaddedBatch(self, source):
		device = next(self.parameters()).device
		m = max(len(s) for s in source)
		sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in source]
		return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)
	
	def generate_square_subsequent_mask(self, seq_len):
		"""
        Creates a causal (no peek) mask of shape (seq_len, seq_len),
        with -inf where future tokens should not be attended.
        """
		device = next(self.parameters()).device
		mask = torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1  # upper triangular
		mask = mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)
		# 0 -inf  -inf
		# 0    0  -inf
		# 0    0    0
		return mask

	def save(self,fileName):
		torch.save(self.state_dict(), fileName)

	def load(self,fileName):
		self.load_state_dict(torch.load(fileName,map_location=params.device))

	def forward(self, source):
		padded_batch = self.preparePaddedBatch(source) # (batch_size, seq_len)
		batch_size, seq_len = padded_batch.shape
		# The sequence length may change if I change the tokenization strategy

		embedded_batch = self.embedding(padded_batch) # (batch_size, seq_len, embed_size)
		
		embedded_position_encoded_batch = self.position_encoding(embedded_batch) # (batch_size, seq_len, embed_size)

		mask = self.generate_square_subsequent_mask(seq_len) # (seq_len, seq_len)

		key_padding_mask = (padded_batch == self.padTokenIdx).to(dtype=torch.bool) # (batch_size, seq_len)

		memory = torch.zeros(batch_size, 1, self.embed_size, device=next(self.parameters()).device) # (batch_size, 1, embed_size) 
		
		decoder_output = self.decoder(
			tgt=embedded_position_encoded_batch,
			memory=memory,
			tgt_mask=mask,
			tgt_key_padding_mask=key_padding_mask
		)
		# shape: (batch_size, seq_len, embed_size)

		logits  = self.out_projection(decoder_output)   # (batch_size, seq_len, vocab_size)

		logits  = logits[:, :-1, :]   # (batch_size, seq_len-1, vocab_size)
		targets = padded_batch[:, 1:] # (batch_size, seq_len-1)

        # # flatten
		logits_flat = logits.reshape(-1, self.vocab_size)
		targets_flat= targets.reshape(-1)

        # # cross-entropy ignoring <PAD>
		loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.padTokenIdx)
		return loss
		
	def generate(self, prefix, limit=1000):
		"""
		Greedy generation from a partial prefix (list of token IDs).
		Args:
			prefix (list[int]): A list of token IDs as a starting prefix.
			limit (int): Max length to generate.

		Returns: A list of token IDs (including the prefix and newly generated tokens).
		"""
		device = next(self.parameters()).device
		self.eval()

		if not prefix:
			prefix = [self.startTokenIdx]

		generated = list(prefix)

		with torch.no_grad():
			for _ in range(limit):
				inp = torch.tensor([generated], dtype=torch.long, device=device)
				seq_len = inp.size(1)

				emb = self.embedding(inp)  # (1, seq_len, embed_size)
				emb = self.position_encoding(emb)  # same shape

				tgt_mask = self.generate_square_subsequent_mask(seq_len)  # shape (seq_len, seq_len)

				tgt_key_padding_mask = (inp == self.padTokenIdx).bool()  # shape (1, seq_len)

				memory = torch.zeros(
					1, 1, self.embed_size,
					device=device
				)  # shape (1, 1, embed_size) since batch_first=True

				dec_out = self.decoder(
					tgt=emb,                # (B=1, seq_len, embed_size)
					memory=memory,          # (B=1, mem_seq_len=1, embed_size)
					tgt_mask=tgt_mask,      # (seq_len, seq_len)
					tgt_key_padding_mask=tgt_key_padding_mask  # (B=1, seq_len)
				)
				# dec_out shape: (1, seq_len, embed_size)

				last_hidden = dec_out[:, -1, :]  # shape: (1, embed_size)
				logits = self.out_projection(last_hidden)  # shape: (1, vocab_size)

				next_token = torch.argmax(logits, dim=-1).item()

				generated.append(next_token)

				if next_token == self.endTokenIdx:
					break

		return generated
	
	def generate_beam(self, prefix, beam_size=3, limit=1000):
		"""
		Beam search generation for a decoder-only Transformer.
		
		Args:
			prefix (list[int]): A list of token IDs as a starting prefix.
			beam_size (int): Number of candidates to keep at each step.
			limit (int): Max length to generate.

		Returns:
			best_seq (list[int]): The best sequence found (token IDs).
		"""
		device = next(self.parameters()).device
		self.eval()

		if not prefix:
			prefix = [self.startTokenIdx]

		beam = [(prefix, 0.0)]
		end_token = self.endTokenIdx

		with torch.no_grad():
			for _ in range(limit):
				new_beam = []

				for (tokens, cum_logprob) in beam:
					if tokens[-1] == end_token:
						new_beam.append((tokens, cum_logprob))
						continue

					inp = torch.tensor([tokens], dtype=torch.long, device=device)
					seq_len = inp.size(1)

					emb = self.embedding(inp)
					emb = self.position_encoding(emb)

					tgt_mask = self.generate_square_subsequent_mask(seq_len)

					tgt_key_padding_mask = (inp == self.padTokenIdx).bool()

					memory = torch.zeros(
						1, 1, self.embed_size,
						device=device
					)

					dec_out = self.decoder(
						tgt=emb,
						memory=memory,
						tgt_mask=tgt_mask,
						tgt_key_padding_mask=tgt_key_padding_mask
					)
					# dec_out shape: (1, seq_len, embed_size)

					last_hidden = dec_out[:, -1, :]
					logits = self.out_projection(last_hidden)

					log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # shape (vocab_size,)

					topk = torch.topk(log_probs, beam_size)
					top_tokens = topk.indices.tolist()
					top_scores = topk.values.tolist()

					for i in range(beam_size):
						tok_id = top_tokens[i]
						tok_logprob = top_scores[i]

						new_seq = tokens + [tok_id]
						new_cum_logprob = cum_logprob + tok_logprob
						new_beam.append((new_seq, new_cum_logprob))

				new_beam.sort(key=lambda x: x[1], reverse=True)

				beam = new_beam[:beam_size]

				if all(hyp[0][-1] == end_token for hyp in beam):
					break

		beam.sort(key=lambda x: x[1], reverse=True)
		best_seq, best_logp = beam[0]

		return best_seq


