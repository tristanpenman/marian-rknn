class Counters:
    """Capture timing and token statistics for an inference pass."""

    def __init__(self):
        self.total_ms = 0.0
        self.encoder_ms = 0.0
        self.decoder_ms = 0.0
        self.lm_head_ms = 0.0
        self.decoder_iterations = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self):
        self.total_ms = 0.0
        self.encoder_ms = 0.0
        self.decoder_ms = 0.0
        self.lm_head_ms = 0.0
        self.decoder_iterations = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def accumulate(self, other):
        self.total_ms += other.total_ms
        self.encoder_ms += other.encoder_ms
        self.decoder_ms += other.decoder_ms
        self.lm_head_ms += other.lm_head_ms
        self.decoder_iterations += other.decoder_iterations
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
