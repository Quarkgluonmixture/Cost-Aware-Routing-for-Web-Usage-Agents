class TokenEstimator:
    def __init__(self):
        # Rough estimation for Qwen/multimodal
        # 1 image token ~ 256 tokens (varies by resolution in Qwen2-VL, but keeping simple)
        self.image_token_cost = 1000 # Conservative upper bound for high res
        self.char_per_token = 3.5 

    def estimate(self, text: str, num_images: int = 0) -> int:
        text_tokens = len(text) / self.char_per_token
        image_tokens = num_images * self.image_token_cost
        return int(text_tokens + image_tokens)
