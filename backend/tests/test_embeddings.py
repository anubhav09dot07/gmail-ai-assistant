import math
import unittest

from backend.app.embeddings import EmbeddingService


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, **kwargs):
        return [[float(len(text)), 1.0, -1.0] + [0.0] * 381 for text in texts]


class EmbeddingTests(unittest.TestCase):
    def test_single_and_batch_embeddings_are_numeric_and_finite(self) -> None:
        service = EmbeddingService(model=FakeEmbeddingModel())
        vector = service.embed_text("email chunk")
        vectors = service.embed_texts(["one", "two"])
        self.assertEqual(service.dimension, 384)
        self.assertEqual(len(vector), 384)
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(math.isfinite(value) for row in vectors for value in row))

    def test_empty_batch_does_not_load_model(self) -> None:
        service = EmbeddingService()
        self.assertEqual(service.embed_texts([]), [])
        self.assertIsNone(service._model)


if __name__ == "__main__":
    unittest.main()