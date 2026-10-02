import unittest

from backend.app.query_normalizer import expand_query, normalize_query


class QueryNormalizerTests(unittest.TestCase):
    def test_common_email_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_query("recent mail msg signin"), "recent email message sign-in")

    def test_abbreviation_expands_with_corpus_term(self) -> None:
        variants = expand_query("recent mail for IG login", ["Instagram login alert"])
        self.assertTrue(any("instagram" in variant for variant in variants))

    def test_abbreviation_stays_unresolved_without_corpus_support(self) -> None:
        variants = expand_query("recent mail for IG login", ["Google security alert"])
        self.assertFalse(any("instagram" in variant for variant in variants))

    def test_corpus_typos_get_conservative_corrections(self) -> None:
        variants = expand_query(
            "recent instagarm securty statment",
            ["Instagram security statement"],
        )
        self.assertTrue(any("instagram" in variant for variant in variants))
        self.assertTrue(any("security" in variant for variant in variants))
        self.assertTrue(any("statement" in variant for variant in variants))

    def test_original_query_is_preserved_as_first_variant(self) -> None:
        self.assertEqual(expand_query("Recent mail") [0], "Recent mail")


if __name__ == "__main__":
    unittest.main()
