import unittest

from rag.chunker import chunk_pages


class ChunkerTests(unittest.TestCase):
    def test_chunks_have_overlap_and_page_metadata(self) -> None:
        pages = [{"source": "paper.pdf", "page": 3, "text": "0123456789ABCDEFGHIJ"}]
        chunks = chunk_pages(pages, chunk_size=10, overlap=3)

        self.assertEqual([chunk["text"] for chunk in chunks], ["0123456789", "789ABCDEFG", "EFGHIJ"])
        self.assertTrue(all(chunk["source"] == "paper.pdf" for chunk in chunks))
        self.assertTrue(all(chunk["page"] == 3 for chunk in chunks))
        self.assertEqual([chunk["chunk_id"] for chunk in chunks], [0, 1, 2])

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            chunk_pages([], chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
