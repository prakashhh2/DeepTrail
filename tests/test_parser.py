import unittest

from engine.parser import HTMLPageParser


class HTMLPageParserTests(unittest.TestCase):
    def test_extracts_title_visible_text_links_and_filters_non_html_assets(self):
        html = """
        <html>
          <head><title>International Scholarships</title><style>.x{}</style></head>
          <body>
            <script>hidden()</script>
            <h1>Scholarships for computer science students</h1>
            <a href="/financial-aid">Financial aid</a>
            <a href="https://example.edu/image.png">Image</a>
            <a href="mailto:test@example.edu">Email</a>
            <a href="/financial-aid#section">Duplicate aid link</a>
          </body>
        </html>
        """

        page = HTMLPageParser().parse(html, "https://example.edu/admissions", depth=0)

        self.assertEqual(page.title, "International Scholarships")
        self.assertIn("Scholarships for computer science students", page.text)
        self.assertNotIn("hidden()", page.text)
        self.assertEqual([link.url for link in page.links], ["https://example.edu/financial-aid"])
        self.assertEqual(page.links[0].anchor_text, "Financial aid")
        self.assertEqual(page.links[0].depth, 1)


if __name__ == "__main__":
    unittest.main()
