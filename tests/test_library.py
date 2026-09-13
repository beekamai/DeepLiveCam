import os
import tempfile
import unittest

from modules import library


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self._orig = library.LIBRARY_FILE
        self._dir = tempfile.TemporaryDirectory()
        library.LIBRARY_FILE = os.path.join(self._dir.name, "library.json")
        self.photo = os.path.join(self._dir.name, "a.png")
        open(self.photo, "wb").close()

    def tearDown(self):
        library.LIBRARY_FILE = self._orig
        self._dir.cleanup()

    def test_save_load_delete_round_trip(self):
        library.save("me", [self.photo, "missing.png"], {self.photo: (10, 20)})
        self.assertEqual(library.names(), ["me"])
        entry = library.load("me")
        self.assertEqual(entry["paths"], [self.photo])       # missing file dropped
        self.assertEqual(entry["picks"], {self.photo: [10, 20]})
        library.delete("me")
        self.assertEqual(library.names(), [])
        self.assertIsNone(library.load("me"))

    def test_empty_name_or_paths_are_ignored(self):
        library.save("  ", [self.photo])
        library.save("x", [])
        self.assertEqual(library.names(), [])


if __name__ == "__main__":
    unittest.main()
