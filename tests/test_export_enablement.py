import unittest

from cove_video_editor.app import export_controls_enabled


class TestExportControlsEnabled(unittest.TestCase):
    def test_clips_only_project_enabled(self):
        self.assertTrue(export_controls_enabled(
            has_clips=True, has_added_audio=False,
            audio_only=False, exporting=False,
        ))

    def test_added_audio_only_project_disabled(self):
        self.assertFalse(export_controls_enabled(
            has_clips=False, has_added_audio=True,
            audio_only=False, exporting=False,
        ))

    def test_added_audio_only_audio_only_enabled(self):
        self.assertTrue(export_controls_enabled(
            has_clips=False, has_added_audio=True,
            audio_only=True, exporting=False,
        ))

    def test_empty_audio_only_disabled(self):
        self.assertFalse(export_controls_enabled(
            has_clips=False, has_added_audio=False,
            audio_only=True, exporting=False,
        ))

    def test_clips_audio_only_enabled(self):
        self.assertTrue(export_controls_enabled(
            has_clips=True, has_added_audio=False,
            audio_only=True, exporting=False,
        ))

    def test_exporting_disables_otherwise_enabled_states(self):
        self.assertFalse(export_controls_enabled(
            has_clips=True, has_added_audio=False,
            audio_only=False, exporting=True,
        ))
        self.assertFalse(export_controls_enabled(
            has_clips=False, has_added_audio=True,
            audio_only=True, exporting=True,
        ))
        self.assertFalse(export_controls_enabled(
            has_clips=True, has_added_audio=True,
            audio_only=True, exporting=True,
        ))

    def test_subtitles_mode_needs_clips(self):
        self.assertTrue(export_controls_enabled(
            has_clips=True, has_added_audio=False,
            audio_only=False, exporting=False,
            subtitles_only=True,
        ))
        self.assertFalse(export_controls_enabled(
            has_clips=False, has_added_audio=False,
            audio_only=False, exporting=False,
            subtitles_only=True,
        ))
        self.assertFalse(export_controls_enabled(
            has_clips=True, has_added_audio=False,
            audio_only=False, exporting=True,
            subtitles_only=True,
        ))


if __name__ == "__main__":
    unittest.main()
